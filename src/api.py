"""Public API for sajepa — Scale-Aware JEPA pipeline."""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Optional

import numpy as np
import torch
import yaml

from src.train import load_config, run_training
from src.utils.memory import OOMSafeTrainer, clear_memory_cache


class ScaleAwareJEPA:
    """Scale-Aware Joint-Embedding Predictive Architecture for physical fields.

    Usage:
        model = ScaleAwareJEPA(config="configs/mhd_turbulence.yaml")
        model.fit(field, epochs=10)
        latent = model.extract(field)
        model.save_session("sessions/my_run")
        # Later:
        model2 = ScaleAwareJEPA.load_session("sessions/my_run")
        latent2 = model2.extract(new_field)
    """

    def __init__(self, config: Optional[dict | str] = None):
        self._config = self._parse_config(config)
        self._session_dir: Optional[str] = None
        self._is_trained: bool = False

    # ── training ────────────────────────────────────────────────

    def fit(self, field: torch.Tensor, epochs: Optional[int] = None) -> "ScaleAwareJEPA":
        """Train on a raw physical field.  Returns self for chaining."""
        cfg = copy.deepcopy(self._config)
        if epochs is not None:
            cfg.setdefault("training", cfg.setdefault("train", {}))["epochs"] = int(epochs)

        sessions_dir = tempfile.mkdtemp(prefix="sajepa_")
        data_dir = os.path.join(sessions_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "_input.npy")
        arr = field.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        np.save(data_path, arr)
        cfg.setdefault("data", {})["npy_pattern"] = "_input.npy"

        # --- auto-batch OOM handling ---
        train_cfg = cfg.setdefault("train", cfg.setdefault("training", {}))
        trainer = OOMSafeTrainer(
            initial_batch=int(train_cfg.get("batch_size", 4)),
            target_batch=int(train_cfg.get("target_batch_size", train_cfg.get("target_batch", 32))),
            scale_mode=str(train_cfg.get("auto_scale_batch_size", "power_of_two")),
            max_retries=int(train_cfg.get("oom_max_retries", 5)),
        )
        train_cfg["batch_size"] = trainer.batch_size
        train_cfg["gradient_accumulation_steps"] = trainer.accumulation_steps

        if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if not torch.cuda.is_available():
            train_cfg["num_workers"] = 0

        print(f"[sajepa] batch={trainer.batch_size} accum={trainer.accumulation_steps}")

        old_cwd = os.getcwd()
        os.chdir(sessions_dir)
        session_dir = None
        try:
            while True:
                try:
                    session_dir = run_training(cfg, config_name="sajepa", sessions_root=".")
                    break
                except RuntimeError as e:
                    if not trainer.handle_oom(e):
                        raise
                    train_cfg["batch_size"] = trainer.batch_size
                    train_cfg["gradient_accumulation_steps"] = trainer.accumulation_steps
                    clear_memory_cache()
        finally:
            self._session_dir = os.path.abspath(session_dir) if session_dir else None
            os.chdir(old_cwd)

        if self._session_dir is None:
            raise RuntimeError("Training failed after all OOM retries.")
        self._is_trained = True
        _print_metrics_summary(self._session_dir)
        return self

    # ── inference ───────────────────────────────────────────────

    @torch.no_grad()
    def project(self, field: torch.Tensor, method: str = "umap") -> dict:
        """Extract latents and project to 2D with fallback.

        Always returns a dict with at least ``"pca"``; ``"umap"`` may be None
        if GPU/torchdr is unavailable or fails.
        """
        latent = self.extract(field)
        C, H, W = latent.shape
        flat = latent.permute(1, 2, 0).reshape(H * W, C).float()

        results: dict[str, Optional[np.ndarray]] = {"pca": None, "umap": None}

        # PCA — always works, always runs
        try:
            _, _, V = torch.pca_lowrank(flat, q=min(2, C))
            results["pca"] = torch.matmul(flat, V[:, :2]).cpu().numpy()
        except Exception as e:
            print(f"[sajepa] PCA fallback failed: {e}")

        # UMAP — best-effort
        if method.lower() == "umap":
            try:
                import torchdr
                reducer = torchdr.UMAP(
                    n_neighbors=self._config.get("diagnostics", {}).get("umap", {}).get("n_neighbors", 50),
                    min_dist=self._config.get("diagnostics", {}).get("umap", {}).get("min_dist", 0.2),
                    device=str(self._get_device()),
                )
                emb = reducer.fit_transform(flat.to(self._get_device()))
                results["umap"] = emb.cpu().numpy()
            except Exception as e:
                print(f"[sajepa] UMAP unavailable ({type(e).__name__}), PCA only.")

        return results

    @torch.no_grad()
    def extract(self, field: torch.Tensor) -> torch.Tensor:
        """Return pixel-registered latent atlas for the given field.  No training."""
        if self._session_dir is None:
            raise RuntimeError("No session available. Call fit() first or load_session().")
        inf_path = os.path.join(self._session_dir, "inference_outputs.pt")
        if not os.path.exists(inf_path):
            # Run inference if not already done
            if not self._is_trained:
                raise RuntimeError("Model not trained. Call fit() before extract().")
            raise RuntimeError("No inference outputs found — call fit() first.")
        outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
        ctx = outputs.get("context_map")
        if ctx is None:
            ctx = outputs.get("pred_map")
        if ctx is None:
            raise RuntimeError("No latent map in inference outputs.")
        return ctx.squeeze(0).cpu()

    # ── persistence ─────────────────────────────────────────────

    def save_session(self, path: str):
        """Save model weights, config, and session artifacts to *path*."""
        if self._session_dir is None:
            raise RuntimeError("No session to save. Call fit() first.")
        import shutil
        os.makedirs(path, exist_ok=True)
        for name in ("config_used.json", "model_last.pt", "metrics.csv",
                     "inference_outputs.pt", "dashboard.html"):
            src = os.path.join(self._session_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(path, name))
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)
        print(f"[sajepa] session saved to {path}")

    @classmethod
    def load_session(cls, path: str) -> "ScaleAwareJEPA":
        """Restore a model from a saved session directory."""
        cfg_path = os.path.join(path, "config.yaml")
        if os.path.exists(cfg_path):
            instance = cls(config=cfg_path)
        else:
            instance = cls(config=os.path.join(path, "config_used.json"))
        instance._session_dir = os.path.abspath(path)
        instance._is_trained = os.path.exists(os.path.join(path, "inference_outputs.pt"))
        return instance

    # ── diagnostics ────────────────────────────────────────────

    def analyze_rank(self) -> dict:
        """Return effective-rank diagnostics for the current session."""
        if self._session_dir is None:
            raise RuntimeError("No session. Call fit() or load_session() first.")
        from scripts.print_effective_rank import rank_summary
        rows = rank_summary([self._session_dir])
        if not rows:
            return {}
        cols = [
            "session", "mode", "mask_scale", "mask_box", "sampling",
            "l2_norm", "psnorm", "final_norm", "sig_type", "sig_spatial_mode", "sig_w", "sig_t",
            "vicreg_var_weight", "vicreg_cov_weight", "sym_loss", "depth", "dilations", "hardcap",
            "energy", "sim_r", "hinge_r", "sig_r",
            "vicreg_var_r", "vicreg_cov_r", "weighted_vicreg_var_r", "weighted_vicreg_cov_r",
            "erank", "context_erank", "predictor_erank", "target_erank",
            "top1", "pred_part", "target_part", "part_ratio", "dead_frac", "dead_ch",
        ]
        return dict(zip(cols, rows[0]))

    # ── dashboard ───────────────────────────────────────────────

    def generate_dashboard(self, output_path: Optional[str] = None):
        """Generate interactive HTML dashboard from the current session.

        If session already has dash artifacts (from post-training inference),
        uses those.  Otherwise falls back to session_to_dash.py.
        """
        if self._session_dir is None:
            raise RuntimeError("No session. Call fit() or load_session() first.")
        try:
            from scripts.session_to_dash import compute_dash_data, plot_dash
            compute_dash_data(self._session_dir, overwrite=False)
            plot_dash(self._session_dir, overwrite=False)
            dash = os.path.join(self._session_dir, "dashboard.html")
            if output_path and os.path.exists(dash):
                import shutil
                shutil.copy2(dash, output_path)
            print(f"[sajepa] dashboard: {output_path or dash}")
        except Exception as e:
            print(f"[sajepa] dashboard failed ({type(e).__name__}), generating minimal dashboard...")
            _generate_minimal_dashboard(self._session_dir, output_path)

    # ── internals ───────────────────────────────────────────────

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _parse_config(config: Optional[dict | str]) -> dict:
        if config is None:
            return load_config(os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "configs", "base_pyramid_scaleaware_convnext.yaml"))
        if isinstance(config, str):
            return load_config(config)
        base = load_config(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "base_pyramid_scaleaware_convnext.yaml"))
        return _deep_merge(base, config)


def _print_metrics_summary(session_dir: str) -> None:
    import csv as _csv
    path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    try:
        epochs: dict[int, dict[str, list[float]]] = {}
        with open(path, "r") as f:
            for row in _csv.DictReader(f):
                ep = int(row.get("epoch", -1))
                if ep < 0:
                    continue
                if ep not in epochs:
                    epochs[ep] = {}
                for k in ("loss_total", "loss_prediction", "loss_spread", "sim", "var", "cov", "lr"):
                    v = row.get(k, "")
                    if v:
                        epochs[ep].setdefault(k, []).append(float(v))
        if not epochs:
            return
        last_ep, first_ep = max(epochs.keys()), min(epochs.keys())
        print(f"\n{'='*60}")
        print(f"Training Metrics (epoch {first_ep} → {last_ep})")
        print(f"{'='*60}")
        keys = [
            ("loss_total", "L(total)     "), ("loss_prediction", "MSE(pred,gt) "),
            ("loss_spread", "sig=relu(1-std)"), ("sim", "cos(pred,gt) "),
            ("var", "var_term    "), ("cov", "cov_term    "), ("lr", "lr          "),
        ]
        for key, label in keys:
            vals_f = epochs[first_ep].get(key, [])
            vals_l = epochs[last_ep].get(key, [])
            first = sum(vals_f) / len(vals_f) if vals_f else 0.0
            last = sum(vals_l) / len(vals_l) if vals_l else 0.0
            ratio = last / first if first > 1e-20 else 1.0
            print(f"  {label}: {first:>8.4f} → {last:>8.4f}  (ratio={ratio:.3f})")
        print(f"{'='*60}")
    except Exception:
        pass


def _generate_minimal_dashboard(session_dir: str, output_path: Optional[str] = None) -> None:
    """Fallback dashboard: reads whatever artifacts exist and renders a simple HTML."""
    dash_path = output_path or os.path.join(session_dir, "dashboard.html")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as _np

        # Try loading what's available
        inf_path = os.path.join(session_dir, "inference_outputs.pt")
        has_inference = os.path.exists(inf_path)
        outputs = torch.load(inf_path, map_location="cpu", weights_only=False) if has_inference else {}

        fig = make_subplots(rows=1, cols=1, subplot_titles=["sajepa Session"])
        if has_inference:
            ctx = outputs.get("context_map")
            if ctx is not None:
                img = ctx.squeeze().cpu().numpy()
                if img.ndim == 3:
                    img = img.mean(0)
                fig.add_trace(go.Heatmap(z=img, colorscale="Viridis"), row=1, col=1)
        fig.write_html(dash_path)
        print(f"[sajepa] minimal dashboard saved: {dash_path}")
    except Exception as e:
        print(f"[sajepa] minimal dashboard failed: {e}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
