"""Public API for sajepa — Scale-Aware JEPA pipeline."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.train import load_config, reject_removed_config_aliases, run_training
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

    @staticmethod
    def _seed_session_from_base(base_session: str, target_session: str, mode: str = "weights") -> None:
        """Seed a new training session from an existing session.

        mode="weights" copies only model_last.pt, resetting epoch/optimizer.
        mode="resume" also copies checkpoint_last.pt for full optimizer resume.
        """
        import shutil

        mode = str(mode).strip().lower()
        if mode not in {"weights", "resume"}:
            raise ValueError("base_session_mode must be 'weights' or 'resume'.")
        base_session = os.path.abspath(base_session)
        target_session = os.path.abspath(target_session)
        if not os.path.isdir(base_session):
            raise FileNotFoundError(f"base_session does not exist: {base_session}")
        if os.path.abspath(base_session) == os.path.abspath(target_session):
            raise ValueError("base_session and target session must be different directories.")

        os.makedirs(target_session, exist_ok=True)
        copied = []
        ckpt_names = ["model_last.pt"]
        if mode == "resume":
            ckpt_names.append("checkpoint_last.pt")
        for ckpt_name in ckpt_names:
            src = os.path.join(base_session, ckpt_name)
            if os.path.exists(src):
                dst = os.path.join(target_session, ckpt_name)
                shutil.copy2(src, dst)
                copied.append(ckpt_name)
        if not copied:
            raise FileNotFoundError(
                f"base_session has no model_last.pt: {base_session}"
            )
        print(f"[sajepa] seeded {target_session} from {base_session} mode={mode}: {', '.join(copied)}")

    # ── training ────────────────────────────────────────────────

    @property
    def session_dir(self) -> Optional[str]:
        """Current training or inference session directory, if one exists."""
        return self._session_dir

    def train(
        self,
        configs: Optional[dict | str] = None,
        *,
        config_name: str = "sajepa",
        sessions_dir: str = "sessions",
        dashboard: bool = False,
        base_session: Optional[str] = None,
        base_session_mode: str = "weights",
    ) -> "ScaleAwareJEPA":
        """Train from the current config plus optional overrides.

        When *base_session* is provided, seeds the new session from that
        session.  ``base_session_mode="weights"`` copies only model weights
        and resets optimizer/epoch; ``"resume"`` also copies optimizer,
        scheduler, scaler, and epoch state.
        """
        cfg = copy.deepcopy(self._config)
        if configs is not None:
            if isinstance(configs, str):
                override = load_config(configs)
            else:
                override = copy.deepcopy(configs)
                reject_removed_config_aliases(override)
            cfg = _deep_merge(cfg, override)

        if base_session is not None:
            new_dir = os.path.join(sessions_dir, config_name)
            self._seed_session_from_base(base_session, new_dir, mode=base_session_mode)

        session_dir = run_training(cfg, config_name=config_name, sessions_root=sessions_dir)
        self._config = cfg
        self._session_dir = os.path.abspath(session_dir)
        self._is_trained = True
        if dashboard:
            self.generate_dashboard(os.path.join(self._session_dir, "dashboard.html"))
        return self

    def fit(
        self,
        field: torch.Tensor,
        epochs: Optional[int] = None,
        *,
        session_dir: Optional[str] = None,
        base_session: Optional[str] = None,
        base_session_mode: str = "weights",
    ) -> "ScaleAwareJEPA":
        """Train on a raw physical field.  Returns self for chaining.

        When *session_dir* is provided, training persists to that directory
        and resumes from the last checkpoint if one exists.  Without it, a
        temporary directory is used and no resume is attempted.

        When *base_session* is provided, seeds the new session from that
        session.  ``base_session_mode="weights"`` copies only model weights
        and resets optimizer/epoch; ``"resume"`` also copies optimizer,
        scheduler, scaler, and epoch state.
        """
        cfg = copy.deepcopy(self._config)
        if epochs is not None:
            cfg.setdefault("train", {})["epochs"] = int(epochs)

        if session_dir is not None:
            sessions_dir = os.path.abspath(session_dir)
            os.makedirs(sessions_dir, exist_ok=True)
        else:
            sessions_dir = tempfile.mkdtemp(prefix="sajepa_")
        if base_session is not None:
            self._seed_session_from_base(
                base_session,
                os.path.join(sessions_dir, "sajepa"),
                mode=base_session_mode,
            )
        data_dir = os.path.join(sessions_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "_input.npy")
        arr = field.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        np.save(data_path, arr)
        cfg.setdefault("data", {})["npy_pattern"] = "_input.npy"
        cfg.setdefault("data", {})["data_root"] = sessions_dir

        # --- auto-batch OOM handling ---
        train_cfg = cfg.setdefault("train", {})
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

        print(f"[sajepa] batch={trainer.batch_size} accum={trainer.accumulation_steps}"
              f" session={session_dir or '(temp)'}")

        old_cwd = os.getcwd()
        os.chdir(sessions_dir)
        sd = None
        try:
            while True:
                try:
                    sd = run_training(cfg, config_name="sajepa", sessions_root=".")
                    break
                except RuntimeError as e:
                    if not trainer.handle_oom(e):
                        raise
                    train_cfg["batch_size"] = trainer.batch_size
                    train_cfg["gradient_accumulation_steps"] = trainer.accumulation_steps
                    clear_memory_cache()
        finally:
            self._session_dir = os.path.abspath(sd) if sd else None
            os.chdir(old_cwd)

        if self._session_dir is None:
            raise RuntimeError("Training failed after all OOM retries.")
        self._is_trained = True
        _print_metrics_summary(self._session_dir)
        return self

    # ── inference ───────────────────────────────────────────────

    @classmethod
    def infer_from_session(
        cls,
        session_dir: str,
        input_path: str,
        output_dir: str,
        **kwargs,
    ) -> str:
        """Create an inference-only session from a trained session and a new ``.npy``."""
        model = cls.load_session(session_dir)
        return model.infer_npy(input_path, output_dir, **kwargs)

    @torch.no_grad()
    def infer_npy(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        *,
        crop_size: Optional[int] = None,
        crop_mode: str = "center",
        mode: str = "image",
        mask_inference: bool = True,
        batch_size: int = 2,
        device: Optional[str | torch.device] = None,
        tta: bool = False,
        tta_mode: str = "flip4",
        allow_partial_load: bool = False,
        make_dashboard: bool = True,
    ) -> str:
        """Run a loaded session on an arbitrary ``.npy`` file.

        The result is saved as an inference-only session directory containing
        ``inference_outputs.pt``, map artifacts, UMAP/PCA artifacts, and
        optionally ``dashboard.html``.
        """
        if self._session_dir is None:
            raise RuntimeError("No session. Call load_session() or fit() first.")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)

        from src.inference_from_session import (
            _collate_pad_spatial,
            _stitch_tiled_outputs,
            load_model_from_session,
            load_raw_data,
            run_inference_on_data,
            save_inference_session,
        )

        run_device = torch.device(device) if device is not None else self._get_device()
        model, config, source_session = load_model_from_session(
            self._session_dir,
            run_device,
            strict_load=not bool(allow_partial_load),
        )
        config["_source_session"] = source_session

        data_tensor, tile_layout = load_raw_data(
            input_path,
            crop_size=crop_size,
            crop_mode=crop_mode,
            mode=mode,
            return_layout=True,
            slab_depth=(
                getattr(model, "required_input_depth", None)
                if str(mode).strip().lower() in ("3d_slab", "3d-slab")
                else None
            ),
        )

        class _TensorDataset(torch.utils.data.Dataset):
            def __init__(self, tensor: torch.Tensor):
                self.tensor = tensor

            def __len__(self) -> int:
                return int(self.tensor.shape[0])

            def __getitem__(self, idx: int) -> torch.Tensor:
                return self.tensor[idx]

        loader = DataLoader(
            _TensorDataset(data_tensor),
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_pad_spatial,
        )
        outputs = run_inference_on_data(
            model,
            loader,
            run_device,
            mask_inference=bool(mask_inference),
            inference_tta_enabled=bool(tta),
            inference_tta_mode=str(tta_mode),
        )
        outputs = _stitch_tiled_outputs(outputs, tile_layout)

        if output_dir is None:
            base = os.path.basename(self._session_dir.rstrip(os.sep))
            stem = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.abspath(os.path.join("sessions", f"inference_{base}_{stem}"))

        out = save_inference_session(
            outputs,
            output_dir,
            config,
            input_path,
            crop_size=crop_size,
            mode=mode,
            mask_inference=bool(mask_inference),
            make_dashboard=bool(make_dashboard),
            umap_cfg=config.get("train", {}).get("umap", {}),
        )
        self._session_dir = os.path.abspath(out)
        self._is_trained = True
        if make_dashboard:
            self.generate_dashboard(os.path.join(self._session_dir, "dashboard.html"))
        return self._session_dir

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
            cfg = cls._parse_config_exact(cfg_path)
        else:
            cfg = cls._parse_config_exact(os.path.join(path, "config_used.json"))
        instance = cls(config=None)
        instance._config = cfg
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
            "l2_norm", "psnorm", "final_norm", "spread_type", "spread_spatial_mode", "spread_w", "spread_t",
            "vicreg_var_weight", "vicreg_cov_weight", "sym_loss", "depth", "dilations", "hardcap", "cdd_scales",
            "energy", "sim_r", "hinge_r", "sig_r",
            "vicreg_var_r", "vicreg_cov_r", "weighted_vicreg_var_r", "weighted_vicreg_cov_r",
            "target_effrank", "context_effrank", "predictor_effrank", "target_branch_effrank",
            "top1", "pred_part", "target_part", "part_ratio", "dead_frac", "dead_ch",
            "loss_total_last", "loss_prediction_last", "loss_spread_last",
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
            self._ensure_inference_umap_artifacts()
            from scripts.session_to_dash import compute_dash_data, plot_dash
            compute_dash_data(self._session_dir, overwrite=False)
            plot_dash(self._session_dir, overwrite=False)
            dash = os.path.join(self._session_dir, "dashboard.html")
            if output_path and os.path.exists(dash):
                import shutil
                if os.path.abspath(dash) != os.path.abspath(output_path):
                    shutil.copy2(dash, output_path)
            print(f"[sajepa] dashboard: {output_path or dash}")
        except Exception as e:
            print(f"[sajepa] dashboard failed ({type(e).__name__}), generating minimal dashboard...")
            _generate_minimal_dashboard(self._session_dir, output_path)

    def _ensure_inference_umap_artifacts(self) -> None:
        if self._session_dir is None:
            return
        cfg_path = os.path.join(self._session_dir, "config_used.json")
        inf_path = os.path.join(self._session_dir, "inference_outputs.pt")
        if not (os.path.exists(cfg_path) and os.path.exists(inf_path)):
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            is_inference_only = bool(cfg.get("_inference", {}).get("inference_only", False))
            if not is_inference_only:
                return
            results_dir = os.path.join(self._session_dir, "results")
            required = os.path.join(results_dir, "predict_umap_xyz.npy")
            if os.path.exists(required):
                return
            from src.utils.viz import save_inference_dashboard

            outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
            umap_cfg = cfg.get("train", {}).get("umap", {})
            save_inference_dashboard(self._session_dir, outputs, umap_cfg=umap_cfg)
        except Exception as e:
            print(f"[sajepa] inference UMAP artifact generation failed: {type(e).__name__}: {e}")

    # ── internals ───────────────────────────────────────────────

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _parse_config(config: Optional[dict | str]) -> dict:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "base_pyramid_scaleaware_convnext.yaml")
        base = load_config(default_path)
        if config is None:
            return base
        if isinstance(config, str):
            return _deep_merge(base, load_config(config))
        override = copy.deepcopy(config)
        reject_removed_config_aliases(override)
        return _deep_merge(base, override)

    @staticmethod
    def _parse_config_exact(config: dict | str) -> dict:
        if isinstance(config, str):
            return load_config(config)
        return copy.deepcopy(config)


def _print_metrics_summary(session_dir: str) -> None:
    import csv as _csv
    path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    try:
        loss_weights = {}
        weights_path = os.path.join(session_dir, "loss_weights.json")
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    loss_weights = json.load(f)
            except Exception:
                loss_weights = {}
        epochs: dict[int, dict[str, list[float]]] = {}
        with open(path, "r") as f:
            for row in _csv.DictReader(f):
                ep = int(row.get("epoch", -1))
                if ep < 0:
                    continue
                if ep not in epochs:
                    epochs[ep] = {}
                for k in ("loss_total", "loss_prediction", "loss_spread", "var", "cov", "lr"):
                    v = row.get(k, "")
                    if v:
                        epochs[ep].setdefault(k, []).append(float(v))
        if not epochs:
            return
        last_ep, first_ep = max(epochs.keys()), min(epochs.keys())
        print(f"\n{'='*60}")
        print(f"Training Metrics (epoch {first_ep} → {last_ep})")
        print(f"{'='*60}")
        def _weight_active(key: str, nested: str | None = None) -> bool:
            if not loss_weights:
                return any(epochs[ep].get(key) for ep in epochs)
            if nested is not None:
                value = loss_weights.get(nested, {}).get("weight")
            else:
                value = loss_weights.get(key)
            try:
                return value is not None and abs(float(value)) > 1e-12
            except Exception:
                return False

        keys = [
            ("loss_total", "L(total)     "), ("loss_prediction", "MSE(pred,gt) "),
        ]
        if _weight_active("loss_spread", "spread_regularizer"):
            keys.append(("loss_spread", "spread      "))
        if _weight_active("vicreg_var_weight"):
            keys.append(("var", "vicreg_var  "))
        if _weight_active("vicreg_cov_weight"):
            keys.append(("cov", "vicreg_cov  "))
        keys.append(("lr", "lr          "))
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
