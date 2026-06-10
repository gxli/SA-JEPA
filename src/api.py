"""Public API for sajepa — modular Scale-Aware JEPA training and inference."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from typing import Optional

import numpy as np
import torch

from src.train import load_config, run_training
from src.utils.memory import OOMSafeTrainer, clear_memory_cache, compute_accumulation_steps


DEFAULT_BASE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "base_pyramid_scaleaware_convnext.json")


class ScaleAwareJEPA:
    """High-level interface for training and extracting latent atlases from physical fields.

    Usage:
        model = ScaleAwareJEPA(config=config_dict)
        latent_atlas = model.fit_and_extract(raw_field, epochs=10)
    """

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        if config_path is not None:
            self._config = load_config(config_path)
        elif config is not None:
            base = load_config(DEFAULT_BASE_CONFIG)
            self._config = _deep_merge(base, config)
        else:
            self._config = load_config(DEFAULT_BASE_CONFIG)

    @classmethod
    def from_config_file(cls, path: str) -> "ScaleAwareJEPA":
        return cls(config_path=path)

    def fit_and_extract(
        self,
        raw_field: torch.Tensor,
        epochs: Optional[int] = None,
        sessions_dir: Optional[str] = None,
        save_dashboard: bool = False,
    ) -> torch.Tensor:
        """Train on a raw physical field and return pixel-registered latent embeddings.

        Args:
            raw_field: (H, W) or (C, H, W) tensor of physical measurements.
            epochs: Override training epochs.
            sessions_dir: Output directory for session artifacts (temp dir if None).
            save_dashboard: If True, generate dashboard HTML via session_to_dash.

        Returns:
            Latent embedding atlas as (C_latent, H, W) tensor.
        """
        cfg = copy.deepcopy(self._config)
        if epochs is not None:
            cfg.setdefault("train", {})["epochs"] = int(epochs)

        if sessions_dir is None:
            sessions_dir = tempfile.mkdtemp(prefix="sajepa_")

        # Save raw field as temporary .npy so the dataset pipeline can load it.
        data_dir = os.path.join(sessions_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "_input.npy")
        arr = raw_field.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        np.save(data_path, arr)

        cfg.setdefault("data", {})["npy_pattern"] = "_input.npy"

        # --- Auto-batch OOM handling ---
        train_cfg = cfg.setdefault("train", {})
        initial_batch = int(train_cfg.get("batch_size", 4))
        target_batch = int(train_cfg.get("target_batch_size", train_cfg.get("target_batch", 32)))
        scale_mode = str(train_cfg.get("auto_scale_batch_size", "power_of_two"))
        max_retries = int(train_cfg.get("oom_max_retries", 5))
        precision = str(train_cfg.get("precision", "bf16"))

        trainer = OOMSafeTrainer(
            initial_batch=initial_batch,
            target_batch=target_batch,
            scale_mode=scale_mode,
            max_retries=max_retries,
        )
        train_cfg["batch_size"] = trainer.batch_size
        train_cfg["gradient_accumulation_steps"] = trainer.accumulation_steps
        if "target_batch_size" not in train_cfg:
            train_cfg["target_batch_size"] = target_batch
        # Enable MPS fallback for ops not yet implemented on Apple Silicon
        # (e.g. avg_pool3d in CDD library). Also disable multiprocessing
        # DataLoader workers on macOS (spawn-based pickling fails for closures).
        if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if not torch.cuda.is_available():
            train_cfg["num_workers"] = 0

        print(
            f"[sajepa] batch={trainer.batch_size} accum={trainer.accumulation_steps} "
            f"target_batch={target_batch} precision={precision}"
        )

        # Point data root to the temp dir so the dataset finds our file.
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
                    cfg["train"] = train_cfg
                    clear_memory_cache()
        finally:
            session_dir = os.path.abspath(session_dir) if session_dir else None
            os.chdir(old_cwd)

        if session_dir is None:
            raise RuntimeError("Training failed after all OOM retries.")

        # Print final epoch metrics summary.
        _print_metrics_summary(session_dir)

        # Generate dashboard if requested
        if save_dashboard:
            _generate_dashboard(session_dir)

        # Load the latent embeddings from the session output.
        inf_path = os.path.join(session_dir, "inference_outputs.pt")
        outputs = torch.load(inf_path, map_location="cpu", weights_only=False)

        context_map = outputs.get("context_map")
        if context_map is None:
            pred_map = outputs.get("pred_map")
            if pred_map is not None:
                return pred_map.squeeze(0).cpu()
            raise RuntimeError("No context_map or pred_map found in inference outputs.")

        return context_map.squeeze(0).cpu()


def _generate_dashboard(session_dir: str) -> None:
    """Generate dashboard HTML from session outputs using session_to_dash."""
    try:
        from scripts.session_to_dash import compute_dash_data, plot_dash
        compute_dash_data(session_dir, overwrite=False)
        plot_dash(session_dir, overwrite=False)
        dash_path = os.path.join(session_dir, "dashboard.html")
        if os.path.exists(dash_path):
            print(f"[sajepa] dashboard saved: {dash_path}")
    except Exception as e:
        print(f"[sajepa] dashboard skipped: {type(e).__name__}: {e}")


def _print_metrics_summary(session_dir: str) -> None:
    """Print final-epoch averaged metrics from metrics.csv."""
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
        last_ep = max(epochs.keys())
        first_ep = min(epochs.keys())
        print(f"\n{'='*60}")
        print(f"Training Metrics (epoch {first_ep} → {last_ep})")
        print(f"{'='*60}")

        def _avg(ep, k):
            vals = epochs[ep].get(k, [])
            return sum(vals) / len(vals) if vals else 0.0

        keys = [
            ("loss_total", "L(total)     "),
            ("loss_prediction", "MSE(pred,gt) "),
            ("loss_spread", "sig=relu(1-std)"),
            ("sim", "cos(pred,gt) "),
            ("var", "var_term    "),
            ("cov", "cov_term    "),
            ("lr", "lr          "),
        ]
        for key, label in keys:
            first = _avg(first_ep, key)
            last = _avg(last_ep, key)
            ratio = last / first if first > 1e-20 else 1.0
            print(f"  {label}: {first:>8.4f} → {last:>8.4f}  (ratio={ratio:.3f})")
        print(f"{'='*60}")
    except Exception:
        pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
