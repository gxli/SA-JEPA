import argparse
import logging
import os
import sys
import copy

# Ensure the project root is on the Python path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.train import load_config, run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run one JEPA config")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--name", type=str, default=None, help="Session name; defaults to config filename stem")
    parser.add_argument("--sessions-dir", type=str, default="sessions", help="Session output root")
    parser.add_argument(
        "--update-effective-rank",
        action="store_true",
        help="Update mode: skip training, reuse session weights, and compute/save effective rank.",
    )
    parser.add_argument(
        "--recompute-inference",
        action="store_true",
        help="When used with --update-effective-rank, force regeneration of inference artifacts.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_config(args.config)
    config_name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    if args.update_effective_rank:
        cfg = copy.deepcopy(config)
        cfg.setdefault("train", {})
        cfg["train"]["epochs"] = 0
        cfg["train"]["compute_effective_rank"] = True
        cfg["train"]["force_recompute_inference"] = bool(args.recompute_inference)
        config = cfg
    elif args.recompute_inference:
        cfg = copy.deepcopy(config)
        cfg.setdefault("train", {})
        cfg["train"]["force_recompute_inference"] = True
        config = cfg
    session_dir = run_training(config, config_name=config_name, sessions_root=args.sessions_dir)
    print(f"session_saved={session_dir}")


if __name__ == "__main__":
    main()
