import argparse
import os

from src.train import load_config, run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run one JEPA config")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--sessions-dir", type=str, default="sessions", help="Session output root")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    session_dir = run_training(config, config_name=config_name, sessions_root=args.sessions_dir)
    print(f"session_saved={session_dir}")


if __name__ == "__main__":
    main()
