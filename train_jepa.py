import os
import sys

from src.train import load_config, run_training


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python train_jepa.py configs/cdd_jepa.json")

    config_path = sys.argv[1]
    config = load_config(config_path)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    session_dir = run_training(config, config_name=config_name)

    print(f"Training complete. Session saved to: {session_dir}")


if __name__ == "__main__":
    main()
