import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt


def read_metrics(csv_path):
    epochs = []
    total = []
    jepa = []
    pixel = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = len(total)
            epochs.append(step)
            total.append(float(row["total_loss"]))
            jepa.append(float(row["loss_jepa"]))
            pixel.append(float(row["loss_pixel"]))
    return epochs, total, jepa, pixel


def plot_session(metrics_path, out_dir):
    session_name = os.path.basename(os.path.dirname(metrics_path))
    x, total, jepa, pixel = read_metrics(metrics_path)

    plt.figure(figsize=(10, 6))
    plt.plot(x, total, label="total_loss")
    plt.plot(x, jepa, label="loss_jepa")
    plt.plot(x, pixel, label="loss_pixel")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title(f"Training Curves: {session_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{session_name}_loss.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert session logs into result plots")
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    metrics_files = glob.glob(os.path.join(args.sessions_dir, "*", "metrics.csv"))
    if not metrics_files:
        print(f"No metrics.csv found under {args.sessions_dir}")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    generated = []
    for metrics in sorted(metrics_files):
        generated.append(plot_session(metrics, args.results_dir))

    print("Generated plots:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
