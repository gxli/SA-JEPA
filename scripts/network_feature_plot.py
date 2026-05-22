from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot precomputed network input structure from data (no CDD/feature regeneration)."
    )
    p.add_argument("input_npy", type=str)
    p.add_argument("--out", type=str, default="results/network_feature_plot.html")
    p.add_argument("--scales", type=str, default="2,4,8,16")
    p.add_argument("--sample-idx", type=int, default=0, help="Batch index for B,C,H,W or B,C,D,H,W")
    p.add_argument("--depth-idx", type=int, default=0, help="Depth index if input is 5D")
    p.add_argument("--low-q", type=float, default=0.5)
    p.add_argument("--high-q", type=float, default=99.5)
    return p.parse_args()


class NetworkFeaturePlotter:
    """
    Plotter for precomputed network input channels.
    Expected channel order:
      [grad*scale for each scale] + [lap*scale^2 for each scale]
    """

    def __init__(self, low_q: float = 0.5, high_q: float = 99.5):
        self.low_q = float(low_q)
        self.high_q = float(high_q)

    def _vis_limits(self, x: np.ndarray, signed: bool) -> tuple[float, float]:
        if signed:
            vmax = max(float(np.percentile(np.abs(x), self.high_q)), 1e-8)
            return -vmax, vmax
        lo = float(np.percentile(x, self.low_q))
        hi = float(np.percentile(x, self.high_q))
        if hi <= lo + 1e-12:
            delta = max(1e-6, abs(lo) * 1e-3 + 1e-6)
            lo, hi = lo - delta, hi + delta
        return lo, hi

    def plot(self, feature_stack: np.ndarray, scales: tuple[float, ...], out_html: str) -> None:
        if feature_stack.ndim != 3:
            raise ValueError(f"feature_stack must be C,H,W after sample/depth selection, got {feature_stack.shape}")
        c, _, _ = feature_stack.shape
        s = len(scales)
        if c != 2 * s:
            raise ValueError(
                f"Expected channels=2*len(scales)={2*s}, got {c}. "
                "Input should be [grad*scale channels][lap*scale^2 channels]."
            )

        grad = feature_stack[:s]
        lap = feature_stack[s : 2 * s]
        # Placeholder image panel (not regenerated): zeros with same H,W.
        img = np.zeros_like(grad[0], dtype=np.float32)

        fig = make_subplots(
            rows=s,
            cols=3,
            subplot_titles=tuple(
                f"scale={scales[i]:g} | {name}"
                for i in range(s)
                for name in ("image(ref)", "grad*scale", "lap*scale^2")
            ),
            horizontal_spacing=0.03,
            vertical_spacing=0.06,
        )

        for i in range(s):
            panels = [img, grad[i], lap[i]]
            for j, panel in enumerate(panels):
                x = np.nan_to_num(panel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                signed = j != 1
                zmin, zmax = self._vis_limits(x, signed=signed)
                colorscale = "RdBu" if signed else "Inferno"
                fig.add_trace(
                    go.Heatmap(
                        z=x,
                        zmin=zmin,
                        zmax=zmax,
                        colorscale=colorscale,
                        colorbar=dict(len=0.22, thickness=10),
                        hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.6e}<extra></extra>",
                    ),
                    row=i + 1,
                    col=j + 1,
                )

        fig.update_layout(
            title="Network Input Structure: [grad*scale, lap*scale^2] (from data, no regeneration)",
            height=max(700, 280 * s),
            width=1300,
        )
        for r in range(1, s + 1):
            for cidx in range(1, 4):
                fig.update_xaxes(showticklabels=False, row=r, col=cidx)
                # Enforce 1:1 aspect ratio for each heatmap panel.
                fig.update_yaxes(
                    showticklabels=False,
                    row=r,
                    col=cidx,
                    scaleanchor="x",
                    scaleratio=1.0,
                    constrain="domain",
                )
        fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    arr = np.asarray(np.load(args.input_npy), dtype=np.float32)
    # Accept C,H,W or B,C,H,W or B,C,D,H,W
    if arr.ndim == 3:
        chw = arr
    elif arr.ndim == 4:
        chw = arr[args.sample_idx]
    elif arr.ndim == 5:
        chw = arr[args.sample_idx, :, args.depth_idx]
    else:
        raise ValueError(f"Unsupported input shape {arr.shape}; expected C,H,W or B,C,H,W or B,C,D,H,W.")

    scales = tuple(float(x.strip()) for x in args.scales.split(",") if x.strip())
    print("input_feature_shape:", tuple(chw.shape))
    plotter = NetworkFeaturePlotter(low_q=args.low_q, high_q=args.high_q)
    plotter.plot(chw, scales=scales, out_html=args.out)
    print(args.out)


if __name__ == "__main__":
    main()
