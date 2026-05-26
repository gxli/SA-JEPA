"""
Compute CDD effectiveness and redundancy maps for a single image.

Effectiveness map (old):
    E(x,y) = sum_i |c_i| * cos(theta_i) / sum_i |c_i|
    where cos(theta_i) = (vec(c_i) · vec(I)) / (||c_i|| * ||I||)

Redundancy map (new):
    R(x,y) = N / D
    where g_i = grad(c_i), g_j = grad(c_j)
    N = sum_{i!=j} angle(g_i, g_j) * sqrt(|c_i * c_j|)
    D = sum_i |c_i|

Output: combined panel with both maps and histograms.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def load_image(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        mid = arr.shape[0] // 2
        arr = arr[mid]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin > 1e-20:
        arr = (arr - vmin) / (vmax - vmin)
    return arr


def run_cdd(image: np.ndarray, constrained: bool) -> np.ndarray:
    import constrained_diffusion as cdd

    result, residual = cdd.constrained_diffusion_decomposition(
        image.astype(np.float32),
        mode="log",
        constrained=constrained,
        sm_mode="reflect",
        return_scales=False,
        verbose=False,
        use_gpu=False,
    )
    return np.asarray(result, dtype=np.float32)


# --- effectiveness map (old) ---

def effectiveness_map(channels: np.ndarray, image: np.ndarray) -> np.ndarray:
    S = channels.shape[0]
    img_flat = image.ravel().astype(np.float64)
    img_norm = float(np.linalg.norm(img_flat))

    cos_theta = np.empty(S, dtype=np.float64)
    for i in range(S):
        ci_flat = channels[i].ravel().astype(np.float64)
        ci_norm = float(np.linalg.norm(ci_flat))
        if ci_norm < 1e-30 or img_norm < 1e-30:
            cos_theta[i] = 0.0
        else:
            dot = float(np.dot(ci_flat, img_flat))
            cos_theta[i] = dot / (ci_norm * img_norm)

    numer = np.zeros(channels.shape[1:], dtype=np.float64)
    denom = np.zeros(channels.shape[1:], dtype=np.float64)
    for i in range(S):
        abs_ci = np.abs(channels[i]).astype(np.float64)
        numer += abs_ci * cos_theta[i]
        denom += abs_ci

    with np.errstate(divide="ignore", invalid="ignore"):
        E = numer / denom
    E[~np.isfinite(E)] = 0.0
    return E.astype(np.float32)


# --- redundancy map (new) ---

def angle_map(gx_i, gy_i, gx_j, gy_j):
    dot = gx_i * gx_j + gy_i * gy_j
    norm_i = np.sqrt(gx_i ** 2 + gy_i ** 2)
    norm_j = np.sqrt(gx_j ** 2 + gy_j ** 2)
    denom = norm_i * norm_j
    cos_angle = np.zeros_like(dot)
    mask = denom > 1e-30
    cos_angle[mask] = dot[mask] / denom[mask]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def redundancy_map(channels: np.ndarray) -> np.ndarray:
    S = channels.shape[0]

    grads = []
    for i in range(S):
        gy, gx = np.gradient(channels[i].astype(np.float64))
        grads.append((gy, gx))

    N = np.zeros(channels.shape[1:], dtype=np.float64)
    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            ang = angle_map(grads[i][0], grads[i][1], grads[j][0], grads[j][1])
            N += ang * np.sqrt(np.abs(channels[i].astype(np.float64) * channels[j].astype(np.float64)))

    D = np.sum(np.abs(channels).astype(np.float64), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        R = N / D
    R[~np.isfinite(R)] = 0.0
    return R.astype(np.float32)


# --- combined plot ---

def plot_combined(
    image: np.ndarray,
    E_con: np.ndarray,
    E_unc: np.ndarray,
    R_con: np.ndarray,
    R_unc: np.ndarray,
    out_path: str,
    num_scales: int,
):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # row 0: original | E constrained | E unconstrained | (spacer)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image, cmap="gray", origin="upper")
    ax_img.set_title("Original Image", fontsize=10)
    ax_img.axis("off")

    # shared vlims for E maps
    vE_min = min(float(E_con.min()), float(E_unc.min()))
    vE_max = max(float(E_con.max()), float(E_unc.max()))
    if vE_max - vE_min < 1e-8:
        vE_min, vE_max = -1.0, 1.0

    ax_Ec = fig.add_subplot(gs[0, 1])
    im_Ec = ax_Ec.imshow(E_con, cmap="coolwarm", origin="upper", vmin=vE_min, vmax=vE_max)
    ax_Ec.set_title(f"Effectiveness (constrained, {num_scales} scales)", fontsize=10)
    ax_Ec.axis("off")
    plt.colorbar(im_Ec, ax=ax_Ec, fraction=0.046)

    ax_Eu = fig.add_subplot(gs[0, 2])
    im_Eu = ax_Eu.imshow(E_unc, cmap="coolwarm", origin="upper", vmin=vE_min, vmax=vE_max)
    ax_Eu.set_title(f"Effectiveness (unconstrained, {num_scales} scales)", fontsize=10)
    ax_Eu.axis("off")
    plt.colorbar(im_Eu, ax=ax_Eu, fraction=0.046)

    # row 1: (empty) | R constrained | R unconstrained | (empty)
    vR_min = min(float(R_con.min()), float(R_unc.min()))
    vR_max = max(float(R_con.max()), float(R_unc.max()))
    if vR_max - vR_min < 1e-8:
        vR_min, vR_max = 0.0, 1.0

    ax_Rc = fig.add_subplot(gs[1, 1])
    im_Rc = ax_Rc.imshow(R_con, cmap="coolwarm", origin="upper", vmin=vR_min, vmax=vR_max)
    ax_Rc.set_title(f"Redundancy (constrained, {num_scales} scales)", fontsize=10)
    ax_Rc.axis("off")
    plt.colorbar(im_Rc, ax=ax_Rc, fraction=0.046)

    ax_Ru = fig.add_subplot(gs[1, 2])
    im_Ru = ax_Ru.imshow(R_unc, cmap="coolwarm", origin="upper", vmin=vR_min, vmax=vR_max)
    ax_Ru.set_title(f"Redundancy (unconstrained, {num_scales} scales)", fontsize=10)
    ax_Ru.axis("off")
    plt.colorbar(im_Ru, ax=ax_Ru, fraction=0.046)

    # row 2: E histogram (col 0-1) | R histogram (col 2-3)
    ax_histE = fig.add_subplot(gs[2, 0:2])
    E_flat_con = E_con.ravel()
    E_flat_unc = E_unc.ravel()
    bins_E = np.linspace(min(E_flat_con.min(), E_flat_unc.min()),
                         max(E_flat_con.max(), E_flat_unc.max()), 80)
    ax_histE.hist(E_flat_con, bins=bins_E, alpha=0.55, label="constrained", color="C0")
    ax_histE.hist(E_flat_unc, bins=bins_E, alpha=0.55, label="unconstrained", color="C1")
    ax_histE.set_xlabel("Effectiveness E")
    ax_histE.set_ylabel("Pixel count")
    ax_histE.legend(fontsize=8)
    ax_histE.set_title("Effectiveness histogram")

    ax_histR = fig.add_subplot(gs[2, 2:4])
    R_flat_con = R_con.ravel()
    R_flat_unc = R_unc.ravel()
    bins_R = np.linspace(min(R_flat_con.min(), R_flat_unc.min()),
                         max(R_flat_con.max(), R_flat_unc.max()), 80)
    ax_histR.hist(R_flat_con, bins=bins_R, alpha=0.55, label="constrained", color="C0")
    ax_histR.hist(R_flat_unc, bins=bins_R, alpha=0.55, label="unconstrained", color="C1")
    ax_histR.set_xlabel("Redundancy R")
    ax_histR.set_ylabel("Pixel count")
    ax_histR.legend(fontsize=8)
    ax_histR.set_title("Redundancy histogram")

    fig.suptitle(f"CDD Effectiveness vs Redundancy Maps ({num_scales} scales)", fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to .npy image file")
    args = parser.parse_args()

    image = load_image(args.image)
    print(f"Image shape: {image.shape}, range: [{image.min():.4f}, {image.max():.4f}]")

    print("Running CDD (constrained)...")
    ch_con = run_cdd(image, constrained=True)
    print("Running CDD (unconstrained)...")
    ch_unc = run_cdd(image, constrained=False)

    S = ch_con.shape[0]
    print(f"CDD auto scales — constrained: {S}, unconstrained: {ch_unc.shape[0]}")

    print("Computing effectiveness maps (old)...")
    E_con = effectiveness_map(ch_con, image)
    E_unc = effectiveness_map(ch_unc, image)
    print(f"Effectiveness constrained:   [{E_con.min():.4f}, {E_con.max():.4f}]")
    print(f"Effectiveness unconstrained: [{E_unc.min():.4f}, {E_unc.max():.4f}]")

    print("Computing redundancy maps (new)...")
    R_con = redundancy_map(ch_con)
    R_unc = redundancy_map(ch_unc)
    print(f"Redundancy constrained:   [{R_con.min():.4f}, {R_con.max():.4f}]")
    print(f"Redundancy unconstrained: [{R_unc.min():.4f}, {R_unc.max():.4f}]")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result_local")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(out_dir, f"{base}_combined.png")
    plot_combined(image, E_con, E_unc, R_con, R_unc, out_path, S)


if __name__ == "__main__":
    main()
