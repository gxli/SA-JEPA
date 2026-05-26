#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def to_native(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.byteorder in (">", "<") and arr.dtype.byteorder != "=":
        # NumPy 2.0 removed ndarray.newbyteorder(); use dtype.newbyteorder instead.
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    return np.asarray(arr)


def pca_3d(latents: np.ndarray) -> np.ndarray:
    x = to_native(latents).astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:3].T
    return x @ comps


def load_or_compute_pca_3d(results_dir: Path, latents: np.ndarray) -> np.ndarray:
    pca_x_path = results_dir / "pca_x.npy"
    pca_y_path = results_dir / "pca_y.npy"
    pca_z_path = results_dir / "pca_z.npy"
    pca_xyz_path = results_dir / "pca_xyz.npy"

    pca_points = None
    if pca_xyz_path.exists():
        try:
            arr = to_native(np.load(pca_xyz_path))
            if arr.ndim == 2 and arr.shape[1] >= 3:
                pca_points = arr[:, :3].astype(np.float32)
        except Exception:
            pca_points = None
    elif pca_x_path.exists() and pca_y_path.exists() and pca_z_path.exists():
        try:
            px = to_native(np.load(pca_x_path)).reshape(-1).astype(np.float32)
            py = to_native(np.load(pca_y_path)).reshape(-1).astype(np.float32)
            pz = to_native(np.load(pca_z_path)).reshape(-1).astype(np.float32)
            if px.shape[0] == py.shape[0] == pz.shape[0]:
                pca_points = np.stack([px, py, pz], axis=1)
        except Exception:
            pca_points = None

    if pca_points is None or pca_points.shape[0] != latents.shape[0]:
        pca_points = pca_3d(latents).astype(np.float32)
        np.save(pca_xyz_path, pca_points)
        np.save(pca_x_path, pca_points[:, 0])
        np.save(pca_y_path, pca_points[:, 1])
        np.save(pca_z_path, pca_points[:, 2])
    return pca_points


def l2_normalize_rows(x: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return x
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def tile_maps_batch(maps: np.ndarray) -> np.ndarray:
    arr = np.asarray(maps, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported map shape for tiling: {arr.shape}")
    n, h, w = arr.shape
    if n <= 1:
        return arr[0]
    cols = int(np.ceil(np.sqrt(float(n))))
    rows = int(np.ceil(float(n) / float(cols)))
    canvas = np.zeros((rows * h, cols * w), dtype=np.float32)
    for i in range(n):
        r = i // cols
        c = i % cols
        y0 = r * h
        x0 = c * w
        canvas[y0 : y0 + h, x0 : x0 + w] = arr[i]
    return canvas


def _load_session_cfg(session_dir: Path, config_override: str = None) -> dict:
    if config_override:
        cfg_path = Path(config_override)
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config override not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    cfg_path = session_dir / "resolved_config.json"
    if not cfg_path.exists():
        cfg_path = session_dir / "config_used.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file in {session_dir} (resolved_config.json or config_used.json)")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_original_data(session_dir: Path, config_override: str = None) -> np.ndarray:
    cfg = _load_session_cfg(session_dir, config_override=config_override)
    data_path = cfg.get("data_path")
    if data_path is None:
        data_cfg = cfg.get("data", {})
        data_root = Path(data_cfg.get("data_root", "data"))
        npy_pattern = data_cfg.get("npy_pattern", "*.npy")
        files = sorted(data_root.glob(npy_pattern))
        if not files:
            raise FileNotFoundError(f"No data files for pattern {data_root / npy_pattern}")
        data_path = str(files[0])
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Original data not found: {data_path}")
    return np.load(data_path)


def make_session_overview_plot(
    original: np.ndarray,
    umap_points: np.ndarray,
    pca_points: np.ndarray,
    umap_x: np.ndarray,
    umap_y: np.ndarray,
    umap_z: np.ndarray,
    out_path: Path,
    image_shape: tuple[int, int] | None = None,
    image_umap_points: np.ndarray | None = None,
    image_pca_points: np.ndarray | None = None,
    visit_freq: np.ndarray | None = None,
    energy_map: np.ndarray | None = None,
    train_curve: dict | None = None,
) -> None:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    umap_points = np.asarray(umap_points, dtype=np.float32)
    pca_points = np.asarray(pca_points, dtype=np.float32)

    if image_shape is not None:
        view_shape = (int(image_shape[0]), int(image_shape[1]))
    elif umap_x.ndim == 1:
        # Legacy sessions may store flattened token arrays.
        n = int(umap_x.shape[0])
        view_shape = (1, max(1, n))
    elif umap_x.ndim == 2:
        view_shape = umap_x.shape
    else:
        view_shape = umap_x.shape[-2], umap_x.shape[-1]

    def rgb_from_ranges(points: np.ndarray, mins: np.ndarray, maxs: np.ndarray):
        rng = np.maximum(maxs - mins, 1e-12)
        norm = np.clip((points - mins) / rng, 0.0, 1.0)
        # NaN sentinels → black (no-data marker)
        nan_mask = ~np.isfinite(norm).all(axis=1)
        norm[nan_mask] = 0.0
        rgb_u8 = (norm * 255.0).astype(np.uint8)
        rgb_hex = [f"rgb({r},{g},{b})" for r, g, b in rgb_u8]
        return rgb_u8, rgb_hex

    umap_raw_all = np.stack([umap_x.reshape(-1), umap_y.reshape(-1), umap_z.reshape(-1)], axis=1).astype(np.float32)
    umap_raw = umap_raw_all if image_umap_points is None else np.asarray(image_umap_points, dtype=np.float32)
    pca_mins, pca_maxs = np.nanmin(pca_points, axis=0), np.nanmax(pca_points, axis=0)
    umap_mins, umap_maxs = np.nanmin(umap_raw, axis=0), np.nanmax(umap_raw, axis=0)

    pca_rgb_u8, pca_colors = rgb_from_ranges(pca_points, pca_mins, pca_maxs)
    umap_rgb_u8, umap_colors = rgb_from_ranges(umap_raw, umap_mins, umap_maxs)

    pca_img_src = pca_points if image_pca_points is None else np.asarray(image_pca_points, dtype=np.float32)
    pca_img_u8, _ = rgb_from_ranges(pca_img_src, pca_mins, pca_maxs)
    pca_img = pca_img_u8.reshape(*view_shape, 3)
    umap_img = umap_rgb_u8.reshape(*view_shape, 3)
    if pca_img.ndim == 4:
        pca_img = pca_img.reshape(-1, *view_shape, 3)[0]
        umap_img = umap_img.reshape(-1, *view_shape, 3)[0]

    have_energy = energy_map is not None
    have_curve = train_curve is not None and len(train_curve.get("x", [])) > 0
    n_rows = 5 if (have_energy and have_curve) else (4 if (have_energy or have_curve) else 3)
    specs = []
    titles = []
    if have_curve:
        specs.append([{"type": "xy"}, {"type": "xy"}])
        titles.extend(["Training Curves (batch)", "Validation Curves (epoch)"])
    specs.extend(
        [
            [{"type": "heatmap"}, {"type": "scene"}],
            [{"type": "heatmap"}, {"type": "scene"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
        ]
    )
    titles.extend(
        [
            "PCA Color Image (PC1 map)",
            "PCA 3D Scatter",
            "UMAP Color Image (UMAP-X map)",
            "UMAP 3D Scatter",
            "Visited Target Frequency",
            "Visited Target Frequency (log1p)",
        ]
    )
    if have_energy:
        specs.append([{"type": "heatmap"}, {"type": "xy"}])
        titles.extend(["JEPA Target Energy Map", "JEPA Target Energy Histogram"])
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    row_offset = 1 if have_curve else 0
    if have_curve:
        tx = train_curve["x"]
        fig.add_trace(go.Scatter(x=tx, y=train_curve.get("total", []), mode="lines", name="total"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tx, y=train_curve.get("jepa", []), mode="lines", name="jepa"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tx, y=train_curve.get("sim", []), mode="lines", name="sim"), row=1, col=1)
        vx = train_curve.get("vx", [])
        if len(vx) > 0:
            fig.add_trace(go.Scatter(x=vx, y=train_curve.get("val_loss", []), mode="lines+markers", name="val_loss"), row=1, col=2)
            fig.add_trace(go.Scatter(x=vx, y=train_curve.get("val_sim", []), mode="lines+markers", name="val_sim"), row=1, col=2)
        fig.update_xaxes(title_text="epoch + 0.001*batch", row=1, col=1)
        fig.update_yaxes(title_text="train metric", row=1, col=1)
        fig.update_xaxes(title_text="epoch", row=1, col=2)
        fig.update_yaxes(title_text="val metric", row=1, col=2)

    fig.add_trace(go.Image(z=pca_img), row=1 + row_offset, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=pca_points[:, 0].tolist(),
            y=pca_points[:, 1].tolist(),
            z=pca_points[:, 2].tolist(),
            mode="markers",
            marker={"size": 2, "color": pca_colors, "opacity": 0.85},
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1 + row_offset,
        col=2,
    )
    fig.add_trace(go.Image(z=umap_img), row=2 + row_offset, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=umap_points[:, 0].tolist(),
            y=umap_points[:, 1].tolist(),
            z=umap_points[:, 2].tolist(),
            mode="markers",
            marker={"size": 2, "color": umap_colors, "opacity": 0.85},
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2 + row_offset,
        col=2,
    )
    if visit_freq is None:
        visit_freq = np.zeros(view_shape, dtype=np.float32)
    visit_freq = np.asarray(visit_freq, dtype=np.float32)
    if visit_freq.shape != tuple(view_shape):
        visit_freq = np.zeros(view_shape, dtype=np.float32)
    fig.add_trace(go.Heatmap(z=visit_freq, colorscale="Inferno", showscale=True), row=3 + row_offset, col=1)
    fig.add_trace(go.Heatmap(z=np.log1p(visit_freq), colorscale="Viridis", showscale=True), row=3 + row_offset, col=2)

    fig.update_layout(
        title="Session Overview: PCA + UMAP",
        template="plotly_white",
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3", "aspectmode": "cube"},
        scene2={"xaxis_title": "U1", "yaxis_title": "U2", "zaxis_title": "U3", "aspectmode": "cube"},
        height=1900 if (have_energy and have_curve) else (1600 if (have_energy or have_curve) else 1250),
        width=1400,
        margin={"l": 50, "r": 40, "t": 90, "b": 40},
        xaxis={"constrain": "domain"},
        yaxis={"constrain": "domain", "scaleanchor": "x", "scaleratio": 1},
        xaxis2={"constrain": "domain"},
        yaxis2={"constrain": "domain", "scaleanchor": "x2", "scaleratio": 1},
    )
    h_img, w_img = view_shape
    fig.update_xaxes(range=[0, w_img], constrain="domain", row=1 + row_offset, col=1)
    fig.update_yaxes(range=[h_img, 0], scaleanchor="x", scaleratio=1, constrain="domain", row=1 + row_offset, col=1)
    fig.update_xaxes(range=[0, w_img], constrain="domain", row=2 + row_offset, col=1)
    fig.update_yaxes(range=[h_img, 0], scaleanchor="x2", scaleratio=1, constrain="domain", row=2 + row_offset, col=1)
    fig.update_xaxes(range=[0, w_img], constrain="domain", row=3 + row_offset, col=1)
    fig.update_yaxes(range=[h_img, 0], scaleanchor="x3", scaleratio=1, constrain="domain", row=3 + row_offset, col=1)
    fig.update_xaxes(range=[0, w_img], constrain="domain", row=3 + row_offset, col=2)
    fig.update_yaxes(range=[h_img, 0], scaleanchor="x4", scaleratio=1, constrain="domain", row=3 + row_offset, col=2)
    if have_energy:
        em = np.asarray(energy_map, dtype=np.float32)
        e_row = 4 + row_offset
        fig.add_trace(
            go.Heatmap(
                z=em,
                colorscale="Magma",
                showscale=True,
                colorbar={"title": "energy", "x": 0.47, "len": 0.2},
            ),
            row=e_row,
            col=1,
        )
        em_valid = em[np.isfinite(em)].reshape(-1)
        fig.add_trace(
            go.Histogram(
                x=em_valid.tolist(),
                nbinsx=80,
                marker={"color": "#4C78A8"},
                name="energy_hist",
                showlegend=False,
            ),
            row=e_row,
            col=2,
        )
        eh, ew = int(em.shape[0]), int(em.shape[1])
        fig.update_xaxes(range=[0, ew], constrain="domain", row=e_row, col=1)
        fig.update_yaxes(range=[eh, 0], scaleanchor="x5", scaleratio=1, constrain="domain", row=e_row, col=1)
        fig.update_xaxes(title_text="energy value", row=e_row, col=2)
        fig.update_yaxes(title_text="count", row=e_row, col=2)
    pio.write_html(fig, str(out_path), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Plotly overview from a training session.")
    parser.add_argument("--session-dir", required=True, help="Path to sessions/<session_id>")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <session-dir>/plots)")
    parser.add_argument("--config", default=None, help="Optional explicit config json path")
    parser.add_argument(
        "--inference",
        default="default",
        choices=["default", "target", "context", "predict", "all"],
        help="Which branch embeddings to plot (uses cached files when present).",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    results_dir = session_dir / "results"
    out_dir = Path(args.out_dir) if args.out_dir else (session_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_session_cfg(session_dir, config_override=args.config)
    plot_l2 = bool(cfg.get("plot_l2_normalize", cfg.get("umap_l2_normalize", False)))

    original = to_native(load_original_data(session_dir, config_override=args.config))

    modes = ["default", "predict", "target", "context"] if args.inference == "all" else [args.inference]
    for mode in modes:
        prefix = "" if mode == "default" else f"{mode}_"
        latent_full_path = results_dir / f"{prefix}latent_vectors_full.npy"
        umap_x_path = results_dir / f"{prefix}umap_x.npy"
        umap_y_path = results_dir / f"{prefix}umap_y.npy"
        umap_z_path = results_dir / f"{prefix}umap_z.npy"
        req = [latent_full_path, umap_x_path, umap_y_path, umap_z_path]
        if any(not p.exists() for p in req):
            if args.inference == "all":
                continue
            missing = [str(p) for p in req if not p.exists()]
            raise FileNotFoundError(f"Missing required files: {missing}")

        latents = to_native(np.load(latent_full_path))
        latents = l2_normalize_rows(latents, enable=plot_l2)
        umap_x = to_native(np.load(umap_x_path))
        umap_y = to_native(np.load(umap_y_path))
        umap_z = to_native(np.load(umap_z_path))

        umap_points = np.stack([umap_x.reshape(-1), umap_y.reshape(-1), umap_z.reshape(-1)], axis=1)
        if umap_points.shape[0] != latents.shape[0]:
            if args.inference == "all":
                continue
            raise ValueError(f"Shape mismatch: latents={latents.shape}, umap_points={umap_points.shape}")

        pca_points = load_or_compute_pca_3d(results_dir, latents) if prefix == "" else None
        if prefix != "":
        # Branch-specific PCA caches.
            pca_xyz = results_dir / f"{prefix}pca_xyz.npy"
            pca_x = results_dir / f"{prefix}pca_x.npy"
            pca_y = results_dir / f"{prefix}pca_y.npy"
            pca_z = results_dir / f"{prefix}pca_z.npy"
            if pca_xyz.exists():
                pca_points = to_native(np.load(pca_xyz)).astype(np.float32)
            elif pca_x.exists() and pca_y.exists() and pca_z.exists():
                pca_points = np.stack(
                    [
                        to_native(np.load(pca_x)).reshape(-1),
                        to_native(np.load(pca_y)).reshape(-1),
                        to_native(np.load(pca_z)).reshape(-1),
                    ],
                    axis=1,
                ).astype(np.float32)
            else:
                pca_points = pca_3d(latents).astype(np.float32)
                np.save(pca_xyz, pca_points)
                np.save(pca_x, pca_points[:, 0])
                np.save(pca_y, pca_points[:, 1])
                np.save(pca_z, pca_points[:, 2])

        # Resolve image shape/subset for the 2D color-map panels.
        image_shape = None
        image_umap_points = None
        image_pca_points = None
        spatial_shape_path = results_dir / f"{prefix}spatial_shape.npy"
        if spatial_shape_path.exists():
            try:
                shp = to_native(np.load(spatial_shape_path)).reshape(-1).astype(np.int64)
                if shp.size >= 2 and int(shp[0]) > 0 and int(shp[1]) > 0:
                    h_img, w_img = int(shp[0]), int(shp[1])
                    n_img = h_img * w_img
                    if n_img <= umap_points.shape[0] and n_img <= pca_points.shape[0]:
                        image_shape = (h_img, w_img)
                        image_umap_points = umap_points[:n_img]
                        image_pca_points = pca_points[:n_img]
            except Exception:
                pass
        if image_shape is None and umap_x.ndim == 1:
            # Legacy/default fallback: use original HxW if compatible.
            h0, w0 = original.shape[-2], original.shape[-1]
            n0 = int(h0 * w0)
            if n0 <= umap_points.shape[0] and n0 <= pca_points.shape[0]:
                image_shape = (int(h0), int(w0))
                image_umap_points = umap_points[:n0]
                image_pca_points = pca_points[:n0]

        out_name = "session_overview_4panel.html" if prefix == "" else f"session_overview_4panel_{mode}.html"
        out_path = out_dir / out_name
        visit_freq = None
        if image_shape is not None:
            vh, vw = int(image_shape[0]), int(image_shape[1])
        visit_npy = session_dir / "visited_target_frequency.npy"
        if visit_npy.exists():
            try:
                vf = to_native(np.load(visit_npy)).astype(np.float32)
                if vf.shape == (vh, vw):
                    visit_freq = vf
            except Exception:
                visit_freq = None
        if visit_freq is None:
            # Legacy fallback for old sessions.
            visit_csv = session_dir / "visited_target_locations.csv"
            if visit_csv.exists():
                vf = np.zeros((vh, vw), dtype=np.float32)
                try:
                    with visit_csv.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            y = int(float(row["y"]))
                            x = int(float(row["x"]))
                            if 0 <= y < vh and 0 <= x < vw:
                                vf[y, x] += 1.0
                    visit_freq = vf
                except Exception:
                    visit_freq = None
        energy_map = None
        epath = session_dir / "target_energy_map.npy"
        if epath.exists():
            try:
                em = to_native(np.load(epath))
                em = tile_maps_batch(em)
                energy_map = em.astype(np.float32)
            except Exception:
                energy_map = None
        train_curve = None
        metrics_path = session_dir / "metrics.csv"
        epoch_summary_path = session_dir / "epoch_summary.csv"
        if metrics_path.exists():
            tx, t_total, t_jepa, t_sim = [], [], [], []
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tx.append(float(row["epoch"]) + 0.001 * float(row["batch"]))
                        t_total.append(float(row["total_loss"]))
                        t_jepa.append(float(row["loss_jepa"]))
                        t_sim.append(float(row.get("sim", 0.0)))
            except Exception:
                tx, t_total, t_jepa, t_sim = [], [], [], []
            if len(tx) > 0:
                train_curve = {"x": tx, "total": t_total, "jepa": t_jepa, "sim": t_sim, "vx": [], "val_loss": [], "val_sim": []}
        if epoch_summary_path.exists():
            vx, vloss, vsim = [], [], []
            try:
                with epoch_summary_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vx.append(float(row["epoch"]))
                        vloss.append(float(row["val_loss"]))
                        vsim.append(float(row["val_sim"]))
            except Exception:
                vx, vloss, vsim = [], [], []
            if train_curve is None and len(vx) > 0:
                train_curve = {"x": [], "total": [], "jepa": [], "sim": [], "vx": vx, "val_loss": vloss, "val_sim": vsim}
            elif train_curve is not None:
                train_curve["vx"] = vx
                train_curve["val_loss"] = vloss
                train_curve["val_sim"] = vsim
        make_session_overview_plot(
            original=original,
            umap_points=umap_points,
            pca_points=pca_points,
            umap_x=umap_x,
            umap_y=umap_y,
            umap_z=umap_z,
            image_shape=image_shape,
            image_umap_points=image_umap_points,
            image_pca_points=image_pca_points,
            visit_freq=visit_freq,
            energy_map=energy_map,
            train_curve=train_curve,
            out_path=out_path,
        )
        print(f"saved={out_path}")


if __name__ == "__main__":
    main()
