#!/usr/bin/env python3
"""Convert all .npy files to chunked HDF5 (.h5) for fast random-access I/O.

Run once on your data directory before training:
    python scripts/convert_npy_to_h5.py --data-dir data
"""

import argparse
import glob
import os
import numpy as np

try:
    import h5py
except ImportError:
    print("h5py not installed. Run: pip install h5py")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert .npy files to chunked .h5")
    parser.add_argument("--data-dir", default="data", help="Directory containing .npy files")
    parser.add_argument(
        "--chunks",
        nargs=3,
        type=int,
        default=(32, 32, 32),
        help="HDF5 chunk size for 3D arrays (D, H, W). 2D arrays get (H, W) version.",
    )
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"], help="HDF5 compression filter")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .h5 files")
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    npy_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))

    if not npy_files:
        print(f"No .npy files found in {args.data_dir}")
        return

    for path in npy_files:
        out_path = path.replace(".npy", ".h5")
        if os.path.exists(out_path) and not args.force:
            print(f"SKIP (exists): {out_path}")
            continue

        arr = np.load(path)
        chunks = tuple(min(s, c) for s, c in zip(arr.shape, args.chunks))
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset("data", data=arr, chunks=chunks, compression=compression)
        print(f"OK  {path} → {out_path}  shape={arr.shape} chunks={chunks}")

    print(f"\nDone. {len(npy_files)} files processed.")


if __name__ == "__main__":
    main()
