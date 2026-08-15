#!/usr/bin/env python3
"""Make a copied CLIP-hidden manifest self-contained inside ``vadclip_data``.

The input manifest supplies keys and metadata only.  Every output hidden path
is rebuilt as ``<hidden-root>/features/<key>.npz`` so no path in the new
repository points into any other project.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import ensure_dir, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a copied hidden manifest with VadCLIP-local paths.")
    parser.add_argument("--input-manifest", required=True, help="Copied source manifest stored under vadclip_data staging.")
    parser.add_argument("--hidden-root", required=True, help="Target directory containing features/<key>.npz.")
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_manifest)
    if output.exists() and not args.clean:
        print(f"reuse staged hidden manifest: {output}", flush=True)
        return
    source = pd.read_csv(args.input_manifest)
    if "key" not in source.columns:
        raise ValueError(f"{args.input_manifest} is missing key column")
    root = Path(args.hidden_root)
    rows, missing = [], []
    header = ["key", "hidden_path", "stride", "num_frames", "fps", "layers", "token_pool"]
    for _, row in tqdm(source.iterrows(), total=len(source), desc="validate copied CLIP hidden", unit="video"):
        key = str(row["key"])
        hidden_path = root / "features" / f"{key}.npz"
        if not hidden_path.exists():
            missing.append(str(hidden_path))
            continue
        artifact = np.load(hidden_path, allow_pickle=False)
        if "hidden" not in artifact.files:
            raise ValueError(f"{hidden_path}: does not contain hidden array")
        hidden = artifact["hidden"]
        if hidden.ndim != 3 or hidden.shape[0] == 0:
            raise ValueError(f"{hidden_path}: expected non-empty [T,L,D], got {hidden.shape}")
        value = lambda name, default: row[name] if name in source.columns and pd.notna(row[name]) else default
        token_pool = str(value("token_pool", artifact["token_pool"].item() if "token_pool" in artifact.files else "cls"))
        rows.append([
            key, str(hidden_path), value("stride", int(artifact["stride"]) if "stride" in artifact.files else ""),
            value("num_frames", int(artifact["num_frames"]) if "num_frames" in artifact.files else ""),
            value("fps", float(artifact["fps"]) if "fps" in artifact.files else ""),
            value("layers", ""), token_pool,
        ])
    if missing:
        raise FileNotFoundError(f"{len(missing)} copied hidden artifacts are missing; first={missing[0]}")
    ensure_dir(output.parent)
    write_csv(output, header, rows)
    print(f"wrote self-contained manifest {output}: {len(rows)} videos", flush=True)


if __name__ == "__main__":
    main()
