#!/usr/bin/env python3
"""Create a VadCLIP-local feature CSV by indexing staged 512D feature files."""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from common import ensure_dir, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a VadCLIP list to paths under the independent vadclip_data feature root.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--feature-root", required=True, help="Staged local 512D feature directory.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if any listed feature is missing from the staged root.")
    parser.add_argument("--clean", action="store_true", help="Replace an existing output CSV.")
    args = parser.parse_args()

    output = Path(args.output_csv)
    if output.exists() and not args.clean:
        print(f"reuse existing local feature CSV: {output}", flush=True)
        return
    root = Path(args.feature_root)
    if not root.exists():
        raise FileNotFoundError(f"staged feature root does not exist: {root}")
    index: dict[str, str] = {}
    for path in tqdm(root.rglob("*.npy"), desc="index staged 512D features", unit="file"):
        if path.name in index:
            raise ValueError(f"duplicate staged feature filename: {path.name}")
        index[path.name] = str(path)
    rows, missing = [], []
    for _, row in read_csv(args.input_csv).iterrows():
        filename = Path(str(row["path"])).name
        resolved = index.get(filename)
        if resolved is None:
            missing.append(filename)
            if args.strict:
                continue
            resolved = str(row["path"])
        rows.append([resolved, str(row["label"])])
    if args.strict and missing:
        raise FileNotFoundError(f"{len(missing)} staged 512D features are missing; first={missing[0]}")
    ensure_dir(output.parent)
    write_csv(output, ["path", "label"], rows)
    print(f"wrote {output}: rows={len(rows)}, missing={len(missing)}", flush=True)


if __name__ == "__main__":
    main()
