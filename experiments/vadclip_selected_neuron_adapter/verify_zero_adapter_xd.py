#!/usr/bin/env python3
"""Verify zero-start online CLIP features against the original 512D cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import baseline_options, clean_dir, ensure_dir, load_feature, save_json
from models import build_model, initialize_frozen_baseline
from online_data import OnlineVideoDataset, one_item_collate


def main() -> None:
    parser = argparse.ArgumentParser(description="Check zero-start selected-neuron Adapter against cached VadCLIP features.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--source-list", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=3e-4)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.samples <= 0 or args.frame_batch_size <= 0 or args.atol < 0 or args.rtol < 0:
        parser.error("samples and frame-batch-size must be positive; tolerances must be non-negative")
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    device = torch.device(args.device)
    options = baseline_options(args.vadclip_root)
    model = build_model(options, args.vadclip_root, str(device), args.neuron_json, adapter_rank=8)
    copied = initialize_frozen_baseline(model, args.init_baseline_model)
    model.to(device).eval()
    dataset = OnlineVideoDataset(args.source_list, args.hidden_manifest, args.vadclip_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=one_item_collate)
    rows = []
    for index, sample in enumerate(tqdm(loader, total=min(args.samples, len(loader)), desc="verify zero Adapter", unit="video")):
        if index >= args.samples:
            break
        reference = load_feature(sample.source_path)
        with torch.no_grad():
            online = model.encode_frame_sequence(sample.frames.to(device), args.frame_batch_size).float().cpu().numpy()
        same_shape = online.shape == reference.shape
        maximum = float(np.max(np.abs(online - reference))) if same_shape else float("inf")
        mean = float(np.mean(np.abs(online - reference))) if same_shape else float("inf")
        close = bool(same_shape and np.allclose(online, reference, rtol=args.rtol, atol=args.atol))
        rows.append({
            "source_path": sample.source_path,
            "video_path": sample.video_path,
            "online_shape": list(online.shape),
            "reference_shape": list(reference.shape),
            "max_abs_error": maximum,
            "mean_abs_error": mean,
            "passed": close,
        })
    report = {
        "baseline_tensors_copied": copied,
        "samples": rows,
        "atol": args.atol,
        "rtol": args.rtol,
        "all_passed": bool(rows) and all(row["passed"] for row in rows),
        "meaning": "All-pass confirms zero Adapter uses the same frame order, preprocessing and 512D CLIP output as the cached VadCLIP source features.",
    }
    save_json(out_dir / "zero_adapter_alignment.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"zero Adapter alignment failed; inspect {out_dir / 'zero_adapter_alignment.json'} before training")
    print(f"zero Adapter alignment passed for {len(rows)} videos; report={out_dir / 'zero_adapter_alignment.json'}", flush=True)


if __name__ == "__main__":
    main()
