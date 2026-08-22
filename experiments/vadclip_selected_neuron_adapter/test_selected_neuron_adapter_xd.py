#!/usr/bin/env python3
"""Resumable online XD test for a trained selected-neuron Adapter model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import baseline_options, clean_dir, ensure_dir, state_dict_from_file
from models import build_model
from online_data import OnlineVideoDataset, one_item_collate
from xd_utils import infer_sample, print_metrics, save_metrics, summarize_records


def record_from_file(path: Path):
    artifact = np.load(path, allow_pickle=False)
    required = {"prob1", "prob2", "logits2", "source_path", "label"}
    if not required.issubset(artifact.files):
        raise ValueError(f"{path}: incomplete resumable per-video output")
    return (
        np.asarray(artifact["prob1"], dtype=np.float32),
        np.asarray(artifact["prob2"], dtype=np.float32),
        np.asarray(artifact["logits2"], dtype=np.float32),
        str(artifact["source_path"].item()),
        str(artifact["label"].item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen-VadCLIP selected-neuron Adapter on XD-Violence.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.frame_batch_size <= 0 or args.adapter_rank <= 0:
        parser.error("frame-batch-size and adapter-rank must be positive")
    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "per_video")
    options = baseline_options(args.vadclip_root)
    model = build_model(options, args.vadclip_root, str(device), args.neuron_json, args.adapter_rank)
    model.load_state_dict(state_dict_from_file(args.model_path), strict=True)
    model.freeze_base()
    model.to(device).eval()
    dataset = OnlineVideoDataset(args.test_list, args.test_hidden_manifest, args.vadclip_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=one_item_collate)
    records, output_paths = [], set()
    with torch.no_grad():
        for sample in tqdm(loader, desc="VadCLIP selected-neuron Adapter test", unit="video"):
            output_path = score_dir / f"{Path(sample.source_path).stem}.npz"
            if output_path in output_paths:
                raise ValueError(f"duplicate test feature stem creates ambiguous resume artifact: {output_path}")
            output_paths.add(output_path)
            if output_path.exists() and not args.no_resume:
                record = record_from_file(output_path)
                if record[3] != sample.source_path:
                    raise ValueError(f"{output_path}: source path differs from CSV; use --clean")
            else:
                record = infer_sample(model, sample, options.visual_length, args.frame_batch_size, device)
                np.savez_compressed(
                    output_path, prob1=record[0], prob2=record[1], logits2=record[2],
                    source_path=np.asarray(record[3]), label=np.asarray(record[4]),
                )
            records.append(record)
    metrics = summarize_records(
        records,
        str(Path(args.vadclip_root) / "list" / "gt.npy"),
        str(Path(args.vadclip_root) / "list" / "gt_segment.npy"),
        str(Path(args.vadclip_root) / "list" / "gt_label.npy"),
        args.vadclip_root,
    )
    save_metrics(out_dir / "metrics.json", metrics)
    print_metrics(metrics)
    print(f"wrote {out_dir / 'metrics.json'} and {len(records)} resumable per-video outputs", flush=True)


if __name__ == "__main__":
    main()
