#!/usr/bin/env python3
"""Standalone, resumable XD evaluation for M1/M2 feature modulation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models import add_vadclip_source, build_film_model
from xd_shared import (
    XD_LABELS,
    build_test_loader,
    clean_dir,
    ensure_dir,
    infer_item,
    load_json,
    print_metrics,
    save_json,
    summarize_records,
)


def baseline_options(vadclip_root: str):
    add_vadclip_source(vadclip_root)
    import xd_option

    return xd_option.parser.parse_args([])


def load_state(path: str) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a M1/M2 model state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def record_from_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    artifact = np.load(path, allow_pickle=False)
    required = {"prob1", "prob2", "logits2", "source_path", "label"}
    if not required.issubset(artifact.files):
        raise ValueError(f"{path}: incomplete resumable test artifact")
    return (
        np.asarray(artifact["prob1"], dtype=np.float32),
        np.asarray(artifact["prob2"], dtype=np.float32),
        np.asarray(artifact["logits2"], dtype=np.float32),
        str(artifact["source_path"].item()),
        str(artifact["label"].item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen-VadCLIP M1/M2 feature modulation on XD.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--module", choices=["m1", "m2"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--num-workers", type=int, default=0, help="Official VadCLIP XD loader default.")
    parser.add_argument("--condition-hidden-dim", type=int, default=256)
    parser.add_argument("--residual-hidden-dim", type=int, default=512)
    parser.add_argument("--temporal-dilations", type=int, nargs="+", default=[1, 2], help="Used by M2 only.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Recompute per-video predictions while retaining output directory.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.condition_hidden_dim <= 0 or args.residual_hidden_dim <= 0 or any(value <= 0 for value in args.temporal_dilations):
        parser.error("hidden dimensions and temporal dilations must be positive")
    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "per_video")
    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    options = baseline_options(args.vadclip_root)
    model = build_film_model(
        options=options,
        vadclip_root=args.vadclip_root,
        device=str(device),
        contract=contract,
        module_variant=args.module,
        condition_hidden_dim=args.condition_hidden_dim,
        residual_hidden_dim=args.residual_hidden_dim,
        temporal_dilations=args.temporal_dilations,
    )
    model.load_state_dict(load_state(args.model_path), strict=True)
    model.freeze_base()
    model.to(device).eval()
    loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    records, output_paths = [], set()
    for item in tqdm(loader, desc=f"VadCLIP {args.module} XD test", unit="video"):
        source_path = str(item[2][0])
        output_path = score_dir / f"{Path(source_path).stem}.npz"
        if output_path in output_paths:
            raise ValueError(f"duplicate test feature stem creates ambiguous resume artifact: {output_path}")
        output_paths.add(output_path)
        if output_path.exists() and not args.no_resume:
            record = record_from_file(output_path)
            if record[3] != source_path:
                raise ValueError(f"{output_path}: source path differs from current test CSV; use --clean")
        else:
            record = infer_item(model, item, options.visual_length, list(XD_LABELS.values()), device)
            np.savez_compressed(
                output_path,
                prob1=record[0],
                prob2=record[1],
                logits2=record[2],
                source_path=np.asarray(record[3]),
                label=np.asarray(record[4]),
            )
        records.append(record)
    metrics = summarize_records(
        records,
        str(Path(args.vadclip_root) / "list" / "gt.npy"),
        str(Path(args.vadclip_root) / "list" / "gt_segment.npy"),
        str(Path(args.vadclip_root) / "list" / "gt_label.npy"),
        args.vadclip_root,
    )
    save_json(out_dir / "metrics.json", {
        **metrics,
        "module": args.module,
        "model_path": args.model_path,
    })
    print_metrics(metrics)
    print(f"wrote {out_dir / 'metrics.json'} and {len(records)} resumable per-video outputs", flush=True)


if __name__ == "__main__":
    main()
