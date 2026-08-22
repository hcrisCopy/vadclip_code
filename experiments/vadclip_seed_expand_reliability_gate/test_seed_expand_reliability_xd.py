#!/usr/bin/env python3
"""Standalone resumable XD test for reliability-gated VadCLIP residual injection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def add_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    source = str(root / "vadclip_neuron_injection")
    if source not in sys.path:
        sys.path.insert(0, source)


add_sources()
from common import clean_dir, ensure_dir, load_json, save_json  # noqa: E402
from models import build_residual_model  # noqa: E402
from test_single_vadclip_style_xd import baseline_options, load_state  # noqa: E402
from xd_evaluation import build_test_loader, print_metrics, save_metrics, summarize_records  # noqa: E402
from xd_reliability import (  # noqa: E402
    infer_reliability_gated_item,
    load_reliability_record,
    reliability_summary,
    runtime_from_contract,
    save_reliability_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a reliability-gated VadCLIP residual model on XD-Violence.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.clean and args.no_resume:
        parser.error("--clean and --no-resume cannot be used together")
    for path in (args.test_list, args.test_hidden_manifest, args.neuron_json, args.model_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing reliability test input: {path}")
    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "per_video")
    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    runtime = runtime_from_contract(contract, args.test_hidden_manifest)
    options = baseline_options(args.vadclip_root)
    model = build_residual_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    model.load_state_dict(load_state(args.model_path), strict=True)
    model.freeze_base()
    model.to(device).eval()
    loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    records, output_paths = [], set()
    for item in tqdm(loader, desc="reliability-gated VadCLIP XD test", unit="video"):
        source_path = str(item[2][0])
        output_path = score_dir / f"{Path(source_path).stem}.npz"
        if output_path in output_paths:
            raise ValueError(f"duplicate test feature stem creates ambiguous resume artifact: {output_path}")
        output_paths.add(output_path)
        if output_path.is_file() and not args.no_resume:
            record = load_reliability_record(output_path)
            if record[3] != source_path:
                raise ValueError(f"{output_path}: source path differs from current test CSV; use --clean")
        else:
            record = infer_reliability_gated_item(model, item, options.visual_length, device, runtime)
            save_reliability_record(output_path, record)
        records.append(record)
    standard_records = [record[:5] for record in records]
    metrics = summarize_records(
        standard_records, str(Path(args.vadclip_root) / "list" / "gt.npy"),
        str(Path(args.vadclip_root) / "list" / "gt_segment.npy"), str(Path(args.vadclip_root) / "list" / "gt_label.npy"),
        args.vadclip_root,
    )
    save_metrics(out_dir / "metrics.json", metrics)
    save_json(out_dir / "reliability_summary.json", {
        "method": "vadclip_seed_expand_reliability_gate_v1",
        "reliability": reliability_summary(records),
        "gate_definition": "base_logits + q * (residual_logits - base_logits)",
        "test_list": args.test_list,
        "test_hidden_manifest": args.test_hidden_manifest,
    })
    print_metrics(metrics)
    print(f"wrote {out_dir / 'metrics.json'}, reliability_summary.json, and {len(records)} resumable per-video outputs", flush=True)


if __name__ == "__main__":
    main()
