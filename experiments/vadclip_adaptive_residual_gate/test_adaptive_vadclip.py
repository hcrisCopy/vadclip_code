#!/usr/bin/env python3
"""Resumable UCF/XD evaluation for a trained adaptive snippet-gate model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from shared import add_injection_source, state_dict_from_file

add_injection_source()
from common import clean_dir, ensure_dir, load_json, save_json
from models import add_vadclip_source, build_adaptive_model


def baseline_options(dataset: str, vadclip_root: str):
    add_vadclip_source(vadclip_root)
    if dataset == "ucf":
        import ucf_option as option_module
    else:
        import xd_option as option_module
    return option_module.parser.parse_args([])


def evaluation_components(dataset: str):
    """Reuse the ordinary inference and metric implementations without edits."""
    if dataset == "ucf":
        from evaluation import build_test_loader, infer_item, print_metrics, save_metrics, summarize_records

        return build_test_loader, infer_item, print_metrics, save_metrics, summarize_records, [
            "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
            "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
        ], ("gt_ucf.npy", "gt_segment_ucf.npy", "gt_label_ucf.npy")
    from xd_evaluation import build_test_loader, infer_item, print_metrics, save_metrics, summarize_records

    return build_test_loader, infer_item, print_metrics, save_metrics, summarize_records, [
        "normal", "fighting", "shooting", "riot", "abuse", "car accident", "explosion",
    ], ("gt.npy", "gt_segment.npy", "gt_label.npy")


def record_from_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    artifact = np.load(path, allow_pickle=False)
    required = {"prob1", "prob2", "logits2", "source_path", "label"}
    if not required.issubset(artifact.files):
        raise ValueError(f"{path}: incomplete resumable test artifact")
    return (
        np.asarray(artifact["prob1"], dtype=np.float32),
        np.asarray(artifact["prob2"], dtype=np.float32),
        np.asarray(artifact["logits2"], dtype=np.float32),
        str(artifact["source_path"].item()), str(artifact["label"].item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adaptive snippet-gated global-768 VadCLIP on UCF or XD.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True)
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

    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    for path in (args.test_list, args.neuron_json, args.model_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing adaptive test input: {path}")
    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "per_video")
    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    options = baseline_options(args.dataset, args.vadclip_root)
    model = build_adaptive_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    model.load_state_dict(state_dict_from_file(args.model_path), strict=True)
    model.freeze_base()
    model.to(device).eval()
    build_loader, infer_item, print_metrics, save_metrics, summarize_records, prompts, annotation_names = evaluation_components(args.dataset)
    loader = build_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    records, output_paths = [], set()
    for item in tqdm(loader, desc="adaptive residual test", unit="video"):
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
            record = infer_item(model, item, options.visual_length, prompts, device)
            np.savez_compressed(
                output_path, prob1=record[0], prob2=record[1], logits2=record[2],
                source_path=np.asarray(record[3]), label=np.asarray(record[4]),
            )
        records.append(record)
    annotation_paths = [str(Path(args.vadclip_root) / "list" / name) for name in annotation_names]
    metrics = summarize_records(records, *annotation_paths, args.vadclip_root)
    save_metrics(out_dir / "metrics.json", metrics)
    save_json(out_dir / "evaluation_config.json", {
        "method": "adaptive_snippet_gate_global768_v1",
        "dataset": args.dataset,
        "test_list": args.test_list,
        "model_path": args.model_path,
        "frame_labels_used_for_training": False,
        "pseudo_ranking_enabled": False,
    })
    print_metrics(metrics)
    print(f"wrote {out_dir / 'metrics.json'} and {len(records)} resumable per-video outputs", flush=True)


if __name__ == "__main__":
    main()
