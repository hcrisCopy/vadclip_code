#!/usr/bin/env python3
"""Measure how much the trained global-768 residual changes VadCLIP inputs and scores.

This is an inference-only diagnostic.  It never reads frame annotations and
never updates model parameters.  The test preprocessing, prompt order and
score definitions match the project's ordinary UCF/XD evaluation helpers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from shared import add_injection_source, state_dict_from_file

add_injection_source()
from common import clean_dir, ensure_dir, load_json, save_json, write_csv
from models import add_vadclip_source, build_residual_model, split_concat


def baseline_options(dataset: str, vadclip_root: str):
    """Load the unchanged official options needed to construct VadCLIP."""
    add_vadclip_source(vadclip_root)
    if dataset == "ucf":
        import ucf_option as option_module
    else:
        import xd_option as option_module
    return option_module.parser.parse_args([])


def evaluation_inputs(dataset: str):
    """Return the official test loader and prompt order for one dataset."""
    if dataset == "ucf":
        from evaluation import build_test_loader

        return build_test_loader, [
            "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
            "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
        ]
    from xd_evaluation import build_test_loader

    return build_test_loader, ["normal", "fighting", "shooting", "riot", "abuse", "car accident", "explosion"]


def build_model_inputs(item, visual_length: int, device: torch.device):
    """Reproduce the regular test-time split, padding-mask and length handling."""
    from common import test_chunk_lengths

    visual = item[0].squeeze(0)
    original_length = int(item[1].item())
    if visual.ndim == 2:
        visual = visual.unsqueeze(0)
    if visual.ndim != 3:
        raise ValueError(f"expected split test feature [chunks,T,D], got {tuple(visual.shape)}")
    visual = visual.to(device, non_blocking=True)
    lengths = torch.as_tensor(test_chunk_lengths(original_length, visual_length), dtype=torch.int64, device=device)
    if visual.shape[0] != len(lengths):
        raise ValueError(
            f"test split has {visual.shape[0]} chunks but official length construction returned {len(lengths)}"
        )
    padding_mask = torch.zeros((len(lengths), visual_length), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths.tolist()):
        if length < visual_length:
            padding_mask[index, max(length, 0):] = True
    return visual, original_length, padding_mask, lengths


def anomaly_probabilities(logits1: torch.Tensor, logits2: torch.Tensor, original_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the same binary anomaly scores that ordinary evaluation reports."""
    logits1 = logits1.reshape(-1, logits1.shape[-1])[:original_length]
    logits2 = logits2.reshape(-1, logits2.shape[-1])[:original_length]
    return torch.sigmoid(logits1.squeeze(-1)), 1.0 - logits2.softmax(dim=-1)[:, 0]


def artifact_from_file(path: Path) -> dict[str, object]:
    """Load one resumable per-video diagnostic artifact with validation."""
    artifact = np.load(path, allow_pickle=False)
    required = {
        "source_path", "label", "clip_l2", "correction_l2", "weighted_residual_l2",
        "residual_to_clip_ratio", "prob1_delta", "prob2_delta",
    }
    if not required.issubset(artifact.files):
        raise ValueError(f"{path}: incomplete residual contribution artifact; use --clean")
    result: dict[str, object] = {
        "source_path": str(artifact["source_path"].item()),
        "label": str(artifact["label"].item()),
    }
    for key in required - {"source_path", "label"}:
        result[key] = np.asarray(artifact[key], dtype=np.float32).reshape(-1)
    lengths = {len(value) for key, value in result.items() if key not in {"source_path", "label"}}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"{path}: invalid or inconsistent per-snippet diagnostic arrays")
    return result


def distribution(values: np.ndarray) -> dict[str, float | int]:
    """Small JSON-safe descriptive summary for one per-snippet quantity."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("cannot summarize an empty diagnostic quantity")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def per_video_row(record: dict[str, object]) -> list[object]:
    """Make a compact CSV row while full per-snippet values stay resumable in NPZ."""
    ratio = np.asarray(record["residual_to_clip_ratio"], dtype=np.float32)
    return [
        record["source_path"], record["label"], len(ratio),
        float(np.asarray(record["clip_l2"]).mean()),
        float(np.asarray(record["correction_l2"]).mean()),
        float(np.asarray(record["weighted_residual_l2"]).mean()),
        float(ratio.mean()), float(np.median(ratio)), float(np.quantile(ratio, 0.95)),
        float(np.abs(np.asarray(record["prob1_delta"])).mean()),
        float(np.abs(np.asarray(record["prob2_delta"])).mean()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only residual contribution diagnostic for global-768 VadCLIP.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True, help="Any concat test CSV, such as the diagnostic held-out CSV.")
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--model-path", required=True, help="Trained integrated residual model checkpoint.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--ratio-epsilon", type=float, default=1e-8)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.num_workers < 0 or args.ratio_epsilon <= 0:
        parser.error("--num-workers must be non-negative and --ratio-epsilon must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    for path in (args.test_list, args.neuron_json, args.model_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing diagnostic input: {path}")

    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "per_video")
    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    options = baseline_options(args.dataset, args.vadclip_root)
    model = build_residual_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    model.load_state_dict(state_dict_from_file(args.model_path), strict=True)
    model.freeze_base()
    model.to(device).eval()
    gate_logit = float(model.gate_logit.detach().cpu().item())
    gate = float(model.residual_gate().detach().cpu().item())
    build_test_loader, prompts = evaluation_inputs(args.dataset)
    loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)

    records: list[dict[str, object]] = []
    output_paths: set[Path] = set()
    for item in tqdm(loader, desc="residual contribution diagnostic", unit="video"):
        source_path, label = str(item[2][0]), str(item[3][0])
        output_path = score_dir / f"{Path(source_path).stem}.npz"
        if output_path in output_paths:
            raise ValueError(f"duplicate test feature stem creates ambiguous resume artifact: {output_path}")
        output_paths.add(output_path)
        if output_path.exists() and not args.no_resume:
            record = artifact_from_file(output_path)
            if record["source_path"] != source_path:
                raise ValueError(f"{output_path}: source path differs from current test CSV; use --clean")
        else:
            visual, original_length, padding_mask, lengths = build_model_inputs(item, options.visual_length, device)
            with torch.no_grad():
                neuron, clip = split_concat(visual, model.neuron_width, model.clip_dim)
                correction = model.neuron_to_clip(model.neuron_norm(neuron))
                weighted_residual = model.residual_gate().to(clip.dtype) * correction.to(clip.dtype)
                _, base_logits1, base_logits2 = model.base(clip, padding_mask, prompts, lengths)
                _, enhanced_logits1, enhanced_logits2 = model(visual, padding_mask, prompts, lengths)
                base_prob1, base_prob2 = anomaly_probabilities(base_logits1, base_logits2, original_length)
                enhanced_prob1, enhanced_prob2 = anomaly_probabilities(enhanced_logits1, enhanced_logits2, original_length)

                # Flatten then truncate exactly as the ordinary test evaluator does.
                flat_clip = clip.reshape(-1, clip.shape[-1])[:original_length]
                flat_correction = correction.reshape(-1, correction.shape[-1])[:original_length]
                flat_weighted = weighted_residual.reshape(-1, weighted_residual.shape[-1])[:original_length]
                clip_l2 = flat_clip.norm(dim=-1)
                correction_l2 = flat_correction.norm(dim=-1)
                weighted_l2 = flat_weighted.norm(dim=-1)
                ratio = weighted_l2 / clip_l2.clamp_min(args.ratio_epsilon)
                record = {
                    "source_path": source_path,
                    "label": label,
                    "clip_l2": clip_l2.cpu().numpy().astype(np.float32),
                    "correction_l2": correction_l2.cpu().numpy().astype(np.float32),
                    "weighted_residual_l2": weighted_l2.cpu().numpy().astype(np.float32),
                    "residual_to_clip_ratio": ratio.cpu().numpy().astype(np.float32),
                    "prob1_delta": (enhanced_prob1 - base_prob1).cpu().numpy().astype(np.float32),
                    "prob2_delta": (enhanced_prob2 - base_prob2).cpu().numpy().astype(np.float32),
                }
            np.savez_compressed(output_path, **record)
        records.append(record)

    quantities = [
        "clip_l2", "correction_l2", "weighted_residual_l2", "residual_to_clip_ratio", "prob1_delta", "prob2_delta",
    ]
    values = {key: np.concatenate([np.asarray(record[key], dtype=np.float32) for record in records]) for key in quantities}
    summary = {
        "method": "global768_residual_contribution_diagnostic_v1",
        "dataset": args.dataset,
        "test_list": args.test_list,
        "model_path": args.model_path,
        "neuron_json": args.neuron_json,
        "videos": len(records),
        "snippets": int(len(values["clip_l2"])),
        "gate_logit": gate_logit,
        "gate_sigmoid": gate,
        "distributions": {key: distribution(value) for key, value in values.items()},
        "absolute_score_change": {
            "prob1": distribution(np.abs(values["prob1_delta"])),
            "prob2": distribution(np.abs(values["prob2_delta"])),
        },
        "meaning": {
            "residual_to_clip_ratio": "||sigmoid(gate_logit) * correction||_2 / max(||original_clip||_2, ratio_epsilon)",
            "probability_delta": "residual-injected anomaly probability minus frozen-base anomaly probability",
            "frame_annotations_used": False,
            "parameters_updated": False,
        },
    }
    save_json(out_dir / "summary.json", summary)
    write_csv(
        out_dir / "per_video.csv",
        [
            "source_path", "label", "snippets", "clip_l2_mean", "correction_l2_mean", "weighted_residual_l2_mean",
            "residual_to_clip_ratio_mean", "residual_to_clip_ratio_p50", "residual_to_clip_ratio_p95",
            "abs_prob1_delta_mean", "abs_prob2_delta_mean",
        ],
        [per_video_row(record) for record in records],
    )
    save_json(out_dir / "run_config.json", vars(args))
    ratio_stats = summary["distributions"]["residual_to_clip_ratio"]
    score_stats = summary["absolute_score_change"]["prob2"]
    print(
        "residual contribution | "
        f"gate=sigmoid({gate_logit:.6f})={gate:.6f} | "
        f"residual/clip p50={ratio_stats['p50']:.6f} p95={ratio_stats['p95']:.6f} | "
        f"|delta prob2| mean={score_stats['mean']:.6f} p95={score_stats['p95']:.6f}",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'} and {len(records)} resumable per-video artifacts", flush=True)


if __name__ == "__main__":
    main()
