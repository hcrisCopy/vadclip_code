#!/usr/bin/env python3
"""Sweep one global residual scale for a fixed, trained global-768 VadCLIP model.

The model checkpoint and frozen VadCLIP baseline are never changed.  For each
candidate ``alpha``, inference uses ``clip + alpha * gate * correction`` and
then calls the unchanged frozen VadCLIP backbone.  Metrics and model-selection
criteria come from the repository's ordinary UCF/XD evaluation helpers.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


INJECTION_DIR = Path(__file__).resolve().parents[1] / "vadclip_neuron_injection"
if str(INJECTION_DIR) not in sys.path:
    sys.path.insert(0, str(INJECTION_DIR))

from common import clean_dir, ensure_dir, load_json, save_json, write_csv
from models import add_vadclip_source, build_residual_model, split_concat


def state_dict_from_file(path: str) -> dict:
    """Load this repository's raw-model or resumable-checkpoint state dict."""
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def baseline_options(dataset: str, vadclip_root: str):
    """Create unchanged official VadCLIP construction options."""
    add_vadclip_source(vadclip_root)
    if dataset == "ucf":
        import ucf_option as option_module
    else:
        import xd_option as option_module
    return option_module.parser.parse_args([])


def evaluation_components(dataset: str):
    """Return the ordinary loader, summarizer and selection rule for one baseline dataset."""
    if dataset == "ucf":
        from evaluation import build_test_loader, print_metrics, save_metrics, summarize_records

        return {
            "build_test_loader": build_test_loader,
            "print_metrics": print_metrics,
            "save_metrics": save_metrics,
            "summarize_records": summarize_records,
            "prompts": [
                "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
                "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
            ],
            "selection_metric": "roc_auc_logits1",
        }
    from xd_evaluation import build_test_loader, print_metrics, save_metrics, summarize_records

    return {
        "build_test_loader": build_test_loader,
        "print_metrics": print_metrics,
        "save_metrics": save_metrics,
        "summarize_records": summarize_records,
        "prompts": ["normal", "fighting", "shooting", "riot", "abuse", "car accident", "explosion"],
        "selection_metric": "ap_logits2",
    }


def alpha_name(alpha: float) -> str:
    """Stable directory name that does not depend on Python's float repr."""
    return f"alpha_{alpha:.8g}".replace("-", "neg_").replace(".", "p")


def model_inputs(item, visual_length: int, device: torch.device):
    """Match ordinary evaluation's split, valid-length and padding-mask construction."""
    from common import test_chunk_lengths

    visual = item[0].squeeze(0)
    original_length = int(item[1].item())
    if original_length < visual_length:
        visual = visual.unsqueeze(0)
    visual = visual.to(device, non_blocking=True)
    lengths = torch.as_tensor(test_chunk_lengths(original_length, visual_length), dtype=torch.int64, device=device)
    padding_mask = torch.zeros((len(lengths), visual_length), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths.tolist()):
        if length < visual_length:
            padding_mask[index, max(length, 0):] = True
    return visual, original_length, padding_mask, lengths


def infer_item_with_alpha(
    model,
    item,
    visual_length: int,
    prompts: list[str],
    alpha: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Run one official test item after scaling only the trained residual."""
    visual, original_length, padding_mask, lengths = model_inputs(item, visual_length, device)
    path, label = str(item[2][0]), str(item[3][0])
    with torch.no_grad():
        neuron, clip = split_concat(visual, model.neuron_width, model.clip_dim)
        correction = model.neuron_to_clip(model.neuron_norm(neuron))
        enhanced_clip = clip + float(alpha) * model.residual_gate().to(clip.dtype) * correction.to(clip.dtype)
        _text, logits1, logits2 = model.base(enhanced_clip, padding_mask, prompts, lengths)
    logits1 = logits1.reshape(-1, logits1.shape[-1])[:original_length]
    logits2 = logits2.reshape(-1, logits2.shape[-1])[:original_length]
    prob1 = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy().astype(np.float32)
    prob2 = (1.0 - logits2.softmax(dim=-1)[:, 0]).cpu().numpy().astype(np.float32)
    return prob1, prob2, logits2.cpu().numpy().astype(np.float32), path, label


def record_from_file(path: Path, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Read one reusable alpha prediction and reject a mismatched artifact."""
    artifact = np.load(path, allow_pickle=False)
    required = {"prob1", "prob2", "logits2", "source_path", "label", "alpha"}
    if not required.issubset(artifact.files):
        raise ValueError(f"{path}: incomplete alpha prediction; use --clean")
    saved_alpha = float(artifact["alpha"].item())
    if not math.isclose(saved_alpha, alpha, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError(f"{path}: saved alpha={saved_alpha} differs from requested alpha={alpha}; use --clean")
    return (
        np.asarray(artifact["prob1"], dtype=np.float32),
        np.asarray(artifact["prob2"], dtype=np.float32),
        np.asarray(artifact["logits2"], dtype=np.float32),
        str(artifact["source_path"].item()),
        str(artifact["label"].item()),
    )


def parse_alphas(values: list[float]) -> list[float]:
    """Validate and preserve the user's deterministic candidate order."""
    if not values:
        raise ValueError("at least one alpha is required")
    result: list[float] = []
    for value in values:
        alpha = float(value)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"alpha must be finite and non-negative, got {value!r}")
        if any(math.isclose(alpha, prior, abs_tol=1e-12, rel_tol=0.0) for prior in result):
            raise ValueError(f"duplicate alpha candidate: {alpha}")
        result.append(alpha)
    return result


def metric_row(alpha: float, metrics: dict[str, object], selection_metric: str) -> list[object]:
    """Write all shared metrics plus optional UCF Ano-AUC fields to one CSV row."""
    return [
        alpha,
        float(metrics[selection_metric]),
        float(metrics["roc_auc_logits1"]), float(metrics["ap_logits1"]),
        float(metrics["roc_auc_logits2"]), float(metrics["ap_logits2"]),
        float(metrics["detection_map_average"]),
        "" if "ano_auc_logits1" not in metrics else float(metrics["ano_auc_logits1"]),
        "" if "ano_auc_logits2" not in metrics else float(metrics["ano_auc_logits2"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline-aligned global residual-alpha sweep for a fixed VadCLIP checkpoint.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--selection-list", required=True, help="Use the same baseline-aligned model-selection CSV.")
    parser.add_argument("--selection-gt-path", required=True)
    parser.add_argument("--selection-segment-path", required=True)
    parser.add_argument("--selection-label-path", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--model-path", required=True, help="Fixed trained residual checkpoint; it is never modified.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.clean and args.no_resume:
        parser.error("--clean and --no-resume cannot be used together")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    alphas = parse_alphas(args.alphas)
    for path in (
        args.selection_list, args.selection_gt_path, args.selection_segment_path, args.selection_label_path,
        args.neuron_json, args.model_path,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing alpha-sweep input: {path}")

    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    if expected_width != int(contract.get("neuron_width", 768)) + int(contract.get("clip_dim", 512)):
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    options = baseline_options(args.dataset, args.vadclip_root)
    model = build_residual_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    model.load_state_dict(state_dict_from_file(args.model_path), strict=True)
    model.freeze_base()
    model.to(device).eval()
    components = evaluation_components(args.dataset)
    selection_metric = str(components["selection_metric"])
    loader = components["build_test_loader"](
        args.selection_list, options.visual_length, expected_width, args.num_workers
    )
    metrics_by_alpha: list[tuple[float, dict[str, object]]] = []

    for alpha in alphas:
        alpha_dir = ensure_dir(out_dir / alpha_name(alpha))
        per_video_dir = ensure_dir(alpha_dir / "per_video")
        records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]] = []
        output_paths: set[Path] = set()
        for item in tqdm(loader, desc=f"alpha={alpha:g} selection inference", unit="video"):
            source_path = str(item[2][0])
            output_path = per_video_dir / f"{Path(source_path).stem}.npz"
            if output_path in output_paths:
                raise ValueError(f"duplicate feature stem creates ambiguous resume artifact: {output_path}")
            output_paths.add(output_path)
            if output_path.exists() and not args.no_resume:
                record = record_from_file(output_path, alpha)
                if record[3] != source_path:
                    raise ValueError(f"{output_path}: source path differs from current selection CSV; use --clean")
            else:
                record = infer_item_with_alpha(
                    model, item, options.visual_length, components["prompts"], alpha, device
                )
                np.savez_compressed(
                    output_path,
                    prob1=record[0], prob2=record[1], logits2=record[2],
                    source_path=np.asarray(record[3]), label=np.asarray(record[4]), alpha=np.asarray(alpha),
                )
            records.append(record)
        metrics = components["summarize_records"](
            records, args.selection_gt_path, args.selection_segment_path, args.selection_label_path, args.vadclip_root
        )
        components["save_metrics"](alpha_dir / "metrics.json", metrics)
        components["print_metrics"](metrics)
        print(f"alpha={alpha:g} | {selection_metric}={float(metrics[selection_metric]):.6f}", flush=True)
        metrics_by_alpha.append((alpha, metrics))

    best_alpha, best_metrics = metrics_by_alpha[0]
    for alpha, metrics in metrics_by_alpha[1:]:
        if float(metrics[selection_metric]) > float(best_metrics[selection_metric]):
            best_alpha, best_metrics = alpha, metrics
    write_csv(
        out_dir / "alpha_metrics.csv",
        [
            "alpha", "selection_metric", "auc1", "ap1", "auc2", "ap2", "detection_map_average",
            "ano_auc1", "ano_auc2",
        ],
        [metric_row(alpha, metrics, selection_metric) for alpha, metrics in metrics_by_alpha],
    )
    save_json(out_dir / "selected_alpha.json", {
        "method": "global768_residual_alpha_calibration_v1",
        "dataset": args.dataset,
        "model_path": args.model_path,
        "selection_list": args.selection_list,
        "selection_metric": selection_metric,
        "candidate_alphas": alphas,
        "selected_alpha": best_alpha,
        "selected_metrics": best_metrics,
        "tie_break": "keep the earliest user-supplied alpha when selection metrics are equal",
        "model_parameters_updated": False,
        "frame_labels_used_for_training": False,
    })
    save_json(out_dir / "run_config.json", vars(args))
    print(
        f"selected alpha={best_alpha:g} by {selection_metric}={float(best_metrics[selection_metric]):.6f} "
        f"-> {out_dir / 'selected_alpha.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
