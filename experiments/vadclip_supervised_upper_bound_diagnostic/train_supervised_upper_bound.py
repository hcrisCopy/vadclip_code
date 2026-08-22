#!/usr/bin/env python3
"""Train a leakage-controlled supervised capacity diagnostic on global-768."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SupervisedSnippetDataset
from models import add_vadclip_source, build_model, correction_statistics, valid_snippet_mask
from shared import add_injection_source, initialize_from_baseline, state_dict_from_file

add_injection_source()
from common import UCF_TEST_LABELS, XD_LABELS, clean_dir, ensure_dir, load_json, save_json


def set_seed(seed: int) -> None:
    """Follow the repository's single-process reproducibility policy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def baseline_options(dataset: str, vadclip_root: str):
    """Read official option defaults without changing the baseline source."""
    add_vadclip_source(vadclip_root)
    if dataset == "ucf":
        import ucf_option as option_module
    else:
        import xd_option as option_module
    return option_module.parser.parse_args([])


def evaluation_components(dataset: str):
    """Use existing VadCLIP evaluation and its usual checkpoint criterion."""
    if dataset == "ucf":
        from evaluation import build_test_loader, print_metrics, run_evaluation

        return build_test_loader, print_metrics, run_evaluation, "roc_auc_logits1", list(UCF_TEST_LABELS.values())
    from xd_evaluation import build_test_loader, print_metrics, run_evaluation

    return build_test_loader, print_metrics, run_evaluation, "ap_logits2", list(XD_LABELS.values())


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over true snippets only, avoiding train-time right padding."""
    weights = valid.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def correction_smoothness(delta: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Optional small regulariser on the new scalar correction only."""
    if delta.shape[1] < 2:
        return delta.new_zeros(())
    valid = valid_snippet_mask(lengths, delta.shape[1])
    pair_valid = valid[:, 1:] & valid[:, :-1]
    if not bool(pair_valid.any()):
        return delta.new_zeros(())
    differences = functional.smooth_l1_loss(delta[:, 1:], delta[:, :-1], reduction="none").squeeze(-1)
    return masked_mean(differences, pair_valid)


def append_history(path: Path, row: dict[str, object]) -> None:
    """Append one resume-safe, human-readable epoch row."""
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def remove_file_if_present(path: Path) -> None:
    """Remove only an explicitly named output file when --clean is requested."""
    if path.exists() and path.is_file():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a supervised global-768 capacity diagnostic; never use this as a formal test result."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True, help="Disjoint diagnostic train CSV only.")
    parser.add_argument("--train-gt-path", required=True, help="Frame GT aligned only to --train-list.")
    parser.add_argument("--validation-list", required=True, help="Disjoint checkpoint-selection CSV.")
    parser.add_argument("--validation-gt-path", required=True)
    parser.add_argument("--validation-segment-path", required=True)
    parser.add_argument("--validation-label-path", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None, help="Default: baseline XD=96, UCF=64.")
    parser.add_argument("--lr", type=float, default=None, help="Default: baseline XD=1e-5, UCF=2e-5.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-kernel-size", type=int, default=5)
    parser.add_argument("--delta-logit-cap", type=float, default=4.0)
    parser.add_argument("--logits1-weight", type=float, default=1.0)
    parser.add_argument("--logits2-weight", type=float, default=1.0)
    parser.add_argument("--delta-smooth-weight", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    if args.max_epoch <= 0 or args.num_workers < 0:
        parser.error("max-epoch must be positive and num-workers must be non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("lr must be positive")
    if args.adapter_hidden_dim <= 0 or args.adapter_kernel_size <= 0 or args.adapter_kernel_size % 2 == 0:
        parser.error("adapter-hidden-dim must be positive and adapter-kernel-size must be a positive odd integer")
    if args.delta_logit_cap <= 0 or min(args.logits1_weight, args.logits2_weight, args.delta_smooth_weight) < 0:
        parser.error("delta-logit-cap must be positive and all loss weights must be non-negative")
    if args.logits1_weight == 0 and args.logits2_weight == 0:
        parser.error("at least one supervised loss weight must be positive")
    required_paths = (
        args.train_list, args.train_gt_path, args.validation_list, args.validation_gt_path,
        args.validation_segment_path, args.validation_label_path, args.neuron_json, args.init_baseline_model,
    )
    for path in required_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing supervised diagnostic input: {path}")

    default_batch, default_lr = (64, 2e-5) if args.dataset == "ucf" else (96, 1e-5)
    batch_size = int(args.batch_size if args.batch_size is not None else default_batch)
    lr = float(args.lr if args.lr is not None else default_lr)
    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    checkpoint_path, model_path = Path(args.checkpoint_path), Path(args.model_path)
    ensure_dir(checkpoint_path.parent)
    ensure_dir(model_path.parent)
    if args.clean:
        remove_file_if_present(checkpoint_path)
        remove_file_if_present(model_path)
    set_seed(args.seed)

    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    neuron_width, clip_dim = int(contract.get("neuron_width", 768)), int(contract.get("clip_dim", 512))
    if expected_width != neuron_width + clip_dim:
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    options = baseline_options(args.dataset, args.vadclip_root)
    options.visual_width, options.visual_length = 512, int(options.visual_length)
    model = build_model(
        options, args.vadclip_root, str(device), contract,
        args.adapter_hidden_dim, args.adapter_kernel_size, args.delta_logit_cap,
    )
    start_epoch, best_metric, global_step, resume_checkpoint = 0, float("-inf"), 0, None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        model.freeze_base()
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint["best_metric"])
        global_step = int(resume_checkpoint.get("global_step", 0))
        print(f"resuming from epoch {start_epoch + 1}, best metric={best_metric:.6f}", flush=True)
    else:
        copied = initialize_from_baseline(model, args.init_baseline_model)
        model.freeze_base()
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    print(f"trainable supervised scalar-adapter parameters: {trainable}", flush=True)

    train_dataset = SupervisedSnippetDataset(args.train_list, args.train_gt_path, options.visual_length, expected_width)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    if len(train_loader) <= 0:
        raise RuntimeError("supervised diagnostic train loader is empty")
    build_test_loader, print_metrics, run_evaluation, selection_name, prompts = evaluation_components(args.dataset)
    validation_loader = build_test_loader(args.validation_list, options.visual_length, expected_width, args.num_workers)
    save_json(out_dir / "run_config.json", {
        "method": "supervised_global768_upper_bound_diagnostic_v1",
        "warning": "Diagnostic only: frame labels from a disjoint portion of the original test set supervise training. Do not report it as a formal result.",
        "dataset": args.dataset, "train_list": args.train_list, "train_gt_path": args.train_gt_path,
        "validation_list": args.validation_list, "validation_gt_path": args.validation_gt_path,
        "neuron_json": args.neuron_json, "vadclip_root": args.vadclip_root,
        "base_frozen": True, "encoder_finetuned": False,
        "adapter": "global-768 temporal scalar correction; anomaly class ordering remains unchanged",
        "initial_prediction": "exactly the supplied baseline because the scalar output head is zero initialized",
        "training_frame_labels_used": True, "training_pseudo_scores_used": False,
        "batch_size": batch_size, "lr": lr, "max_epoch": args.max_epoch, "seed": args.seed,
        "adapter_hidden_dim": args.adapter_hidden_dim, "adapter_kernel_size": args.adapter_kernel_size,
        "delta_logit_cap": args.delta_logit_cap, "logits1_weight": args.logits1_weight,
        "logits2_weight": args.logits2_weight, "delta_smooth_weight": args.delta_smooth_weight,
        "checkpoint_selection": f"maximum {selection_name}, matching official VadCLIP {args.dataset}",
    })

    def validate(description: str) -> dict[str, object]:
        _records, metrics = run_evaluation(
            model, validation_loader, options.visual_length, device,
            args.validation_gt_path, args.validation_segment_path, args.validation_label_path,
            args.vadclip_root, description=description,
        )
        print_metrics(metrics)
        return metrics

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "smooth": 0.0, "positive": 0.0, "delta": 0.0, "prob2_shift": 0.0}
        progress = tqdm(train_loader, desc=f"supervised diagnostic epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for iteration, (visual, target, lengths) in enumerate(progress):
            visual = visual.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            _text, logits1, logits2, details = model.forward_with_details(visual, None, prompts, lengths)
            valid = valid_snippet_mask(lengths, logits1.shape[1])
            binary_loss = masked_mean(
                functional.binary_cross_entropy_with_logits(logits1.squeeze(-1), target, reduction="none"), valid
            )
            anomaly_probability = 1.0 - logits2.softmax(dim=-1)[..., 0]
            language_loss = masked_mean(
                functional.binary_cross_entropy(anomaly_probability, target, reduction="none"), valid
            )
            smooth_loss = correction_smoothness(details["delta"], lengths)
            loss = args.logits1_weight * binary_loss + args.logits2_weight * language_loss + args.delta_smooth_weight * smooth_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            statistics = correction_statistics(details, lengths)
            values = {
                "loss": loss, "loss1": binary_loss, "loss2": language_loss, "smooth": smooth_loss,
                "positive": masked_mean(target, valid), "delta": statistics["delta_abs_mean"],
                "prob2_shift": statistics["prob2_shift_abs_mean"],
            }
            for key, value in values.items():
                totals[key] += float(value.detach().item())
            progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})

        metrics = validate(f"supervised diagnostic validation epoch {epoch + 1}")
        metric = float(metrics[selection_name])
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), model_path)
            print(f"new best diagnostic model: {selection_name}={best_metric:.6f} -> {model_path}", flush=True)
        if not model_path.is_file():
            raise RuntimeError("no best model was written; inspect validation metrics")
        scheduler.step()
        # Retain the established experiments' best-model carry-forward behaviour.
        model.load_state_dict(state_dict_from_file(str(model_path)), strict=True)
        torch.save({
            "epoch": epoch, "global_step": global_step, "best_metric": best_metric,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "metrics": metrics,
        }, checkpoint_path)
        append_history(out_dir / "history.csv", {
            "epoch": epoch + 1, "global_step": global_step,
            **{key: value / len(train_loader) for key, value in totals.items()},
            "selection_metric": metric, "roc_auc_logits1": metrics["roc_auc_logits1"],
            "ap_logits1": metrics["ap_logits1"], "roc_auc_logits2": metrics["roc_auc_logits2"],
            "ap_logits2": metrics["ap_logits2"], "detection_map_average": metrics["detection_map_average"],
        })
    print(f"finished diagnostic; best {selection_name}={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
