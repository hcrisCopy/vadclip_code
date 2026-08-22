#!/usr/bin/env python3
"""Train M1/M2 neuron-conditioned feature modulation on XD-Violence.

The XD data loader, three official weak losses, optimiser schedule, validation
cadence and AP2 model-selection rule follow ``VadCLIP/src/xd_train.py``.
Only the frozen-baseline input is changed from the original cached 512D CLIP
feature to its identity-start M1 or M2 enhanced version.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from models import add_vadclip_source, build_film_model
from xd_shared import (
    XD_LABELS,
    build_test_loader,
    build_train_loader,
    clas2_loss,
    clean_dir,
    clasm_loss,
    ensure_dir,
    label_tensor,
    load_json,
    print_metrics,
    run_evaluation,
    save_json,
    text_separation_loss,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def baseline_options(vadclip_root: str):
    add_vadclip_source(vadclip_root)
    import xd_option

    return xd_option.parser.parse_args([])


def state_dict_from_file(path: str) -> dict:
    """Load plain, checkpoint-wrapped, or DDP-prefixed state dictionaries."""
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def initialize_from_baseline(model, baseline_path: str) -> int:
    """Copy only official VadCLIP tensors into the wrapped frozen baseline."""
    source, target = state_dict_from_file(baseline_path), model.state_dict()
    missing, mismatched, copied = [], [], 0
    for name, tensor in target.items():
        if not name.startswith("base."):
            continue
        source_name = name.removeprefix("base.")
        candidate = source.get(source_name)
        if candidate is None:
            missing.append(source_name)
        elif tuple(candidate.shape) != tuple(tensor.shape):
            mismatched.append((source_name, tuple(candidate.shape), tuple(tensor.shape)))
        else:
            target[name] = candidate
            copied += 1
    if missing or mismatched:
        raise RuntimeError(
            "baseline model is incompatible with official 512D XD VadCLIP: "
            f"missing={missing[:4]}, shape_mismatch={mismatched[:2]}"
        )
    model.load_state_dict(target, strict=True)
    return copied


def append_history(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def remove_if_file(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-VadCLIP M1/M2 neuron-conditioned feature modulation on XD.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--module", choices=["m1", "m2"], required=True, help="m1: per-snippet FiLM; m2: FiLM plus temporal mixer.")
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96, help="Official VadCLIP XD batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Official VadCLIP XD learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="Official VadCLIP XD loader default.")
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--condition-hidden-dim", type=int, default=256)
    parser.add_argument("--residual-hidden-dim", type=int, default=512)
    parser.add_argument("--temporal-dilations", type=int, nargs="+", default=[1, 2], help="Used by M2 only.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    if args.max_epoch <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("max-epoch, batch-size and lr must be positive")
    if args.condition_hidden_dim <= 0 or args.residual_hidden_dim <= 0 or any(value <= 0 for value in args.temporal_dilations):
        parser.error("hidden dimensions and temporal dilations must be positive")

    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    checkpoint_path, model_path = Path(args.checkpoint_path), Path(args.model_path)
    ensure_dir(checkpoint_path.parent)
    ensure_dir(model_path.parent)
    if args.clean:
        remove_if_file(checkpoint_path)
        remove_if_file(model_path)
    set_seed(args.seed)

    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    if expected_width != int(contract.get("neuron_width", 768)) + int(contract.get("clip_dim", 512)):
        raise ValueError("neuron JSON has an invalid concat contract")
    options = baseline_options(args.vadclip_root)
    options.visual_width, options.visual_length = 512, int(options.visual_length)
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

    start_epoch, best_metric = 0, 0.0
    resume_checkpoint = None
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if str(resume_checkpoint.get("module", args.module)) != args.module:
            raise ValueError("checkpoint module does not match --module")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint["best_metric"])
        model.freeze_base()
        print(f"resuming {args.module} from epoch {start_epoch + 1}, best AP2={best_metric:.6f}", flush=True)
    else:
        copied = initialize_from_baseline(model, args.init_baseline_model)
        model.freeze_base()
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)

    # The wrapped official CLIP can be constructed on CUDA while new modules
    # start on CPU; move the full wrapper before creating the optimiser.
    model.to(device)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    print(f"trainable {args.module} parameters: {trainable}", flush=True)

    train_loader = build_train_loader(args.train_list, options.visual_length, expected_width, args.batch_size, args.num_workers)
    if not len(train_loader):
        raise RuntimeError("XD training loader is empty; inspect CSV or batch size")
    test_loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    gt_path = str(Path(args.vadclip_root) / "list" / "gt.npy")
    segment_path = str(Path(args.vadclip_root) / "list" / "gt_segment.npy")
    label_path = str(Path(args.vadclip_root) / "list" / "gt_label.npy")
    save_json(out_dir / "run_config.json", {
        "dataset": args.dataset,
        "module": args.module,
        "train_list": args.train_list,
        "test_list": args.test_list,
        "neuron_json": args.neuron_json,
        "vadclip_root": args.vadclip_root,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epoch": args.max_epoch,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "condition_hidden_dim": args.condition_hidden_dim,
        "residual_hidden_dim": args.residual_hidden_dim,
        "temporal_dilations": args.temporal_dilations,
        "training_loss": "official XD VadCLIP weak losses only; no pseudo-score or ground-truth auxiliary loss",
        "baseline": "all original VadCLIP parameters are frozen",
        "checkpoint_selection": "maximum XD language-branch AP2, matching official VadCLIP xd_train.py",
    })

    prompts = list(XD_LABELS.values())
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0, "ratio": 0.0, "gamma": 0.0}
        progress = tqdm(train_loader, desc=f"{args.module} train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for iteration, (visual, raw_labels, lengths) in enumerate(progress):
            visual = visual.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = label_tensor(list(raw_labels), device)
            text_features, logits1, logits2 = model(visual, None, prompts, lengths)
            loss1 = clas2_loss(logits1, labels, lengths)
            loss2 = clasm_loss(logits2, labels, lengths)
            loss3 = text_separation_loss(text_features)
            loss = loss1 + loss2 + loss3 * 1e-4
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats = model.residual_statistics()
            values = {
                "loss": float(loss.item()),
                "loss1": float(loss1.item()),
                "loss2": float(loss2.item()),
                "loss3": float(loss3.item()),
                "ratio": float(stats["residual_ratio"]),
                "gamma": float(stats["gamma_abs"]),
            }
            for key, value in values.items():
                totals[key] += value
            progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})

        scheduler.step()
        _records, metrics = run_evaluation(
            model, test_loader, options.visual_length, device, gt_path, segment_path, label_path,
            args.vadclip_root, description=f"{args.module} validation epoch {epoch + 1}",
        )
        print_metrics(metrics)
        metric = float(metrics["ap_logits2"])
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), model_path)
            print(f"new best {args.module} model: AP2={best_metric:.6f} -> {model_path}", flush=True)
        if not model_path.exists():
            raise RuntimeError("No best model was written; AP2 must be a finite positive value")
        # This preserves the official XD rule: the next epoch begins at the
        # best validation checkpoint, retaining its optimiser/scheduler state.
        model.load_state_dict(state_dict_from_file(str(model_path)), strict=True)
        torch.save({
            "epoch": epoch,
            "module": args.module,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }, checkpoint_path)
        append_history(out_dir / "history.csv", {
            "epoch": epoch + 1,
            **{key: totals[key] / len(train_loader) for key in totals},
            "roc_auc_logits1": metrics["roc_auc_logits1"],
            "ap_logits1": metrics["ap_logits1"],
            "roc_auc_logits2": metrics["roc_auc_logits2"],
            "ap_logits2": metrics["ap_logits2"],
            "detection_map_average": metrics["detection_map_average"],
        })
    print(f"finished {args.module}; best XD language AP2={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
