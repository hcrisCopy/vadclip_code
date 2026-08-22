#!/usr/bin/env python3
"""Train the zero-start, normal-anchored neuron residual on XD-Violence.

The data loader, official VadCLIP losses, optimiser, scheduler, per-epoch
validation, and AP2 checkpoint rule are reused unchanged.  This script adds
only a residual-feature anchor on labelled normal *training* videos.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def add_shared_source() -> None:
    """Import shared VadCLIP experiment utilities from this repository only."""
    source = str(Path(__file__).resolve().parents[1] / "vadclip_neuron_injection")
    if source not in sys.path:
        # Keep this directory first so ``from models`` resolves to this M2
        # adapter rather than the earlier gated-adapter implementation.
        sys.path.append(source)


add_shared_source()
from common import XD_LABELS, clean_dir, ensure_dir, is_normal_label, load_json, save_json  # noqa: E402
from train_single_vadclip_style_xd import (  # noqa: E402
    XDConcatTrainDataset,
    append_history,
    baseline_options,
    clas2_loss,
    clasm_loss,
    initialize_from_baseline,
    label_tensor,
    remove_if_file,
    set_seed,
    state_dict_from_file,
    text_separation_loss,
)
from xd_evaluation import build_test_loader, print_metrics, run_evaluation  # noqa: E402
from models import build_residual_model  # noqa: E402


def normal_residual_anchor(
    residual: torch.Tensor,
    normal_mask: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Penalise residual changes only on valid snippets from normal videos."""
    if residual.ndim != 3:
        raise ValueError(f"expected residual [B,T,D], got {tuple(residual.shape)}")
    time = torch.arange(residual.shape[1], device=residual.device).unsqueeze(0)
    valid = time < lengths.clamp(min=0, max=residual.shape[1]).unsqueeze(1)
    selected = valid & normal_mask.unsqueeze(1)
    count = int(selected.sum().item())
    if count == 0:
        return residual.new_zeros(()), 0
    return residual[selected].square().mean(), count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train frozen VadCLIP with zero-start neuron residual scale and normal-video feature anchor."
    )
    parser.add_argument("--dataset", choices=["xd"], required=True)
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
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument(
        "--residual-scale-init",
        type=float,
        default=0.0,
        help="Initial signed residual scale; zero preserves the supplied baseline exactly.",
    )
    parser.add_argument(
        "--normal-anchor-weight",
        type=float,
        default=0.10,
        help="Weight of the normal-training-video residual feature anchor.",
    )
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
    if args.normal_anchor_weight < 0:
        parser.error("--normal-anchor-weight must be non-negative")

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
    model = build_residual_model(
        options,
        args.vadclip_root,
        str(device),
        contract,
        args.residual_hidden_dim,
        args.residual_depth,
        args.residual_scale_init,
    )

    start_epoch, best_metric = 0, 0.0
    resume_checkpoint = None
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint["best_metric"])
        model.freeze_base()
        print(f"resuming from epoch {start_epoch + 1}, best AP2={best_metric:.6f}", flush=True)
    else:
        copied = initialize_from_baseline(model, args.init_baseline_model)
        model.freeze_base()
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)

    model.to(device)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr
    )
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    print(f"trainable zero-start residual parameters: {trainable}", flush=True)
    print(
        f"M2 normal anchor: weight={args.normal_anchor_weight:g}, "
        f"initial_scale={args.residual_scale_init:g}",
        flush=True,
    )

    train_loader = DataLoader(
        XDConcatTrainDataset(args.train_list, options.visual_length, expected_width),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    if not len(train_loader):
        raise RuntimeError("XD training loader is empty; inspect CSV or batch size")
    test_loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    gt_path = str(Path(args.vadclip_root) / "list" / "gt.npy")
    segment_path = str(Path(args.vadclip_root) / "list" / "gt_segment.npy")
    label_path = str(Path(args.vadclip_root) / "list" / "gt_label.npy")
    save_json(out_dir / "run_config.json", {
        "dataset": args.dataset,
        "train_list": args.train_list,
        "test_list": args.test_list,
        "neuron_json": args.neuron_json,
        "vadclip_root": args.vadclip_root,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epoch": args.max_epoch,
        "num_workers": args.num_workers,
        "residual_hidden_dim": args.residual_hidden_dim,
        "residual_depth": args.residual_depth,
        "residual_scale_init": args.residual_scale_init,
        "normal_anchor_weight": args.normal_anchor_weight,
        "normal_anchor_definition": "mean squared injected 512D residual on valid snippets from normal training videos only",
        "initial_prediction": "exactly the supplied frozen baseline when residual_scale_init=0",
        "validation": "once per epoch, matching official VadCLIP XD",
        "checkpoint_selection": "maximum XD language-branch AP2, matching official VadCLIP xd_train.py",
        "test_gt_used_in_training": False,
    })

    prompt_text = list(XD_LABELS.values())
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0, "loss_anchor": 0.0}
        normal_snippets = 0
        progress = tqdm(train_loader, desc=f"train M2 epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for iteration, (visual, raw_labels, lengths, _unused_teacher) in enumerate(progress):
            visual = visual.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            normal_mask = torch.tensor(
                [is_normal_label("xd", str(label)) for label in raw_labels], dtype=torch.bool, device=device
            )
            labels = label_tensor(list(raw_labels), device)
            text_features, logits1, logits2, residual = model.forward_with_residual(
                visual, None, prompt_text, lengths
            )
            loss1 = clas2_loss(logits1, labels, lengths)
            loss2 = clasm_loss(logits2, labels, lengths)
            loss3 = text_separation_loss(text_features)
            anchor, count = normal_residual_anchor(residual, normal_mask, lengths)
            loss = loss1 + loss2 + loss3 * 1e-4 + args.normal_anchor_weight * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            normal_snippets += count
            for key, value in (
                ("loss", loss), ("loss1", loss1), ("loss2", loss2), ("loss3", loss3), ("loss_anchor", anchor),
            ):
                totals[key] += float(value.item())
            postfix = {key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()}
            postfix["scale"] = f"{float(model.residual_scale_value().detach().cpu()):.5f}"
            progress.set_postfix(postfix)

        scheduler.step()
        _records, metrics = run_evaluation(
            model,
            test_loader,
            options.visual_length,
            device,
            gt_path,
            segment_path,
            label_path,
            args.vadclip_root,
            description=f"validation M2 epoch {epoch + 1}",
        )
        print_metrics(metrics)
        metric = float(metrics["ap_logits2"])
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), model_path)
            print(f"new best model: AP2={best_metric:.6f} -> {model_path}", flush=True)
        if not model_path.exists():
            raise RuntimeError("No best model was written; AP2 must be a finite positive value")
        model.load_state_dict(state_dict_from_file(str(model_path)), strict=True)
        torch.save({
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }, checkpoint_path)
        append_history(out_dir / "history.csv", {
            "epoch": epoch + 1,
            **{key: totals[key] / len(train_loader) for key in totals},
            "normal_anchor_snippets": normal_snippets,
            "roc_auc_logits1": metrics["roc_auc_logits1"],
            "ap_logits1": metrics["ap_logits1"],
            "roc_auc_logits2": metrics["roc_auc_logits2"],
            "ap_logits2": metrics["ap_logits2"],
            "detection_map_average": metrics["detection_map_average"],
            "residual_scale": float(model.residual_scale_value().detach().cpu()),
        })
    print(f"finished; best XD language AP2={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
