#!/usr/bin/env python3
"""Train original VadCLIP residual loss with reliability-gated XD validation.

Training loss, optimiser, scheduler, model mode transitions, per-epoch
validation cadence and AP2 checkpoint selection follow the established
VadCLIP residual experiment.  The only new operation is a frozen q gate on
the residual logit delta during validation, using selector artifacts fixed
before training starts.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def add_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    source = str(root / "vadclip_neuron_injection")
    if source not in sys.path:
        sys.path.insert(0, source)


add_sources()
from common import clean_dir, ensure_dir, load_json, save_json  # noqa: E402
from models import build_residual_model  # noqa: E402
from train_single_vadclip_style_xd import (  # noqa: E402
    XDConcatTrainDataset,
    baseline_options,
    clas2_loss,
    clasm_loss,
    initialize_from_baseline,
    label_tensor,
    set_seed,
    state_dict_from_file,
    text_separation_loss,
)
from xd_evaluation import build_test_loader, print_metrics  # noqa: E402
from xd_reliability import run_reliability_evaluation, runtime_from_contract  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Train original residual loss and select a reliability-gated XD model by AP2.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True, help="Official XD validation/test CSV, as in the baseline protocol.")
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
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
    for path in (args.train_list, args.test_list, args.test_hidden_manifest, args.neuron_json, args.init_baseline_model):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing training input: {path}")

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
        raise ValueError("neuron JSON has an invalid concat width contract")
    runtime = runtime_from_contract(contract, args.test_hidden_manifest)
    options = baseline_options(args.vadclip_root)
    options.visual_width, options.visual_length = 512, int(options.visual_length)
    model = build_residual_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    start_epoch, best_metric, resume_checkpoint = 0, 0.0, None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_epoch, best_metric = int(resume_checkpoint["epoch"]) + 1, float(resume_checkpoint["best_metric"])
        model.freeze_base()
        print(f"resuming from epoch {start_epoch + 1}, best gated AP2={best_metric:.6f}", flush=True)
    else:
        copied = initialize_from_baseline(model, args.init_baseline_model)
        model.freeze_base()
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)
    model.to(device)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    print(f"trainable original residual parameters: {trainable}", flush=True)

    train_loader = DataLoader(
        XDConcatTrainDataset(args.train_list, options.visual_length, expected_width),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    if not len(train_loader):
        raise RuntimeError("XD training loader is empty; inspect the concat train CSV or batch size")
    test_loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    gt_path = str(Path(args.vadclip_root) / "list" / "gt.npy")
    segment_path = str(Path(args.vadclip_root) / "list" / "gt_segment.npy")
    label_path = str(Path(args.vadclip_root) / "list" / "gt_label.npy")
    save_json(out_dir / "run_config.json", {
        "method": "vadclip_seed_expand_reliability_gate_v1", "dataset": args.dataset,
        "train_list": args.train_list, "test_list": args.test_list, "test_hidden_manifest": args.test_hidden_manifest,
        "neuron_json": args.neuron_json, "vadclip_root": args.vadclip_root, "max_epoch": args.max_epoch,
        "batch_size": args.batch_size, "lr": args.lr, "num_workers": args.num_workers, "seed": args.seed,
        "residual_hidden_dim": args.residual_hidden_dim, "residual_depth": args.residual_depth,
        "training_loss": "unchanged original VadCLIP residual-injection loss (classification + language + text separation)",
        "baseline_weights": "frozen", "validation": "once per epoch, baseline-aligned official XD metrics with frozen reliability gating",
        "checkpoint_selection": "maximum gated XD language-branch AP2, matching the baseline AP2 criterion",
        "reliability_gate": "base_logits + q * (residual_logits - base_logits); q uses only frozen score, training normal calibration, and shared hidden features",
    })

    prompts = ["normal", "fighting", "shooting", "riot", "abuse", "car accident", "explosion"]
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0}
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for iteration, (visual, raw_labels, lengths, _teacher_scores) in enumerate(progress):
            visual, lengths = visual.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
            labels = label_tensor(list(raw_labels), device)
            text_features, logits1, logits2 = model(visual, None, prompts, lengths)
            loss1, loss2, loss3 = clas2_loss(logits1, labels, lengths), clasm_loss(logits2, labels, lengths), text_separation_loss(text_features)
            loss = loss1 + loss2 + loss3 * 1e-4
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in (("loss", loss), ("loss1", loss1), ("loss2", loss2), ("loss3", loss3)):
                totals[key] += float(value.item())
            progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})
        scheduler.step()
        _records, metrics = run_reliability_evaluation(
            model, test_loader, options.visual_length, device, runtime, gt_path, segment_path, label_path,
            args.vadclip_root, description=f"reliability-gated validation epoch {epoch + 1}",
        )
        print_metrics(metrics)
        metric = float(metrics["ap_logits2"])
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), model_path)
            print(f"new best reliability-gated model: AP2={best_metric:.6f} -> {model_path}", flush=True)
        if not model_path.exists():
            raise RuntimeError("No best model was written; AP2 must be finite and positive")
        # Same baseline protocol: next epoch begins from the selected best state,
        # while the optimizer and scheduler retain their current states.
        model.load_state_dict(state_dict_from_file(str(model_path)), strict=True)
        torch.save({
            "epoch": epoch, "best_metric": best_metric, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }, checkpoint_path)
        append_history(out_dir / "history.csv", {
            "epoch": epoch + 1, **{key: value / len(train_loader) for key, value in totals.items()},
            "roc_auc_logits1": metrics["roc_auc_logits1"], "ap_logits1": metrics["ap_logits1"],
            "roc_auc_logits2": metrics["roc_auc_logits2"], "ap_logits2": metrics["ap_logits2"],
            "detection_map_average": metrics["detection_map_average"],
            "gate": float(model.residual_gate().detach().cpu()),
        })
    print(f"finished; best reliability-gated XD language AP2={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
