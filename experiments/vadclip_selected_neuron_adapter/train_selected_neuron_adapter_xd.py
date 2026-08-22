#!/usr/bin/env python3
"""Train zero-start adapters at global-768 CLIP neuron locations on XD-Violence.

VadCLIP, including its CLIP visual encoder and all scoring heads, remains
frozen.  Only the per-layer selected-neuron Adapter parameters are optimised.
The online visual pass is required because cached hidden activations cannot
carry a gradient through later CLIP layers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import baseline_options, clean_dir, ensure_dir, save_json, set_seed, state_dict_from_file
from models import build_model, initialize_frozen_baseline
from online_data import OnlineVideoDataset, one_item_collate, process_train_feature
from xd_utils import (
    append_history,
    clas2_loss,
    clasm_loss,
    infer_sample,
    label_tensor,
    print_metrics,
    prompt_text,
    summarize_records,
    text_separation_loss,
)


def validation_records(model, dataset, visual_length: int, frame_batch_size: int, device: torch.device, epoch: int):
    """Evaluate every epoch as the official XD training launcher does."""
    model.eval()
    records = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=one_item_collate)
    with torch.no_grad():
        for sample in tqdm(loader, desc=f"validation epoch {epoch}", unit="video"):
            records.append(infer_sample(model, sample, visual_length, frame_batch_size, device))
    return records


def remove_if_file(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-VadCLIP selected-neuron Adapters on XD-Violence.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--train-list", required=True, help="Original 512D VadCLIP train CSV, used only for paths/labels.")
    parser.add_argument("--train-hidden-manifest", required=True, help="Reusable hidden manifest with original-video and frame-index metadata.")
    parser.add_argument("--test-list", required=True, help="Original 512D VadCLIP test CSV, used only for paths/labels.")
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96, help="Official XD batch size, implemented as gradient accumulation over online videos.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Official XD learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="Raw video decoding workers; 0 is the safest official-style setting.")
    parser.add_argument("--frame-batch-size", type=int, default=128, help="CLIP frames per GPU forward; changes memory use only, never temporal data.")
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument(
        "--skip-missing-train-manifest",
        action="store_true",
        help="Explicitly omit training rows whose original video is absent from the reusable hidden manifest. Never applies to test rows.",
    )
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    if min(args.max_epoch, args.batch_size, args.frame_batch_size, args.adapter_rank) <= 0 or args.lr <= 0:
        parser.error("max-epoch, batch-size, frame-batch-size, adapter-rank and lr must be positive")

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    checkpoint_path, model_path = Path(args.checkpoint_path), Path(args.model_path)
    ensure_dir(checkpoint_path.parent)
    ensure_dir(model_path.parent)
    if args.clean:
        remove_if_file(checkpoint_path)
        remove_if_file(model_path)
    set_seed(args.seed)
    device = torch.device(args.device)
    options = baseline_options(args.vadclip_root)
    train_dataset = OnlineVideoDataset(
        args.train_list, args.train_hidden_manifest, args.vadclip_root,
        skip_missing_manifest=args.skip_missing_train_manifest,
    )
    test_dataset = OnlineVideoDataset(args.test_list, args.test_hidden_manifest, args.vadclip_root)
    if len(train_dataset.missing_manifest_rows):
        train_dataset.missing_manifest_rows.to_csv(out_dir / "skipped_train_missing_manifest.csv", index=False)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
        collate_fn=one_item_collate,
    )
    if len(train_loader) == 0:
        raise RuntimeError("online XD training loader is empty")
    model = build_model(options, args.vadclip_root, str(device), args.neuron_json, args.adapter_rank)
    start_epoch, best_metric, checkpoint = 0, 0.0, None
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.freeze_base()
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        print(f"resuming at epoch {start_epoch + 1}; best AP2={best_metric:.6f}", flush=True)
    else:
        copied = initialize_frozen_baseline(model, args.init_baseline_model)
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)
    model.to(device)
    trainable = model.trainable_adapter_names()
    if not trainable or any(name.startswith("base.") for name in trainable):
        raise RuntimeError(f"only selected-neuron adapters may train; got {trainable}")
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"trainable selected-neuron Adapter parameters: {trainable}", flush=True)
    save_json(out_dir / "run_config.json", {
        "dataset": args.dataset,
        "train_list": args.train_list,
        "train_hidden_manifest": args.train_hidden_manifest,
        "test_list": args.test_list,
        "test_hidden_manifest": args.test_hidden_manifest,
        "neuron_json": args.neuron_json,
        "init_baseline_model": args.init_baseline_model,
        "baseline_frozen": True,
        "trainable": "only zero-start per-layer selected-neuron adapters",
        "adapter_rank": args.adapter_rank,
        "skip_missing_train_manifest": args.skip_missing_train_manifest,
        "skipped_train_rows": int(len(train_dataset.missing_manifest_rows)),
        "frame_batch_size": args.frame_batch_size,
        "batch_size": args.batch_size,
        "batch_implementation": "one variable-length video at a time with gradients accumulated over official batch-size samples",
        "loss": "unaltered official XD CLAS2 + CLASM + 1e-4 text separation",
        "checkpoint_selection": "maximum language-branch AP2, matching official VadCLIP xd_train.py",
    })

    gt_path = str(Path(args.vadclip_root) / "list" / "gt.npy")
    segment_path = str(Path(args.vadclip_root) / "list" / "gt_segment.npy")
    label_path = str(Path(args.vadclip_root) / "list" / "gt_label.npy")
    prompts = prompt_text()
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0}
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="video")
        for iteration, sample in enumerate(progress):
            group_size = min(args.batch_size, len(train_loader) - (iteration // args.batch_size) * args.batch_size)
            frames = sample.frames.to(device, non_blocking=True)
            features = model.encode_frame_sequence(frames, args.frame_batch_size)
            visual, length = process_train_feature(features, options.visual_length)
            lengths = torch.as_tensor([length], dtype=torch.int64, device=device)
            labels = label_tensor(sample.label, device)
            text_features, logits1, logits2 = model(visual.unsqueeze(0), None, prompts, lengths)
            loss1 = clas2_loss(logits1, labels, lengths)
            loss2 = clasm_loss(logits2, labels, lengths)
            loss3 = text_separation_loss(text_features)
            loss = loss1 + loss2 + loss3 * 1e-4
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, video={iteration + 1}")
            (loss / group_size).backward()
            if (iteration + 1) % args.batch_size == 0 or iteration + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key, value in (("loss", loss), ("loss1", loss1), ("loss2", loss2), ("loss3", loss3)):
                totals[key] += float(value.detach().item())
            progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})

        scheduler.step()
        records = validation_records(model, test_dataset, options.visual_length, args.frame_batch_size, device, epoch + 1)
        metrics = summarize_records(records, gt_path, segment_path, label_path, args.vadclip_root)
        print_metrics(metrics)
        metric = float(metrics["ap_logits2"])
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), model_path)
            print(f"new best model: AP2={best_metric:.6f} -> {model_path}", flush=True)
        if not model_path.exists():
            raise RuntimeError("no best model exists; validation AP2 must be finite")
        # This mirrors the official launcher: the next epoch starts from the
        # best observed weights, while AdamW/scheduler state continues.
        model.load_state_dict(state_dict_from_file(model_path), strict=True)
        model.freeze_base()
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
            **{key: value / len(train_loader) for key, value in totals.items()},
            "roc_auc_logits1": metrics["roc_auc_logits1"],
            "ap_logits1": metrics["ap_logits1"],
            "roc_auc_logits2": metrics["roc_auc_logits2"],
            "ap_logits2": metrics["ap_logits2"],
            "detection_map_average": metrics["detection_map_average"],
        })
    print(f"finished; best XD language AP2={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
