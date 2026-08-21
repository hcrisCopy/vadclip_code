#!/usr/bin/env python3
"""Train the unmodified global-768 residual method on a test-video subset.

This is a diagnostic-only launcher.  Its model, residual module, frozen
baseline, original VadCLIP losses, optimiser and checkpoint criterion match
the ordinary global-768 experiment.  The only difference is that validation
uses an explicitly supplied, disjoint subset GT artifact.  No frame-level
annotation is loaded by the training dataset or loss.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from shared import add_injection_source, initialize_residual_from_baseline, state_dict_from_file

add_injection_source()
from common import (
    UCF_TEST_LABELS,
    UCF_TRAIN_LABELS,
    XD_LABELS,
    clean_dir,
    ensure_dir,
    is_normal_label,
    load_clip_feature,
    load_json,
    process_train_feature,
    read_csv,
    save_json,
)
from models import add_vadclip_source, build_residual_model


class OriginalConcatTrainDataset(Dataset):
    """Original 1280D train dataset without pseudo or frame-label targets."""

    def __init__(
        self, csv_path: str, visual_length: int, expected_width: int, dataset: str, normal: bool | None = None
    ) -> None:
        frame = read_csv(csv_path)
        if normal is not None:
            mask = frame["label"].astype(str).map(lambda label: is_normal_label(dataset, label))
            frame = frame.loc[mask if normal else ~mask]
        self.frame = frame.reset_index(drop=True)
        self.visual_length, self.expected_width = int(visual_length), int(expected_width)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.loc[index]
        path = str(row["path"])
        feature = load_clip_feature(path)
        if feature.shape[1] != self.expected_width:
            raise ValueError(f"{path}: expected {self.expected_width}D concat feature, got {feature.shape[1]}D")
        feature, length = process_train_feature(feature, self.visual_length)
        return torch.from_numpy(feature), str(row["label"]), int(length)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def baseline_options(dataset: str, vadclip_root: str):
    add_vadclip_source(vadclip_root)
    if dataset == "ucf":
        import ucf_option as option_module
    else:
        import xd_option as option_module
    return option_module.parser.parse_args([])


def label_tensor(dataset: str, labels: list[str], device: torch.device) -> torch.Tensor:
    prompts = list(UCF_TRAIN_LABELS.values() if dataset == "ucf" else XD_LABELS.values())
    lookup = {name: index for index, name in enumerate(prompts)}
    target = torch.zeros((len(labels), len(prompts)), dtype=torch.float32, device=device)
    for row, label in enumerate(labels):
        if dataset == "ucf":
            if label not in UCF_TRAIN_LABELS:
                raise ValueError(f"unrecognised UCF label {label!r}")
            target[row, lookup[UCF_TRAIN_LABELS[label]]] = 1.0
        else:
            matched = False
            for code in str(label).split("-"):
                if code in XD_LABELS:
                    target[row, lookup[XD_LABELS[code]]] = 1.0
                    matched = True
            if not matched:
                raise ValueError(f"unrecognised XD label {label!r}")
    return target


def clas2_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    instance_logits = []
    probabilities = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for index in range(probabilities.shape[0]):
        valid = max(1, min(int(lengths[index]), probabilities.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(probabilities[index, :valid].topk(count).values.mean())
    return functional.binary_cross_entropy(torch.stack(instance_logits), 1.0 - labels[:, 0])


def clasm_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    instance_logits = []
    labels = labels / labels.sum(dim=1, keepdim=True).clamp(min=1e-6)
    for index in range(logits.shape[0]):
        valid = max(1, min(int(lengths[index]), logits.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(logits[index, :valid].topk(count, dim=0).values.mean(dim=0))
    return -torch.mean(torch.sum(labels * functional.log_softmax(torch.stack(instance_logits), dim=1), dim=1))


def text_separation_loss(dataset: str, text_features: torch.Tensor) -> torch.Tensor:
    normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    loss = torch.zeros((), device=text_features.device)
    for index in range(1, text_features.shape[0]):
        anomaly = text_features[index] / text_features[index].norm(dim=-1, keepdim=True)
        loss = loss + torch.abs(normal @ anomaly)
    return loss / 13.0 * 1e-1 if dataset == "ucf" else loss / 6.0


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
    parser = argparse.ArgumentParser(description="Train original frozen-VadCLIP global-768 residual on a diagnostic split.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True, help="Video-disjoint diagnostic train CSV.")
    parser.add_argument("--validation-list", required=True, help="Video-disjoint diagnostic validation CSV.")
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-interval-samples", type=int, default=1280)
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
    if args.max_epoch <= 0 or args.eval_interval_samples <= 0:
        parser.error("max-epoch and eval-interval-samples must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("batch-size must be positive when specified")
    if args.lr is not None and args.lr <= 0:
        parser.error("lr must be positive when specified")
    for path in (args.validation_gt_path, args.validation_segment_path, args.validation_label_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing aligned validation annotation: {path}")

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
    options = baseline_options(args.dataset, args.vadclip_root)
    options.visual_width, options.visual_length = 512, int(options.visual_length)
    default_batch, default_lr = (64, 2e-5) if args.dataset == "ucf" else (96, 1e-5)
    batch_size = int(args.batch_size if args.batch_size is not None else default_batch)
    lr = float(args.lr if args.lr is not None else default_lr)
    model = build_residual_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    start_epoch, best_metric, global_step = 0, 0.0, 0
    resume_checkpoint = None
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint["best_metric"])
        global_step = int(resume_checkpoint.get("global_step", 0))
        model.freeze_base()
        print(f"resuming from epoch {start_epoch + 1}, best metric={best_metric:.6f}", flush=True)
    else:
        copied = initialize_residual_from_baseline(model, args.init_baseline_model)
        model.freeze_base()
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    print(f"trainable original residual parameters: {trainable}", flush=True)

    if args.dataset == "ucf":
        from evaluation import build_test_loader, print_metrics, run_evaluation

        normal_dataset = OriginalConcatTrainDataset(args.train_list, options.visual_length, expected_width, "ucf", normal=True)
        anomaly_dataset = OriginalConcatTrainDataset(args.train_list, options.visual_length, expected_width, "ucf", normal=False)
        normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        batches = min(len(normal_loader), len(anomaly_loader))
        if batches <= 0:
            raise RuntimeError("UCF diagnostic train subset has an empty paired loader; increase train fraction or lower batch-size")
        train_loader = None
        validation_loader = build_test_loader(args.validation_list, options.visual_length, expected_width, args.num_workers)
        prompt_text = list(UCF_TRAIN_LABELS.values())
        selection_name = "roc_auc_logits1"
    else:
        from xd_evaluation import build_test_loader, print_metrics, run_evaluation

        train_dataset = OriginalConcatTrainDataset(args.train_list, options.visual_length, expected_width, "xd")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        if not len(train_loader):
            raise RuntimeError("XD diagnostic train loader is empty; increase train fraction or inspect the split")
        normal_loader = anomaly_loader = None
        batches = len(train_loader)
        validation_loader = build_test_loader(args.validation_list, options.visual_length, expected_width, args.num_workers)
        prompt_text = list(XD_LABELS.values())
        selection_name = "ap_logits2"

    def validate(description: str) -> dict[str, object]:
        _records, metrics = run_evaluation(
            model, validation_loader, options.visual_length, device,
            args.validation_gt_path, args.validation_segment_path, args.validation_label_path,
            args.vadclip_root, description=description,
        )
        print_metrics(metrics)
        return metrics

    save_json(out_dir / "run_config.json", {
        "method": "original_global768_no_trick_split_diagnostic_v1",
        "dataset": args.dataset,
        "train_list": args.train_list,
        "validation_list": args.validation_list,
        "validation_annotations": {
            "gt": args.validation_gt_path,
            "segment": args.validation_segment_path,
            "label": args.validation_label_path,
        },
        "neuron_json": args.neuron_json,
        "vadclip_root": args.vadclip_root,
        "batch_size": batch_size,
        "lr": lr,
        "max_epoch": args.max_epoch,
        "num_workers": args.num_workers,
        "eval_interval_samples": args.eval_interval_samples,
        "residual_hidden_dim": args.residual_hidden_dim,
        "residual_depth": args.residual_depth,
        "frame_labels_used_for_training": False,
        "ranking_or_other_tricks_enabled": False,
        "checkpoint_selection": f"maximum {selection_name} on the disjoint validation split",
        "ucf_small_split_fallback": "validate once at epoch end only when official 1280-sample cadence produced no validation",
    })

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0}
        last_metrics = None
        if args.dataset == "ucf":
            normal_iter, anomaly_iter = iter(normal_loader), iter(anomaly_loader)
            progress = tqdm(range(batches), desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
            for iteration in progress:
                normal_feature, normal_label, normal_length = next(normal_iter)
                anomaly_feature, anomaly_label, anomaly_length = next(anomaly_iter)
                visual = torch.cat([normal_feature, anomaly_feature], dim=0).to(device, non_blocking=True)
                lengths = torch.cat([normal_length, anomaly_length], dim=0).to(device, non_blocking=True)
                labels = label_tensor("ucf", list(normal_label) + list(anomaly_label), device)
                text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
                loss1 = clas2_loss(logits1, labels, lengths)
                loss2 = clasm_loss(logits2, labels, lengths)
                loss3 = text_separation_loss("ucf", text_features)
                loss = loss1 + loss2 + loss3
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                for key, value in (("loss", loss), ("loss1", loss1), ("loss2", loss2), ("loss3", loss3)):
                    totals[key] += float(value.item())
                progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})
                official_step = iteration * normal_loader.batch_size * 2
                if official_step != 0 and official_step % args.eval_interval_samples == 0:
                    last_metrics = validate(f"validation epoch {epoch + 1}, step {official_step}")
                    metric = float(last_metrics[selection_name])
                    if metric > best_metric:
                        best_metric = metric
                        torch.save(model.state_dict(), model_path)
                        print(f"new best model: {selection_name}={best_metric:.6f} -> {model_path}", flush=True)
        else:
            progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
            for visual, raw_labels, lengths in progress:
                visual, lengths = visual.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
                labels = label_tensor("xd", list(raw_labels), device)
                text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
                loss1 = clas2_loss(logits1, labels, lengths)
                loss2 = clasm_loss(logits2, labels, lengths)
                loss3 = text_separation_loss("xd", text_features)
                loss = loss1 + loss2 + loss3 * 1e-4
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch={epoch + 1}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for key, value in (("loss", loss), ("loss1", loss1), ("loss2", loss2), ("loss3", loss3)):
                    totals[key] += float(value.item())
                progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})
            last_metrics = validate(f"validation epoch {epoch + 1}")
            metric = float(last_metrics[selection_name])
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), model_path)
                print(f"new best model: {selection_name}={best_metric:.6f} -> {model_path}", flush=True)

        scheduler.step()
        if args.dataset == "ucf" and last_metrics is None:
            # A 60% test split can have fewer than two official 1280-sample
            # intervals.  Preserve the normal cadence whenever it exists, but
            # produce one validation checkpoint otherwise so this diagnostic is
            # resumable and the held-out set remains untouched.
            last_metrics = validate(f"validation epoch {epoch + 1}, small-split fallback")
            metric = float(last_metrics[selection_name])
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), model_path)
                print(f"new best model: {selection_name}={best_metric:.6f} -> {model_path}", flush=True)
        if not model_path.exists():
            raise RuntimeError("No validation checkpoint was written; inspect the validation metrics")
        model.load_state_dict(state_dict_from_file(str(model_path)), strict=True)
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": last_metrics,
        }
        torch.save(checkpoint, checkpoint_path)
        append_history(out_dir / "history.csv", {
            "epoch": epoch + 1,
            "global_step": global_step,
            **{key: totals[key] / batches for key in totals},
            "selection_metric": None if last_metrics is None else last_metrics[selection_name],
            "ap_logits1": None if last_metrics is None else last_metrics["ap_logits1"],
            "ap_logits2": None if last_metrics is None else last_metrics["ap_logits2"],
            "detection_map_average": None if last_metrics is None else last_metrics["detection_map_average"],
            "gate": float(model.residual_gate().detach().cpu()),
        })
    print(f"finished; best validation {selection_name}={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
