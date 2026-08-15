#!/usr/bin/env python3
"""Single-GPU UCF training for VadCLIP gated neuron residual injection.

The loss definitions, paired normal/abnormal UCF loaders, optimiser and
multi-step schedule follow the official VadCLIP UCF launcher.  The only model
change is the trainable 768D-to-512D residual immediately before frozen
VadCLIP; the baseline directory is never modified.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import UCF_TRAIN_LABELS, clean_dir, ensure_dir, load_clip_feature, load_json, process_train_feature, read_csv, save_json
from evaluation import build_test_loader, print_metrics, run_evaluation
from models import add_vadclip_source, build_residual_model


class UCFConcatTrainDataset(Dataset):
    def __init__(self, csv_path: str, visual_length: int, expected_width: int, normal: bool) -> None:
        frame = read_csv(csv_path)
        is_normal = frame["label"].astype(str) == "Normal"
        self.frame = frame.loc[is_normal if normal else ~is_normal].reset_index(drop=True)
        self.visual_length, self.expected_width = int(visual_length), int(expected_width)
        self.role = "normal" if normal else "abnormal"

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


def baseline_options(vadclip_root: str):
    add_vadclip_source(vadclip_root)
    import ucf_option

    return ucf_option.parser.parse_args([])


def state_dict_from_file(path: str) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def initialize_from_baseline(model, baseline_path: str) -> int:
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
            "baseline model is incompatible with official 512D VadCLIP: "
            f"missing={missing[:4]}, shape_mismatch={mismatched[:2]}"
        )
    model.load_state_dict(target, strict=True)
    return copied


def label_tensor(labels: list[str], device: torch.device) -> torch.Tensor:
    prompts = list(UCF_TRAIN_LABELS.values())
    lookup = {name: index for index, name in enumerate(prompts)}
    target = torch.zeros((len(labels), len(prompts)), dtype=torch.float32, device=device)
    for row, label in enumerate(labels):
        if label not in UCF_TRAIN_LABELS:
            raise ValueError(f"unrecognised UCF label {label!r}")
        target[row, lookup[UCF_TRAIN_LABELS[label]]] = 1.0
    return target


def clas2_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    instance_logits = []
    probabilities = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for index in range(probabilities.shape[0]):
        valid = max(1, min(int(lengths[index]), probabilities.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(probabilities[index, :valid].topk(count).values.mean())
    target = 1.0 - labels[:, 0]
    return functional.binary_cross_entropy(torch.stack(instance_logits), target)


def clasm_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    instance_logits = []
    labels = labels / labels.sum(dim=1, keepdim=True).clamp(min=1e-6)
    for index in range(logits.shape[0]):
        valid = max(1, min(int(lengths[index]), logits.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(logits[index, :valid].topk(count, dim=0).values.mean(dim=0))
    return -torch.mean(torch.sum(labels * functional.log_softmax(torch.stack(instance_logits), dim=1), dim=1))


def text_separation_loss(text_features: torch.Tensor) -> torch.Tensor:
    normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    loss = torch.zeros((), device=text_features.device)
    for index in range(1, text_features.shape[0]):
        anomaly = text_features[index] / text_features[index].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        loss = loss + torch.abs(normal @ anomaly)
    return loss / 13.0 * 1e-1


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
    parser = argparse.ArgumentParser(description="Train frozen-VadCLIP gated neuron residual injection on UCF.")
    parser.add_argument("--dataset", choices=["ucf"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="Per normal/anomaly loader batch, matching the supplied UCF command.")
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--num-workers", type=int, default=4)
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
    options.visual_width = 512
    options.visual_length = int(options.visual_length)
    model = build_residual_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    start_epoch, best_metric, global_step = 0, float("-inf"), 0
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
        print(f"resuming from epoch {start_epoch + 1}, best ROC-AUC1={best_metric:.6f}", flush=True)
    else:
        copied = initialize_from_baseline(model, args.init_baseline_model)
        model.freeze_base()
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr
    )
    scheduler = MultiStepLR(optimizer, milestones=[4, 8], gamma=0.1)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    print(f"trainable residual parameters: {trainable}", flush=True)
    normal_dataset = UCFConcatTrainDataset(args.train_list, options.visual_length, expected_width, normal=True)
    anomaly_dataset = UCFConcatTrainDataset(args.train_list, options.visual_length, expected_width, normal=False)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, pin_memory=True)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, pin_memory=True)
    batches = min(len(normal_loader), len(anomaly_loader))
    if batches <= 0:
        raise RuntimeError("one UCF loader is empty after batch/drop_last; inspect CSV or lower --batch-size")
    gt_path = str(Path(args.vadclip_root) / "list" / "gt_ucf.npy")
    segment_path = str(Path(args.vadclip_root) / "list" / "gt_segment_ucf.npy")
    label_path = str(Path(args.vadclip_root) / "list" / "gt_label_ucf.npy")
    test_loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    save_json(out_dir / "run_config.json", {
        "dataset": args.dataset, "train_list": args.train_list, "test_list": args.test_list,
        "neuron_json": args.neuron_json, "vadclip_root": args.vadclip_root,
        "batch_size_per_loader": args.batch_size, "lr": args.lr, "max_epoch": args.max_epoch,
        "residual_hidden_dim": args.residual_hidden_dim, "residual_depth": args.residual_depth,
        "checkpoint_selection": "maximum UCF logits1 ROC-AUC, matching the official VadCLIP UCF train loop",
    })

    prompt_text = list(UCF_TRAIN_LABELS.values())
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        normal_iter, anomaly_iter = iter(normal_loader), iter(anomaly_loader)
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0}
        progress = tqdm(range(batches), desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for iteration in progress:
            normal_feature, normal_label, normal_length = next(normal_iter)
            anomaly_feature, anomaly_label, anomaly_length = next(anomaly_iter)
            visual = torch.cat([normal_feature, anomaly_feature], dim=0).to(device, non_blocking=True)
            lengths = torch.cat([normal_length, anomaly_length], dim=0).to(device, non_blocking=True)
            labels = label_tensor(list(normal_label) + list(anomaly_label), device)
            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
            loss1 = clas2_loss(logits1, labels, lengths)
            loss2 = clasm_loss(logits2, labels, lengths)
            loss3 = text_separation_loss(text_features)
            loss = loss1 + loss2 + loss3
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            for key, value in (("loss", loss), ("loss1", loss1), ("loss2", loss2), ("loss3", loss3)):
                totals[key] += float(value.item())
            progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})
        scheduler.step()
        _records, metrics = run_evaluation(
            model, test_loader, options.visual_length, device, gt_path, segment_path, label_path,
            args.vadclip_root, description=f"validation epoch {epoch + 1}/{args.max_epoch}",
        )
        print_metrics(metrics)
        metric = float(metrics["roc_auc_logits1"])
        checkpoint = {
            "epoch": epoch, "global_step": global_step, "best_metric": max(best_metric, metric),
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "metrics": metrics,
        }
        torch.save(checkpoint, checkpoint_path)
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), model_path)
            print(f"new best model: ROC-AUC1={best_metric:.6f} -> {model_path}", flush=True)
        append_history(out_dir / "history.csv", {
            "epoch": epoch + 1, "global_step": global_step,
            **{key: totals[key] / batches for key in totals},
            "roc_auc_logits1": metrics["roc_auc_logits1"], "ap_logits1": metrics["ap_logits1"],
            "roc_auc_logits2": metrics["roc_auc_logits2"], "ap_logits2": metrics["ap_logits2"],
            "detection_map_average": metrics["detection_map_average"], "gate": float(model.residual_gate().detach().cpu()),
        })
    print(f"finished; best UCF logits1 ROC-AUC={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
