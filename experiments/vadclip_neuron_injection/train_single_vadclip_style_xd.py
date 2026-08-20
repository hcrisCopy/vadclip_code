#!/usr/bin/env python3
"""Single-GPU XD-Violence training for VadCLIP neuron residual injection.

The data loader, losses, optimiser, scheduler, per-epoch validation and AP2
model selection reproduce ``VadCLIP/src/xd_train.py``.  Only the small
768D-to-512D residual before the otherwise frozen baseline is new.
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

from common import XD_LABELS, clean_dir, ensure_dir, is_normal_label, load_clip_feature, load_json, process_train_feature, read_csv, save_json
from models import add_vadclip_source, build_residual_model
from ranking import PseudoRankingTargets, dual_branch_temporal_ranking_loss
from xd_evaluation import build_test_loader, print_metrics, run_evaluation


class XDConcatTrainDataset(Dataset):
    """Official single XD train loader with concat-width validation."""

    def __init__(
        self,
        csv_path: str,
        visual_length: int,
        expected_width: int,
        ranking_targets: PseudoRankingTargets | None = None,
    ) -> None:
        self.frame = read_csv(csv_path).reset_index(drop=True)
        self.visual_length, self.expected_width = int(visual_length), int(expected_width)
        self.ranking_targets = ranking_targets

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.loc[index]
        path = str(row["path"])
        feature = load_clip_feature(path)
        if feature.shape[1] != self.expected_width:
            raise ValueError(f"{path}: expected {self.expected_width}D concat feature, got {feature.shape[1]}D")
        feature, length = process_train_feature(feature, self.visual_length)
        label = str(row["label"])
        teacher = (
            self.ranking_targets.target_for(path)
            if self.ranking_targets is not None and not is_normal_label("xd", label)
            else np.zeros(self.visual_length, dtype=np.float32)
        )
        return torch.from_numpy(feature), label, int(length), torch.from_numpy(teacher)


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
            "baseline model is incompatible with official 512D XD VadCLIP: "
            f"missing={missing[:4]}, shape_mismatch={mismatched[:2]}"
        )
    model.load_state_dict(target, strict=True)
    return copied


def label_tensor(labels: list[str], device: torch.device) -> torch.Tensor:
    prompts = list(XD_LABELS.values())
    lookup = {name: index for index, name in enumerate(prompts)}
    target = torch.zeros((len(labels), len(prompts)), dtype=torch.float32, device=device)
    for row, text in enumerate(labels):
        matched = False
        for code in str(text).split("-"):
            if code in XD_LABELS:
                target[row, lookup[XD_LABELS[code]]] = 1.0
                matched = True
        if not matched:
            raise ValueError(f"unrecognised XD label {text!r}")
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
    # All XD labels contain at least one official class; this exactly mirrors
    # ``labels / torch.sum(labels, dim=1, keepdim=True)`` in xd_train.py.
    labels = labels / labels.sum(dim=1, keepdim=True)
    for index in range(logits.shape[0]):
        valid = max(1, min(int(lengths[index]), logits.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(logits[index, :valid].topk(count, dim=0).values.mean(dim=0))
    return -torch.mean(torch.sum(labels * functional.log_softmax(torch.stack(instance_logits), dim=1), dim=1))


def text_separation_loss(text_features: torch.Tensor) -> torch.Tensor:
    normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    loss = torch.zeros((), device=text_features.device)
    for index in range(1, text_features.shape[0]):
        anomaly = text_features[index] / text_features[index].norm(dim=-1, keepdim=True)
        loss = loss + torch.abs(normal @ anomaly)
    return loss / 6.0


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
    parser = argparse.ArgumentParser(description="Train frozen-VadCLIP gated neuron residual injection on XD-Violence.")
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
        "--ranking-pseudo-csv", default="",
        help="Optional frozen-baseline group_scores.csv. Enables M1 pseudo-score ranking supervision.",
    )
    parser.add_argument("--rank-loss-weight", type=float, default=0.10)
    parser.add_argument("--rank-top-p", type=float, default=0.10)
    parser.add_argument("--rank-intra-margin", type=float, default=0.10)
    parser.add_argument("--rank-cross-margin", type=float, default=0.10)
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
    if args.rank_loss_weight < 0 or not 0.0 < args.rank_top_p <= 0.5:
        parser.error("rank-loss-weight must be non-negative and rank-top-p must be in (0, 0.5]")
    if args.rank_intra_margin < 0 or args.rank_cross_margin < 0:
        parser.error("rank-intra-margin and rank-cross-margin must be non-negative")
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
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
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

    # The residual is constructed on CPU, while the official CLIP backbone may
    # be constructed directly on CUDA.  Move the complete wrapper before
    # optimisation so their parameters always share a device.
    model.to(device)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr
    )
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    print(f"trainable residual parameters: {trainable}", flush=True)

    ranking_targets = (
        PseudoRankingTargets(args.dataset, args.train_list, args.ranking_pseudo_csv, options.visual_length)
        if args.ranking_pseudo_csv else None
    )
    if ranking_targets is not None:
        print(
            f"M1 ranking supervision: weight={args.rank_loss_weight:g}, top_p={args.rank_top_p:.3f}, "
            f"margins=({args.rank_intra_margin:g},{args.rank_cross_margin:g})",
            flush=True,
        )
    train_loader = DataLoader(
        XDConcatTrainDataset(args.train_list, options.visual_length, expected_width, ranking_targets=ranking_targets),
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
        "ranking_pseudo_csv": args.ranking_pseudo_csv,
        "ranking_enabled": ranking_targets is not None,
        "rank_loss_weight": args.rank_loss_weight,
        "rank_top_p": args.rank_top_p,
        "rank_intra_margin": args.rank_intra_margin,
        "rank_cross_margin": args.rank_cross_margin,
        "ranking_definition": "confidence-weighted top/bottom pseudo ranking plus hard-normal ranking on both official anomaly outputs",
        "validation": "once per epoch, matching official VadCLIP XD",
        "checkpoint_selection": "maximum XD language-branch AP2, matching official VadCLIP xd_train.py",
    })

    prompt_text = list(XD_LABELS.values())
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {
            "loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0,
            "rank_intra": 0.0, "rank_cross": 0.0, "rank": 0.0,
        }
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for iteration, (visual, raw_labels, lengths, teacher_scores) in enumerate(progress):
            visual = visual.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            teacher_scores = teacher_scores.to(device, non_blocking=True)
            abnormal_mask = torch.tensor(
                [not is_normal_label("xd", str(label)) for label in raw_labels], dtype=torch.bool, device=device
            )
            labels = label_tensor(list(raw_labels), device)
            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
            loss1 = clas2_loss(logits1, labels, lengths)
            loss2 = clasm_loss(logits2, labels, lengths)
            loss3 = text_separation_loss(text_features)
            loss = loss1 + loss2 + loss3 * 1e-4
            rank_intra = rank_cross = rank_loss = torch.zeros_like(loss)
            if ranking_targets is not None and args.rank_loss_weight > 0:
                rank_intra, rank_cross, rank_loss, _rank_stats = dual_branch_temporal_ranking_loss(
                    logits1, logits2, teacher_scores, lengths, abnormal_mask,
                    args.rank_top_p, args.rank_intra_margin, args.rank_cross_margin,
                )
                loss = loss + args.rank_loss_weight * rank_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in (
                ("loss", loss), ("loss1", loss1), ("loss2", loss2), ("loss3", loss3),
                ("rank_intra", rank_intra), ("rank_cross", rank_cross), ("rank", rank_loss),
            ):
                totals[key] += float(value.item())
            progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})

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
            description=f"validation epoch {epoch + 1}",
        )
        print_metrics(metrics)
        metric = float(metrics["ap_logits2"])
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), model_path)
            print(f"new best model: AP2={best_metric:.6f} -> {model_path}", flush=True)
        if not model_path.exists():
            raise RuntimeError("No best model was written; AP2 must be a finite positive value")
        # Match xd_train.py: begin the next epoch from the best model while
        # retaining the current optimiser/scheduler state.
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
            "roc_auc_logits1": metrics["roc_auc_logits1"],
            "ap_logits1": metrics["ap_logits1"],
            "roc_auc_logits2": metrics["roc_auc_logits2"],
            "ap_logits2": metrics["ap_logits2"],
            "detection_map_average": metrics["detection_map_average"],
            "gate": float(model.residual_gate().detach().cpu()),
        })
    print(f"finished; best XD language AP2={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
