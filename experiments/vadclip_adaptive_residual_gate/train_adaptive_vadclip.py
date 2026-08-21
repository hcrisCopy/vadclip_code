#!/usr/bin/env python3
"""Train snippet-adaptive global-768 residual injection with baseline-aligned VadCLIP rules."""
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

from shared import add_injection_source, initialize_adaptive_from_baseline, state_dict_from_file

add_injection_source()
from common import UCF_TRAIN_LABELS, XD_LABELS, clean_dir, ensure_dir, load_json, save_json
from datasets import AdaptiveConcatTrainDataset
from models import add_vadclip_source, adaptive_regularization, build_adaptive_model


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


def evaluation_components(dataset: str):
    """Use the unchanged metrics and selection definition for each baseline dataset."""
    if dataset == "ucf":
        from evaluation import build_test_loader, print_metrics, run_evaluation

        return build_test_loader, print_metrics, run_evaluation, "roc_auc_logits1"
    from xd_evaluation import build_test_loader, print_metrics, run_evaluation

    return build_test_loader, print_metrics, run_evaluation, "ap_logits2"


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
    """Exact instance-level binary loss used by the existing global-768 scripts."""
    instance_logits = []
    probabilities = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for index in range(probabilities.shape[0]):
        valid = max(1, min(int(lengths[index]), probabilities.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(probabilities[index, :valid].topk(count).values.mean())
    return functional.binary_cross_entropy(torch.stack(instance_logits), 1.0 - labels[:, 0])


def clasm_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Exact instance-level multi-class loss used by the existing global-768 scripts."""
    instance_logits = []
    labels = labels / labels.sum(dim=1, keepdim=True).clamp(min=1e-6)
    for index in range(logits.shape[0]):
        valid = max(1, min(int(lengths[index]), logits.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(logits[index, :valid].topk(count, dim=0).values.mean(dim=0))
    return -torch.mean(torch.sum(labels * functional.log_softmax(torch.stack(instance_logits), dim=1), dim=1))


def text_separation_loss(dataset: str, text_features: torch.Tensor) -> torch.Tensor:
    """Keep the original UCF/XD text-separation formula and weighting."""
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
    if path.is_file():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adaptive per-snippet residual gates without changing VadCLIP baseline losses.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True, help="Baseline-aligned validation/model-selection CSV.")
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
    parser.add_argument("--residual-ratio-cap", type=float, default=0.01)
    parser.add_argument("--residual-amp-weight", type=float, default=0.10)
    parser.add_argument("--residual-tv-weight", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    if args.max_epoch <= 0 or args.eval_interval_samples <= 0 or args.num_workers < 0:
        parser.error("max-epoch and eval-interval-samples must be positive; num-workers must be non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("batch-size must be positive when specified")
    if args.lr is not None and args.lr <= 0:
        parser.error("lr must be positive when specified")
    if args.residual_ratio_cap <= 0 or args.residual_amp_weight < 0 or args.residual_tv_weight < 0:
        parser.error("residual-ratio-cap must be positive; regularization weights must be non-negative")
    for path in (args.train_list, args.test_list, args.neuron_json, args.init_baseline_model):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing adaptive-residual input: {path}")

    default_batch, default_lr = (64, 2e-5) if args.dataset == "ucf" else (96, 1e-5)
    batch_size = int(args.batch_size if args.batch_size is not None else default_batch)
    lr = float(args.lr if args.lr is not None else default_lr)
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
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    options = baseline_options(args.dataset, args.vadclip_root)
    options.visual_width, options.visual_length = 512, int(options.visual_length)
    model = build_adaptive_model(
        options, args.vadclip_root, str(device), contract, args.residual_hidden_dim, args.residual_depth
    )
    start_epoch, best_metric, global_step = 0, 0.0, 0
    resume_checkpoint = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint["best_metric"])
        global_step = int(resume_checkpoint.get("global_step", 0))
        model.freeze_base()
        print(f"resuming from epoch {start_epoch + 1}, best metric={best_metric:.6f}", flush=True)
    else:
        copied = initialize_adaptive_from_baseline(model, args.init_baseline_model)
        model.freeze_base()
        print(f"initialized frozen VadCLIP baseline tensors: {copied}", flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_milestones, gamma=options.scheduler_rate)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    print(f"trainable adaptive residual parameters: {trainable}", flush=True)

    build_test_loader, print_metrics, run_evaluation, selection_name = evaluation_components(args.dataset)
    if args.dataset == "ucf":
        normal_dataset = AdaptiveConcatTrainDataset(args.train_list, options.visual_length, expected_width, "ucf", normal=True)
        anomaly_dataset = AdaptiveConcatTrainDataset(args.train_list, options.visual_length, expected_width, "ucf", normal=False)
        normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        batches = min(len(normal_loader), len(anomaly_loader))
        if batches <= 0:
            raise RuntimeError("one UCF paired loader is empty; inspect CSV or lower --batch-size")
        train_loader = None
        prompts = list(UCF_TRAIN_LABELS.values())
        gt_path = str(Path(args.vadclip_root) / "list" / "gt_ucf.npy")
        segment_path = str(Path(args.vadclip_root) / "list" / "gt_segment_ucf.npy")
        label_path = str(Path(args.vadclip_root) / "list" / "gt_label_ucf.npy")
    else:
        train_dataset = AdaptiveConcatTrainDataset(args.train_list, options.visual_length, expected_width, "xd")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        batches = len(train_loader)
        if batches <= 0:
            raise RuntimeError("XD training loader is empty; inspect CSV")
        normal_loader = anomaly_loader = None
        prompts = list(XD_LABELS.values())
        gt_path = str(Path(args.vadclip_root) / "list" / "gt.npy")
        segment_path = str(Path(args.vadclip_root) / "list" / "gt_segment.npy")
        label_path = str(Path(args.vadclip_root) / "list" / "gt_label.npy")
    test_loader = build_test_loader(args.test_list, options.visual_length, expected_width, args.num_workers)
    save_json(out_dir / "run_config.json", {
        "method": "adaptive_snippet_gate_global768_v1",
        "dataset": args.dataset,
        "train_list": args.train_list,
        "test_list": args.test_list,
        "neuron_json": args.neuron_json,
        "vadclip_root": args.vadclip_root,
        "batch_size": batch_size,
        "lr": lr,
        "max_epoch": args.max_epoch,
        "num_workers": args.num_workers,
        "eval_interval_samples": args.eval_interval_samples,
        "residual_hidden_dim": args.residual_hidden_dim,
        "residual_depth": args.residual_depth,
        "residual_ratio_cap": args.residual_ratio_cap,
        "residual_amp_weight": args.residual_amp_weight,
        "residual_tv_weight": args.residual_tv_weight,
        "base_frozen": True,
        "frame_labels_used_for_training": False,
        "pseudo_ranking_enabled": False,
        "checkpoint_selection": f"maximum {selection_name}, matching official VadCLIP {args.dataset}",
    })

    def validate(description: str) -> dict[str, object]:
        _records, metrics = run_evaluation(
            model, test_loader, options.visual_length, device, gt_path, segment_path, label_path,
            args.vadclip_root, description=description,
        )
        print_metrics(metrics)
        return metrics

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0, "loss_amp": 0.0, "loss_tv": 0.0, "gate": 0.0, "ratio": 0.0}
        last_metrics = None
        if args.dataset == "ucf":
            normal_iter, anomaly_iter = iter(normal_loader), iter(anomaly_loader)
            progress = tqdm(range(batches), desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
            iterator = (
                (next(normal_iter), next(anomaly_iter), iteration)
                for iteration in progress
            )
            for normal_batch, anomaly_batch, iteration in iterator:
                normal_feature, normal_label, normal_length = normal_batch
                anomaly_feature, anomaly_label, anomaly_length = anomaly_batch
                visual = torch.cat([normal_feature, anomaly_feature], dim=0).to(device, non_blocking=True)
                lengths = torch.cat([normal_length, anomaly_length], dim=0).to(device, non_blocking=True)
                labels = label_tensor("ucf", list(normal_label) + list(anomaly_label), device)
                text_features, logits1, logits2, details = model(visual, None, prompts, lengths, return_details=True)
                loss1, loss2 = clas2_loss(logits1, labels, lengths), clasm_loss(logits2, labels, lengths)
                loss3 = text_separation_loss("ucf", text_features)
                loss_amp, loss_tv, statistics = adaptive_regularization(details, lengths, args.residual_ratio_cap)
                loss = loss1 + loss2 + loss3 + args.residual_amp_weight * loss_amp + args.residual_tv_weight * loss_tv
                official_step = iteration * normal_loader.batch_size * 2
                should_validate = official_step != 0 and official_step % args.eval_interval_samples == 0
                label_for_error = f"epoch={epoch + 1}, batch={iteration + 1}"
                progress_target = progress
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at {label_for_error}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                values = {"loss": loss, "loss1": loss1, "loss2": loss2, "loss3": loss3, "loss_amp": loss_amp, "loss_tv": loss_tv, "gate": statistics["gate_mean"], "ratio": statistics["ratio_mean"]}
                for key, value in values.items():
                    totals[key] += float(value.detach().item())
                progress_target.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})
                if should_validate:
                    last_metrics = validate(f"validation epoch {epoch + 1}, step {official_step}")
                    metric = float(last_metrics[selection_name])
                    if metric > best_metric:
                        best_metric = metric
                        torch.save(model.state_dict(), model_path)
                        print(f"new best model: {selection_name}={best_metric:.6f} -> {model_path}", flush=True)
        else:
            progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.max_epoch}", unit="batch")
            for iteration, (visual, raw_labels, lengths) in enumerate(progress):
                visual, lengths = visual.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
                labels = label_tensor("xd", list(raw_labels), device)
                text_features, logits1, logits2, details = model(visual, None, prompts, lengths, return_details=True)
                loss1, loss2 = clas2_loss(logits1, labels, lengths), clasm_loss(logits2, labels, lengths)
                loss3 = text_separation_loss("xd", text_features)
                loss_amp, loss_tv, statistics = adaptive_regularization(details, lengths, args.residual_ratio_cap)
                loss = loss1 + loss2 + loss3 * 1e-4 + args.residual_amp_weight * loss_amp + args.residual_tv_weight * loss_tv
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch={epoch + 1}, batch={iteration + 1}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                values = {"loss": loss, "loss1": loss1, "loss2": loss2, "loss3": loss3, "loss_amp": loss_amp, "loss_tv": loss_tv, "gate": statistics["gate_mean"], "ratio": statistics["ratio_mean"]}
                for key, value in values.items():
                    totals[key] += float(value.detach().item())
                progress.set_postfix({key: f"{value / (iteration + 1):.4f}" for key, value in totals.items()})
            last_metrics = validate(f"validation epoch {epoch + 1}")
            metric = float(last_metrics[selection_name])
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), model_path)
                print(f"new best model: {selection_name}={best_metric:.6f} -> {model_path}", flush=True)

        scheduler.step()
        if args.dataset == "ucf" and last_metrics is None:
            raise RuntimeError("No UCF validation occurred; check train batches and --eval-interval-samples")
        if not model_path.is_file():
            raise RuntimeError("No best model was written; inspect validation metrics")
        model.load_state_dict(state_dict_from_file(str(model_path)), strict=True)
        checkpoint = {
            "epoch": epoch, "global_step": global_step, "best_metric": best_metric,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "metrics": last_metrics,
        }
        torch.save(checkpoint, checkpoint_path)
        append_history(out_dir / "history.csv", {
            "epoch": epoch + 1, "global_step": global_step,
            **{key: totals[key] / batches for key in totals},
            "selection_metric": float(last_metrics[selection_name]),
            "roc_auc_logits1": last_metrics["roc_auc_logits1"], "ap_logits1": last_metrics["ap_logits1"],
            "roc_auc_logits2": last_metrics["roc_auc_logits2"], "ap_logits2": last_metrics["ap_logits2"],
            "detection_map_average": last_metrics["detection_map_average"],
        })
    print(f"finished; best {selection_name}={best_metric:.6f}; model={model_path}", flush=True)


if __name__ == "__main__":
    main()
