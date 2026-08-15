#!/usr/bin/env python3
"""Write VadCLIP anomaly-classification pseudo scores for UCF or XD videos.

Scores come from VadCLIP's original sigmoid ``logits1`` branch.  Any feature
variants bearing a common ``__<index>`` suffix are grouped by video exactly as
in the DSANet-side pseudo-score stage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import XD_LABELS, UCF_TEST_LABELS, clean_dir, ensure_dir, grouped_rows, load_clip_feature, read_csv, write_csv


def add_vadclip_source(vadclip_root: str) -> None:
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def load_vadclip_model(vadclip_root: str, model_path: str, device: torch.device, dataset: str):
    add_vadclip_source(vadclip_root)
    from model import CLIPVAD
    if dataset == "ucf":
        import ucf_option as option_module
    else:
        import xd_option as option_module

    options = option_module.parser.parse_args([])
    model = CLIPVAD(
        options.classes_num, options.embed_dim, options.visual_length, options.visual_width,
        options.visual_head, options.visual_layers, options.attn_window,
        options.prompt_prefix, options.prompt_postfix, str(device),
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"{model_path} does not contain a VadCLIP state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, options


def score_feature(model, feature: np.ndarray, visual_length: int, prompt_text: list[str], device: torch.device) -> np.ndarray:
    chunks, lengths = [], []
    for start in range(0, feature.shape[0], visual_length):
        part = feature[start:start + visual_length]
        if part.shape[0] == 0:
            continue
        lengths.append(part.shape[0])
        if part.shape[0] < visual_length:
            part = np.pad(part, ((0, visual_length - part.shape[0]), (0, 0)), mode="constant")
        chunks.append(part.reshape(1, visual_length, feature.shape[1]))
    visual = torch.from_numpy(np.concatenate(chunks, axis=0)).to(device)
    valid_lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    with torch.no_grad():
        _text, logits1, _logits2 = model(visual, None, prompt_text, valid_lengths)
    return torch.sigmoid(logits1.reshape(-1)[:feature.shape[0]]).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score training videos with the frozen VadCLIP classifier branch.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Remove only this stage's previous scores before scoring.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute existing per-video score files without deleting the directory.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "scores")
    if args.clean:
        group_csv = out_dir / "group_scores.csv"
        if group_csv.exists():
            group_csv.unlink()

    model, options = load_vadclip_model(args.vadclip_root, args.model_path, device, args.dataset)
    prompt_text = list(UCF_TEST_LABELS.values() if args.dataset == "ucf" else XD_LABELS.values())
    rows = []
    groups = grouped_rows(read_csv(args.train_list))
    for key, group in tqdm(groups.items(), desc="VadCLIP pseudo scores", unit="video"):
        output_path = score_dir / f"{key}.npy"
        label = str(group.iloc[0]["label"])
        if output_path.exists() and not args.no_resume:
            scores = np.load(output_path)
        else:
            variants = []
            for _, row in group.iterrows():
                feature = load_clip_feature(str(row["path"]))
                if feature.shape[1] != options.visual_width:
                    raise ValueError(
                        f"{row['path']}: expected VadCLIP {options.visual_width}D input, got {feature.shape[1]}D"
                    )
                variants.append(score_feature(model, feature, options.visual_length, prompt_text, device))
            scores = np.concatenate(variants, axis=0)
            np.save(output_path, scores)
        rows.append([key, label, str(output_path), int(len(scores)), int(len(group))])

    write_csv(out_dir / "group_scores.csv", ["key", "label", "score_path", "score_len", "num_chunks"], rows)
    print(f"wrote {out_dir / 'group_scores.csv'} for {len(rows)} video groups", flush=True)


if __name__ == "__main__":
    main()
