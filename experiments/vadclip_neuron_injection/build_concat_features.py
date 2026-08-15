#!/usr/bin/env python3
"""Build 1280D features as [768D selected neurons | 512D final CLIP]."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    base_key,
    clean_dir,
    ensure_dir,
    load_clip_feature,
    load_hidden,
    load_json,
    read_csv,
    write_csv,
)


def hidden_by_key(path: str) -> dict[str, str]:
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing manifest columns: {sorted(missing)}")
    return {str(row["key"]): str(row["hidden_path"]) for _, row in frame.iterrows()}


def manifest_token_pool(path: str) -> str:
    frame = pd.read_csv(path)
    pools = set(frame["token_pool"].astype(str)) if "token_pool" in frame.columns else {"cls"}
    if pools - {"cls", "patch_mean"} or len(pools) != 1:
        raise ValueError(f"{path}: expected one valid token_pool, got {sorted(pools)}")
    return next(iter(pools))


def target_lengths(path: str, prefix_from: str = "", prefix_to: str = "") -> dict[str, int]:
    """Return local target lengths, optionally rewriting an official CSV root."""
    if not path:
        return {}
    output: dict[str, int] = {}
    for _, row in read_csv(path).iterrows():
        source = str(row["path"])
        mapped = source.replace(prefix_from, prefix_to) if prefix_from else source
        length = int(load_clip_feature(mapped).shape[0])
        output[Path(source).stem] = length
        output.setdefault(base_key(source), length)
    return output


def align_to(feature: np.ndarray, target_length: int, pad_short: bool) -> tuple[np.ndarray, str]:
    if feature.shape[0] == target_length:
        return feature, "exact"
    if feature.shape[0] > target_length:
        return feature[:target_length], "crop"
    if pad_short:
        return np.concatenate([feature, np.repeat(feature[-1:], target_length - feature.shape[0], axis=0)]), "pad"
    return feature, "short"


def selected_neuron_features(hidden: np.ndarray, normal_mean: np.ndarray, normal_std: np.ndarray, selected: list[tuple[int, np.ndarray]]) -> np.ndarray:
    if hidden.ndim != 3 or hidden.shape[1:] != normal_mean.shape:
        raise ValueError(f"hidden shape {hidden.shape} does not match normal stats {normal_mean.shape}")
    z_hidden = (hidden - normal_mean) / (normal_std + 1e-6)
    return np.concatenate([z_hidden[:, layer, dims] for layer, dims in selected], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct 1280D VadCLIP residual-injection features.")
    parser.add_argument("--source-csv", required=True, help="Local path,label CSV; each path is a staged 512D CLIP .npy.")
    parser.add_argument("--hidden-manifest", required=True, help="Reused hidden manifest from the shared data cache.")
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--target-feature-csv", default="", help="Optional local CSV whose feature lengths define test alignment.")
    parser.add_argument("--target-prefix-from", default="", help="Optional root in target CSV paths to rewrite.")
    parser.add_argument("--target-prefix-to", default="", help="Local replacement for --target-prefix-from.")
    parser.add_argument("--pad-short", action="store_true", help="Repeat the final row only when an explicitly chosen target is longer.")
    parser.add_argument("--l2-norm-clip", action="store_true")
    parser.add_argument("--keep-missing", action="store_true", help="Skip source rows missing a hidden artifact instead of failing.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    output_csv = Path(args.out_csv)
    if args.clean and output_csv.exists():
        output_csv.unlink()
    cfg = load_json(args.neuron_json)
    neuron_width, clip_dim = int(cfg.get("neuron_width", 768)), int(cfg.get("clip_dim", 512))
    expected_width = int(cfg.get("input_width", cfg.get("visual_width", neuron_width + clip_dim)))
    if clip_dim != 512 or neuron_width + clip_dim != expected_width:
        raise ValueError("neuron JSON is not a valid 768D/512D concat contract")
    hidden_paths = hidden_by_key(args.hidden_manifest)
    token_pool = manifest_token_pool(args.hidden_manifest)
    if str(cfg.get("token_pool", "cls")) != token_pool:
        raise ValueError("selected-neuron token_pool differs from staged hidden manifest")
    normal_mean = np.load(cfg["normal_mean_path"]).astype(np.float32)
    normal_std = np.load(cfg["normal_std_path"]).astype(np.float32)
    selected = [(int(item["layer_index"]), np.asarray(item["dims"], dtype=np.int64)) for item in cfg["selected"]]
    if sum(len(dims) for _layer, dims in selected) != neuron_width:
        raise ValueError("selected dimension count does not match neuron_width")
    target_by_stem = target_lengths(args.target_feature_csv, args.target_prefix_from, args.target_prefix_to)
    source = read_csv(args.source_csv)
    cache: dict[str, np.ndarray] = {}
    output_paths: set[Path] = set()
    rows, skipped = [], []
    stats = {"exact": 0, "crop": 0, "pad": 0, "short": 0, "reused": 0}

    for _, row in tqdm(source.iterrows(), total=len(source), desc="build concat features", unit="feature"):
        clip_path, label = str(row["path"]), str(row["label"])
        key, stem = base_key(clip_path), Path(clip_path).stem
        out_path = out_dir / f"{stem}.npy"
        if out_path in output_paths:
            raise ValueError(f"duplicate output feature name: {out_path}")
        output_paths.add(out_path)
        if out_path.exists() and not args.no_resume:
            resumed = load_clip_feature(out_path)
            if resumed.shape[1] != expected_width:
                raise ValueError(f"{out_path}: existing feature width {resumed.shape[1]} != {expected_width}")
            rows.append([str(out_path), label])
            stats["reused"] += 1
            continue
        if key not in hidden_paths:
            if args.keep_missing:
                skipped.append([clip_path, label, "missing_hidden"])
                continue
            raise FileNotFoundError(f"missing staged hidden artifact for {key}: {clip_path}")
        if key not in cache:
            hidden, _metadata = load_hidden(hidden_paths[key])
            cache[key] = selected_neuron_features(hidden, normal_mean, normal_std, selected)
        neuron = cache[key].copy()
        clip = load_clip_feature(clip_path)
        if clip.shape[1] != clip_dim:
            raise ValueError(f"{clip_path}: expected {clip_dim}D final CLIP feature, got {clip.shape[1]}D")
        target_length = int(target_by_stem.get(stem, target_by_stem.get(key, clip.shape[0])))
        neuron, neuron_how = align_to(neuron, target_length, args.pad_short)
        clip, clip_how = align_to(clip, target_length, args.pad_short)
        how = neuron_how if neuron_how != "exact" else clip_how
        stats[how] += 1
        if neuron.shape[0] != clip.shape[0]:
            raise ValueError(
                f"{key}: neuron length {neuron.shape[0]} and CLIP length {clip.shape[0]} differ; "
                "fix staged feature alignment or use --pad-short deliberately"
            )
        if args.l2_norm_clip:
            clip = clip / np.maximum(np.linalg.norm(clip, axis=1, keepdims=True), 1e-8)
        concat = np.concatenate([neuron, clip], axis=1).astype(np.float32)
        if concat.shape != (target_length, expected_width):
            raise ValueError(f"{key}: output {concat.shape}, expected {(target_length, expected_width)}")
        np.save(out_path, concat)
        rows.append([str(out_path), label])

    write_csv(output_csv, ["path", "label"], rows)
    write_csv(out_dir / "skipped_rows.csv", ["source_path", "label", "reason"], skipped)
    print(f"alignment: exact={stats['exact']} crop={stats['crop']} pad={stats['pad']} short={stats['short']} reused={stats['reused']}")
    print(f"wrote {output_csv} with {len(rows)} rows of [T,{expected_width}] features", flush=True)


if __name__ == "__main__":
    main()
