#!/usr/bin/env python3
"""Record the [selected-neuron | CLIP] input contract without reselecting neurons."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import clean_dir, ensure_dir, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrap a selected-neuron JSON in the 768D+512D VadCLIP concat contract.")
    parser.add_argument("--source-neuron-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--neuron-width", type=int, required=True)
    parser.add_argument("--clip-dim", type=int, default=512)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    out_path = out_dir / "selected_neurons.json"
    if out_path.exists() and not args.no_resume:
        print(f"reuse completed concat contract: {out_path}", flush=True)
        return
    source = load_json(args.source_neuron_json)
    missing = {"selected", "normal_mean_path", "normal_std_path"} - set(source)
    if missing:
        raise ValueError(f"{args.source_neuron_json} is missing: {sorted(missing)}")
    selected_width = sum(len(item["dims"]) for item in source["selected"])
    if selected_width != args.neuron_width:
        raise ValueError(f"selected dimensions={selected_width}, not --neuron-width={args.neuron_width}")
    if args.clip_dim != 512:
        raise ValueError("VadCLIP baseline final feature width is fixed at 512")

    content = dict(source)
    input_width = args.neuron_width + args.clip_dim
    content.update({
        "method": "vadclip_concat_clip512_v1",
        "description": "Frame-wise [selected-neuron z-score | staged official CLIP final feature] input for VadCLIP residual injection.",
        "source_neuron_json": args.source_neuron_json,
        "neuron_width": int(args.neuron_width), "clip_dim": int(args.clip_dim),
        "visual_width": int(input_width), "input_width": int(input_width),
        "concat_contract": {
            "order": "neuron_first_then_clip", "neuron_width": int(args.neuron_width),
            "clip_dim": int(args.clip_dim), "input_width": int(input_width),
            "clip_source": "staged VadCLIP-compatible 512D CLIP feature",
            "neuron_normalization": "z-score with selected-neuron normal_mean/normal_std",
            "clip_normalization": "raw feature values",
        },
    })
    save_json(out_path, content)
    print(f"wrote {out_path}: {args.neuron_width}D neuron + {args.clip_dim}D CLIP = {input_width}D", flush=True)


if __name__ == "__main__":
    main()
