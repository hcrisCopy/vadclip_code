#!/usr/bin/env python3
"""Cheap preflight check: a fresh M1/M2 must equal frozen VadCLIP exactly."""
from __future__ import annotations

import argparse

import torch

from models import add_vadclip_source, build_film_model, split_concat
from train_m1_m2_xd import baseline_options, initialize_from_baseline
from xd_shared import XD_LABELS, build_test_loader, load_json, test_chunk_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the zero-initialised M1/M2 residual is exactly identity on one XD item.")
    parser.add_argument("--dataset", choices=["xd"], required=True)
    parser.add_argument("--module", choices=["m1", "m2"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--condition-hidden-dim", type=int, default=256)
    parser.add_argument("--residual-hidden-dim", type=int, default=512)
    parser.add_argument("--temporal-dilations", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device(args.device)
    contract = load_json(args.neuron_json)
    expected_width = int(contract.get("input_width", contract.get("visual_width", 1280)))
    options = baseline_options(args.vadclip_root)
    model = build_film_model(
        options, args.vadclip_root, str(device), contract, args.module,
        args.condition_hidden_dim, args.residual_hidden_dim, args.temporal_dilations,
    )
    copied = initialize_from_baseline(model, args.init_baseline_model)
    model.freeze_base()
    model.to(device).eval()
    item = next(iter(build_test_loader(args.test_list, options.visual_length, expected_width, num_workers=0)))
    visual = item[0].squeeze(0).to(device)
    original_length = int(item[1].item())
    if original_length < options.visual_length:
        visual = visual.unsqueeze(0)
    lengths = torch.as_tensor(test_chunk_lengths(original_length, options.visual_length), dtype=torch.long, device=device)
    padding_mask = torch.zeros((len(lengths), options.visual_length), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths.tolist()):
        if length < options.visual_length:
            padding_mask[index, length:] = True
    _neuron, clip = split_concat(visual, int(contract["neuron_width"]), int(contract["clip_dim"]))
    with torch.no_grad():
        enhanced = model.enhancement(_neuron, clip, lengths)
        baseline = model.base(clip, padding_mask, list(XD_LABELS.values()), lengths)
        wrapped = model(visual, padding_mask, list(XD_LABELS.values()), lengths)
    feature_error = float((enhanced - clip).abs().max().cpu())
    output_errors = [float((left - right).abs().max().cpu()) for left, right in zip(baseline, wrapped)]
    report = {"copied_baseline_tensors": copied, "max_feature_error": feature_error, "max_output_errors": output_errors}
    print(f"zero-identity {args.module}: {report}", flush=True)
    if feature_error != 0.0 or any(error != 0.0 for error in output_errors):
        raise RuntimeError("fresh zero-initialised M1/M2 is not exactly identical to frozen VadCLIP")


if __name__ == "__main__":
    main()
