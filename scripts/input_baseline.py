#!/usr/bin/env python
"""
scripts/input_baseline.py
-------------------------
Scores the UNENHANCED input images with the same metrics, as a floor for the
cross-dataset table.

    python scripts/input_baseline.py --tag input

Why this is not optional
------------------------
NIQE is a no-reference metric: it scores how natural an image's statistics
look, with nothing to compare against. That makes "arm B has lower NIQE than
arm A" easy to over-read. The question a reader should be able to answer is
whether enhancement improved naturalness *at all* — and the only way to know
is to score the input images the model was given.

A model can post a respectable NIQE and still be worse than doing nothing.
Reporting the enhanced numbers without this row would hide that, which is
exactly the kind of selective reporting this project committed to avoiding.

The "model" here is the identity function, run through the same engine, with
the same clamping, quantisation and metric objects as every other arm, so the
numbers are directly comparable rather than approximately so.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from data.datasets import build_dataset
from evaluation.engine import evaluate_dataset
from evaluation.metrics import LPIPSMetric, NoReferenceMetric

CROSS_EVAL = ["LIME", "MEF", "DICM"]


class Identity(nn.Module):
    """Returns the input unchanged, in the shape evaluate_dataset expects."""

    def forward(self, batch):
        low = batch["low"]
        ones = torch.ones_like(low[:, :1])
        return {"S": low, "R_low": low, "I_low": ones, "I_delta": ones}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="input")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--lol-root", default="data/lol")
    parser.add_argument("--cross-root", default="data/cross_eval")
    parser.add_argument("--no-reference", default="niqe", choices=["niqe", "brisque"])
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    model = Identity()
    lpips_metric = LPIPSMetric(net=args.lpips_net, device=device)
    no_ref, backend = NoReferenceMetric.best_available(device=device, preferred=args.no_reference)
    print(f"Identity baseline -- no-reference metric: {backend}\n")

    summary = {
        "tag": args.tag,
        "checkpoint": "(identity -- unenhanced input)",
        "device": device,
        "no_reference_backend": backend,
        "lpips_net": args.lpips_net,
        "padding_mode": "n/a",
        "max_side": 0,
        "benchmarks": {},
    }

    print("LOL eval15")
    lol = build_dataset("lol", root=args.lol_root, split="eval")
    rows, means = evaluate_dataset(model, lol, device=device, save_dir=None,
                                   lpips_metric=lpips_metric, no_reference_metric=no_ref)
    summary["benchmarks"]["LOL"] = {"n_images": len(lol), **means}
    (out_root / "LOL.json").write_text(json.dumps(rows, indent=2))
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))

    for name in CROSS_EVAL:
        print(f"\n{name}")
        ds = build_dataset("unpaired", root=args.cross_root, subset=name)
        rows, means = evaluate_dataset(model, ds, device=device, save_dir=None,
                                       lpips_metric=None, no_reference_metric=no_ref)
        summary["benchmarks"][name] = {"n_images": len(ds), **means}
        (out_root / f"{name}.json").write_text(json.dumps(rows, indent=2))
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
