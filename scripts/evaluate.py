#!/usr/bin/env python
"""
scripts/evaluate.py
-------------------
Scores one checkpoint on every benchmark and writes the numbers to disk.
Run it once per model arm; `scripts/results_table.py` then assembles the
comparison table from whatever it finds.

    python scripts/evaluate.py --checkpoint runs/enhance_l1/last.pth   --tag l1
    python scripts/evaluate.py --checkpoint runs/enhance_ssim/last.pth --tag ssim
    python scripts/evaluate.py --checkpoint checkpoints/tf_reference.pth --tag tf_reference

Benchmarks
----------
    LOL eval15   paired, has ground truth  -> PSNR, SSIM, LPIPS
    LIME         unpaired, no ground truth -> no-reference metric only
    MEF          unpaired                  -> no-reference metric only
    DICM         unpaired                  -> no-reference metric only

Outputs, under outputs/eval/<tag>/
    summary.json          mean of every metric per benchmark, plus the exact
                          configuration the numbers were produced under
    <benchmark>.json      per-image scores, so a mean can be interrogated
                          rather than trusted
    images/<benchmark>/   the enhanced PNGs themselves

Everything that could differ between two runs and change a number -- the
no-reference backend, the LPIPS backbone, the device, the checkpoint path --
is recorded in summary.json. A metric whose protocol is not written down next
to it is not reproducible.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.datasets import build_dataset
from evaluation.denoise import BM3DReflectanceDenoiser
from evaluation.engine import evaluate_dataset
from evaluation.metrics import LPIPSMetric, NoReferenceMetric
from models import build_model

CROSS_EVAL = ["LIME", "MEF", "DICM"]


def load_model(checkpoint: str, device: str, padding_mode: str, layer_num: int, channels: int):
    """Build a RetinexNet and load a full stage-2 checkpoint into it."""
    model = build_model(
        "retinex", channels=channels, layer_num=layer_num,
        padding_mode=padding_mode, freeze_decom=True,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint (.pth)")
    parser.add_argument("--tag", required=True, help="Short name for this arm, e.g. l1 / ssim")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--lol-root", default="data/lol")
    parser.add_argument("--cross-root", default="data/cross_eval")
    parser.add_argument("--no-reference", default="niqe", choices=["niqe", "brisque"],
                        help="Preferred no-reference backend; falls back to whichever is "
                             "installed and records the choice in summary.json")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    parser.add_argument("--padding-mode", default="tf_same", choices=["tf_same", "symmetric"])
    parser.add_argument("--layer-num", type=int, default=5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--max-side", type=int, default=0,
                        help="Downscale unpaired images whose longer edge exceeds this. "
                             "0 (default) keeps native resolution; the engine falls back "
                             "to CPU for any image that does not fit on the GPU.")
    parser.add_argument("--skip-cross", action="store_true", help="LOL only (quick check)")
    parser.add_argument("--save-decomposition", action="store_true",
                        help="Also write R, I and I_delta for the qualitative figures")
    parser.add_argument("--denoise", choices=["none", "bm3d"], default="none",
                        help="BM3D post-process on reflectance (paper Section 3.3). "
                             "Inference-time only -- the same checkpoint is scored with "
                             "and without it, so the comparison isolates the denoiser.")
    parser.add_argument("--denoise-sigma", type=float, default=0.04,
                        help="BM3D noise sigma in [0,1] image scale. Tune it with "
                             "scripts/tune_denoise.py on TRAINING pairs, never on eval15.")
    parser.add_argument("--denoise-gamma", type=float, default=1.0,
                        help="Illumination-relative blend exponent; 0 denoises uniformly")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device, args.padding_mode, args.layer_num, args.channels)

    denoiser = None
    if args.denoise == "bm3d":
        denoiser = BM3DReflectanceDenoiser(sigma=args.denoise_sigma, gamma=args.denoise_gamma)
        print(f"Denoiser: {denoiser}  (adds roughly 10 s per 400x600 image)")

    # Built once and reused; LPIPS in particular loads a network per construction.
    lpips_metric = LPIPSMetric(net=args.lpips_net, device=device)
    no_ref, no_ref_backend = NoReferenceMetric.best_available(
        device=device, preferred=args.no_reference
    )
    if no_ref_backend != args.no_reference:
        print(f"NOTE: '{args.no_reference}' is not installed; using '{no_ref_backend}' instead. "
              f"This is recorded in summary.json.")
    print(f"No-reference metric: {no_ref_backend}    LPIPS backbone: {args.lpips_net}\n")

    summary = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "no_reference_backend": no_ref_backend,
        "lpips_net": args.lpips_net,
        "padding_mode": args.padding_mode,
        "max_side": args.max_side,
        "denoise": args.denoise,
        "denoise_sigma": args.denoise_sigma if args.denoise != "none" else None,
        "denoise_gamma": args.denoise_gamma if args.denoise != "none" else None,
        "benchmarks": {},
    }

    # ------------------------------------------------------------------
    # In-distribution: LOL eval15 (paired)
    # ------------------------------------------------------------------
    print("LOL eval15 (paired: PSNR / SSIM / LPIPS)")
    lol = build_dataset("lol", root=args.lol_root, split="eval")
    rows, means = evaluate_dataset(
        model, lol, device=device,
        save_dir=out_root / "images" / "LOL",
        lpips_metric=lpips_metric,
        no_reference_metric=no_ref,
        save_decomposition=args.save_decomposition,
        denoiser=denoiser,
    )
    summary["benchmarks"]["LOL"] = {"n_images": len(lol), **means}
    (out_root / "LOL.json").write_text(json.dumps(rows, indent=2))
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))

    # ------------------------------------------------------------------
    # Cross-dataset: LIME / MEF / DICM (unpaired, no ground truth)
    # ------------------------------------------------------------------
    if not args.skip_cross:
        for name in CROSS_EVAL:
            print(f"\n{name} (unpaired: {no_ref_backend} only)")
            ds = build_dataset("unpaired", root=args.cross_root, subset=name,
                               max_side=args.max_side)
            rows, means = evaluate_dataset(
                model, ds, device=device,
                save_dir=out_root / "images" / name,
                lpips_metric=None,            # no ground truth to compare against
                no_reference_metric=no_ref,
                denoiser=denoiser,
            )
            summary["benchmarks"][name] = {"n_images": len(ds), **means}
            (out_root / f"{name}.json").write_text(json.dumps(rows, indent=2))
            print("  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
