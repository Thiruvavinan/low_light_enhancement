#!/usr/bin/env python
"""
scripts/tune_denoise.py
-----------------------
Selects the BM3D `sigma` and illumination-blend `gamma` for
`evaluation/denoise.py`, on TRAINING pairs.

    python scripts/tune_denoise.py --checkpoint runs/enhance_l1/last.pth --n 12

Why it tunes on training data
-----------------------------
The denoiser has two free parameters. Choosing them by looking at eval15
would make the reported LOL numbers a best-of-N over the test set, and the
cross-dataset numbers would inherit that choice. So the sweep runs on images
drawn from `our485` — the training split — and the selected values are then
applied unchanged to eval15, LIME, MEF and DICM.

The honest caveat: those images were seen by the network during training, so
its noise characteristics on them may not perfectly match unseen data, and
the chosen sigma is therefore not guaranteed optimal at evaluation time. What
matters is the direction of the leak, and there is none — nothing in the
evaluation sets influenced the choice, so the reported numbers are not
inflated by it. A cleaner design would hold out 20 pairs from training
specifically for this; that costs a retrain and is noted as a limitation
rather than silently skipped.

Cost
----
BM3D depends only on `sigma`, and the illumination blend only on `gamma`, so
each BM3D pass is computed once and reused across every gamma. That turns an
S x G sweep into S passes instead of S*G — roughly `n * len(sigmas) * 10 s`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.datasets import build_dataset
from evaluation.denoise import BM3DReflectanceDenoiser, illumination_blend
from evaluation.metrics import LPIPSMetric, psnr, ssim
from models import build_model

DEFAULT_SIGMAS = [0.01, 0.02, 0.04, 0.08]
DEFAULT_GAMMAS = [0.0, 0.5, 1.0, 2.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n", type=int, default=12, help="Training pairs to sweep over")
    parser.add_argument("--sigmas", type=float, nargs="*", default=DEFAULT_SIGMAS)
    parser.add_argument("--gammas", type=float, nargs="*", default=DEFAULT_GAMMAS)
    parser.add_argument("--select-on", default="lpips", choices=["lpips", "ssim", "psnr"],
                        help="Metric used to pick the winner. LPIPS by default: the "
                             "denoiser targets perceived noise, and PSNR actively "
                             "rewards over-smoothing.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model("retinex", freeze_decom=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model = model.to(device).eval()

    # Training split, evenly spaced so the subset is deterministic and not
    # cherry-picked. eval15 is never touched here.
    train = build_dataset("lol", root="data/lol", split="eval", splits=["our485"])
    step = max(1, len(train) // args.n)
    indices = list(range(0, len(train), step))[: args.n]
    print(f"Sweeping on {len(indices)} pairs from our485 (training split), device={device}")
    print(f"sigmas={args.sigmas}  gammas={args.gammas}  selecting on {args.select_on}\n")

    lpips_metric = LPIPSMetric(net="alex", device=device)

    # Cache the model outputs once -- they do not depend on sigma or gamma
    cached = []
    with torch.no_grad():
        for i in indices:
            sample = train[i]
            out = model({"low": sample["low"].unsqueeze(0).to(device)})
            cached.append((out["R_low"], out["I_low"], out["I_delta"],
                           sample["high"].unsqueeze(0).to(device)))

    def score(pred, high):
        return {"psnr": psnr(pred, high), "ssim": ssim(pred, high),
                "lpips": lpips_metric(pred, high)}

    results = {}

    # Baseline: no denoising at all
    base = [score(r * d, h) for r, _, d, h in cached]
    results[("off", "-")] = {k: sum(x[k] for x in base) / len(base) for k in base[0]}

    for sigma in args.sigmas:
        denoiser = BM3DReflectanceDenoiser(sigma=sigma, gamma=0.0)
        # One BM3D pass per image at this sigma, reused for every gamma
        passes = [(denoiser.denoise_only(r), r, i, d, h) for r, i, d, h in cached]
        for gamma in args.gammas:
            rows = [score(illumination_blend(dn, r, illum, gamma) * delta, h)
                    for dn, r, illum, delta, h in passes]
            results[(sigma, gamma)] = {k: sum(x[k] for x in rows) / len(rows) for k in rows[0]}
        print(f"  sigma={sigma} done", flush=True)

    print(f"\n{'sigma':>7} {'gamma':>7} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    for (sigma, gamma), m in results.items():
        print(f"{str(sigma):>7} {str(gamma):>7} {m['psnr']:>8.3f} {m['ssim']:>8.4f} {m['lpips']:>8.4f}")

    tuned = {k: v for k, v in results.items() if k[0] != "off"}
    higher_better = args.select_on in ("psnr", "ssim")
    best = (max if higher_better else min)(tuned, key=lambda k: tuned[k][args.select_on])
    off = results[("off", "-")]

    print(f"\nBest by {args.select_on}: sigma={best[0]}  gamma={best[1]}")
    print(f"  vs no denoising:  {args.select_on} {off[args.select_on]:.4f} -> "
          f"{tuned[best][args.select_on]:.4f}")
    print(f"\n  python scripts/evaluate.py --checkpoint {args.checkpoint} --tag <name> \\")
    print(f"      --denoise bm3d --denoise-sigma {best[0]} --denoise-gamma {best[1]}")


if __name__ == "__main__":
    main()
