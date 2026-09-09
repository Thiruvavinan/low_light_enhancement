#!/usr/bin/env python
"""
scripts/verify_port.py
----------------------
Regression test for the architecture port: loads the authors' released
TensorFlow weights into this PyTorch implementation and checks the outputs
against reference images produced by the ORIGINAL TensorFlow graph.

Why this exists
---------------
"Reimplemented from the paper" is not verifiable by reading the code -- a
transposed kernel, a concatenated skip connection, or the wrong padding
convention all produce code that looks right and results that are merely
plausible. Transplanting the reference weights makes the claim falsifiable:
if the architecture differs anywhere, the outputs diverge visibly.

It caught a real bug. PyTorch's `padding=1` on a stride-2 3x3 conv pads
symmetrically; TensorFlow's 'SAME' pads bottom/right only when the input is
even. With symmetric padding the transplanted weights produced outputs
differing from the reference by up to 0.27 (out of 1.0) -- clearly wrong,
but not so wrong that it would have been noticed by looking at pictures.
See models/enhance_net.py, `padding_mode`.

Reference images
----------------
docs/tf_reference/*.png are 8-bit renders of the original TensorFlow graph's
R, I_delta and S outputs on LOL eval images 1, 22 and 55. Stored as PNGs
rather than raw tensors so the repository stays small; that caps the
achievable precision at one 8-bit level, which is far tighter than any
architectural error would survive. The float32 comparison, run at the time
these were generated, agreed to 1.4e-05.

Prerequisites
-------------
    # once, in an environment with tensorflow:
    python scripts/convert_tf_weights.py export --out checkpoints/tf_reference.npz
    # then, here:
    python scripts/convert_tf_weights.py convert --npz checkpoints/tf_reference.npz \
        --out checkpoints/tf_reference.pth

Usage
-----
    python scripts/verify_port.py
    python scripts/verify_port.py --tolerance 2   # allow 2/255 per pixel
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from models import build_model

REFERENCE_DIR = Path("docs/tf_reference")
IMAGES = ["1", "22", "55"]
# reference file suffix -> (model output key, whether it is single-channel)
TENSORS = {"R": ("R_low", False), "I_delta": ("I_delta", True), "S": ("S", False)}


def load_reference(path: Path) -> np.ndarray:
    """8-bit PNG -> float HWC (or HW for single-channel) in [0, 1]."""
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tf_reference.pth")
    parser.add_argument("--data-root", default="data/lol/eval15/low")
    parser.add_argument("--padding-mode", default="tf_same", choices=["tf_same", "symmetric"],
                        help="'symmetric' is expected to FAIL; it is here to show the "
                             "test actually discriminates")
    parser.add_argument("--tolerance", type=int, default=1,
                        help="Maximum allowed per-pixel difference, in 8-bit levels")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Missing {ckpt_path}. See the Prerequisites section of this file's docstring.",
              file=sys.stderr)
        return 2

    model = build_model("retinex", channels=64, layer_num=5,
                        padding_mode=args.padding_mode, freeze_decom=True).eval()
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])

    tolerance = args.tolerance / 255.0
    failures = 0
    print(f"padding_mode={args.padding_mode}   tolerance={args.tolerance}/255\n")
    print(f"{'image':>6} {'tensor':>9} {'max diff (8-bit levels)':>24}   result")

    for name in IMAGES:
        low = np.asarray(Image.open(Path(args.data_root) / f"{name}.png").convert("RGB"),
                         dtype=np.float32) / 255.0
        with torch.no_grad():
            outputs = model({"low": torch.from_numpy(low.transpose(2, 0, 1)).unsqueeze(0)})

        for suffix, (key, single_channel) in TENSORS.items():
            reference = load_reference(REFERENCE_DIR / f"{name}_{suffix}.png")
            predicted = outputs[key][0].clamp(0, 1).numpy()
            predicted = predicted[0] if single_channel else predicted.transpose(1, 2, 0)

            diff = np.abs(predicted - reference).max()
            ok = diff <= tolerance
            failures += not ok
            print(f"{name:>6} {suffix:>9} {diff * 255:>24.2f}   {'ok' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"{failures} tensor(s) outside tolerance -- the port does NOT match the "
              f"reference implementation.")
        return 1
    print("All outputs match the original TensorFlow graph within 8-bit precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
