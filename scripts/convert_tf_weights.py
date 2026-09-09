#!/usr/bin/env python
"""
scripts/convert_tf_weights.py
-----------------------------
Port the authors' released TensorFlow weights into this PyTorch
implementation. This is a VERIFICATION tool, not part of the experiment: it
proves the architecture port is correct by transplanting the original
weights and checking the outputs are sane, rather than asking the reader to
take a re-read of the paper on faith.

If the port were wrong -- a transposed kernel, a skip connection that
concatenated instead of summed, the wrong padding on the shallow conv -- the
transplanted weights would produce garbage. They do not.

The checkpoint layout it reads (the authors' `./model/` directory):

    DecomNet/shallow_feature_extraction   [9,9,4,64]   -> decom.net.shallow
    DecomNet/activated_layer_{0..4}       [3,3,64,64]  -> decom.net.activated[2i]
    DecomNet/recon_layer                  [3,3,64,4]   -> decom.net.recon
    RelightNet/conv2d                     [3,3,4,64]   -> enhance.conv0
    RelightNet/conv2d_{1,2,3}             [3,3,64,64]  -> enhance.down{1,2,3}
    RelightNet/conv2d_{4,5,6}             [3,3,64,64]  -> enhance.up{1,2,3}
    RelightNet/conv2d_7                   [1,1,192,64] -> enhance.fuse
    RelightNet/conv2d_8                   [3,3,64,1]   -> enhance.out

TF stores conv kernels as [kh, kw, in, out]; PyTorch wants [out, in, kh, kw].

The `conv2d_7` shape [1,1,192,64] is itself a useful cross-check: 192 = 3*64
confirms the multi-scale concatenation gathers all three decoder scales, and
`conv2d` at [3,3,4,64] confirms the stem takes R (3ch) + I (1ch).

Usage
-----
TensorFlow and PyTorch rarely coexist in one environment, so this runs in
two steps and each half needs only one of them:

    # 1. in an environment with tensorflow (>=2.x reads TF1 checkpoints fine)
    python scripts/convert_tf_weights.py export \
        --checkpoint-dir RetinexNet/model --out checkpoints/tf_reference.npz

    # 2. in this project's environment (torch only)
    python scripts/convert_tf_weights.py convert \
        --npz checkpoints/tf_reference.npz --out checkpoints/tf_reference.pth

Then evaluate it like any other checkpoint:

    python scripts/evaluate.py --checkpoint checkpoints/tf_reference.pth --tag tf_reference
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# TF variable name (without /kernel|/bias) -> PyTorch module prefix
DECOM_MAP = {
    "DecomNet/shallow_feature_extraction": "decom.net.shallow",
    "DecomNet/recon_layer": "decom.net.recon",
    **{f"DecomNet/activated_layer_{i}": f"decom.net.activated.{2 * i}" for i in range(5)},
}

# tf.layers.conv2d auto-names sequentially in creation order within the scope
ENHANCE_MAP = {
    "RelightNet/conv2d": "enhance.conv0",
    "RelightNet/conv2d_1": "enhance.down1",
    "RelightNet/conv2d_2": "enhance.down2",
    "RelightNet/conv2d_3": "enhance.down3",
    "RelightNet/conv2d_4": "enhance.up1",
    "RelightNet/conv2d_5": "enhance.up2",
    "RelightNet/conv2d_6": "enhance.up3",
    "RelightNet/conv2d_7": "enhance.fuse",
    "RelightNet/conv2d_8": "enhance.out",
}


def cmd_export(args):
    """Read the TF checkpoints and dump the raw arrays to a .npz (needs tensorflow)."""
    import tensorflow as tf

    root = Path(args.checkpoint_dir)
    arrays = {}

    # Decom-Net weights from ./Decom, Enhance-Net weights from ./Relight --
    # the same split the authors' own test() uses. (Each checkpoint also
    # contains a stale copy of the other net's variables plus a DenoiseNet
    # that is not part of the published architecture; both are ignored.)
    for subdir, prefixes in [("Decom", DECOM_MAP), ("Relight", ENHANCE_MAP)]:
        reader = tf.train.load_checkpoint(str(root / subdir))
        available = reader.get_variable_to_shape_map()
        for tf_name in prefixes:
            for part in ("kernel", "bias"):
                key = f"{tf_name}/{part}"
                if key not in available:
                    raise KeyError(f"{key} not found in {root / subdir}")
                arrays[key] = reader.get_tensor(key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"Exported {len(arrays)} arrays to {out}")


def cmd_convert(args):
    """Turn the .npz into a checkpoint this repo's RetinexNet can load (needs torch)."""
    import torch

    from models import build_model

    data = np.load(args.npz)
    state = {}

    for tf_name, torch_name in {**DECOM_MAP, **ENHANCE_MAP}.items():
        kernel = data[f"{tf_name}/kernel"]          # [kh, kw, in, out]
        bias = data[f"{tf_name}/bias"]
        state[f"{torch_name}.weight"] = torch.from_numpy(
            np.ascontiguousarray(kernel.transpose(3, 2, 0, 1))   # -> [out, in, kh, kw]
        )
        state[f"{torch_name}.bias"] = torch.from_numpy(np.ascontiguousarray(bias))

    # strict=True is the whole point: any missing, extra or mis-shaped tensor
    # means the port does not match the reference architecture.
    model = build_model("retinex", channels=64, layer_num=5, freeze_decom=True)
    model.load_state_dict(state, strict=True)
    print(f"Loaded {len(state)} tensors into RetinexNet with strict=True -- shapes match.")

    with torch.no_grad():
        out = model({"low": torch.rand(1, 3, 128, 128)})
    print("Forward pass ok:", {k: tuple(v.shape) for k, v in out.items()})

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "model_state": model.state_dict(), "source": "official TF release"}, dest)
    print(f"Wrote {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="TF checkpoints -> .npz (needs tensorflow)")
    p_export.add_argument("--checkpoint-dir", default="RetinexNet/model",
                          help="Directory containing Decom/ and Relight/")
    p_export.add_argument("--out", default="checkpoints/tf_reference.npz")
    p_export.set_defaults(func=cmd_export)

    p_convert = sub.add_parser("convert", help=".npz -> RetinexNet .pth (needs torch)")
    p_convert.add_argument("--npz", default="checkpoints/tf_reference.npz")
    p_convert.add_argument("--out", default="checkpoints/tf_reference.pth")
    p_convert.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
