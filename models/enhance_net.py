"""
models/enhance_net.py
---------------------
Enhance-Net (called Relight-Net in the authors' code): Stage 2 of
Retinex-Net (Wei et al., BMVC 2018, Section 3.2).

Takes the low-light decomposition (R_low, I_low) produced by a *frozen*
Decom-Net and predicts an enhanced illumination map I_delta. The final
enhanced image is R_low * I_delta.

Architecture notes
------------------
* Encoder-decoder, 3 stride-2 down-sampling blocks and 3 up-sampling blocks.

* Up-sampling is "resize-convolution": nearest-neighbour interpolation
  followed by a stride-1 conv, NOT a transposed convolution. The paper
  cites Odena et al. 2016 explicitly — transposed convs produce
  checkerboard artifacts, which on an illumination map would show up as
  periodic brightness banding in the enhanced output.

* Skip connections are element-wise SUMMATION (conv_i is added to the
  matching decoder output), not U-Net-style channel concatenation. This
  matters: concatenation would double the decoder's input width and change
  the parameter count, so it is not a free substitution.

* Multi-scale concatenation: all three decoder outputs are resized to full
  resolution by nearest-neighbour interpolation and concatenated along
  channels (C x M), fused back down to C by a 1x1 conv, then a final 3x3
  conv emits the single-channel illumination map. Fusing features from
  every scale is what lets the network reason about brightness that is
  consistent both locally and across the whole frame.

* The output conv has no activation and I_delta is NOT sigmoid-bounded --
  the reference implementation leaves it unbounded so the network can
  brighten past 1.0 during training. Values are clamped to [0, 1] only at
  inference/save time (see evaluation/engine.py), never inside the loss.

* Input is R_low and I_low concatenated (4 channels). The paper's Figure 1
  shows both being fed in; the reference implementation concatenates them
  in that order.

Resolution handling
-------------------
Interpolation targets are read from the actual encoder tensors rather than
computed from H/W, so odd input sizes survive the three stride-2 halvings
(e.g. 400x600 -> 200x300 -> 100x150 -> 50x75) without an off-by-one shape
mismatch at the skip connections. That is what lets evaluation run on
full-size images while training runs on 96x96 patches.

Padding
-------
`padding_mode="tf_same"` (the default) reproduces TensorFlow's 'SAME'
padding, which for a stride-2 3x3 conv over an EVEN-sized input pads one
row/column on the bottom/right only -- not symmetrically. PyTorch's
`padding=1` pads both sides, which shifts every encoder feature map by half
a stride relative to the reference implementation.

This is not a cosmetic detail. Porting the authors' released weights into
this class and comparing against the original TensorFlow graph on LOL eval
images gives:

    symmetric padding   max |diff| ~ 2.7e-01     (visibly different output)
    tf_same padding     max |diff| ~ 6e-06       (float32 round-off)

so the padding convention was the single difference between "looks like the
paper" and "is the paper". See scripts/convert_tf_weights.py, which is how
those numbers were produced.

Stride-1 convs are unaffected -- for stride 1 and an odd kernel, TF 'SAME'
and symmetric padding are identical -- so only the three down-sampling
convs take the special path. `padding_mode="symmetric"` selects idiomatic
PyTorch padding; it trains to a comparable result but is no longer
weight-compatible with the reference.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .norm import build_norm


def _resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour resize of x to ref's spatial size (paper: no transposed conv)."""
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


def _tf_same_padding(size: int, kernel: int, stride: int) -> Tuple[int, int]:
    """
    (before, after) padding for one axis under TensorFlow's 'SAME' rule:
    output = ceil(size / stride), and any odd padding goes on the far side.
    """
    out = -(-size // stride)                              # ceil division
    total = max((out - 1) * stride + kernel - size, 0)
    before = total // 2
    return before, total - before


def _tf_same_conv(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Apply `conv` (built with padding=0) using TF 'SAME' padding."""
    k, s = conv.kernel_size[0], conv.stride[0]
    top, bottom = _tf_same_padding(x.shape[-2], k, s)
    left, right = _tf_same_padding(x.shape[-1], k, s)
    return conv(F.pad(x, (left, right, top, bottom)))


class EnhanceNet(BaseModel):
    """
    Parameters
    ----------
    channels     : width of every conv layer (paper/reference: 64)
    kernel_size  : 3, as in the paper
    padding_mode : "tf_same" (default, weight-compatible with the authors'
                   release) or "symmetric" -- see the module docstring
    norm         : "none" (published, default), "group", "batch" or
                   "instance", inserted after each encoder/decoder conv and
                   before its ReLU. See models/norm.py -- "group" is the one
                   to try; "batch" is actively risky here because the
                   cross-dataset evaluation is a distribution shift and
                   BatchNorm's running statistics come from LOL.
    num_groups   : groups for norm="group"

    forward
    -------
    batch["R"] : [B, 3, H, W]   reflectance from a frozen Decom-Net
    batch["I"] : [B, 1, H, W]   illumination from a frozen Decom-Net
      ->  {"I_delta": [B, 1, H, W]}  unbounded (see module docstring)
    """

    PADDING_MODES = ("tf_same", "symmetric")

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        padding_mode: str = "tf_same",
        norm: str = "none",
        num_groups: int = 8,
    ):
        super().__init__()
        if padding_mode not in self.PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {self.PADDING_MODES}, got {padding_mode!r}"
            )
        self.padding_mode = padding_mode
        p = kernel_size // 2

        # Stem: 4 in-channels = R (3) + I (1). Stride 1, so TF 'SAME' and
        # symmetric padding coincide and the built-in padding is used either way.
        self.conv0 = nn.Conv2d(4, channels, kernel_size, padding=p)

        # Encoder: 3 stride-2 down-sampling blocks, each conv + ReLU.
        # Under tf_same these are built with padding=0 and padded explicitly
        # in forward, because the correct padding is asymmetric and depends on
        # the input size, which is not known at construction time.
        down_pad = 0 if padding_mode == "tf_same" else p
        self.down1 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=down_pad)
        self.down2 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=down_pad)
        self.down3 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=down_pad)

        # Decoder: resize-conv blocks; the resize itself happens in forward
        self.up1 = nn.Conv2d(channels, channels, kernel_size, padding=p)
        self.up2 = nn.Conv2d(channels, channels, kernel_size, padding=p)
        self.up3 = nn.Conv2d(channels, channels, kernel_size, padding=p)

        # Multi-scale fusion: C*3 -> C via 1x1, then 3x3 -> 1-channel illumination
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.out = nn.Conv2d(channels, 1, kernel_size, padding=p)

        self.relu = nn.ReLU(inplace=True)

        # One normaliser per activated conv (3 down + 3 up). ModuleDict rather
        # than a list so the state_dict keys name the layer they belong to,
        # which keeps a normalised checkpoint readable. Empty when norm="none",
        # so the published model has no extra parameters or buffers at all.
        self.norm = norm
        self.norms = nn.ModuleDict()
        if norm != "none":
            for key in ("down1", "down2", "down3", "up1", "up2", "up3"):
                self.norms[key] = build_norm(norm, channels, num_groups)

    def _act(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """[norm ->] ReLU. With norm="none" this is exactly the published net."""
        if key in self.norms:
            x = self.norms[key](x)
        return self.relu(x)

    def _down(self, key: str, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        y = _tf_same_conv(conv, x) if self.padding_mode == "tf_same" else conv(x)
        return self._act(key, y)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = torch.cat([batch["R"], batch["I"]], dim=1)

        conv0 = self.conv0(x)                             # full res, no activation
        conv1 = self._down("down1", self.down1, conv0)    # 1/2
        conv2 = self._down("down2", self.down2, conv1)    # 1/4
        conv3 = self._down("down3", self.down3, conv2)    # 1/8

        # Resize-conv up-sampling with additive skips. The skip is added AFTER
        # the activation, so it is never normalised -- normalising a residual
        # sum would rescale the encoder signal the skip exists to preserve.
        deconv1 = self._act("up1", self.up1(_resize_to(conv3, conv2))) + conv2    # 1/4
        deconv2 = self._act("up2", self.up2(_resize_to(deconv1, conv1))) + conv1  # 1/2
        deconv3 = self._act("up3", self.up3(_resize_to(deconv2, conv0))) + conv0  # full

        # Multi-scale concatenation at full resolution
        gathered = torch.cat(
            [_resize_to(deconv1, deconv3), _resize_to(deconv2, deconv3), deconv3],
            dim=1,
        )
        fused = self.fuse(gathered)

        return {"I_delta": self.out(fused)}
