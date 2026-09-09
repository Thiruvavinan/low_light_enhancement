"""
models/decom_net.py
-------------------
Decom-Net: Stage 1 of Retinex-Net (Wei et al., BMVC 2018, Section 3.1).

Splits an sRGB image S into a reflectance map R (3-channel, illumination-
invariant) and an illumination map I (1-channel, spatially smooth), such
that S ~= R * I. The same weights process the low-light and the normal-light
image of a pair independently — the network is shared, not siamese-with-
different-weights, and the pairing only enters through the loss
(training/losses.py).

Architecture notes / deviations
-------------------------------
* Input is 4-channel, not 3. The channel-wise max of the RGB input is
  concatenated in front of it. The paper's text does not mention this;
  the authors' released implementation does it, and it is a sensible
  initialisation prior — max(R,G,B) is the classic hand-crafted
  illumination estimate that Retinex methods like LIME start from, so
  the network is handed a first guess at I rather than having to
  discover it. Kept for fidelity to the reference implementation.

* The paper (Section 4.1) describes "5 convolutional layers with a ReLU
  activation between 2 conv-layers without ReLU". The released code reads
  that as 5 *activated* layers sandwiched between an unactivated shallow
  feature-extraction conv and an unactivated reconstruction conv, i.e. 7
  convs total. We follow the released code (`layer_num=5`) because that is
  the configuration that produced the published numbers; `layer_num=3`
  reproduces the stricter reading of the text if you want to compare.

* The shallow feature-extraction conv uses a 9x9 kernel (the reference code
  writes it as `kernel_size * 3`), while every other conv is 3x3.

* Both heads are sigmoid-bounded to [0, 1]: R because reflectance is a
  bounded albedo, I because illumination is expressed in the same
  normalised range as the input image.

* No normalisation layers, as published. `norm` allows an ablation; read
  models/norm.py before turning it on, because normalising the features
  feeding the illumination head discards the brightness magnitude that head
  exists to predict.
"""

from typing import Dict

import torch
import torch.nn as nn

from .base import BaseModel
from .norm import build_norm


class DecomNet(BaseModel):
    """
    Parameters
    ----------
    channels   : width of every hidden conv layer (paper/reference: 64)
    layer_num  : number of ReLU-activated 3x3 convs between the shallow
                 feature-extraction conv and the reconstruction conv
    kernel_size: kernel of the hidden convs (the shallow conv uses 3x this)
    norm       : "none" (published, default), "group", "batch" or "instance",
                 inserted between each hidden conv and its ReLU
    num_groups : groups for norm="group"

    forward
    -------
    batch["image"] : [B, 3, H, W] in [0, 1]
      ->  {"R": [B, 3, H, W], "I": [B, 1, H, W]}, both in [0, 1]
    """

    def __init__(
        self,
        channels: int = 64,
        layer_num: int = 5,
        kernel_size: int = 3,
        norm: str = "none",
        num_groups: int = 8,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.norm = norm

        # 4 in-channels = channel-wise max prior + RGB (see module docstring)
        shallow_k = kernel_size * 3
        self.shallow = nn.Conv2d(4, channels, shallow_k, padding=shallow_k // 2)

        # conv -> [norm] -> ReLU, repeated. Built as a flat Sequential so that
        # with norm="none" the module indices are exactly 2i (conv) and 2i+1
        # (ReLU) -- which is what scripts/convert_tf_weights.py maps the
        # authors' `activated_layer_i` weights onto.
        blocks = []
        for _ in range(layer_num):
            blocks.append(nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2))
            norm_layer = build_norm(norm, channels, num_groups)
            if norm_layer is not None:
                blocks.append(norm_layer)
            blocks.append(nn.ReLU(inplace=True))
        self.activated = nn.Sequential(*blocks)

        # 4 out-channels: 3 reflectance + 1 illumination, split after the conv
        self.recon = nn.Conv2d(channels, 4, kernel_size, padding=kernel_size // 2)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = batch["image"]

        input_max, _ = torch.max(image, dim=1, keepdim=True)
        x = torch.cat([input_max, image], dim=1)

        x = self.shallow(x)          # no activation (paper: first conv has no ReLU)
        x = self.activated(x)
        x = self.recon(x)            # no activation (paper: last conv has no ReLU)

        return {
            "R": torch.sigmoid(x[:, 0:3, :, :]),
            "I": torch.sigmoid(x[:, 3:4, :, :]),
        }
