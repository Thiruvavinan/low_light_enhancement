"""
models/norm.py
--------------
Optional normalisation layers for Decom-Net and Enhance-Net.

Retinex-Net as published has NO normalisation anywhere, and "none" is the
default everywhere in this repo. This module exists so the choice can be
ablated deliberately rather than assumed, and so the L1-vs-SSIM comparison
is never accidentally run against a differently-normalised pair of models.

Which normaliser, and why it is not BatchNorm
---------------------------------------------
"group" (GroupNorm) is the recommended non-default. The reasons are specific
to this model, not general preference:

* Train/test batch mismatch. Training runs batch=16 on 96x96 patches;
  inference runs batch=1 on whole images. BatchNorm's running statistics are
  estimated under the first regime and applied under the second, and the two
  distributions genuinely differ -- a 96x96 crop of a dark room and a full
  frame containing both a lamp and deep shadow do not have the same channel
  statistics. GroupNorm computes its statistics per-sample, so training and
  inference behave identically.

* Distribution shift is the thing being measured. The cross-dataset
  evaluation (LIME/MEF/DICM) deliberately feeds the model images unlike its
  training set. BatchNorm's frozen running statistics come from LOL and
  would silently mis-normalise exactly the out-of-distribution case whose
  behaviour the project is trying to measure. That would confound the
  headline result with a BatchNorm artifact.

* "instance" (InstanceNorm) is available and is GroupNorm with one group per
  channel. It is the most aggressive at removing per-image contrast, which
  makes it the most likely to hurt here -- see the caveat below.

The caveat that applies to ALL of them
--------------------------------------
Normalisation removes per-feature mean and scale. Absolute brightness is not
a nuisance variable in this model -- it is the signal. Decom-Net's whole job
is to separate an image into reflectance (brightness-invariant) and
illumination (brightness-carrying), and its illumination head is a sigmoid
whose output is compared directly against input pixel values through
S = R * I. Normalising the features feeding that head throws away the
magnitude information it needs to reconstruct, and the reconstruction loss
then has to recover it through the final conv's bias alone.

So the honest expectation is that normalisation helps Enhance-Net's deeper
encoder-decoder more than it helps Decom-Net, and may measurably hurt
Decom-Net. That is a prediction worth testing, which is the point of making
it a flag; it is not a recommendation to turn it on.
"""

from typing import Optional

import torch.nn as nn

NORM_TYPES = ("none", "group", "batch", "instance")


def build_norm(norm: str, channels: int, num_groups: int = 8) -> Optional[nn.Module]:
    """
    Returns a normalisation layer for `channels` feature maps, or None for
    "none" -- callers should treat None as "insert nothing" rather than
    substituting nn.Identity, so a model built with norm="none" has exactly
    the published parameter count and module list.

    num_groups is clamped to a divisor of `channels`; GroupNorm requires
    channels % num_groups == 0 and the default 8 divides the 64 used
    throughout, but a narrower ablation (channels=32, 16) should not have to
    remember to change it too.
    """
    if norm not in NORM_TYPES:
        raise ValueError(f"norm must be one of {NORM_TYPES}, got {norm!r}")

    if norm == "none":
        return None
    if norm == "group":
        groups = num_groups
        while groups > 1 and channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    return nn.InstanceNorm2d(channels, affine=True)
