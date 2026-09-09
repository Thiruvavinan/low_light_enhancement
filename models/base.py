"""
models/base.py
--------------
Abstract base class for the two networks that make up Retinex-Net.

Contract
--------
Retinex-Net is not a single end-to-end network — it is two networks trained
in two separate stages against two different losses (Wei et al., BMVC 2018,
Sections 3.1-3.2). So the contract here is deliberately looser than a
one-model-one-tensor benchmark: `forward` takes a dict of named tensors and
returns a dict of named tensors. That keeps the trainer, the losses and the
evaluation engine architecture-agnostic across both stages — a stage is
just "some module mapping a batch dict to an outputs dict".

    Stage 1  DecomNet    {"image": [B,3,H,W]}
                      -> {"R": [B,3,H,W], "I": [B,1,H,W]}

    Stage 2  EnhanceNet  {"R": [B,3,H,W], "I": [B,1,H,W]}
                      -> {"I_delta": [B,1,H,W]}

All image tensors are float32 in [0, 1], NCHW. Reflectance R is 3-channel,
illumination I is 1-channel (broadcast over RGB wherever it multiplies R).

Anything downstream — training/trainer.py, training/losses.py,
evaluation/engine.py — only ever touches these dicts by key, so adding a
stage or swapping an architecture requires no changes outside models/.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : dict of named float32 tensors in [0, 1], NCHW

        Returns
        -------
        dict of named float32 tensors (keys documented per subclass)
        """
        ...

    # ------------------------------------------------------------------
    # Convenience: parameter count (logged at training start)
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
