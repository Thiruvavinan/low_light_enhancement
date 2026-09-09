"""
models/retinexnet.py
--------------------
The two trainable stages, each packaged as a single module with the
batch-dict-in / dict-out contract from models/base.py.

Why a wrapper instead of using DecomNet / EnhanceNet directly
-------------------------------------------------------------
Neither stage is "one image in, one image out":

  * Stage 1 runs the SAME Decom-Net over both images of a low/normal pair,
    because the decomposition loss is defined across the pair (invariable
    reflectance, cross-reconstruction). One `forward` therefore has to
    produce four tensors from two inputs.

  * Stage 2 needs a frozen, already-trained Decom-Net in front of the
    Enhance-Net. Freezing is a property of the *stage*, not of either
    network, so it belongs here rather than inside EnhanceNet.

Packaging both as `batch dict -> outputs dict` keeps training/trainer.py
completely stage-agnostic: it never learns that stage 1 has a pair and
stage 2 has a frozen prefix.

RetinexNet is also the inference model — the stage-2 training module and
the full enhancement pipeline are literally the same object, so what gets
evaluated is exactly what was trained.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseModel
from .decom_net import DecomNet
from .enhance_net import EnhanceNet


class DecomStage(BaseModel):
    """
    Stage 1 training module: shared-weight Decom-Net over a low/normal pair.

    forward
    -------
    batch["low"]  : [B, 3, H, W]           required
    batch["high"] : [B, 3, H, W]           optional (absent at inference)
      ->  {"R_low", "I_low"}  and, when "high" is present,
          {"R_high", "I_high"}
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
        self.net = DecomNet(
            channels=channels, layer_num=layer_num, kernel_size=kernel_size,
            norm=norm, num_groups=num_groups,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        low = self.net({"image": batch["low"]})
        out = {"R_low": low["R"], "I_low": low["I"]}

        if "high" in batch:
            high = self.net({"image": batch["high"]})
            out["R_high"] = high["R"]
            out["I_high"] = high["I"]

        return out


class RetinexNet(BaseModel):
    """
    Stage 2 training module and the full inference pipeline:
    frozen Decom-Net -> Enhance-Net -> enhanced image.

    Parameters
    ----------
    padding_mode : passed to Enhance-Net; "tf_same" keeps the model
        weight-compatible with the authors' release (see
        models/enhance_net.py). Identical in both arms of the experiment.
    norm : normalisation for ENHANCE-Net only ("none" by default; see
        models/norm.py). Decom-Net's normalisation is set separately by
        `decom_norm` and must match whatever the stage-1 checkpoint was
        trained with, otherwise its weights will not load.
    decom_norm : normalisation Decom-Net was trained with; None means "none".
    decom_checkpoint : path to a stage-1 checkpoint. Required for stage-2
        training and for inference; None leaves Decom-Net randomly
        initialised, which is only ever useful for a shape smoke test.
    freeze_decom : True reproduces the paper's two-stage protocol — stage 2
        optimises Enhance-Net only, against a fixed decomposition. Setting
        it False would make the two variants' comparison confounded (the
        decomposition itself would drift differently under each loss), so
        it should stay True for the L1-vs-SSIM experiment.

    forward
    -------
    batch["low"] : [B, 3, H, W]
      ->  {"R_low", "I_low", "I_delta", "S"}
          S = R_low * I_delta, the enhanced image, UNCLAMPED
          (clamping happens at save/metric time, not in the loss)
    """

    def __init__(
        self,
        channels: int = 64,
        layer_num: int = 5,
        kernel_size: int = 3,
        padding_mode: str = "tf_same",
        norm: str = "none",
        decom_norm: Optional[str] = None,
        num_groups: int = 8,
        decom_checkpoint: Optional[str] = None,
        freeze_decom: bool = True,
    ):
        super().__init__()
        # decom_norm defaults to whatever the stage-1 checkpoint was trained
        # with, which is usually "none" even when Enhance-Net is being
        # normalised -- passing `norm` down blindly would make the checkpoint
        # fail to load, with a confusing key error rather than a clear one.
        self.decom = DecomStage(
            channels=channels, layer_num=layer_num, kernel_size=kernel_size,
            norm=decom_norm if decom_norm is not None else "none",
            num_groups=num_groups,
        )
        self.enhance = EnhanceNet(
            channels=channels, kernel_size=kernel_size, padding_mode=padding_mode,
            norm=norm, num_groups=num_groups,
        )
        self.freeze_decom = freeze_decom

        if decom_checkpoint is not None:
            self.load_decom(decom_checkpoint)

        if freeze_decom:
            for p in self.decom.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------

    def load_decom(self, path: str) -> None:
        """
        Load stage-1 weights from a Trainer checkpoint or a bare state_dict.

        Also accepts a full RetinexNet checkpoint, taking just its `decom.*`
        entries -- convenient for re-running stage 2 from an existing trained
        model without first having to split its checkpoint apart.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

        if any(k.startswith("decom.") for k in state):
            state = {k[len("decom."):]: v for k, v in state.items() if k.startswith("decom.")}

        self.decom.load_state_dict(state)

    def train(self, mode: bool = True):
        """Keep a frozen Decom-Net in eval mode even when the stage is training."""
        super().train(mode)
        if self.freeze_decom:
            self.decom.eval()
        return self

    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # No gradient needs to reach Decom-Net when it is frozen; skipping the
        # graph there saves memory, which matters on a 4GB card.
        if self.freeze_decom:
            with torch.no_grad():
                decom = self.decom({"low": batch["low"]})
            decom = {k: v.detach() for k, v in decom.items()}
        else:
            decom = self.decom({"low": batch["low"]})

        i_delta = self.enhance({"R": decom["R_low"], "I": decom["I_low"]})["I_delta"]

        return {
            "R_low": decom["R_low"],
            "I_low": decom["I_low"],
            "I_delta": i_delta,
            "S": decom["R_low"] * i_delta,
        }
