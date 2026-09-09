"""
training/losses.py
------------------
Every loss term in Retinex-Net, plus the SSIM variant this project tests.

Both stages' losses are ports of the authors' TensorFlow implementation,
matched operation-for-operation (padding conventions included -- see
`gradient` and `ave_gradient`), so the baseline is a faithful reproduction
rather than a reimplementation-from-the-text that happens to look similar.

Paper vs. released code
-----------------------
The paper's text and the authors' released code disagree on two weights.
Where they differ, the defaults here follow the RELEASED CODE, because that
is the configuration that produced the published results:

    term                         paper text     released code   default here
    invariable reflectance       0.001          0.01            0.01
    Enhance-Net smoothness       1.0            3.0             3.0
    cross-reconstruction         0.001          0.001           0.001
    illumination smoothness      0.1            0.1             0.1
    gradient sensitivity         10             10              10

All of them are config knobs (configs/*.yaml), so reproducing the paper's
stated values is a config edit, not a code edit.

The comparison this project runs
--------------------------------
`EnhanceLoss` covers BOTH arms of the experiment. The baseline sets
ssim_weight=0 and the variant sets it to 1.0; nothing else about the loss,
the architecture, the data, the optimiser or the schedule differs. Making
the two arms the same class with a different weight -- rather than two
classes -- is deliberate: it makes it structurally impossible for an
unrelated difference to sneak into the variant.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Gradient helpers -- exact ports of the TF reference ops
# ----------------------------------------------------------------------

# The reference uses a 2x2 forward-difference kernel, not a centred Sobel.
# Horizontal: out(y,x) = in(y+1,x+1) - in(y+1,x)
_KERNEL_X = torch.tensor([[0.0, 0.0], [-1.0, 1.0]]).view(1, 1, 2, 2)
_KERNEL_Y = _KERNEL_X.transpose(2, 3).contiguous()

# TF's rgb_to_grayscale luma coefficients. Reflectance is converted to
# grayscale before its gradient is measured, so a colour edge with no
# luminance step does not relax the smoothness constraint.
_LUMA = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)


def gradient(x: torch.Tensor, direction: str) -> torch.Tensor:
    """
    Absolute forward difference of a 1-channel map, matching TF's 'SAME'
    padding for a 2x2 stride-1 kernel: TF pads bottom/right only, so
    replicate that with F.pad(..., (left, right, top, bottom)) = (0, 1, 0, 1)
    and an unpadded conv. Symmetric padding would shift the gradient map by
    half a pixel relative to the reference implementation.
    """
    kernel = (_KERNEL_X if direction == "x" else _KERNEL_Y).to(x.device, x.dtype)
    return torch.abs(F.conv2d(F.pad(x, (0, 1, 0, 1)), kernel))


def ave_gradient(x: torch.Tensor, direction: str) -> torch.Tensor:
    """
    3x3 box-average of the gradient magnitude. `count_include_pad=False`
    matches tf.layers.average_pooling2d with 'SAME' padding, which averages
    over valid elements only -- with the PyTorch default (True) border pixels
    would be biased toward zero, weakening the smoothness relaxation exactly
    at the frame edge.
    """
    return F.avg_pool2d(
        gradient(x, direction), kernel_size=3, stride=1, padding=1, count_include_pad=False
    )


def to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B,1,H,W] using TF's luma weights."""
    return (rgb * _LUMA.to(rgb.device, rgb.dtype)).sum(dim=1, keepdim=True)


def smoothness_loss(illumination: torch.Tensor, reflectance: torch.Tensor,
                    lambda_g: float = 10.0) -> torch.Tensor:
    """
    Structure-aware illumination smoothness (paper Eq. 4):

        L_is = mean( |grad_x I| * exp(-lambda_g * |grad_x R|)
                   + |grad_y I| * exp(-lambda_g * |grad_y R|) )

    This is the term that makes the decomposition work. Plain total variation
    is structure-blind: it penalises every illumination gradient equally, so
    it smooths across real object boundaries and leaves their edges baked into
    the illumination map instead of the reflectance. The exp(-lambda_g * grad R)
    weight collapses toward zero wherever reflectance has a strong edge, which
    is precisely where illumination is *allowed* to be discontinuous -- so
    smoothness is enforced on flat regions and released on structure.

    lambda_g controls how sharply that release kicks in; 10 is the paper's value.
    """
    r_gray = to_grayscale(reflectance)
    return torch.mean(
        gradient(illumination, "x") * torch.exp(-lambda_g * ave_gradient(r_gray, "x"))
        + gradient(illumination, "y") * torch.exp(-lambda_g * ave_gradient(r_gray, "y"))
    )


def _expand(illumination: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] -> [B,3,H,W] so illumination can multiply RGB reflectance."""
    return illumination.expand(-1, 3, -1, -1)


# ----------------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------------

class DecomLoss(nn.Module):
    """
    Decom-Net loss (paper Eq. 2-4):

        L_decom = L_recon + lambda_ir * L_ir + lambda_is * L_is

    L_recon is a 2x2 cross-reconstruction: each reflectance is recombined
    with each illumination and compared against the corresponding input.
    The two matched terms (R_low*I_low vs S_low, R_high*I_high vs S_high)
    carry weight 1; the two crossed terms carry 0.001. The crossed terms are
    what force the decomposition to be *transferable* -- without them the
    network could learn any factorisation that happens to reconstruct each
    image, including ones where "reflectance" silently absorbs lighting.

    L_ir = |R_low - R_high| states the Retinex assumption directly:
    reflectance is a property of the scene, so it must not change with
    illumination.

    Returns
    -------
    dict with "loss" (the scalar to backprop) plus every component, so the
    trainer can log which term is actually moving.
    """

    def __init__(
        self,
        cross_weight: float = 0.001,
        smooth_weight: float = 0.1,
        equal_r_weight: float = 0.01,
        lambda_g: float = 10.0,
    ):
        super().__init__()
        self.cross_weight = cross_weight
        self.smooth_weight = smooth_weight
        self.equal_r_weight = equal_r_weight
        self.lambda_g = lambda_g

    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s_low, s_high = batch["low"], batch["high"]
        r_low, i_low = outputs["R_low"], _expand(outputs["I_low"])
        r_high, i_high = outputs["R_high"], _expand(outputs["I_high"])

        recon_low = torch.mean(torch.abs(r_low * i_low - s_low))
        recon_high = torch.mean(torch.abs(r_high * i_high - s_high))
        # Crossed: the OTHER image's reflectance with this image's illumination
        recon_cross_low = torch.mean(torch.abs(r_high * i_low - s_low))
        recon_cross_high = torch.mean(torch.abs(r_low * i_high - s_high))

        equal_r = torch.mean(torch.abs(r_low - r_high))

        smooth_low = smoothness_loss(outputs["I_low"], r_low, self.lambda_g)
        smooth_high = smoothness_loss(outputs["I_high"], r_high, self.lambda_g)

        loss = (
            recon_low + recon_high
            + self.cross_weight * (recon_cross_low + recon_cross_high)
            + self.smooth_weight * (smooth_low + smooth_high)
            + self.equal_r_weight * equal_r
        )

        return {
            "loss": loss,
            "recon_low": recon_low,
            "recon_high": recon_high,
            "recon_cross": recon_cross_low + recon_cross_high,
            "equal_r": equal_r,
            "smooth": smooth_low + smooth_high,
        }


# ----------------------------------------------------------------------
# Stage 2 -- both arms of the experiment
# ----------------------------------------------------------------------

class EnhanceLoss(nn.Module):
    """
    Enhance-Net loss. Covers the baseline and the SSIM variant.

        L = l1_weight     * ||R_low * I_delta - S_normal||_1
          + ssim_weight   * (1 - SSIM(R_low * I_delta, S_normal))
          + smooth_weight * L_is(I_delta, R_low)

    ssim_weight = 0  -> the paper's baseline, exactly
    ssim_weight = 1  -> the variant under test

    Hypothesis behind the SSIM term
    -------------------------------
    L1 scores every pixel independently, so it is indifferent to whether the
    error it leaves behind is structured. A prediction that is uniformly a
    little too dim and one that has lost all local contrast can carry the
    same L1. SSIM compares local luminance, contrast and structure jointly,
    so it penalises exactly the failure mode L1 cannot see. Whether that
    helps out-of-distribution -- where there is no ground truth to fit and
    only the learned notion of "looks right" transfers -- is the open
    question this project measures.

    Note on the SSIM input range
    ----------------------------
    I_delta is intentionally unbounded (see models/enhance_net.py), so
    R_low * I_delta can exceed 1.0 early in training. SSIM is computed on
    the raw prediction rather than a clamped copy: clamping would zero the
    gradient wherever the prediction overshoots, which is precisely where a
    correction is needed. `data_range` only sets SSIM's stabilising
    constants, so a brief excursion above 1.0 is well-defined, not a silent
    error. Set clamp_ssim_input=True for the clamped behaviour.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.0,
        smooth_weight: float = 3.0,
        lambda_g: float = 10.0,
        ssim_window_size: int = 11,
        clamp_ssim_input: bool = False,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.smooth_weight = smooth_weight
        self.lambda_g = lambda_g
        self.clamp_ssim_input = clamp_ssim_input
        self._ssim = None

        if ssim_weight > 0:
            # Imported lazily so the baseline arm has no dependency on it at all.
            from pytorch_msssim import SSIM

            self._ssim = SSIM(
                data_range=1.0, size_average=True, channel=3, win_size=ssim_window_size
            )

    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target = batch["high"]
        enhanced = outputs["S"]  # R_low * I_delta, unclamped

        l1 = torch.mean(torch.abs(enhanced - target))
        smooth = smoothness_loss(outputs["I_delta"], outputs["R_low"], self.lambda_g)

        components = {"l1": l1, "smooth": smooth}
        loss = self.l1_weight * l1 + self.smooth_weight * smooth

        if self._ssim is not None:
            pred = enhanced.clamp(0.0, 1.0) if self.clamp_ssim_input else enhanced
            ssim_value = self._ssim(pred, target)
            ssim_term = 1.0 - ssim_value
            loss = loss + self.ssim_weight * ssim_term
            components["ssim"] = ssim_value
            components["ssim_term"] = ssim_term

        components["loss"] = loss
        return components


# ----------------------------------------------------------------------

LOSSES = {
    "decom": DecomLoss,
    "enhance": EnhanceLoss,
}


def build_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(LOSSES)}")
    return LOSSES[name](**kwargs)
