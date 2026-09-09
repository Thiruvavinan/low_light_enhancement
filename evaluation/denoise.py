"""
evaluation/denoise.py
---------------------
BM3D denoising of the reflectance map -- the secondary contribution of
Wei et al. 2018 (Section 3.3), applied as a post-process at inference.

Why reflectance, and why it matters here
----------------------------------------
Retinex-Net reconstructs the enhanced image as `R_low * I_delta`, where
`R_low = S_low / I_low` is implicitly a division by a small number in dark
regions. That division amplifies whatever sensor noise was in the input by
roughly `1/I_low`. The brightening is the point; the noise amplification is
the cost, and it lands entirely in the reflectance.

This project measured that cost: the L1 baseline scores *worse* NIQE than the
untouched input image on two of three cross-datasets. Denoising reflectance
is the paper's own answer to it, which makes this the natural test of whether
noise amplification is the whole story.

The illumination-relative part
------------------------------
The paper notes that denoising strength should track illumination, because
noise is amplified more where the scene was darker. BM3D takes a single
scalar noise level and cannot vary it per pixel, so applying it at a strength
suitable for the darkest region would over-smooth the well-lit parts of the
same frame.

The approach here is one BM3D pass at `sigma`, blended back per-pixel:

    R_out = w * R_denoised + (1 - w) * R,      w = (1 - I_low) ** gamma

`w` approaches 1 where `I_low` is near zero -- exactly the pixels whose noise
was amplified most -- and approaches 0 in well-lit regions, which are left
untouched. `gamma` controls how sharply the blend transitions; `gamma=0`
disables the illumination dependence and denoises uniformly, which is the
ablation worth comparing against.

One BM3D pass rather than several is a deliberate cost decision: BM3D runs
about 10 s on a 400x600 image, and the alternative (several passes at
different strengths, selected per pixel) would multiply an already expensive
evaluation by the number of strengths for a second-order gain.

Denoising is applied to reflectance BEFORE recombination with `I_delta`, as
the paper specifies -- denoising the final image instead would smooth
structure that the illumination map legitimately introduced.
"""

from typing import Optional

import numpy as np
import torch


def illumination_blend(
    denoised: torch.Tensor,
    original: torch.Tensor,
    illumination: Optional[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    """
    Blend a denoised reflectance back toward the original, weighted by how
    dark the region was: w = (1 - I) ** gamma.

    Separated from the denoiser class so a parameter sweep can reuse ONE
    expensive BM3D pass across many gamma values -- BM3D depends only on
    sigma, the blend only on gamma. See scripts/tune_denoise.py.
    """
    if illumination is None or gamma <= 0:
        return denoised
    weight = (1.0 - illumination.clamp(0, 1)) ** gamma
    return weight * denoised + (1.0 - weight) * original


class BM3DReflectanceDenoiser:
    """
    Parameters
    ----------
    sigma : BM3D noise standard deviation, in the same [0, 1] scale as the
        image. Tuned on held-out TRAINING pairs, never on the evaluation set
        -- see scripts/tune_denoise.py.
    gamma : exponent of the illumination-relative blend. 0 disables it
        (uniform denoising everywhere); larger values confine denoising more
        tightly to dark regions.
    enabled : False makes this a no-op passthrough, so callers can hold one
        object and switch behaviour without branching.

    __call__(reflectance, illumination) -> denoised reflectance, same shape
    """

    def __init__(self, sigma: float = 0.04, gamma: float = 1.0, enabled: bool = True):
        self.sigma = sigma
        self.gamma = gamma
        self.enabled = enabled
        self._bm3d = None

        if enabled:
            import bm3d  # optional dependency; only imported when actually used

            self._bm3d = bm3d

    def __repr__(self) -> str:
        if not self.enabled:
            return "BM3DReflectanceDenoiser(disabled)"
        return f"BM3DReflectanceDenoiser(sigma={self.sigma}, gamma={self.gamma})"

    @torch.no_grad()
    def __call__(
        self, reflectance: torch.Tensor, illumination: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        reflectance  : [1,3,H,W] in [0,1]
        illumination : [1,1,H,W] in [0,1]; None means uniform denoising
        """
        if not self.enabled or self.sigma <= 0:
            return reflectance

        device, dtype = reflectance.device, reflectance.dtype
        # BM3D is a numpy/CPU routine expecting HWC float
        rgb = reflectance[0].detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

        denoised = self._bm3d.bm3d_rgb(rgb.astype(np.float64), self.sigma)
        denoised = np.clip(denoised, 0.0, 1.0).astype(np.float32)

        out = torch.from_numpy(denoised.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype)

        # w -> 1 in the dark (where 1/I amplified the noise most), -> 0 in
        # well-lit regions, which keep their original detail.
        return illumination_blend(out, reflectance, illumination, self.gamma)

    @torch.no_grad()
    def denoise_only(self, reflectance: torch.Tensor) -> torch.Tensor:
        """The BM3D pass without the illumination blend, for sweeps that vary
        gamma while holding sigma fixed."""
        device, dtype = reflectance.device, reflectance.dtype
        rgb = reflectance[0].detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        out = np.clip(self._bm3d.bm3d_rgb(rgb.astype(np.float64), self.sigma), 0.0, 1.0)
        return torch.from_numpy(out.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(device, dtype)
