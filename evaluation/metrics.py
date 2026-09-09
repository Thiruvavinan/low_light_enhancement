"""
evaluation/metrics.py
---------------------
Thin wrappers over established metric implementations. Nothing here computes
a metric from scratch -- every number in the results table comes from a
published, independently-tested library, so the comparison is trustworthy and
comparable to other people's numbers rather than to a private reimplementation.

    PSNR   scikit-image
    SSIM   scikit-image, configured to match Wang et al.'s reference MATLAB
    LPIPS  the `lpips` package (AlexNet backbone), Zhang et al. 2018
    NIQE / BRISQUE  no-reference; see NO_REFERENCE below

What this file does own is the *protocol*: the choices that make two numbers
comparable at all. Those are stated explicitly rather than left to a default.

Protocol
--------
* Predictions are clamped to [0, 1] before scoring. Enhance-Net's output is
  deliberately unbounded during training (models/enhance_net.py), but the
  thing a user would actually look at is the saved 8-bit image, so metrics
  are computed on the clamped value. Not clamping would let a model bank
  credit for detail stored above pure white, which no viewer ever sees.

* Predictions are quantised to 8 bits before scoring, for the same reason:
  the deliverable is a PNG, and float32-vs-uint8 is worth a few hundredths
  of a dB. Both arms are quantised identically.

* SSIM uses gaussian_weights=True, sigma=1.5, use_sample_covariance=False --
  the settings that reproduce the original MATLAB ssim(). scikit-image's
  plain defaults use a uniform 7x7 window and the sample covariance.

* TWO SSIM numbers are reported, because the choice of colour handling moves
  the value by more than the effect being measured, and neither convention
  is "the" right one:

      ssim       RGB, averaged over channels
      ssim_gray  luma only

  Measured here with the authors' own released weights on LOL eval15:

      RGB,  matlab-equivalent   0.4189
      RGB,  skimage defaults    0.4248
      gray, matlab-equivalent   0.5369
      gray, skimage defaults    0.5400
      published for RetinexNet  0.560

  So the widely-quoted 0.56 is a GRAYSCALE number; quoting an RGB SSIM
  against it would look like a 0.14 regression that is purely a protocol
  difference. `ssim_gray` is the literature-comparable column.

  `ssim` (RGB) is the primary column here, because it is the quantity the
  SSIM loss variant actually optimises -- the loss runs pytorch_msssim over
  3 channels. Scoring the variant on a grayscale metric would measure
  something adjacent to, rather than the same as, its training objective.

* LPIPS wants [-1, 1]; the conversion happens inside the wrapper so no
  caller can get it wrong.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ----------------------------------------------------------------------
# Shared preprocessing
# ----------------------------------------------------------------------

def to_uint8_float(image: torch.Tensor) -> np.ndarray:
    """
    [1,3,H,W] or [3,H,W] float tensor -> HWC float64 numpy in [0, 1],
    clamped and 8-bit quantised (see the protocol note above).
    """
    if image.dim() == 4:
        if image.shape[0] != 1:
            raise ValueError("metrics score one image at a time; got a batch")
        image = image[0]
    arr = image.detach().float().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
    return np.round(arr * 255.0) / 255.0


# ----------------------------------------------------------------------
# Full-reference: needs ground truth (LOL eval15 only)
# ----------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(peak_signal_noise_ratio(to_uint8_float(target), to_uint8_float(pred), data_range=1.0))


_SSIM_KWARGS = dict(data_range=1.0, gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RGB SSIM, averaged over channels. The quantity the SSIM loss optimises."""
    return float(
        structural_similarity(
            to_uint8_float(target), to_uint8_float(pred), channel_axis=2, **_SSIM_KWARGS
        )
    )


def ssim_gray(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Luma SSIM. The convention the published RetinexNet number uses -- see
    the module docstring; this is the column to compare against the paper."""
    return float(
        structural_similarity(
            rgb2gray(to_uint8_float(target)), rgb2gray(to_uint8_float(pred)), **_SSIM_KWARGS
        )
    )


class LPIPSMetric:
    """
    Perceptual distance (lower is better). Lazily constructed because it
    builds an AlexNet and downloads its linear-head weights on first use --
    the unpaired benchmarks never need it and should not pay for it.
    """

    def __init__(self, net: str = "alex", device: str = "cpu"):
        import lpips  # imported here so the package is optional until used

        self.net_name = net
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        def prep(x: torch.Tensor) -> torch.Tensor:
            arr = to_uint8_float(x)                      # same clamp/quantise as PSNR/SSIM
            t = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
            return (t * 2.0 - 1.0).to(self.device)       # LPIPS expects [-1, 1]

        return float(self.model(prep(pred), prep(target)).item())


# ----------------------------------------------------------------------
# No-reference: the cross-dataset benchmarks have no ground truth
# ----------------------------------------------------------------------

class NoReferenceMetric:
    """
    A no-reference image quality score (lower is better for both backends).

    Backends
    --------
    "niqe"    Natural Image Quality Evaluator (Mittal et al. 2013), via
              `pyiqa`. This is what the low-light literature reports, and
              what the project brief asks for. pyiqa is the only maintained
              Python implementation that ships the pristine-model parameters
              NIQE needs -- NIQE is not computable without them, which is why
              there is no self-contained fallback.

    "brisque" BRISQUE (Mittal et al. 2012), via `piq`. Same authors, same
              natural-scene-statistics family, also no-reference and also
              lower-is-better, but it is a *trained* opinion-aware model
              rather than NIQE's opinion-unaware distance to a pristine
              corpus. Used when pyiqa is not installed. Numbers are NOT
              interchangeable with published NIQE values and the results
              table records which backend produced them.

    Both are computed on the clamped, 8-bit-quantised prediction, exactly as
    the full-reference metrics are, and identically for both model arms.
    """

    BACKENDS = ("niqe", "brisque")

    def __init__(self, backend: str = "niqe", device: str = "cpu"):
        if backend not in self.BACKENDS:
            raise KeyError(f"Unknown no-reference metric '{backend}'. Available: {self.BACKENDS}")

        self.backend = backend
        self.device = device

        if backend == "niqe":
            import pyiqa  # optional dependency; see class docstring

            self._model = pyiqa.create_metric("niqe", device=device)
        else:
            import piq

            self._model = piq

    @staticmethod
    def available() -> List[str]:
        """Which backends this environment can actually run, best first."""
        found = []
        try:
            import pyiqa  # noqa: F401
            found.append("niqe")
        except ImportError:
            pass
        try:
            import piq  # noqa: F401
            found.append("brisque")
        except ImportError:
            pass
        return found

    @classmethod
    def best_available(cls, device: str = "cpu", preferred: Optional[str] = None):
        """
        Build the preferred backend if importable, else the best one that is.
        Returns (metric, backend_name) so the caller can record which was used.
        """
        options = cls.available()
        if not options:
            raise ImportError(
                "No no-reference metric backend available. Install `pyiqa` for NIQE "
                "(preferred) or `piq` for BRISQUE."
            )
        order = ([preferred] if preferred in options else []) + [o for o in options if o != preferred]
        return cls(order[0], device=device), order[0]

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> float:
        arr = to_uint8_float(image)
        t = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        if self.backend == "niqe":
            return float(self._model(t).item())
        return float(self._model.brisque(t, data_range=1.0, reduction="none").item())


# ----------------------------------------------------------------------

def summarise(per_image: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across images, ignoring images where it was absent."""
    keys = sorted({k for row in per_image for k in row if isinstance(row[k], (int, float))})
    out = {}
    for k in keys:
        values = [row[k] for row in per_image if isinstance(row.get(k), (int, float))]
        if values:
            out[k] = float(np.mean(values))
    return out
