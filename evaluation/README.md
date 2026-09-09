# `evaluation/` — metrics and the scoring engine

**Why this folder exists:** to keep scoring completely separate from
training. Evaluation must be able to run against any checkpoint — including
one produced by a different loss, or ported from the authors' TensorFlow
release — without importing anything training-specific. Nothing here imports
from `training/`.

| file | what it is |
|---|---|
| [`metrics.py`](metrics.py) | thin wrappers over established metric libraries |
| [`engine.py`](engine.py) | runs a model over a dataset and scores it |

## Nothing here is hand-rolled

Every number in the results table comes from a published, independently
tested implementation, so it is comparable to other people's numbers rather
than to a private reimplementation:

| metric | source | direction |
|---|---|---|
| PSNR | `scikit-image` | ↑ |
| SSIM (RGB + grayscale) | `scikit-image` | ↑ |
| LPIPS | `lpips` (AlexNet) | ↓ |
| NIQE | `pyiqa` | ↓ |
| BRISQUE | `piq` — fallback only | ↓ |

What this folder *does* own is the **protocol** — the choices that make two
numbers comparable at all. They are stated explicitly rather than left to a
library default.

## Protocol

- **Predictions are clamped to `[0,1]` and quantised to 8 bits before
  scoring.** Enhance-Net's output is deliberately unbounded during training,
  but the deliverable is a PNG — and without clamping, a model could bank
  credit for detail stored above pure white that no viewer ever sees. Both
  arms are treated identically.

- **SSIM matches the reference MATLAB `ssim()`**: `gaussian_weights=True,
  sigma=1.5, use_sample_covariance=False`. scikit-image's *plain defaults*
  use a uniform 7×7 window and the sample covariance.

- **Two SSIM columns are reported**, because the colour convention moves the
  value by more than the effect being measured. Measured with the authors'
  own released weights on LOL eval15:

  | protocol | value |
  |---|---:|
  | RGB, matlab-equivalent | 0.4189 |
  | RGB, scikit-image defaults | 0.4248 |
  | grayscale, matlab-equivalent | **0.5369** |
  | grayscale, scikit-image defaults | 0.5400 |
  | *published for RetinexNet on LOL-v1* | *0.560* |

  The widely-quoted 0.56 is a **grayscale** number. Reporting an RGB SSIM
  against it would look like a 0.14 regression that is entirely protocol.

  `ssim` (RGB) is the primary column, because it is what the SSIM loss
  variant actually optimises — the loss runs `pytorch_msssim` over three
  channels, so scoring it on luma would measure something adjacent to,
  rather than the same as, its training objective. `ssim_gray` is the
  literature-comparable column.

- **LPIPS input conversion to `[-1,1]` happens inside the wrapper**, so no
  caller can get it wrong.

- **Native resolution, one image at a time.** The benchmarks contain images of
  differing sizes, and resizing to a common shape would change the very
  statistics NIQE measures.

## Which metric applies where

The dataset decides, not a hard-coded list. A sample carrying `high` gets the
full-reference metrics; one without gets the no-reference metric only. A
missing `high` is a supported state, not an error.

| benchmark | ground truth | metrics |
|---|:---:|---|
| LOL eval15 | yes | PSNR, SSIM, LPIPS (+ NIQE) |
| LIME / MEF / DICM | no | NIQE only |

## NIQE, and the fallback

NIQE requires the pristine multivariate-Gaussian model parameters fitted by
its authors. Those parameters are data, not code — NIQE is **not computable
without them**, which is why there is no self-contained implementation here
and why the project brief's instruction not to hand-roll it is the right call.
`pyiqa` is the maintained Python package that ships them.

If `pyiqa` is unavailable, `NoReferenceMetric.best_available()` falls back to
`piq`'s BRISQUE: same authors, same natural-scene-statistics family, also
no-reference and lower-is-better — but a *trained, opinion-aware* model rather
than NIQE's opinion-unaware distance to a pristine corpus. The values are not
interchangeable with published NIQE numbers.

Whichever backend ran is recorded in `summary.json` and printed in the
results table. A metric whose protocol is not written down next to it is not
reproducible.

## Memory

Enhance-Net's multi-scale concatenation holds `3 × 64` channels at *full input
resolution*, so a single multi-megapixel LIME or DICM image can exceed a 4GB
card while the rest of the benchmark fits comfortably. Rather than downscaling
(which would change the statistics NIQE measures) or moving everything to CPU
(needlessly slow), `engine.py` catches the OOM and retries **that image** on
CPU. Every benchmark stays at native resolution, and both arms hit the same
images, so it cannot skew the comparison.
