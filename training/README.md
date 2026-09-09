# `training/` — the loop, the losses, the diagnostics

**Why this folder exists:** to hold everything about *how* a model is
optimised, with no knowledge of *what* is being optimised. Nothing here
imports from `models/`. The Trainer receives a module satisfying the
`models/base.py` contract and trains it; both Retinex-Net stages run through
it unchanged.

| file | what it is |
|---|---|
| [`trainer.py`](trainer.py) | stage-agnostic training loop, checkpointing, `history.json` |
| [`losses.py`](losses.py) | every loss term, both arms of the experiment |
| [`optim.py`](optim.py) | optimizer and LR-schedule factories |
| [`diagnostics.py`](diagnostics.py) | per-module gradient norms |

## The comparison is controlled by construction

`EnhanceLoss` covers **both** arms. The baseline is `ssim_weight=0`; the
variant is `ssim_weight=1.0`. Nothing else differs.

Making them one class with a different weight, rather than two classes, is
deliberate — it makes it structurally impossible for an unrelated difference
to drift into the variant. And at `ssim_weight=0` the class does not even
import `pytorch_msssim`, so the baseline is exactly the paper's loss with no
incidental numerical difference.

Because the Trainer is stage-agnostic, the two arms also cannot differ in
anything the loop does. That is the second half of the control; the first
half is in [`../configs/README.md`](../configs/README.md).

## Losses

Ports of the authors' TensorFlow implementation, matched operation for
operation — **including the padding conventions**, which are not incidental:

- `gradient()` uses a 2×2 forward-difference kernel with TF's `SAME` padding
  for a 2×2 stride-1 kernel, which pads **bottom/right only**. Symmetric
  padding would shift the gradient map by half a pixel.
- `ave_gradient()` uses `count_include_pad=False` to match
  `tf.layers.average_pooling2d`. With PyTorch's default (`True`), border
  pixels would be biased toward zero, weakening the smoothness relaxation
  exactly at the frame edge.

### The term that makes the decomposition work

`smoothness_loss` is `mean(|∇I| · exp(−λ_g·|∇R|))`, and the exponential
weight is the whole idea. Plain total variation is *structure-blind*: it
penalises every illumination gradient equally, so it smooths across real
object boundaries and leaves their edges baked into the illumination map
instead of the reflectance. Weighting by `exp(−λ_g·∇R)` collapses the penalty
toward zero wherever reflectance has a strong edge — precisely where
illumination *is* allowed to be discontinuous. Smoothness is enforced on flat
regions and released on structure.

### Why the SSIM term might help

L1 scores every pixel independently, so it is indifferent to whether the error
it leaves behind is structured. A prediction that is uniformly slightly too
dim and one that has lost all local contrast can carry the same L1. SSIM
compares local luminance, contrast and structure jointly, so it penalises
exactly the failure mode L1 cannot see.

Whether that transfers **out of distribution** — where there is no ground
truth to fit and only the learned notion of "looks right" carries over — is
the open question this project measures.

**SSIM is computed on the raw, unclamped prediction.** `I_delta` is
deliberately unbounded, so `R·I_delta` can exceed 1.0 early in training.
Clamping would zero the gradient exactly where the prediction overshoots,
which is where a correction is needed; `data_range` only sets SSIM's
stabilising constants, so a brief excursion above 1.0 is well-defined.
`clamp_ssim_input=True` selects the clamped behaviour.

## Gradient diagnostics

Off by default; enable with `--set training.grad_log_every=50`.

Reports gradient L2 norms **per module group**, not one global number — a
single global norm hides ten dead layers behind one large one. The key
readout is `grad_norm/spread`, the ratio of largest to smallest group norm; a
spread in the thousands is the vanishing-gradient signature, and the loss
curve looks like a plateau either way.

Recorded after `backward()` and **before** any clipping, so an exploding
gradient is still visible rather than already truncated. It never writes to
`.grad`, so enabling it cannot change what the model learns.

See [`../models/README.md`](../models/README.md) for the two failure modes
this architecture actually makes available (sigmoid saturation in Decom-Net's
heads; smoothness-term spikes while `I` is still noisy).

## Checkpoints

`best.pth` (lowest validation loss) and `last.pth` (final epoch) per run, plus
`history.json` and `config.resolved.yaml` — the fully-resolved config
including any `--set` overrides, written next to the weights it produced.

**The results table uses `last.pth`**, the fixed-budget final epoch, for both
arms. `best.pth` exists but selecting it per arm would compare two models
trained for different lengths against two non-comparable validation losses.
