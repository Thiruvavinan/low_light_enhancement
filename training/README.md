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
| [`weighting.py`](weighting.py) | learned loss weights by homoscedastic uncertainty |

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

## Learned loss weights (`weighting.py`)

The Enhance-Net loss has three terms weighted 1.0 / 1.0 / 3.0 by hand. Those
numbers are asserted, not derived. `UncertaintyWeighting` learns them instead
(Kendall, Gal & Cipolla, CVPR 2018), by treating each term as the negative
log-likelihood of an observation model with its own learned noise scale:

```
L = sum_i [ 0.5 * exp(-s_i) * L_i  +  0.5 * s_i ]        s_i = log(sigma_i^2)
```

The `+0.5*s_i` log-barrier is what stops the trivial solution — without it
every weight collapses to zero and the reported loss goes to zero having
learned nothing.

Enable with `configs/enhance_uw.yaml`, or `--set loss.weighting=uncertainty`.

**Learned weights start at the hand-set ones.** `init_at_fixed_weights` (on
by default) solves each log-variance backwards from the fixed weight, so step
0 reproduces the hand-tuned configuration exactly and anything that moves
afterwards was genuinely learned. Skipping this starts every term at the same
weight regardless of scale — which is badly wrong when the hand-set weights
span orders of magnitude. Decom-Net is exactly that case: its weights run from
1.0 to 0.001, so a uniform start puts the cross-reconstruction term at **500×**
its intended strength, and that term is small on purpose. "Learned beats
fixed" measured from a start 500× away from fixed is not a comparison of
weighting schemes; it is a measurement of how well the optimiser recovers from
a bad initialisation.

**Two caveats that are easy to inherit unexamined:**

- *The Gaussian form is exact only for an L2 term.* `mode="gaussian"`
  reproduces the paper as published and is what everyone cites, but an L1
  term is a **Laplace** NLL, whose correct form is `exp(-s)*L + s`.
  `mode="laplace"` gives that. Applying either to SSIM is a heuristic in both
  cases — `1 - SSIM` is not a likelihood, so its learned "variance" is a free
  scale parameter with a log-barrier, not an uncertainty estimate.
- *Regularisers are not observations.* A data-fit term is pinned by the data:
  downweight it and its loss rises. A regulariser has no such counter-pressure,
  so uncertainty weighting will drift it toward whatever the barrier alone
  permits. `fixed_terms` therefore holds the smoothness prior at the paper's
  3.0 by default; `learn_smooth_weight=true` runs the ablation.

The module is Retinex-agnostic — it weights any dict of named scalar losses.
Because it holds `nn.Parameter`s it must reach the optimiser, which
`scripts/train.py` does by passing `loss_fn.parameters()` alongside the
model's, and the Trainer saves `loss_state` so learned weights survive a
checkpoint round-trip. Both are easy to forget, and forgetting either makes
the weights silently stay at their initial value.

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
