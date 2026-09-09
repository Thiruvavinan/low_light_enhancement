# `models/` — architectures only

**Why this folder exists:** to hold network definitions and nothing else. No
training loop, no loss, no metric, no dataset. Every model satisfies the
`batch dict -> outputs dict` contract in [`base.py`](base.py), so swapping an
architecture never requires touching `training/`, `evaluation/` or `scripts/`.

| file | what it is |
|---|---|
| [`base.py`](base.py) | the contract every model obeys |
| [`decom_net.py`](decom_net.py) | Decom-Net — splits an image into reflectance + illumination |
| [`enhance_net.py`](enhance_net.py) | Enhance-Net — predicts the enhanced illumination map |
| [`retinexnet.py`](retinexnet.py) | the two trainable stages; `RetinexNet` is also the inference model |
| [`norm.py`](norm.py) | optional normalisation layers (**off by default**) |

Registered in [`__init__.py`](__init__.py) as `decom` (stage 1) and `retinex`
(stage 2 + inference).

---

## The port is verified, not asserted

`scripts/verify_port.py` loads the authors' released TensorFlow weights into
these classes and compares the outputs against the original TensorFlow graph
on LOL eval images. They agree to **1.4e-05** (float32 round-off).

That test earned its keep immediately: with PyTorch's ordinary symmetric
`padding=1`, the transplanted weights differed from the reference by up to
**0.27** out of 1.0. TensorFlow's `SAME` padding on a stride-2 3×3 conv over
an even-sized input pads the **bottom/right only**, not both sides. The
output *looked* like a plausible enhanced image either way — which is exactly
why a visual check would not have caught it. See `padding_mode` in
[`enhance_net.py`](enhance_net.py).

---

## What is already unusual about this architecture

Two swaps people normally propose are already in the paper, so proposing them
again is a no-op:

- **No pooling anywhere.** Down-sampling is done by stride-2 convolutions
  (`down1..3`). There is no maxpool to replace.
- **No transposed convolution.** Up-sampling is *resize-convolution* —
  nearest-neighbour interpolation followed by a stride-1 conv. The paper cites
  Odena et al. 2016 explicitly: transposed convs produce checkerboard
  artifacts, and on an *illumination map* that shows up as periodic brightness
  banding across the whole enhanced frame.

A third choice is easy to miss and easy to break:

- **Skip connections are addition, not concatenation.** `deconv_i + conv_i`,
  not `cat([deconv_i, conv_i])`. Switching to U-Net-style concatenation
  doubles each decoder conv's input width and changes the parameter count, so
  it is not a drop-in substitution.

---

## Layer-level variations worth trying

These are **not** part of the L1-vs-SSIM comparison, which requires both arms
to be architecturally identical (see [`../configs/README.md`](../configs/README.md)).
Run them as separate, clearly-labelled ablations.

Ordered by expected value for the effort.

### 1. Widen the receptive field before the bottleneck — *implemented as `channels`, but see below*

Enhance-Net sees 1/8 resolution at its deepest point, which on a 400×600 LOL
image is 50×75 — a 3×3 conv there covers ~24 input pixels. Illumination is a
*global* property: a lamp in one corner changes the correct exposure across
the frame. Options, cheapest first:

- **Dilated convs in the bottleneck.** Replace `down3` or add a dilated conv
  after it (`dilation=2,4`). Multiplies receptive field with no parameter cost
  and no resolution loss. This is the one I would try first.
- **A fourth down-sampling stage.** Doubles receptive field but costs a
  stage's parameters and makes small inputs awkward (96×96 patches become 6×6).
- **Global context branch.** Global-average-pool → 1×1 conv → broadcast-add
  into the fusion. Very cheap; directly supplies the whole-frame brightness
  signal the multi-scale concat only approximates.

### 2. Replace the 1×1 fusion conv with attention over scales

`fuse` currently collapses the three decoder scales with a *fixed* learned 1×1
projection — every pixel weights the three scales identically. A per-pixel
softmax over the three scales (a small conv predicting 3 weights) lets a dark
region draw on coarse context while a well-lit region trusts fine detail. Very
cheap, and it targets a real limitation.

### 3. Depthwise-separable convs

Replace 3×3 convs with depthwise + pointwise. Cuts parameters ~8×. The model
is already tiny (445K), so this is only interesting if the target is embedded
inference — worth stating in the README as a deployment note rather than an
accuracy experiment.

### 4. Residual blocks in Decom-Net's trunk

Decom-Net's five activated convs are a plain stack. Making them residual
(`x + conv(relu(conv(x)))`) would help if gradient flow to the early layers is
poor — **check the gradient diagnostics first** (see below) rather than
assuming it. At 5 layers deep, it probably is not the bottleneck.

### 5. Larger kernel in the shallow conv

Already 9×9, which is unusual and deliberate — it is what gives the first
layer enough context to estimate illumination. Shrinking it to 3×3 is a good
*ablation* to demonstrate the choice matters.

---

## Normalisation (`norm.py`) — off by default, and here is why

Available as `--set model.norm={none,group,batch,instance}`. Default `none`,
matching the paper.

**Use GroupNorm, not BatchNorm, if you try this.** The reasons are specific to
this project, not general preference:

- **Train/test batch mismatch.** Training is batch=16 on 96×96 crops;
  inference is batch=1 on whole images. BatchNorm estimates running statistics
  under the first regime and applies them under the second, and those
  distributions genuinely differ — a dark 96×96 crop and a full frame
  containing both a lamp and deep shadow do not share channel statistics.
  GroupNorm normalises per-sample, so train and test behave identically.
- **Distribution shift is the measurement.** The cross-dataset evaluation
  deliberately feeds the model LIME/MEF/DICM images unlike its training set.
  BatchNorm's frozen statistics come from LOL and would mis-normalise exactly
  the out-of-distribution case this project exists to measure — confounding
  the headline result with a BatchNorm artifact.

**The caveat that applies to all of them.** Normalisation removes per-feature
mean and scale. In this model, absolute brightness is not a nuisance variable
— it is the signal. Decom-Net's job is to separate brightness-invariant
reflectance from brightness-carrying illumination, and its illumination head
is a sigmoid compared directly against input pixels through `S = R * I`.
Normalising the features feeding that head discards the magnitude information
it needs.

So the honest prediction: normalisation should help **Enhance-Net's** deeper
encoder-decoder more than it helps **Decom-Net**, and may measurably *hurt*
Decom-Net. That is a testable claim, which is why it is a flag —
`RetinexNet` takes `norm` (Enhance-Net) and `decom_norm` separately so the two
can be ablated independently. Note that `decom_norm` must match whatever the
stage-1 checkpoint was actually trained with, or its weights will not load.

---

## Debugging exploding / vanishing gradients

`training/diagnostics.py` reports gradient norms **per module group**
(`decom.shallow`, `decom.activated`, `enhance.down`, `enhance.up`,
`enhance.fuse`, `enhance.out`), not just one global number — a single global
norm hides ten dead layers behind one large one.

```bash
python scripts/train.py --config configs/enhance_l1.yaml --set training.grad_log_every=50
python scripts/plot_history.py --run runs/enhance_l1 --grad
```

The number to watch is **`grad_norm/spread`** — the ratio of the largest group
norm to the smallest. A spread in the thousands means the small end is
receiving effectively nothing, which is the vanishing-gradient signature; the
total loss looks like a plateau either way.

Two failure modes are genuinely available in this architecture, and they are
not hypothetical:

- **Sigmoid saturation.** Both Decom-Net heads end in a sigmoid. Once a
  pre-activation is far from zero the local gradient is ≈0 and everything
  behind it stops learning.
- **Smoothness-term spikes.** `L_is` contains `exp(-10 · ∇R)`, which is
  well-behaved, but its gradient scales with the illumination gradient
  magnitude and can spike early while `I` is still noisy.

The monitor records after `backward()` and **before** any gradient clipping,
so an exploding gradient is still visible rather than already truncated. It
never touches `.grad`, so turning it on cannot change what the model learns.
