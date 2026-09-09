# Retinex-Net in PyTorch — does an SSIM loss term help?

A from-scratch PyTorch reimplementation of **Retinex-Net** (Wei et al., BMVC
2018), plus a controlled experiment: add an SSIM term to Enhance-Net's
reconstruction loss and measure it both in-distribution and across three
unseen datasets — which the original paper compares only with side-by-side
pictures, no metrics.

---

## Technique

**Decom-Net** splits an image into 3-channel reflectance `R` and 1-channel
illumination `I`, both sigmoid-bounded, so `S ≈ R · I`. Shared weights process
the low- and normal-light image of a pair; the pairing enters only through the
loss. **Enhance-Net** takes the frozen low-light decomposition and predicts an
enhanced illumination map `Î`, output `R_low · Î`. Encoder-decoder with
stride-2 down-sampling, resize-convolution up-sampling (not transposed conv),
additive skip connections (not concatenation), and a multi-scale fusion of all
three decoder scales. 445K parameters total.

The term that makes the decomposition work is a structure-aware smoothness
penalty, `mean(|∇I| · exp(−λ_g·|∇R|))`. Plain total variation is
structure-blind and smooths across real object boundaries, leaving their edges
baked into the illumination map. The exponential weight releases the penalty
exactly where reflectance has a strong edge — where illumination *is* allowed
to be discontinuous.

**The experiment.** The only difference between the two arms is one loss weight:

```
L = 1.0·‖R_low·Î − S_normal‖₁ + ssim_weight·(1 − SSIM(R_low·Î, S_normal)) + 3.0·L_is(Î, R_low)
```

`ssim_weight = 0` is the paper's baseline; `1.0` is the variant. Both arms load
the same frozen Decom-Net, the same seed, the same fixed 100-epoch budget.
`scripts/check_determinism.py` verifies they see identical training data.

Trained on LOL (485 real + 1000 synthetic pairs), 96×96 crops, batch 16, Adam
1e-3 (×0.1 at epoch 20), on an RTX 2050.

**The port is verified against the original.** `scripts/verify_port.py`
transplants the authors' released TensorFlow weights into these PyTorch
classes and diffs the outputs against the original TensorFlow graph — they
agree to 1.4e-05. That test caught a real bug: PyTorch's symmetric
`padding=1` on a stride-2 conv is not TensorFlow's `SAME` (which pads
bottom/right only on even inputs), and the mismatch produced outputs off by
0.27/1.0 that still looked like plausible enhanced images.

---

## Results

| Model | LOL PSNR ↑ | SSIM ↑ | LPIPS ↓ | LIME NIQE ↓ | MEF NIQE ↓ | DICM NIQE ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| input (no enhancement) | 7.77 | 0.195 | 0.560 | 4.35 | 5.19 | 3.86 |
| **Retinex-Net (L1)** | 18.26 | 0.595 | 0.442 | 5.03 | 4.87 | 4.12 |
| **Retinex-Net (+SSIM)** | **18.68** | **0.722** | **0.265** | **4.24** | **3.74** | **3.05** |
| authors' TF weights | 16.79 | 0.419 | 0.474 | 4.62 | 5.17 | 4.53 |

The `input` row is the identity function scored through the same pipeline.
NIQE is a no-reference metric, so without it "B beats A" says nothing about
whether either beat doing nothing. Full table with grayscale SSIM and
per-benchmark margins: [`outputs/results_table.md`](outputs/results_table.md).

![LOL comparison](outputs/qualitative/LOL_l1_ssim.png)

*LOL eval15, evenly-spaced sample (not hand-picked). Input · L1 · +SSIM ·
ground truth. Cross-dataset grids in [`outputs/qualitative/`](outputs/qualitative/).*

The SSIM variant wins on every metric. The baseline is where it gets
interesting — against the untouched input, the paper's L1 loss makes NIQE
**worse** on two of three cross-datasets:

| | input | L1 | +SSIM |
|---|:---:|:---:|:---:|
| LIME | 4.35 | 5.03 ✗ | 4.24 ✓ |
| MEF | 5.19 | 4.87 ✓ | 3.74 ✓ |
| DICM | 3.86 | 4.12 ✗ | 3.05 ✓ |

---

## Probable explanation

**The gain is structural, not per-pixel.** Both arms end at almost identical
training L1 (0.1116 vs 0.1114) and PSNR moves only +0.43 dB, while SSIM moves
+0.127 and LPIPS −0.177. The SSIM term did not make the model fit pixels
better — it changed how the residual error is *distributed*, and only the
metrics that can see structure register it.

**Why L1 alone loses to doing nothing.** L1 rewards getting the brightness
right and is indifferent to what the brightening does to noise. Retinex-Net
divides by a small illumination estimate in dark regions, which amplifies
sensor noise along with signal — and a natural-scene-statistics metric
penalises that more than it rewards the extra visibility. The paper itself
proposes an optional BM3D step for this reason.

**Why the SSIM term suppresses it.** SSIM's contrast and structure terms are
computed over local windows, and additive noise depresses local structural
correlation without moving local means much. So noise is visible to SSIM in a
way it is not to L1. The logs support the mechanism rather than a
blurring explanation: the SSIM arm's illumination smoothness loss settles 3.6×
*higher* (0.016 vs 0.0045), meaning a **less** smooth illumination map. It is
spending the smoothness budget differently, not flattening the image.

**What not to read into it.** LIME improves 4.35 → 4.24 over 10 images — a
wash. MEF (−1.44) and DICM (−0.81) are substantial; LIME is not, and its
inputs are the least dark to begin with. There is one seed per arm and no
error bars. And this L1 baseline *beats* the authors' released checkpoint
(18.26 vs 16.79 PSNR), most likely from training at 96×96 patches per the
paper's text rather than the released code's 48×48 default — so the SSIM
margin sits on top of a strong baseline, not a weakened one.

The obvious next experiment is a denoised L1 baseline: if noise amplification
is the whole story, BM3D should close most of the cross-dataset gap.

---

## Repository

Structured so every model plugs into the same pipeline — changing an
architecture does not touch training, evaluation or visualisation code. Each
folder has a README explaining why it exists.

| folder | contents |
|---|---|
| [`configs/`](configs/) | one YAML per run; the two arms differ in 2 lines |
| [`data/`](data/) | dataset loaders and split definitions |
| [`models/`](models/) | architectures, port verification, variations worth trying |
| [`training/`](training/) | trainer, losses, optimizers, gradient diagnostics |
| [`evaluation/`](evaluation/) | metric wrappers and the scoring engine |
| [`scripts/`](scripts/) | CLI entry points |

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python scripts/prepare_data.py            # needs the 3 dataset zips in the repo root

python scripts/train.py --config configs/decom.yaml          # stage 1, once
python scripts/train.py --config configs/enhance_l1.yaml     # baseline
python scripts/train.py --config configs/enhance_ssim.yaml   # variant

python scripts/evaluate.py --checkpoint runs/enhance_l1/last.pth   --tag l1
python scripts/evaluate.py --checkpoint runs/enhance_ssim/last.pth --tag ssim
python scripts/input_baseline.py --tag input
python scripts/results_table.py --tags input l1 ssim --margin l1 ssim
```

Per-module gradient diagnostics (`--set training.grad_log_every=50`) and
optional GroupNorm (`--set model.norm=group`) are available, both off by
default so the published architecture stays the default path.

---

Retinex-Net: Wei, Wang, Yang & Liu, *Deep Retinex Decomposition for Low-Light
Enhancement*, BMVC 2018. Original TensorFlow implementation:
<https://github.com/weichen582/RetinexNet> (MIT).
