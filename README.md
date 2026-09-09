# Retinex-Net in PyTorch — what actually improves it?

A from-scratch PyTorch reimplementation of **Retinex-Net** (Wei et al., BMVC
2018), verified against the authors' released TensorFlow weights, plus four
controlled experiments on the loss: an SSIM reconstruction term, learned loss
weights, a fixed-weight control, and the paper's BM3D denoising step —
evaluated in-distribution **and across three unseen datasets**, which the
original paper compares only with pictures.

**Headline.** The best model is not the most sophisticated one. Adding SSIM to
the reconstruction loss helps a lot; learning the loss weights by homoscedastic
uncertainty helps slightly more — but a *fixed* config at the weights it
learned matches or beats it. And the paper's own BM3D denoising **improves
in-distribution scores while making cross-dataset naturalness worse**, every
time, on every arm.

---

## Results

Clean view below; the full table with grayscale-SSIM and per-benchmark margins
is in [`outputs/results_table.md`](outputs/results_table.md). Nothing is
omitted from either — the layout groups rows, it does not filter them.

| Model | LOL PSNR ↑ | LOL SSIM ↑ | LOL LPIPS ↓ | LIME NIQE ↓ | MEF NIQE ↓ | DICM NIQE ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| *input (no enhancement)* | 7.77 | 0.1952 | 0.5595 | 4.35 | 5.19 | 3.86 |
| *authors' TF weights* | 16.79 | 0.4189 | 0.4740 | 4.62 | 5.17 | 4.53 |
| **Retinex-Net (L1)** — the paper | 18.26 | 0.5949 | 0.4421 | 5.03 | 4.87 | 4.12 |
| **+ SSIM term** | 18.68 | 0.7222 | 0.2651 | 4.24 | 3.74 | 3.05 |
| **+ learned weights** | 19.05 | 0.7542 | 0.2215 | **3.94** | 3.50 | **2.69** |
| **+ fixed rebalanced weights** | **19.11** | 0.7627 | 0.2055 | 4.00 | **3.47** | **2.69** |
| L1 + BM3D | 18.54 | 0.7734 | 0.2323 | 5.15 | 5.53 | 4.42 |
| SSIM + BM3D | 18.79 | **0.7910** | **0.1900** | 4.47 | 3.91 | 3.21 |
| learned + BM3D | 19.07 | 0.7717 | 0.2059 | 4.09 | 3.59 | 2.76 |

Two rows exist so the rest can be read honestly. **`input`** is the identity
function scored through the same pipeline — NIQE is no-reference, so without it
"B beats A" says nothing about whether either beat doing nothing. **`authors'
TF weights`** anchors the reimplementation against the published model.

![LOL comparison](outputs/qualitative/LOL_l1_ssim_rebalanced.png)

*LOL eval15, evenly-spaced sample (not hand-picked). The L1 column carries the
amplified sensor noise Retinex-Net is known for. More grids, including the
BM3D effect, in [`outputs/qualitative/`](outputs/qualitative/).*

---

## Technique

**Decom-Net** splits an image into 3-channel reflectance `R` and 1-channel
illumination `I`, both sigmoid-bounded, so `S ≈ R · I`. Shared weights process
the low- and normal-light image of a pair; the pairing enters only through the
loss. **Enhance-Net** takes the frozen low-light decomposition and predicts an
enhanced illumination map `Î`, output `R_low · Î` — encoder-decoder with
stride-2 down-sampling, resize-convolution up-sampling (not transposed conv),
additive skips (not concatenation), and multi-scale fusion of all three decoder
scales. 445K parameters.

The term that makes the decomposition work is a structure-aware smoothness
penalty, `mean(|∇I| · exp(−λ_g·|∇R|))`. Plain total variation is
structure-blind and smooths across object boundaries, leaving their edges baked
into the illumination map; the exponential weight releases the penalty exactly
where reflectance has a strong edge.

**The four arms** differ only in the Enhance-Net loss, over a shared frozen
Decom-Net, same seed, same 100-epoch budget:

```
L = w_l1·‖R_low·Î − S_normal‖₁ + w_ssim·(1 − SSIM(R_low·Î, S_normal)) + w_sm·L_is(Î, R_low)
```

| arm | w_l1 | w_ssim | w_sm |
|---|:---:|:---:|:---:|
| L1 (paper) | 1.0 | 0 | 3.0 |
| + SSIM | 1.0 | 1.0 | 3.0 |
| + learned | *learned* | *learned* | 3.0 |
| + rebalanced | 1.0 | 0.65 | 0.70 |

**The port is verified, not asserted.** `scripts/verify_port.py` transplants
the authors' TensorFlow weights into these classes and diffs against the
original graph — they agree to **1.4e-05**. That test caught a real bug:
PyTorch's symmetric `padding=1` on a stride-2 conv is not TensorFlow's `SAME`,
and the mismatch produced outputs off by 0.27/1.0 that still looked like
plausible enhanced images. `scripts/check_determinism.py` separately proves
both arms see identical training data.

---

## Probable explanation

**The gain is structural, not per-pixel.** The L1 and SSIM arms end at almost
identical training L1 (0.1116 vs 0.1114) and PSNR moves only +0.43 dB, while
SSIM moves +0.127 and LPIPS −0.177. The SSIM term did not fit pixels better —
it changed how the residual error is *distributed*, and only the metrics that
can see structure register it.

**Learned weights discovered something real, then proved redundant.**
Homoscedastic uncertainty weighting (Kendall et al. 2018) converged to
`w_l1 = 4.28, w_ssim = 2.77`, i.e. `1.00 : 0.65 : 0.70` after normalising.
Two readable claims: SSIM wants *less* weight than L1, and smoothness wants
**0.70, not the reference implementation's 3.0**. But typing those three
numbers as fixed weights matches or beats the learned arm on 5 of 6 metrics.
The method's contribution was finding the weights, not applying them.

On Decom-Net it found nothing at all: with all five terms learned, every weight
scaled **13× uniformly** and the ratios stayed within **3%** of the paper's.
That is an independent validation of Wei et al.'s hand-set constants, and a
negative result for the method — a global scale is absorbed by the learning
rate.

**BM3D helps in-distribution and hurts out-of-distribution.** This was the
opposite of the prediction. Denoising reflectance nearly doubles the L1 arm's
LOL SSIM (0.595 → 0.773), then degrades NIQE on all three cross-datasets — to
*worse than doing nothing*. The mechanism is a metric conflict: σ was tuned by
LPIPS, which rewards aggressive smoothing when a clean reference exists, while
NIQE scores naturalness by natural-scene statistics and penalises missing
high-frequency detail as much as it penalises noise. Selecting a
hyperparameter on one metric family and reporting it on another is a design
error, and it is inherited by every BM3D row here. Tuning σ *by NIQE* — which
needs no ground truth — would settle whether denoising can help out of
distribution at all. Not yet run.

**What not to read into it.** One seed per arm, no error bars. The BM3D σ was
tuned on training images the network had already seen. Cross-dataset counts
(LIME 10 / MEF 79 / DICM 44) differ from the paper's text, so absolute NIQE is
not comparable to papers using other bundles — both arms see identical images,
so the comparison between them holds. And this L1 baseline *beats* the
authors' released checkpoint (18.26 vs 16.79 PSNR), most likely from training
at 96×96 patches rather than the released code's 48×48 default, so the margins
sit on top of a strong baseline.

---

## Repository

Structured so every model plugs into the same pipeline — changing an
architecture does not touch training, evaluation or visualisation code. Each
folder has a README explaining why it exists.

| folder | contents |
|---|---|
| [`configs/`](configs/) | one YAML per run; the arms differ in a couple of lines |
| [`data/`](data/) | dataset loaders and split definitions |
| [`models/`](models/) | architectures, port verification, variations worth trying |
| [`training/`](training/) | trainer, losses, learned weighting, gradient diagnostics |
| [`evaluation/`](evaluation/) | metric wrappers, BM3D denoising, scoring engine |
| [`scripts/`](scripts/) | CLI entry points |

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python scripts/prepare_data.py            # needs the 3 dataset zips in the repo root

python scripts/train.py --config configs/decom.yaml             # stage 1, once
python scripts/train.py --config configs/enhance_l1.yaml        # paper baseline
python scripts/train.py --config configs/enhance_ssim.yaml      # + SSIM term
python scripts/train.py --config configs/enhance_uw.yaml        # + learned weights
python scripts/train.py --config configs/enhance_rebalanced.yaml # + fixed rebalanced

python scripts/evaluate.py --checkpoint runs/enhance_l1/last.pth --tag l1
python scripts/input_baseline.py --tag input
python scripts/results_table.py
```

Optional BM3D on any arm: `--denoise bm3d --denoise-sigma S --denoise-gamma G`,
with `scripts/tune_denoise.py` selecting `S`/`G` on training pairs. Per-module
gradient diagnostics (`--set training.grad_log_every=50`) and optional GroupNorm
(`--set model.norm=group`) are available, both off by default so the published
architecture stays the default path.

---

Retinex-Net: Wei, Wang, Yang & Liu, *Deep Retinex Decomposition for Low-Light
Enhancement*, BMVC 2018. Original TensorFlow implementation:
<https://github.com/weichen582/RetinexNet> (MIT).
