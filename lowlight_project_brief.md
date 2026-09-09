# Low-Light Enhancement — Retinex-Net with SSIM-Augmented Loss

## Project Goal

Reimplement Retinex-Net (Wei et al., BMVC 2018, "Deep Retinex Decomposition for
Low-Light Enhancement") from the paper specification, train a baseline (paper's
original L1 loss) and a variant (L1 + SSIM-augmented reconstruction loss), and
run a quantitative comparison — both in-distribution and cross-dataset — that
the original paper does not provide.

**The gap this fills:** the paper evaluates cross-dataset generalisation (LIME,
MEF, DICM datasets) only via visual side-by-side comparison — no metrics
reported. This project adds quantitative cross-dataset evaluation, and tests
whether a structurally-aware loss term improves generalisation beyond
in-distribution accuracy.

This is a personal portfolio project. Priorities in order: (1) correct,
faithful reimplementation of the paper's architecture and baseline loss,
(2) a clean, well-motivated SSIM loss variant, (3) rigorous, reproducible
evaluation, (4) a clear README documenting the hypothesis and honest results
— including if the result is mixed or negative.

---

## Architecture (from the paper — implement exactly as specified)

### Decom-Net
- Input: RGB image, low-light and normal-light passed through the SAME
  network (shared weights), each independently
- 5 convolutional layers total:
  - 1st conv layer: extract features from RGB input (3×3 kernel)
  - Several 3×3 conv layers with ReLU activation
  - Final 3×3 conv layer projects features into 4 channels
  - Split into R (3-channel reflectance) and I (1-channel illumination)
  - Sigmoid activation on both R and I to bound outputs to [0, 1]
- Note: the paper's Section 4.1 states "5 convolutional layers with a ReLU
  activation between 2 conv-layers without ReLU" — the first and last conv
  layers have no ReLU; the middle layers do.

### Enhance-Net
- Encoder-decoder architecture, 3 down-sampling blocks + 3 up-sampling blocks
- Down-sampling block: conv layer with stride 2, followed by ReLU
- Up-sampling block: "resize-convolution" — nearest-neighbor interpolation,
  then a stride-1 conv layer, then ReLU (NOT transposed convolution — the
  paper explicitly avoids this to prevent checkerboard artifacts, citing
  Odena et al. 2016)
- Skip connections: element-wise SUMMATION from each down-sampling block to
  its mirrored up-sampling block (this differs from U-Net's concatenation —
  confirm this is implemented as addition, not concat)
- Multi-scale concatenation: resize all M up-sampling block outputs to the
  final scale via nearest-neighbor interpolation, concatenate along channels
  (C × M total channels), reduce via 1×1 conv to C channels, then a final
  3×3 conv reconstructs the illumination map Î
- Input to Enhance-Net: the low-light illumination map I_low (optionally
  concatenated with R_low — check paper Figure 1 for exact input)

### Denoising (optional — lower priority, implement only if time permits)
- BM3D applied to reflectance R_low before final reconstruction
- Illumination-relative strategy: since noise is amplified more in darker
  regions, denoising strength should be related to illumination intensity
- This is a classical (non-learned) post-processing step, not a network

---

## Loss Functions

### Baseline (paper's original, Decom-Net)

```
L_decom = L_recon + λ_ir · L_ir + λ_is · L_is
```

**Reconstruction loss** (L1, cross-reconstruction across low/normal pairs):
```
L_recon = Σ_{i∈{low,normal}} Σ_{j∈{low,normal}} λ_ij · ||R_i ∘ I_j − S_j||_1
```
- λ_ij = 1 when i == j (correct pairing), λ_ij = 0.001 when i ≠ j
  (cross pairing — small weight, per paper Section 4.1)
- ∘ denotes element-wise multiplication

**Invariable reflectance loss:**
```
L_ir = ||R_low − R_normal||_1
```

**Illumination smoothness loss** (structure-aware, weighted by reflectance
gradient):
```
L_is = Σ_{i∈{low,normal}} || ∇I_i ∘ exp(−λ_g · ∇R_i) ||
```
- ∇ includes both horizontal and vertical gradients
- This is the KEY architectural detail: standard total-variation loss is
  "structure-blind" — it blurs illumination uniformly. Weighting by
  exp(−λ_g·∇R) relaxes the smoothness constraint exactly where reflectance
  gradients are steep (i.e., where real image structure is), so illumination
  can be discontinuous there without penalty.

**Hyperparameters from paper:** λ_ir = 0.001, λ_is = 0.1, λ_g = 10

### Enhance-Net loss (baseline)
```
L_enhance = L_recon_enhance + L_is
```
```
L_recon_enhance = ||R_low ∘ Î − S_normal||_1
```
- L_is here is the same smoothness loss, but applied to Î weighted by
  ∇R_low (not R_normal)

### SSIM Variant — the experiment

Replace or augment the Enhance-Net's reconstruction loss:

```
L_recon_enhance_ssim = λ_l1 · ||R_low ∘ Î − S_normal||_1
                       + λ_ssim · (1 − SSIM(R_low ∘ Î, S_normal))
```

- Start with λ_l1 = 1.0, λ_ssim = 1.0 as a first pass; treat as a tunable
  hyperparameter and note in the README what was tried
- Use a standard differentiable SSIM implementation (e.g. `pytorch-msssim`
  or `kornia.losses.ssim_loss`) — do not hand-roll SSIM math, use a
  well-tested library implementation
- Everything else (Decom-Net, Decom-Net's own loss, Enhance-Net architecture)
  stays IDENTICAL between baseline and SSIM variant — this must be a
  controlled comparison where only the reconstruction loss term changes

**Rationale to document in README:** L1 reconstruction loss penalizes
per-pixel intensity error uniformly and has no notion of structural or
perceptual similarity. SSIM measures luminance, contrast, and structural
similarity jointly, and correlates better with human-perceived quality. The
paper's own comparisons against other methods (Fig. 6) are evaluated
visually, not with a perceptual metric — this motivates testing whether an
explicitly structure-aware loss term improves outcomes, particularly on
out-of-distribution generalization.

---

## Dataset

**Training / in-distribution eval: LOL dataset**
- 500 low/normal-light image pairs total
- Split: 485 pairs training, 15 pairs evaluation (paper's exact split —
  use the same split if the dataset provides a canonical one, otherwise
  document how the 15-image eval split was chosen)
- Source: search for "LOL dataset low-light enhancement" — available on
  Kaggle, HuggingFace, or the original authors' release
- Training patch size: 96×96 (random crops from full images)
- Batch size: 16 (paper's setting; adjust downward if GPU memory is
  constrained, and document the change)

**Cross-dataset evaluation (no ground truth, quantitative via no-reference
metrics only):**
- LIME dataset (10 test images)
- MEF dataset (17 image sequences, multiple exposure levels)
- DICM dataset (69 images)
- These are the same three datasets the original paper uses for its
  (purely visual) generalization comparison in Section 4.3 — search for
  each by name, they are small public datasets commonly bundled together
  in low-light enhancement benchmark repos
- If any of these three is difficult to source, substitute with another
  small, standard, publicly available unpaired low-light test set and
  document the substitution clearly in the README

---

## Evaluation Metrics

### In-distribution (LOL eval set, 15 images — paired, has ground truth)
- **SSIM** — structural similarity vs. ground truth normal-light image
- **PSNR** — peak signal-to-noise ratio vs. ground truth
- **LPIPS** — learned perceptual similarity (use the standard `lpips`
  Python package, AlexNet or VGG backbone — either is fine, just state
  which)

### Cross-dataset (LIME, MEF, DICM — unpaired, no ground truth)
- **NIQE** — Natural Image Quality Evaluator, no-reference metric
  (use `piq` or `niqe` Python package — do not reimplement from scratch)

### Final comparison table (this is the core deliverable)

| Model              | SSIM ↑ (LOL) | PSNR ↑ (LOL) | LPIPS ↓ (LOL) | NIQE ↓ (LIME) | NIQE ↓ (MEF) | NIQE ↓ (DICM) |
|---------------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Retinex-Net (L1)     |     |     |     |     |     |     |
| Retinex-Net (+SSIM)  |     |     |     |     |     |     |

Report all six numbers for both models. Do not cherry-pick metrics where
the SSIM variant wins — report everything, and let the README's discussion
section interpret the pattern honestly, including if results are mixed.

---

## Repository Structure

```
lowlight-retinex/
├── README.md                  # hypothesis, method, results, honest discussion
├── data/
│   ├── lol/                   # LOL dataset (train/eval split)
│   └── cross_eval/            # LIME, MEF, DICM
├── models/
│   ├── decom_net.py
│   └── enhance_net.py
├── losses.py                  # all loss terms, both baseline and SSIM variant
├── train_decom.py
├── train_enhance.py           # supports --loss={l1,ssim} flag to switch variant
├── eval.py                    # computes all metrics, produces the table above
├── configs/
│   ├── baseline.yaml
│   └── ssim_variant.yaml
├── checkpoints/                # trained weights, both variants
└── outputs/
    ├── qualitative/           # before/after image grids for both variants
    └── results_table.md       # the final metrics table
```

---

## README Requirements

The README must include, at minimum:

1. **One-paragraph summary** — what this is, what was tested, what was found
2. **Architecture summary** — brief description with reference to the paper
3. **The hypothesis** — why SSIM loss might help, stated before showing results
4. **Method** — training setup, hyperparameters, what differs between the
   two variants (should be ONLY the reconstruction loss term)
5. **Results table** — the six-metric comparison above
6. **Qualitative comparison** — a few side-by-side image grids (input / L1
   output / SSIM output / ground truth where available)
7. **Discussion** — honest interpretation. If SSIM variant wins on some
   metrics and not others, say so and reason about why. If it's a clear
   win or a clear loss, say that too. Do not oversell an ambiguous result.
8. **Limitations** — what wasn't tested, what would be done with more time
   (e.g. hyperparameter sweep on λ_ssim, denoising step, testing on real
   industrial/surveillance footage as the original motivating use case)

---

## What NOT to Do

- Do not implement denoising (BM3D) unless the core comparison is done with
  time remaining — it's in the paper but is a secondary contribution
- Do not add architectural changes beyond the loss function — this must
  stay a controlled comparison (same Decom-Net, same Enhance-Net
  architecture, same hyperparameters except the loss term and its weights)
- Do not skip the cross-dataset evaluation — it is the actual novel
  contribution over the paper, not an optional extra
- Do not hand-roll SSIM, LPIPS, or NIQE implementations — use established
  library implementations so numbers are trustworthy and comparable
- Do not report only metrics that favor the SSIM variant

---

## Timeline (target: 2–3 days)

**Day 1:** Repo structure, data pipeline for LOL, Decom-Net + Enhance-Net
implemented, baseline (L1) training running end-to-end, at least one
checkpoint saved.

**Day 2:** SSIM loss variant implemented behind the `--loss` flag, second
training run complete, both checkpoints producing visibly reasonable
enhanced outputs.

**Day 3:** Cross-dataset images sourced, full evaluation script run for
both variants across all metrics and all datasets, results table populated,
qualitative grids generated, README written.
