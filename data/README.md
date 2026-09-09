# `data/` — datasets and their loaders

**Why this folder exists:** to own everything about getting pixels into a
tensor — download layout, splits, cropping, augmentation — so that no model,
loss or metric ever has to know where an image came from.

```
data/
├── datasets/          the loaders (this is the code)
│   ├── base.py        image IO, augmentation, shared conventions
│   ├── lol.py         LOL — paired, has ground truth
│   └── unpaired.py    LIME / MEF / DICM — no ground truth
├── lol/               populated by scripts/prepare_data.py (gitignored)
└── cross_eval/        populated by scripts/prepare_data.py (gitignored)
```

## Setup

Put these three archives in the repo root and run one command:

| archive | contents | source |
|---|---|---|
| `LOLdataset.zip` | `our485/` + `eval15/` | [project page](https://daooshee.github.io/BMVC2018website/) |
| `BrighteningTrain.zip` | 1000 synthetic pairs | same |
| `Test.zip` | LIME, MEF, DICM (and NPE, VV, Fusion) | same |

```bash
python scripts/prepare_data.py
```

Idempotent; `--force` re-extracts. It also drops the macOS `__MACOSX/`
resource forks and `.DS_Store` entries these archives carry, which are not
images and which PIL refuses to open.

## What ends up where

| path | images | role |
|---|---:|---|
| `lol/our485/{low,high}` | 485 | training (real) |
| `lol/eval15/{low,high}` | 15 | **in-distribution evaluation** |
| `lol/syn/{low,high}` | 1000 | training (synthetic) |
| `cross_eval/LIME` | 10 | cross-dataset evaluation |
| `cross_eval/MEF` | 79 | cross-dataset evaluation |
| `cross_eval/DICM` | 44 | cross-dataset evaluation |

**The split is the authors', not invented here.** LOL ships pre-split as
`our485/` and `eval15/`; that is the paper's 485/15 split, so there was no
choice to make and no selection to document. `eval15` is never sampled during
training.

**Training uses `our485` + `syn`**, which is what the paper does (Section 4.1)
— 1485 pairs. Real-only is a one-line ablation: drop `syn` from
`dataset.splits` in the config.

**Cross-dataset counts differ from the paper's text.** The paper cites LIME
(10), MEF (17 sequences) and DICM (69). The commonly-redistributed `Test.zip`
bundle contains LIME 10, MEF 79 individual images, and DICM 44. Those are the
counts actually evaluated and they are recorded in every `summary.json`, so
the numbers here are reproducible — but the DICM and MEF columns are **not**
directly comparable to a paper reporting a differently-sized version of those
sets. Both model arms see exactly the same images, so the comparison between
them is unaffected.

## Conventions every loader obeys

- **`float32` CHW in `[0, 1]`. No mean/std normalisation, ever.** This is a
  hard requirement, not a default. Retinex-Net's formulation is `S = R · I`
  with both factors sigmoid-bounded to `[0,1]`, and the reconstruction loss
  compares `R · I` directly against the input image. Standardising the input
  would break that identity.
- **Every sample carries a `name`** (the file stem), so predictions can be
  written back out under the same filename and lined up across arms.
- **Paired datasets return `low` and `high`; unpaired ones return `low` only.**
  Losses and metrics branch on whether `high` is present, never on a dataset
  class name — which is why adding a benchmark touches no other folder.
- **Augmentation is applied identically to both images of a pair.** The eight
  dihedral transforms, applied *after* cropping so rotations act on a square.
  Desynchronising the pair would destroy the pixel correspondence the
  reconstruction loss depends on.
- **Training crops, evaluation does not.** 96×96 random crops for training;
  full native resolution for every metric, because resizing changes the
  statistics that no-reference metrics measure.
