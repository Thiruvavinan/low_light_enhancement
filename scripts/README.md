# `scripts/` — entry points

**Why this folder exists:** every script here is a thin command-line wrapper.
The logic lives in `data/`, `models/`, `training/` and `evaluation/`; these
files parse arguments, wire components together, and write results to disk.
None of them contains a hyperparameter default worth overriding — those are in
`configs/`.

## The pipeline, in order

```bash
python scripts/prepare_data.py                              # unpack archives
python scripts/train.py --config configs/decom.yaml         # stage 1, once
python scripts/train.py --config configs/enhance_l1.yaml    # stage 2, baseline
python scripts/train.py --config configs/enhance_ssim.yaml  # stage 2, variant

python scripts/evaluate.py --checkpoint runs/enhance_l1/last.pth   --tag l1   --save-decomposition
python scripts/evaluate.py --checkpoint runs/enhance_ssim/last.pth --tag ssim
python scripts/input_baseline.py --tag input          # the identity row -- see below
python scripts/results_table.py

python scripts/visualize_predictions.py --benchmark LOL  --tags l1 ssim --n 4
python scripts/visualize_predictions.py --benchmark LIME --tags l1 ssim --n 4
python scripts/plot_history.py --run runs/enhance_l1 --run runs/enhance_ssim --metric l1
```

| script | what it does |
|---|---|
| [`prepare_data.py`](prepare_data.py) | unpack the archives into the layout `data/datasets/` expects |
| [`train.py`](train.py) | train either stage; the stage is whatever the config says |
| [`evaluate.py`](evaluate.py) | score one checkpoint on every benchmark, write `summary.json` |
| [`input_baseline.py`](input_baseline.py) | score the *unenhanced* input through the same pipeline |
| [`tune_denoise.py`](tune_denoise.py) | pick BM3D sigma/gamma on **training** pairs, never on eval |
| [`results_table.py`](results_table.py) | assemble the comparison table from the summaries |
| [`visualize_predictions.py`](visualize_predictions.py) | side-by-side qualitative grids |
| [`plot_history.py`](plot_history.py) | loss curves and per-module gradient norms |
| [`convert_tf_weights.py`](convert_tf_weights.py) | port the authors' TF weights into this implementation |
| [`verify_port.py`](verify_port.py) | **check the port against the original TF graph** |
| [`check_determinism.py`](check_determinism.py) | **check that every configuration sees identical training data** |
| [`verify_reported.py`](verify_reported.py) | **check that every published number is backed by a `summary.json`** |

## `verify_port.py` is the one to notice

"Reimplemented from the paper" is not verifiable by reading code. A
transposed kernel, a concatenated skip connection, or the wrong padding
convention all produce code that looks right and results that are merely
plausible.

`verify_port.py` transplants the authors' released weights into this
implementation and compares the outputs against reference renders from the
original TensorFlow graph. It agrees to within one 8-bit level (the float32
comparison at generation time agreed to 1.4e-05).

```bash
python scripts/verify_port.py                          # passes
python scripts/verify_port.py --padding-mode symmetric # FAILS, by design
```

The second command is the negative control — it shows the test actually
discriminates rather than passing vacuously. It also documents a real bug the
test caught: see `padding_mode` in [`../models/enhance_net.py`](../models/enhance_net.py).

## `check_determinism.py` guards the other half of the control

The comparison assumes the two arms differ only in the loss. That is only
true if they also see the same data in the same order. This script builds the
training DataLoader twice under the same seed and hashes the batches:

```bash
python scripts/check_determinism.py --config configs/enhance_l1.yaml
```

Same seed must give identical batches; a *different* seed must give different
ones — the second check is the control, since a loader ignoring the seed
entirely would pass the first.

It was written after finding that the dataset originally drew crops from
`np.random.default_rng()`, which seeds itself from OS entropy and silently
ignored the configured seed. Every run saw a different crop stream, nothing
crashed, no output looked wrong, and the docs claimed identical data across
arms. Seed handling is either tested or quietly untrue.

## `input_baseline.py` is not optional

It scores the unenhanced input through the identical pipeline. NIQE is a
no-reference metric, so without that row "B beats A" says nothing about
whether either beat doing nothing — and here it turned out that the paper's
own L1 loss scores *worse* NIQE than the untouched input on two of three
cross-datasets. Drop the row and the table reads "both work, one works
better", which is wrong.

## `tune_denoise.py` tunes on TRAINING pairs

Choosing BM3D's sigma on eval15 would make the reported LOL numbers a
best-of-N over the test set, and the cross-dataset numbers would inherit that
choice. It sweeps `our485` instead. Known weakness, stated rather than hidden:
those images were seen during training, so the chosen sigma is not guaranteed
optimal at eval time — but nothing in an evaluation set influenced it.

It selects on LPIPS by default, and that choice turned out to matter: the
resulting sigma improves full-reference metrics and *degrades* no-reference
NIQE on every arm. Selecting a hyperparameter on one metric family and
reporting it on another is a design error; see the root README.

## `verify_reported.py` catches stale numbers

Numbers in prose go stale silently. A model gets re-evaluated, the table is
regenerated, and a sentence three sections away still quotes the old value —
nothing errors, nothing looks wrong, and the document is now wrong.

```bash
python scripts/verify_reported.py     # exit 1 on any problem
```

It checks that every evaluated model appears in the table, that every value in
every `summary.json` appears there, that every metric-shaped number in the
README is either a real measurement or reconstructible from one (the
cross-dataset means and the spread figures are recomputed, so the arithmetic
is verified too), and that no relative link is broken.

It has already caught two real problems: rows added without updating the
layout lists, which rendered under a catch-all heading with misleading
bolding; and it fails correctly on an injected fake value, which is how the
check itself was validated.

## Figures come from saved predictions, not fresh inference

`visualize_predictions.py` reads the PNGs `evaluate.py` already wrote rather
than re-running the models. That guarantees the pictures in the README are the
same tensors the numbers in the table were computed from — regenerating
predictions for figures is how a figure and a metric quietly stop agreeing.

`--n` takes an **evenly-spaced** sample sorted by name, not the best-looking
N, and the same indices for every arm. `--names` picks specific images when a
particular failure is worth showing; the README should say when a figure was
hand-picked.
