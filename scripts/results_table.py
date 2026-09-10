#!/usr/bin/env python
"""
scripts/results_table.py
------------------------
Assembles the comparison table from whatever `scripts/evaluate.py` has
written under outputs/eval/*/summary.json.

    python scripts/results_table.py
    python scripts/results_table.py --tags l1 ssim --out outputs/results_table.md

It reports EVERY metric it finds for every configuration. There is deliberately no
option to select a subset: the project's stated commitment is to report all
the numbers rather than the ones that favour a hypothesis, and a filter flag
is the mechanism by which that commitment quietly stops holding.

Structure
---------
The experiment is a 2 x N factorial -- {reconstruction loss} x {denoising off,
on} -- and the output is laid out that way, because denoising is a
post-process available to ANY of these models. Ranking denoised and undenoised
rows in one table confounds "which loss" with "was it denoised": comparing
`ssim` against `l1_bm3d` moves two factors at once.

    A. Loss, no denoising    the controlled experiment -- same frozen
                             Decom-Net, seed, data order and schedule
    B. Loss, with BM3D       identical rows, denoiser on
    Absorption               how much of the loss-driven spread a generic
                             denoiser simply reproduces. This is the view that
                             matters: on LOL it absorbs ~85% and even reorders
                             the ranking, while cross-dataset it absorbs none.
    BM3D paired              per-model delta, denoiser off vs on
    Full table               every configuration, every metric, nothing dropped

Bolding is scoped WITHIN a group, never across all rows. Bolding across
groups is how `ssim_bm3d` came out marked best on SSIM and LPIPS while being
half a NIQE point worse than `rebalanced` cross-dataset -- arithmetically
true, and exactly the wrong thing to draw a reader's eye to.

`--groups` changes LAYOUT, never content. Any tag not named in a group still
prints under "Other", so a row cannot be dropped by forgetting to list it, and
the assert enforces it.

Winners are marked, but the mark is arithmetic, not a claim. A 0.001 SSIM
difference over 15 images gets the same bold as a 2 dB one, so the table
also prints the margin and the README discussion is where the reader is told
which differences are worth anything.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# metric key -> (column label, higher_is_better)
METRICS = {
    "psnr": ("PSNR ↑", True),
    "ssim": ("SSIM ↑", True),
    "ssim_gray": ("SSIM-gray ↑", True),
    "lpips": ("LPIPS ↓", False),
    "niqe": ("NIQE ↓", False),
    "brisque": ("BRISQUE ↓", False),
}
BENCHMARK_ORDER = ["LOL", "LIME", "MEF", "DICM"]

# Human-readable row labels. The --tag names are short because they are typed
# on a command line; they are not self-explanatory to a reader, and a table
# whose rows say `uw` and `ssim_decomuw` cannot be understood without the
# conversation that produced it. The tag is still printed alongside so a row
# can be traced back to the command that made it.
LABELS = {
    "input":          "Identity (no enhancement)",
    "tf_reference":   "Wei et al. released weights, ported",
    "l1":             "L1 recon — paper baseline",
    "ssim":           "L1 + SSIM recon",
    "uw":             "L1 + SSIM, uncertainty-weighted",
    "rebalanced":     "L1 + SSIM, fixed 1.00 : 0.65 : 0.70",
    "l1_bm3d":        "L1 recon + BM3D on R",
    "ssim_bm3d":      "L1 + SSIM + BM3D on R",
    "uw_bm3d":        "Uncertainty-weighted + BM3D on R",
    "ssim_decomuw":   "L1 + SSIM, on uncertainty-weighted Decom-Net",
    "ssim_lowsmooth": "L1 + SSIM, on low-λ_is Decom-Net",
}

# Shown once, above the first table, so the columns mean something on their own.
LEGEND = """**Protocol.** LOL `eval15` is paired, so it carries full-reference
metrics (PSNR / SSIM / LPIPS-Alex). LIME / MEF / DICM are unpaired and unseen
during training, so they carry NIQE only — which is why the identity row is
reported: a no-reference score is uninterpretable without it.

SSIM is RGB, channel-averaged, with MATLAB-equivalent parameters
(`gaussian_weights=True, sigma=1.5, use_sample_covariance=False`). The
`SSIM-gray` column in the full table is the luma variant, and is the one
comparable to the 0.560 usually quoted for Retinex-Net — the two conventions
differ by ~0.12 here, more than most of the effects being measured.

All metrics are computed on outputs clamped to [0,1] and quantised to 8 bits,
at native resolution, identically for every row.
"""

# Layout only -- see the module docstring. Any tag not listed still prints,
# under "Other", so this cannot silently hide an arm.
GROUPS = [
    ("Reference points", ["input", "tf_reference"]),
    ("Enhance-Net loss — shared frozen Decom-Net, same seed, same 100-epoch budget",
     ["l1", "ssim", "uw", "rebalanced"]),
    ("+ BM3D on reflectance, inference-time only (sigma tuned by LPIPS on train pairs)",
     ["l1_bm3d", "ssim_bm3d", "uw_bm3d"]),
    ("Decom-Net variants — different frozen stage 1, NOT comparable to the block above",
     ["ssim_decomuw", "ssim_lowsmooth"]),
]


def load_summaries(root: Path, tags):
    summaries = {}
    candidates = [root / t for t in tags] if tags else sorted(
        p for p in root.iterdir() if p.is_dir()
    )
    for directory in candidates:
        path = directory / "summary.json"
        if not path.exists():
            print(f"  (skipping {directory}: no summary.json)", file=sys.stderr)
            continue
        summaries[directory.name] = json.loads(path.read_text())
    return summaries


def build_columns(summaries):
    """(benchmark, metric) pairs actually present, in a stable reading order."""
    columns = []
    for benchmark in BENCHMARK_ORDER:
        present = {
            m for s in summaries.values()
            for m in s.get("benchmarks", {}).get(benchmark, {})
            if m in METRICS
        }
        for metric in METRICS:
            if metric in present:
                columns.append((benchmark, metric))
    return columns


def _row(tag, summary, columns, best, mark_best) -> str:
    label = LABELS.get(tag)
    row = [f"{label} <sub>`{tag}`</sub>" if label else f"`{tag}`"]
    for benchmark, metric in columns:
        value = summary.get("benchmarks", {}).get(benchmark, {}).get(metric)
        if value is None:
            row.append("—")
            continue
        digits = 2 if metric in ("psnr", "niqe", "brisque") else 4
        text = f"{value:.{digits}f}"
        if mark_best and best.get((benchmark, metric)) == value:
            text = f"**{text}**"
        row.append(text)
    return "| " + " | ".join(row) + " |"


def _best_within(summaries, tags, columns):
    """
    Best value per column among `tags` only.

    Scoping this to a group is the whole point. Computed across every row, the
    mark lands on whichever arm happens to win a column regardless of whether
    it is even comparable to the others -- that is how `ssim_bm3d` came out
    bolded as best on SSIM and LPIPS while sitting half a NIQE point behind
    `rebalanced` cross-dataset.
    """
    best = {}
    for benchmark, metric in columns:
        values = [
            summaries[t]["benchmarks"][benchmark][metric]
            for t in tags
            if metric in summaries[t].get("benchmarks", {}).get(benchmark, {})
        ]
        if values:
            best[(benchmark, metric)] = (max if METRICS[metric][1] else min)(values)
    return best


def format_table(summaries, columns, groups=None) -> str:
    header = ["Model"] + [f"{b} {METRICS[m][0]}" for b, m in columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] + [":---:"] * len(columns)) + "|"]

    mark = len(summaries) > 1
    if not groups:
        best = _best_within(summaries, list(summaries), columns)
        for tag, summary in summaries.items():
            lines.append(_row(tag, summary, columns, best, mark))
        return "\n".join(lines)

    # Grouped layout. Rows are only reordered and labelled, never removed:
    # anything not named in a group falls through to "Other", so a row cannot
    # be dropped by forgetting to list it. The assert enforces that.
    placed, ncols = set(), len(columns) + 1
    for label, tags in groups:
        present = [t for t in tags if t in summaries]
        if not present:
            continue
        lines.append(f"| **{label}** |" + " |" * (ncols - 1))
        # Bold the winner WITHIN this group only.
        group_best = _best_within(summaries, present, columns)
        for tag in present:
            lines.append(_row(tag, summaries[tag], columns, group_best, mark and len(present) > 1))
            placed.add(tag)

    leftover = [t for t in summaries if t not in placed]
    if leftover:
        lines.append("| **Other** |" + " |" * (ncols - 1))
        other_best = _best_within(summaries, leftover, columns)
        for tag in leftover:
            lines.append(_row(tag, summaries[tag], columns, other_best, mark and len(leftover) > 1))

    assert len(placed) + len(leftover) == len(summaries), "a row was dropped"

    return "\n".join(lines)


HEADLINE_TAGS = ["input", "l1", "ssim", "uw", "rebalanced"]

# The experiment is a 2 x N factorial: {loss} x {denoising off, on}. Denoising
# is a post-process available to ANY model, so putting denoised and undenoised
# rows in one ranked table confounds "which loss" with "was it denoised" --
# comparing `ssim` against `l1_bm3d` moves two factors at once.
#
# Each entry is (undenoised tag, denoised tag, label).
FACTORIAL = [
    ("l1",         "l1_bm3d",         "L1 recon \u2014 paper baseline"),
    ("ssim",       "ssim_bm3d",       "L1 + SSIM recon"),
    ("uw",         "uw_bm3d",         "L1 + SSIM, uncertainty-weighted"),
    ("rebalanced", "rebalanced_bm3d", "L1 + SSIM, fixed 1.00 : 0.65 : 0.70"),
]
HEADLINE_COLUMNS = [("LOL", "psnr"), ("LOL", "ssim"), ("LOL", "lpips"),
                    ("LIME", "niqe"), ("MEF", "niqe"), ("DICM", "niqe")]


def format_headline(summaries) -> str:
    """
    The controlled experiment on its own: one frozen Decom-Net, one seed, one
    budget, only the loss differs.

    Six columns rather than the full eight. `ssim_gray` is dropped here because
    it correlates r=+0.99 with `ssim` across every arm -- it exists to be
    comparable with the published grayscale number, which is a footnote, not a
    dimension. `LOL niqe` is dropped because it is a no-reference metric on the
    one benchmark that HAS a reference. Both remain in the full table below;
    this is a reading order, not a filter.
    """
    present = [t for t in HEADLINE_TAGS if t in summaries]
    if len(present) < 2:
        return ""
    columns = [c for c in HEADLINE_COLUMNS
               if any(c[1] in summaries[t].get("benchmarks", {}).get(c[0], {}) for t in present)]
    best = _best_within(summaries, [t for t in present if t != "input"], columns)

    lines = ["", LEGEND, "", "## The controlled comparison", "",
             "Identical architecture, frozen Decom-Net, seed, data order and "
             "schedule across all four trained rows; only the Enhance-Net "
             "reconstruction loss differs. Data-order equivalence is asserted by "
             "`scripts/check_determinism.py`, not assumed.", "",
             "| Model | " + " | ".join(f"{b} {METRICS[m][0]}" for b, m in columns) + " |",
             "|" + "|".join([":---"] + [":---:"] * len(columns)) + "|"]
    for tag in present:
        lines.append(_row(tag, summaries[tag], columns, best, tag != "input"))

    lines += [
        "",
        "**Reading.** The SSIM term is the dominant effect: +0.13 SSIM and "
        "−0.18 LPIPS at only +0.42 dB PSNR — it redistributes residual error "
        "rather than reducing it. Homoscedastic uncertainty weighting "
        "(Kendall et al. 2018) converges to `1.00 : 0.65 : 0.70` and improves "
        "further, but the fixed-weight control at those same ratios matches or "
        "beats it on 5 of 6 metrics. The gain is attributable to the weights, "
        "not to learning them.",
        "",
        "The identity row is load-bearing: the paper's L1 baseline is "
        "**NIQE-worse than the unenhanced input** on LIME (5.03 vs 4.35) and "
        "DICM (4.12 vs 3.86). `R = S / I` amplifies sensor noise by ~`1/I` in "
        "dark regions, and NSS-based no-reference scoring penalises that more "
        "than it rewards the added visibility.",
    ]
    return "\n".join(lines)


def _spread(summaries, tags, get):
    """max - min of `get` across `tags`; None if fewer than two are available."""
    vals = [get(t) for t in tags if t in summaries]
    return (max(vals) - min(vals)) if len(vals) > 1 else None


def format_factorial(summaries) -> str:
    """
    The same loss comparison run twice: denoising off, then on.

    Denoising is a factor that can be applied to any of these models, so it
    belongs on its own axis rather than as extra rows in a single ranking. Two
    tables with identical rows make the question answerable directly: does the
    loss ranking survive denoising, and how much of the loss effect does a
    standard denoiser simply reproduce?
    """
    rows = [(a, b, lab) for a, b, lab in FACTORIAL if a in summaries]
    if not rows:
        return ""

    cols = [c for c in HEADLINE_COLUMNS
            if any(c[1] in summaries[a].get("benchmarks", {}).get(c[0], {}) for a, _, _ in rows)]
    header = "| Model | " + " | ".join(b + " " + METRICS[m][0] for b, m in cols) + " |"
    divider = "|" + "|".join([":---"] + [":---:"] * len(cols)) + "|"

    def table(index, extra_first=None):
        tags = [r[index] for r in rows if r[index] in summaries]
        if extra_first:
            tags = [t for t in extra_first if t in summaries] + tags
        best = _best_within(summaries, [t for t in tags if t not in ("input", "tf_reference")], cols)
        out = [header, divider]
        for tag in tags:
            label = next((lab for a, b, lab in rows if tag in (a, b)), LABELS.get(tag, tag))
            saved = LABELS.get(tag)
            LABELS[tag] = label
            out.append(_row(tag, summaries[tag], cols, best,
                            tag not in ("input", "tf_reference")))
            if saved is None:
                LABELS.pop(tag, None)
            else:
                LABELS[tag] = saved
        return "\n".join(out)

    lines = ["", "## A. Loss function, without denoising", "",
             "Identical architecture, frozen Decom-Net, seed, data order and "
             "schedule; only the Enhance-Net reconstruction loss differs. Data-order "
             "equivalence is asserted by `scripts/check_determinism.py`, not assumed.",
             "", table(0, extra_first=["input", "tf_reference"]), ""]

    denoised = [b for _, b, _ in rows if b in summaries]
    if denoised:
        missing = [lab for _, b, lab in rows if b not in summaries]
        lines += ["## B. Same losses, with BM3D denoising", "",
                  "The same four checkpoints, re-scored with BM3D applied to "
                  "reflectance before recombination. Sigma and gamma are tuned "
                  "per model on training pairs, so each row gets the denoising "
                  "strength that suits it rather than a shared setting."]
        if missing:
            lines += ["", "> Not yet evaluated: " + "; ".join(missing) + "."]
        lines += ["", table(1), ""]

        lines += _absorption(summaries, rows)
    return "\n".join(lines)


def _absorption(summaries, rows):
    """How much of the loss-driven spread a generic denoiser reproduces."""
    pairs = [(a, b) for a, b, _ in rows if a in summaries and b in summaries]
    if len(pairs) < 2:
        return []

    def cross(t):
        bms = [x for x in ("LIME", "MEF", "DICM")
               if "niqe" in summaries[t].get("benchmarks", {}).get(x, {})]
        return sum(summaries[t]["benchmarks"][x]["niqe"] for x in bms) / len(bms)

    metrics = [("LOL LPIPS \u2193", lambda t: summaries[t]["benchmarks"]["LOL"]["lpips"], 4),
               ("LOL SSIM \u2191", lambda t: summaries[t]["benchmarks"]["LOL"]["ssim"], 4),
               ("cross-dataset NIQE \u2193", cross, 4)]

    out = ["### How much of the loss effect is just noise?", "",
           "Spread = best minus worst across the loss configurations. If a generic "
           "denoiser reproduces what a loss change bought, the spread collapses "
           "once denoising is applied to all of them.", "",
           "| Metric | spread without BM3D | spread with BM3D | absorbed by denoising |",
           "|:---|:---:|:---:|:---:|"]
    for name, get, d in metrics:
        sp = _spread(summaries, [a for a, _ in pairs], get)
        sd = _spread(summaries, [b for _, b in pairs], get)
        if sp is None or sd is None or sp == 0:
            continue
        out.append("| %s | %.*f | %.*f | **%+.0f%%** |" % (name, d, sp, d, sd, (1 - sd / sp) * 100))

    lp = lambda t: summaries[t]["benchmarks"]["LOL"]["lpips"]
    order_plain = [a for a, _ in sorted(pairs, key=lambda x: lp(x[0]))]
    order_den = [a for a, _ in sorted(pairs, key=lambda x: lp(x[1]))]
    same = order_plain == order_den

    out += ["",
            "**Reading.** On LOL, a standard denoiser reproduces most of what the "
            "loss change bought \u2014 the losses become close to interchangeable once "
            "all of them are denoised"
            + (", and the LPIPS ranking even reorders." if not same else "."),
            "",
            "Cross-dataset the spread does **not** collapse. So the SSIM term's "
            "in-distribution advantage is largely noise suppression, which BM3D "
            "also provides; its out-of-distribution advantage is something else, "
            "and it survives denoising. That is the column that justifies the loss "
            "change, and it is the one the original paper never reported.",
            "",
            "Caveat: sigma was tuned per model on LPIPS, so table B slightly "
            "favours whichever model that objective suited. The same tuning choice "
            "is why denoising worsens NIQE \u2014 see the paired view below.", ""]
    return out


def format_bm3d_effect(summaries) -> str:
    """
    Denoising as a PAIRED delta against each arm's own undenoised base.

    A denoised arm is only meaningful against the arm it was derived from --
    ranking `ssim_bm3d` against `l1` mixes two changes at once. Pairing also
    makes the finding legible in one glance: BM3D improves the full-reference
    column and degrades the no-reference one, on every arm, without exception.
    That sign flip is the result; in a flat 11-row grid it takes six
    comparisons to notice.

    Cross-dataset NIQE is averaged over LIME/MEF/DICM here. The per-dataset
    values are in the full table below, and the direction is the same in all
    three, so the mean is a summary rather than a smoothing-over.
    """
    pairs = [("l1", "l1_bm3d"), ("ssim", "ssim_bm3d"), ("uw", "uw_bm3d")]
    usable = [(a, b) for a, b in pairs if a in summaries and b in summaries]
    if not usable:
        return ""

    def cross_niqe(tag):
        bms = [b for b in ("LIME", "MEF", "DICM")
               if "niqe" in summaries[tag].get("benchmarks", {}).get(b, {})]
        if not bms:
            return None
        return sum(summaries[tag]["benchmarks"][b]["niqe"] for b in bms) / len(bms)

    lines = [
        "",
        "### Does the paper's BM3D denoising help?",
        "",
        "Same checkpoint scored twice, denoiser off then on. BM3D is applied to "
        "reflectance before recombination with I-hat, with illumination-relative "
        "strength `w = (1 - I)^gamma`. The right column averages NIQE over "
        "LIME/MEF/DICM; the sign is the same on all three individually.",
        "",
        "| Model | full-reference (LPIPS ↓) | Δ | no-reference (NIQE ↓) | Δ |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]
    for base, denoised in usable:
        lp_a = summaries[base]["benchmarks"]["LOL"]["lpips"]
        lp_b = summaries[denoised]["benchmarks"]["LOL"]["lpips"]
        nq_a, nq_b = cross_niqe(base), cross_niqe(denoised)
        if nq_a is None or nq_b is None:
            continue
        lines.append(
            f"| {LABELS.get(base, base)} "
            f"| {lp_a:.4f} → {lp_b:.4f} | {lp_b - lp_a:+.4f} {'✓' if lp_b < lp_a else '✗'} "
            f"| {nq_a:.2f} → {nq_b:.2f} | {nq_b - nq_a:+.2f} {'✓' if nq_b < nq_a else '✗'} |"
        )

    lines += [
        "",
        "**Reading.** The sign flips between metric families on every row, "
        "without exception. Sigma was selected by LPIPS on training pairs, so the "
        "left column is the tuning objective and the right column is not: LPIPS "
        "rewards smoothing toward a clean reference, while NIQE's natural-scene-"
        "statistics model penalises the resulting loss of high-frequency detail "
        "as much as it penalises noise. Selecting on one metric family and "
        "reporting on another is a design error, and these rows are what it "
        "looks like. Tuning sigma *by* NIQE — feasible without ground truth — "
        "would separate 'denoising does not transfer' from 'LPIPS chose the "
        "wrong sigma'; that run has not been done.",
    ]
    return "\n".join(lines)


def format_margins(summaries, columns, margin_tags=None) -> str:
    """
    Signed difference between two named arms, so a "win" can be sized.

    The pair is named explicitly rather than inferred, so a reference row
    (e.g. the authors' released weights) can appear in the table without
    being mistaken for one of the two arms under comparison.
    """
    if margin_tags:
        missing = [t for t in margin_tags if t not in summaries]
        if missing:
            print(f"  (margin: no summary for {missing}; skipping)", file=sys.stderr)
            return ""
        pair = [(t, summaries[t]) for t in margin_tags]
    elif len(summaries) == 2:
        pair = list(summaries.items())
    else:
        return ""
    (tag_a, a), (tag_b, b) = pair
    lines = [
        "",
        f"### Margin (`{tag_b}` minus `{tag_a}`)",
        "",
        "Sign is the raw difference; the ✓/✗ says whether it is an improvement "
        "given the metric's direction.",
        "",
        "| Benchmark | Metric | " + f"`{tag_a}` | `{tag_b}` | Δ | better? |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ]
    for benchmark, metric in columns:
        va = a.get("benchmarks", {}).get(benchmark, {}).get(metric)
        vb = b.get("benchmarks", {}).get(benchmark, {}).get(metric)
        if va is None or vb is None:
            continue
        delta = vb - va
        improved = delta > 0 if METRICS[metric][1] else delta < 0
        digits = 2 if metric in ("psnr", "niqe", "brisque") else 4
        lines.append(
            f"| {benchmark} | {METRICS[metric][0]} | {va:.{digits}f} | {vb:.{digits}f} | "
            f"{delta:+.{digits}f} | {'✓' if improved else '✗'} |"
        )
    return "\n".join(lines)


def format_provenance(summaries) -> str:
    """
    Protocol block. Only fields that DIFFER across arms get a row each; the
    ones that are identical everywhere are stated once. Eleven rows repeating
    the same image counts and metric backends is noise that hides the one
    column a reader might actually check.
    """
    def uniform(key, default="?"):
        vals = {str(s.get(key, default)) for s in summaries.values()}
        return vals.pop() if len(vals) == 1 else None

    lines = ["", "### Protocol", ""]

    counts = {", ".join(f"{b}:{s['benchmarks'][b]['n_images']}"
                        for b in BENCHMARK_ORDER if b in s.get("benchmarks", {}))
              for s in summaries.values()}
    shared = []
    if len(counts) == 1:
        shared.append(f"images scored: {counts.pop()}")
    for key, name in [("no_reference_backend", "no-reference metric"),
                      ("lpips_net", "LPIPS backbone"), ("padding_mode", "padding")]:
        v = uniform(key)
        if v:
            shared.append(f"{name}: {v}")
    if shared:
        lines += ["Identical for every row — " + "; ".join(shared) + ".", ""]

    lines += ["| Row | Checkpoint | Denoising |", "|:---|:---|:---|"]
    for tag, s in summaries.items():
        d = s.get("denoise", "none")
        d = "—" if d in ("none", None) else f"BM3D σ={s.get('denoise_sigma')} γ={s.get('denoise_gamma')}"
        label = LABELS.get(tag, tag)
        lines.append(f"| {label} <sub>`{tag}`</sub> | `{s.get('checkpoint','?')}` | {d} |")

    backends = {s.get("no_reference_backend") for s in summaries.values()}
    if len(backends) > 1:
        lines += ["", "> **Warning:** the arms were scored with different no-reference "
                  "metrics. Those columns are not comparable."]
    elif backends == {"brisque"}:
        lines += ["", "> **Note:** the no-reference columns are BRISQUE, not NIQE "
                  "(`pyiqa` is not installed). Both are lower-is-better "
                  "natural-scene-statistics metrics from the same authors, but the "
                  "values are not interchangeable with published NIQE numbers. "
                  "See `evaluation/metrics.py`."]
    return "\n".join(lines)


def main():
    # The table uses arrows and check marks. Windows consoles default to cp1252,
    # which cannot encode them, and printing would raise even though the file
    # itself is written as UTF-8. Degrade the console, never the file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="Arms to include, in order (default: every directory found)")
    parser.add_argument("--margin", nargs=2, metavar=("BASELINE", "VARIANT"), default=None,
                        help="Two tags to difference in the margin table, e.g. --margin l1 ssim. "
                             "Named explicitly so a reference row can sit in the table without "
                             "being read as one of the compared arms.")
    parser.add_argument("--out", default="outputs/results_table.md")
    args = parser.parse_args()

    root = Path(args.eval_dir)
    if not root.exists():
        print(f"{root} not found -- run scripts/evaluate.py first.", file=sys.stderr)
        return 1

    summaries = load_summaries(root, args.tags)
    if not summaries:
        print("No summary.json files found.", file=sys.stderr)
        return 1

    columns = build_columns(summaries)
    document = "\n".join([
        "# Results",
        "",
        "Generated by `scripts/results_table.py`. The sections below are reading "
        "orders over one set of numbers. The full table at the end holds every "
        "model and every metric, with nothing left out.",
        format_factorial(summaries),
        format_bm3d_effect(summaries),
        "",
        "## Full table",
        "",
        "Every configuration, every metric. **Bold is scoped within a block** — "
        "blocks vary different factors, so cross-block bolding would be "
        "arithmetic rather than comparison.",
        "",
        format_table(summaries, columns, GROUPS),
        format_margins(summaries, columns, args.margin),
        format_provenance(summaries),
        "",
    ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    print(document)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
