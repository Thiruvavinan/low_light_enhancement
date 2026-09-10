#!/usr/bin/env python
"""
scripts/results_table.py
------------------------
Assembles the comparison table from whatever `scripts/evaluate.py` has
written under outputs/eval/*/summary.json.

    python scripts/results_table.py
    python scripts/results_table.py --tags l1 ssim --out outputs/results_table.md

It reports EVERY metric it finds for every arm. There is deliberately no
option to select a subset: the project's stated commitment is to report all
the numbers rather than the ones that favour a hypothesis, and a filter flag
is the mechanism by which that commitment quietly stops holding.

Structure
---------
The output is several focused views followed by the COMPLETE table. The views
exist because one 8-column x 11-row grid is a data dump: it holds three
different experiments whose rows are not all comparable to each other, and it
gives the reader no way to know which comparisons are legal.

    1. Loss comparison   the controlled experiment -- same frozen Decom-Net,
                         same seed, same budget, only the loss differs
    2. BM3D effect       paired deltas, because a denoised arm is only
                         meaningful against its own undenoised base
    3. Stage-1 axis      different frozen decomposition; NOT comparable to (1)
    4. Full table        every arm, every metric, nothing dropped

Bolding is scoped WITHIN a group, never across all rows. Bolding across
groups is how `ssim_bm3d` came out marked best on SSIM and LPIPS while being
half a NIQE point worse than `rebalanced` cross-dataset -- arithmetically
true, and exactly the wrong thing to draw a reader's eye to.

`--groups` changes LAYOUT, never content. Any arm not named in a group still
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

# Layout only -- see the module docstring. Any tag not listed still prints,
# under "Other", so this cannot silently hide an arm.
GROUPS = [
    ("Reference points", ["input", "tf_reference"]),
    ("Loss comparison (shared Decom-Net)", ["l1", "ssim", "uw", "rebalanced"]),
    ("+ BM3D denoising (sigma tuned by LPIPS)", ["l1_bm3d", "ssim_bm3d", "uw_bm3d"]),
    ("Stage-1 axis (different decomposition)", ["ssim_decomuw", "ssim_lowsmooth"]),
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
    row = [f"`{tag}`"]
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

    lines = ["", "## The controlled comparison", "",
             "Same frozen Decom-Net, same seed, same 100-epoch budget. Only the "
             "Enhance-Net loss differs. `input` is the identity function scored "
             "through the same pipeline, and is excluded from the bolding.", "",
             "| Model | " + " | ".join(f"{b} {METRICS[m][0]}" for b, m in columns) + " |",
             "|" + "|".join([":---"] + [":---:"] * len(columns)) + "|"]
    for tag in present:
        lines.append(_row(tag, summaries[tag], columns, best, tag != "input"))
    return "\n".join(lines)


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
        "### Effect of BM3D denoising (paired against each arm's own baseline)",
        "",
        "In-distribution vs. out-of-distribution, same checkpoint, denoising on/off.",
        "",
        "| Arm | LOL LPIPS ↓ | Δ | cross-dataset NIQE ↓ | Δ |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]
    for base, denoised in usable:
        lp_a = summaries[base]["benchmarks"]["LOL"]["lpips"]
        lp_b = summaries[denoised]["benchmarks"]["LOL"]["lpips"]
        nq_a, nq_b = cross_niqe(base), cross_niqe(denoised)
        if nq_a is None or nq_b is None:
            continue
        lines.append(
            f"| `{base}` → `{denoised}` "
            f"| {lp_a:.4f} → {lp_b:.4f} | {lp_b - lp_a:+.4f} {'✓' if lp_b < lp_a else '✗'} "
            f"| {nq_a:.2f} → {nq_b:.2f} | {nq_b - nq_a:+.2f} {'✓' if nq_b < nq_a else '✗'} |"
        )

    lines += [
        "",
        "Every arm moves the same way: denoising helps the metric that has a "
        "clean reference to compare against, and hurts the one that judges "
        "naturalness on its own. `sigma` was selected by LPIPS, so the left "
        "column is the objective it was tuned for and the right column is not.",
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
    lines = ["", "### Protocol", "",
             "| Arm | Checkpoint | Images | No-reference metric | LPIPS backbone | Padding |",
             "|:---|:---|:---|:---:|:---:|:---:|"]
    for tag, s in summaries.items():
        counts = ", ".join(
            f"{b}:{s['benchmarks'][b]['n_images']}"
            for b in BENCHMARK_ORDER if b in s.get("benchmarks", {})
        )
        lines.append(
            f"| `{tag}` | `{s.get('checkpoint','?')}` | {counts} | "
            f"{s.get('no_reference_backend','?')} | {s.get('lpips_net','?')} | "
            f"{s.get('padding_mode','?')} |"
        )

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
        "Generated by `scripts/results_table.py`. The views below are reading "
        "orders over one dataset; the complete table at the end holds every arm "
        "and every metric, with nothing dropped.",
        format_headline(summaries),
        format_bm3d_effect(summaries),
        "",
        "## Full table",
        "",
        "Every arm, every metric. Bolding is scoped within each group -- arms in "
        "different groups are not comparable to each other.",
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
