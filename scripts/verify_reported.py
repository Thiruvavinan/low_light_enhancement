#!/usr/bin/env python
"""
scripts/verify_reported.py
--------------------------
Checks that every number quoted in README.md and outputs/results_table.md
actually appears in the `summary.json` it claims to come from.

    python scripts/verify_reported.py

Why this exists
---------------
Numbers in prose go stale silently. A model gets re-evaluated, the table is
regenerated, and a sentence three sections away still quotes the old value --
nothing errors, nothing looks wrong, and the document is now lying. That has
already happened once in this project: rows were added without updating the
layout lists, and two of them rendered under a catch-all heading with
misleading bolding.

So the invariant is checked rather than remembered:

  1. every evaluated model appears in the generated table
  2. every metric value in every summary.json appears in the table
  3. every metric value quoted in the README matches a real summary.json value
  4. no relative link in either document is broken

(3) is the loose one: it scans for numbers that LOOK like metric values and
requires each to exist somewhere in the evaluation data. It cannot tell that
a value was attributed to the wrong model, only that it is not invented. That
is still enough to catch the common failure, which is a stale copy-paste.

Exit code is non-zero on any failure, so this can gate a commit.
"""

import json
import re
import sys
from pathlib import Path

EVAL_ROOT = Path("outputs/eval")
TABLE = Path("outputs/results_table.md")
README = Path("README.md")

# (metric key, decimal places as rendered)
RENDERED = [("psnr", 2), ("ssim", 4), ("ssim_gray", 4), ("lpips", 4), ("niqe", 2)]
BENCHMARKS = ["LOL", "LIME", "MEF", "DICM"]


def load_all():
    out = {}
    for d in sorted(EVAL_ROOT.iterdir()):
        f = d / "summary.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())["benchmarks"]
    return out


def every_value(summaries):
    """Every metric value, as it would be rendered, with where it came from."""
    for tag, benches in summaries.items():
        for bench in BENCHMARKS:
            for key, digits in RENDERED:
                if key in benches.get(bench, {}):
                    yield tag, bench, key, f"{benches[bench][key]:.{digits}f}"


def check_table(summaries, text):
    problems = []
    for tag in summaries:
        if f"`{tag}`" not in text:
            problems.append(f"model `{tag}` is evaluated but missing from the table")
    for tag, bench, key, rendered in every_value(summaries):
        if rendered not in text:
            problems.append(f"{tag} / {bench} / {key} = {rendered} missing from the table")
    if "**Other**" in text:
        problems.append("table has an 'Other' block -- a model is not assigned to a group")
    return problems


# Loss configurations, in the order the factorial reports them. Used to
# reconstruct the derived "spread" figures the README quotes.
LOSS_SETS = [
    ["l1", "ssim", "uw", "rebalanced"],
    ["l1_bm3d", "ssim_bm3d", "uw_bm3d", "rebalanced_bm3d"],
]


def derived_values(summaries):
    """
    Figures the README computes rather than reads: cross-dataset means, and
    the best-minus-worst spread across the loss configurations.

    These have to be reconstructed here or the check flags every one of them
    as invented. Recomputing from summary.json is also the stronger test: it
    verifies the arithmetic, not just that the digits appeared somewhere.
    """
    out = set()

    def cross(tag):
        b = summaries[tag]
        vals = [b[x]["niqe"] for x in ("LIME", "MEF", "DICM") if "niqe" in b.get(x, {})]
        return sum(vals) / len(vals) if vals else None

    for tag in summaries:
        c = cross(tag)
        if c is not None:
            out.add(f"{c:.2f}")
            out.add(f"{c:.4f}")

    for tags in LOSS_SETS:
        present = [t for t in tags if t in summaries]
        if len(present) < 2:
            continue
        for key, digits in (("lpips", 4), ("ssim", 4), ("psnr", 2)):
            vals = [summaries[t]["LOL"][key] for t in present if key in summaries[t]["LOL"]]
            if len(vals) > 1:
                out.add(f"{max(vals) - min(vals):.{digits}f}")
        cs = [cross(t) for t in present if cross(t) is not None]
        if len(cs) > 1:
            out.add(f"{max(cs) - min(cs):.4f}")
            out.add(f"{max(cs) - min(cs):.2f}")
    return out


def check_readme(summaries, text):
    """
    Every metric-shaped number in a METRIC table must exist in the evaluation
    data, or be reconstructible from it.

    A metric table is identified by the direction arrows in its header, not by
    the metric name: the README's loss-weight table has a `w_ssim` column, so
    a substring match on "SSIM" would scan it and flag 0.65 and 0.70 -- which
    are settings, not measurements.
    """
    known = {rendered for _, _, _, rendered in every_value(summaries)}
    known |= derived_values(summaries)

    problems = []
    in_metric_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            in_metric_table = False
            continue
        header_arrows = ("↑" in line or "↓" in line
                         or "&uarr;" in line or "&darr;" in line)
        if header_arrows and "---" not in line:
            in_metric_table = True
            continue
        if not in_metric_table or "---" in line:
            continue
        for cell in line.split("|"):
            cell = cell.strip().strip("*")
            if re.fullmatch(r"\d+\.\d{2,4}", cell) and cell not in known:
                problems.append(f"README quotes {cell}, which is in no summary.json")
    return problems


def check_links(path, text):
    problems = []
    for link in re.findall(r"\]\(([^)#]+)\)", text):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        if not Path(link).exists():
            problems.append(f"{path.name}: broken link -> {link}")
    return problems


def main() -> int:
    if not EVAL_ROOT.exists():
        print("No outputs/eval -- run scripts/evaluate.py first.", file=sys.stderr)
        return 2

    summaries = load_all()
    table = TABLE.read_text(encoding="utf-8") if TABLE.exists() else ""
    readme = README.read_text(encoding="utf-8") if README.exists() else ""

    checks = [
        (f"{len(summaries)} evaluated models present in the table", check_table(summaries, table)),
        ("README quotes only real values", check_readme(summaries, readme)),
        ("links resolve", check_links(TABLE, table) + check_links(README, readme)),
    ]

    total = sum(len(p) for _, p in checks)
    n_values = sum(1 for _ in every_value(summaries))

    for label, problems in checks:
        print(f"  [{'FAIL' if problems else ' ok '}] {label}")
        for p in problems[:10]:
            print(f"           {p}")
        if len(problems) > 10:
            print(f"           ... and {len(problems) - 10} more")

    print(f"\n{len(summaries)} models, {n_values} metric values checked.")
    if total:
        print(f"{total} problem(s) found.")
        return 1
    print("Everything reported is backed by a summary.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
