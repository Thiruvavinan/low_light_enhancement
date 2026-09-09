#!/usr/bin/env python
"""
scripts/visualize_predictions.py
--------------------------------
Side-by-side image grids for the qualitative comparison.

    # LOL: input | L1 | SSIM | ground truth
    python scripts/visualize_predictions.py --benchmark LOL --tags l1 ssim --n 4

    # cross-dataset (no ground truth column exists): input | L1 | SSIM
    python scripts/visualize_predictions.py --benchmark LIME --tags l1 ssim --n 4

    # what the decomposition itself learned: input | R | I | I_delta | output
    python scripts/visualize_predictions.py --benchmark LOL --tags l1 --decomposition

Reads the PNGs `scripts/evaluate.py` already wrote, rather than re-running
the models. That guarantees the pictures in the README are the same tensors
the numbers in the table were computed from -- regenerating predictions for
the figures is how a figure and a metric quietly stop agreeing.

Image selection
---------------
`--n` takes an evenly-spaced sample across the benchmark, sorted by name, so
the same indices are picked every time and for every arm. It is not the
best-looking N. `--names` picks specific images when a particular failure is
worth showing; the README should say when a figure was hand-picked.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BENCHMARK_SOURCES = {
    "LOL": ("data/lol/eval15/low", "data/lol/eval15/high"),
    "LIME": ("data/cross_eval/LIME", None),
    "MEF": ("data/cross_eval/MEF", None),
    "DICM": ("data/cross_eval/DICM", None),
}


def find_source(directory: Path, stem: str):
    """Source images are .png/.bmp/.jpg depending on the benchmark."""
    for candidate in sorted(directory.glob(f"{stem}.*")):
        return candidate
    return None


def pick_names(eval_dir: Path, benchmark: str, n: int, names):
    available = sorted(p.stem for p in (eval_dir / "images" / benchmark).glob("*.png"))
    if names:
        missing = [x for x in names if x not in available]
        if missing:
            raise SystemExit(f"No prediction for {missing} in {eval_dir / 'images' / benchmark}")
        return names
    if n >= len(available):
        return available
    step = len(available) / n
    return [available[int(i * step)] for i in range(n)]


def build_rows(names, benchmark, tags, eval_root: Path, decomposition: bool):
    """One row per image: list of (column title, PIL image)."""
    low_dir, high_dir = BENCHMARK_SOURCES[benchmark]
    rows = []

    for stem in names:
        columns = []
        source = find_source(Path(low_dir), stem)
        if source is not None:
            columns.append(("input (low-light)", Image.open(source).convert("RGB")))

        if decomposition:
            decom_dir = eval_root / tags[0] / "images" / benchmark / "decomposition"
            for suffix, title in [("R", "reflectance R"), ("I", "illumination I"),
                                  ("I_delta", "enhanced illum. Î")]:
                path = decom_dir / f"{stem}_{suffix}.png"
                if path.exists():
                    columns.append((title, Image.open(path).convert("RGB")))

        for tag in tags:
            path = eval_root / tag / "images" / benchmark / f"{stem}.png"
            if path.exists():
                columns.append((f"{tag}", Image.open(path).convert("RGB")))

        if high_dir:
            truth = find_source(Path(high_dir), stem)
            if truth is not None:
                columns.append(("ground truth", Image.open(truth).convert("RGB")))

        rows.append((stem, columns))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="LOL", choices=sorted(BENCHMARK_SOURCES))
    parser.add_argument("--tags", nargs="+", required=True,
                        help="Arms to show, in column order (must match evaluate.py --tag)")
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--n", type=int, default=4, help="Evenly-spaced sample size")
    parser.add_argument("--names", nargs="*", default=None, help="Specific image stems")
    parser.add_argument("--decomposition", action="store_true",
                        help="Insert R / I / Î columns (needs evaluate.py --save-decomposition)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    eval_root = Path(args.eval_dir)
    reference_dir = eval_root / args.tags[0]
    if not reference_dir.exists():
        raise SystemExit(f"{reference_dir} not found -- run scripts/evaluate.py --tag {args.tags[0]} first.")

    names = pick_names(reference_dir, args.benchmark, args.n, args.names)
    rows = build_rows(names, args.benchmark, args.tags, eval_root, args.decomposition)
    if not rows or not rows[0][1]:
        raise SystemExit("Nothing to plot -- no matching predictions found.")

    n_cols = max(len(cols) for _, cols in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.1 * n_cols, 2.4 * len(rows)),
                             squeeze=False)

    for r, (stem, columns) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_axis_off()
            if c < len(columns):
                title, image = columns[c]
                ax.imshow(image)
                if r == 0:
                    ax.set_title(title, fontsize=9)
        # set_ylabel would be invisible with the axis off, so annotate instead
        axes[r][0].text(-0.04, 0.5, stem, transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)

    out = Path(args.out) if args.out else Path("outputs/qualitative") / (
        f"{args.benchmark}_{'_'.join(args.tags)}"
        f"{'_decomposition' if args.decomposition else ''}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Wrote {out}  ({len(rows)} rows x {n_cols} columns)")


if __name__ == "__main__":
    main()
