#!/usr/bin/env python
"""
scripts/make_qualitative.py
---------------------------
Generates the full set of qualitative grids, one per question the results
table asks, and verifies that **every evaluated model appears in at least one
grid**.

    python scripts/make_qualitative.py
    python scripts/make_qualitative.py --n 3 --benchmarks LOL LIME

Why a driver rather than calling visualize_predictions.py by hand
-----------------------------------------------------------------
There are more evaluated models than fit in one readable grid, so the figures
have to be split. Splitting by hand is how a model quietly ends up with no
picture at all: the table grows, the figures do not, and nobody notices
because nothing errors. This script derives the grids from one declared
layout and then asserts coverage against `outputs/eval/`, so adding an
evaluation either lands in a grid or fails the check.

Grids mirror the README's structure: one per experiment, plus the
cross-dataset views, plus the decomposition. `input` is not a model -- it is
added as the first column of every grid by visualize_predictions.py, and the
ground truth as the last where a benchmark has one.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PY = sys.executable
VIS = "scripts/visualize_predictions.py"

# name -> (benchmarks, tags, extra flags)
GRIDS = [
    # Experiment 1: does an SSIM term help?
    ("exp1_ssim", ["LOL", "LIME", "MEF", "DICM"], ["l1", "ssim"], []),
    # Experiment 2: learned vs fixed weights
    ("exp2_weights", ["LOL", "LIME"], ["ssim", "uw", "rebalanced"], []),
    # Experiment 3: denoising, both tuning objectives
    ("exp3_denoise_l1", ["LOL", "DICM"], ["l1", "l1_bm3d", "l1_bm3d_niqe"], []),
    ("exp3_denoise_all", ["LOL"], ["l1_bm3d", "ssim_bm3d", "uw_bm3d", "rebalanced_bm3d"], []),
    ("exp3_denoise_niqe", ["LOL"],
     ["l1_bm3d_niqe", "ssim_bm3d_niqe", "uw_bm3d_niqe", "rebalanced_bm3d_niqe"], []),
    # Side axes
    ("decomnet_variants", ["LOL"], ["ssim", "ssim_decomuw", "ssim_lowsmooth"], []),
    ("reference", ["LOL"], ["tf_reference", "l1", "rebalanced"], []),
    # What the decomposition itself learned
    ("decomposition", ["LOL"], ["l1"], ["--decomposition"]),
]


def evaluated_models():
    root = Path("outputs/eval")
    return {p.name for p in root.iterdir()
            if (p / "summary.json").exists() and p.name != "input"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="Images per grid")
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help="Restrict to these benchmarks (default: all in the layout)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    made, covered, failed = [], set(), []

    for name, benches, tags, extra in GRIDS:
        for bench in benches:
            if args.benchmarks and bench not in args.benchmarks:
                continue
            out = "outputs/qualitative/%s_%s.png" % (bench, name)
            cmd = [PY, VIS, "--benchmark", bench, "--tags", *tags,
                   "--n", str(args.n), "--out", out, *extra]
            if args.dry_run:
                print("  would run:", " ".join(cmd[2:]))
                covered.update(tags)
                continue
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode:
                failed.append((out, (r.stderr or r.stdout).strip().splitlines()[-1:]))
            else:
                made.append(out)
                covered.update(tags)

    for out in made:
        print("  wrote", out)
    for out, err in failed:
        print("  FAILED", out, err)

    # The point of the script: no evaluated model may be missing from every grid.
    missing = sorted(evaluated_models() - covered)
    print("\n%d grids, %d models covered." % (len(made), len(covered)))
    if missing:
        print("NOT PICTURED ANYWHERE: %s" % ", ".join(missing))
        print("Add them to GRIDS above.")
        return 1
    print("Every evaluated model appears in at least one grid.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
