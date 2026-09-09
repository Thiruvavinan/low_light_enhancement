#!/usr/bin/env python
"""
scripts/plot_history.py
-----------------------
Plot what a training run recorded in <run>/history.json.

    # loss curves for one run
    python scripts/plot_history.py --run runs/decom

    # both arms on one axis, to see whether the SSIM term changed the trajectory
    python scripts/plot_history.py --run runs/enhance_l1 --run runs/enhance_ssim

    # per-module gradient norms (needs training.grad_log_every > 0 during training)
    python scripts/plot_history.py --run runs/enhance_l1 --grad

A note on reading the loss panel across arms
--------------------------------------------
The two Enhance-Net arms optimise different objectives, so their TOTAL losses
are not comparable -- the SSIM arm's is larger by construction because it has
an extra non-negative term. The comparable line is `train_l1`, which both
arms compute identically. `--metric` selects which key to plot for exactly
this reason; it defaults to the total loss, which is the right choice within
one arm and the wrong one across two.

The gradient panel uses a log y-axis, because the interesting failures
(vanishing, exploding) are orders of magnitude, not percentages.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")            # no display on a training box
import matplotlib.pyplot as plt


def load_history(run: Path) -> list:
    path = run / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- has this run trained yet?")
    with open(path) as f:
        return json.load(f)


def plot_losses(runs, metric: str, ax) -> None:
    for run in runs:
        history = load_history(Path(run))
        epochs = [r["epoch"] for r in history]

        train_key = f"train_{metric}"
        if any(train_key in r for r in history):
            ax.plot(
                [r["epoch"] for r in history if train_key in r],
                [r[train_key] for r in history if train_key in r],
                label=f"{Path(run).name} train",
            )
        val_key = f"val_{metric}"
        val_points = [(r["epoch"], r[val_key]) for r in history if val_key in r]
        if val_points:
            ax.plot(*zip(*val_points), linestyle="--", marker="o", markersize=3,
                    label=f"{Path(run).name} val")

        del epochs

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{metric} (log scale)")


def plot_gradients(run: Path, ax) -> bool:
    history = load_history(run)
    keys = sorted({
        k for r in history for k in r
        if k.startswith("grad_norm/") and k not in
        ("grad_norm/n_samples", "grad_norm/n_warnings", "grad_norm/spread")
    })
    if not keys:
        return False

    for key in keys:
        points = [(r["epoch"], r[key]) for r in history if key in r]
        if points:
            label = key.split("/", 1)[1]
            ax.plot(*zip(*points),
                    linewidth=2.0 if label == "global" else 1.2,
                    color="black" if label == "global" else None,
                    label=label)

    spread = [(r["epoch"], r["grad_norm/spread"]) for r in history if "grad_norm/spread" in r]
    if spread:
        twin = ax.twinx()
        twin.plot(*zip(*spread), color="crimson", linestyle=":", label="spread (max/min)")
        twin.set_yscale("log")
        twin.set_ylabel("spread = max group / min group", color="crimson")
        twin.tick_params(axis="y", labelcolor="crimson")

    ax.set_xlabel("epoch")
    ax.set_ylabel("gradient L2 norm")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{run.name}: per-module gradient norms")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="Run directory containing history.json; repeatable")
    parser.add_argument("--metric", default="loss",
                        help="Loss component to plot, e.g. loss, l1, ssim, smooth")
    parser.add_argument("--grad", action="store_true",
                        help="Also plot per-module gradient norms (first run only)")
    parser.add_argument("--out", default=None, help="Output PNG (default: docs/images/<name>.png)")
    args = parser.parse_args()

    n_panels = 2 if args.grad else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)

    plot_losses(args.run, args.metric, axes[0][0])

    if args.grad:
        first = Path(args.run[0])
        if not plot_gradients(first, axes[0][1]):
            axes[0][1].text(0.5, 0.5,
                            "No gradient records in history.json.\n"
                            "Re-train with training.grad_log_every > 0.",
                            ha="center", va="center", wrap=True)
            axes[0][1].set_axis_off()

    out = Path(args.out) if args.out else Path("docs/images") / (
        "_".join(Path(r).name for r in args.run) + ("_grad" if args.grad else "") + ".png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
