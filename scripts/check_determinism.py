#!/usr/bin/env python
"""
scripts/check_determinism.py
----------------------------
Verifies the claim the whole experiment rests on: that two runs with the same
seed see the SAME training data in the SAME order, so a difference in results
is attributable to the loss and not to which crops each arm happened to draw.

    python scripts/check_determinism.py --config configs/enhance_l1.yaml

It builds the training DataLoader exactly as `scripts/train.py` does, twice,
and hashes the tensors the first N batches produce. Same seed must give
identical hashes; a different seed must give different ones -- the second
check is the control, since a loader that returned the same thing regardless
of seed would also "pass" the first.

Why this script exists
----------------------
It was written after finding that the dataset originally drew crops from
`np.random.default_rng()`, which seeds itself from OS entropy and therefore
ignored the configured seed completely. Every run saw a different crop
stream. Nothing crashed, no result looked wrong, and the README claimed
identical data across arms -- the failure was invisible from the outside.
Seed handling is exactly the kind of thing that is either tested or quietly
untrue.
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import build_dataset
from train import set_seed  # noqa: E402  (same directory)


def batch_hashes(cfg: dict, seed: int, n_batches: int, num_workers: int):
    """Hash the first `n_batches` training batches under `seed`."""
    set_seed(seed)

    ds_cfg = cfg["dataset"]
    dataset = build_dataset(
        ds_cfg["name"],
        root=ds_cfg["root"],
        split="train",
        splits=ds_cfg.get("splits"),
        patch_size=ds_cfg.get("patch_size", 96),
        augment=ds_cfg.get("augment", True),
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    hashes = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Hash the pixels AND the sample names: identical pixels from a
        # different image order would still be a reproducibility failure.
        digest = hashlib.sha256()
        digest.update(batch["low"].numpy().tobytes())
        digest.update(batch["high"].numpy().tobytes())
        digest.update("|".join(batch["name"]).encode())
        hashes.append(digest.hexdigest()[:16])
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/enhance_l1.yaml")
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Defaults to the config's value; 0 tests the "
                             "single-process path instead")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"].get("seed", 0)
    workers = args.num_workers if args.num_workers is not None else cfg["dataloader"].get("num_workers", 4)
    print(f"config={args.config}  seed={seed}  num_workers={workers}  batches={args.batches}\n")

    run_a = batch_hashes(cfg, seed, args.batches, workers)
    run_b = batch_hashes(cfg, seed, args.batches, workers)
    run_c = batch_hashes(cfg, seed + 1, args.batches, workers)

    print(f"{'batch':>6}  {'seed=' + str(seed) + ' run A':>20}  "
          f"{'seed=' + str(seed) + ' run B':>20}  {'seed=' + str(seed + 1):>20}")
    for i, (a, b, c) in enumerate(zip(run_a, run_b, run_c)):
        print(f"{i:>6}  {a:>20}  {b:>20}  {c:>20}")

    same_seed_matches = run_a == run_b
    diff_seed_differs = run_a != run_c

    print()
    print(f"  same seed  -> identical batches : {'PASS' if same_seed_matches else 'FAIL'}")
    print(f"  diff seed  -> different batches : {'PASS' if diff_seed_differs else 'FAIL'}"
          f"{'' if diff_seed_differs else '   (loader is ignoring the seed entirely)'}")

    if same_seed_matches and diff_seed_differs:
        print("\nThe two experiment arms will see identical training data.")
        return 0
    print("\nThe crop stream is NOT reproducible -- the arms are not controlled on "
          "the data axis. See the reproducibility note in data/datasets/lol.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
