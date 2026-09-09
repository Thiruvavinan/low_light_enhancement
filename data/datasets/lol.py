"""
data/datasets/lol.py
--------------------
LOL (LOw-Light) paired dataset -- Wei et al., BMVC 2018.

Splits
------
The dataset ships with the authors' canonical split already applied, so no
split is invented here:

    our485/  485 real low/normal pairs, captured by varying exposure and
             ISO on the same scene            -> training
    eval15/   15 real pairs, held out         -> in-distribution evaluation
    syn/     1000 synthetic pairs, low-light  -> training
             synthesised from raw images

`splits` selects which of these to draw from. The paper trains on
our485 + syn together (Section 4.1); evaluation is always eval15 alone.
Nothing is ever sampled from eval15 during training.

Modes
-----
train : one random patch_size x patch_size crop per image per epoch, with a
        random dihedral augmentation applied identically to both images of
        the pair. An epoch therefore visits every image exactly once, which
        matches the reference implementation's epoch definition.
eval  : the full image, unaugmented and uncropped -- metrics have to be
        computed at native resolution to be comparable with published
        numbers.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .base import (
    LowLightDataset, augment, list_images, load_image, random_crop_coords, to_tensor,
)


class LOLDataset(LowLightDataset):
    """
    Parameters
    ----------
    root       : directory holding our485/, eval15/, syn/
    split      : "train" or "eval" -- controls cropping/augmentation
    splits     : which subdirectories to draw from. Defaults to
                 ("our485", "syn") for train and ("eval15",) for eval.
    patch_size : train-mode crop size (paper: 96)
    augment    : enable the 8 dihedral transforms (train only)
    seed       : base seed for crop/augmentation sampling -- see the note below

    __getitem__ -> {"low": [3,h,w], "high": [3,h,w], "name": str}

    Reproducibility of the crop stream
    ----------------------------------
    Crops are drawn from `torch.randint` on the default generator, NOT from
    `np.random.default_rng()`. The difference matters and is not stylistic:
    `default_rng()` with no argument seeds itself from OS entropy, so it
    ignores the run's seed entirely and produces a different crop stream on
    every run. PyTorch's DataLoader, by contrast, seeds each worker's torch
    generator deterministically from the main process's generator, which
    `scripts/train.py` seeds from the config.

    That is what makes the L1-vs-SSIM comparison controlled on the data axis
    as well as the loss axis: with the same seed, both arms see the same
    crops and the same augmentations in the same order, so any difference in
    the result is attributable to the loss rather than to sampling noise.
    `scripts/check_determinism.py` verifies this rather than assuming it.
    """

    DEFAULT_SPLITS = {"train": ("our485", "syn"), "eval": ("eval15",)}

    def __init__(
        self,
        root: str = "data/lol",
        split: str = "train",
        splits: Sequence[str] = None,
        patch_size: int = 96,
        augment: bool = True,
        seed: int = 0,
    ):
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.seed = seed

        self.subsets = tuple(splits) if splits is not None else self.DEFAULT_SPLITS[split]
        self.samples: List[Tuple[Path, Path]] = []

        for subset in self.subsets:
            lows = list_images(self.root / subset / "low")
            highs = list_images(self.root / subset / "high")
            by_stem = {p.stem: p for p in highs}
            missing = [p.name for p in lows if p.stem not in by_stem]
            if missing:
                raise FileNotFoundError(
                    f"{subset}: {len(missing)} low images have no matching high image "
                    f"(e.g. {missing[:3]})"
                )
            self.samples += [(p, by_stem[p.stem]) for p in lows]

        if not self.samples:
            raise RuntimeError(f"No pairs found under {self.root} for subsets {self.subsets}")

    def __repr__(self) -> str:
        return (
            f"LOLDataset(split={self.split}, subsets={self.subsets}, "
            f"pairs={len(self.samples)}, patch_size={self.patch_size if self.split=='train' else 'full'})"
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        low_path, high_path = self.samples[index]
        low = load_image(low_path)
        high = load_image(high_path)

        if self.split == "train":
            # torch's generator, not numpy's: the DataLoader seeds it per worker
            # from the run seed, so the crop stream is reproducible (see the
            # class docstring). Workers still get decorrelated streams.
            y, x = random_crop_coords(low.shape, self.patch_size)
            ps = self.patch_size
            low = low[y:y + ps, x:x + ps, :]
            high = high[y:y + ps, x:x + ps, :]

            if self.augment:
                mode = int(torch.randint(0, 8, (1,)).item())
                low = augment(low, mode)
                high = augment(high, mode)

        return {"low": to_tensor(low), "high": to_tensor(high), "name": low_path.stem}


def collate_full_size(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Collate for eval mode. Full-size images in a dataset are not all the same
    resolution, so they cannot be stacked; evaluation runs one image at a time
    and this simply asserts that and unwraps the singleton.
    """
    if len(batch) != 1:
        raise ValueError(
            "full-size evaluation requires batch_size=1 (images have differing shapes)"
        )
    item = batch[0]
    return {
        k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
        for k, v in item.items()
    }
