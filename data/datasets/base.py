"""
data/datasets/base.py
---------------------
Shared image IO and augmentation for every low-light dataset.

Conventions every dataset in this package obeys
-----------------------------------------------
* Images are returned as float32 CHW tensors in [0, 1]. No mean/std
  normalisation anywhere: Retinex-Net's whole formulation is S = R * I with
  both factors sigmoid-bounded to [0, 1], and its reconstruction loss
  compares R * I directly against the input image. Standardising the input
  would break that identity, so [0, 1] is a hard requirement, not a default.

* Every sample carries a "name" (the file stem) so predictions can be
  written back out under the same filename for qualitative comparison.

* Paired datasets return "low" and "high"; unpaired ones return "low" only.
  Losses and metrics branch on presence, never on a dataset class name.

Augmentation
------------
The eight dihedral transforms (identity, 3 rotations, each with and without
a vertical flip) used by the authors' implementation. Applied identically to
both images of a pair -- an augmentation that desynchronised the pair would
destroy the pixel correspondence the reconstruction loss depends on. Applied
after cropping, so the rotations always act on a square patch.
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(directory: Path) -> List[Path]:
    """Every image file in `directory`, sorted by name. Non-images are ignored."""
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} not found. Run `python scripts/prepare_data.py` first."
        )
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_image(path: Path, max_side: int = 0) -> np.ndarray:
    """
    Read an image as float32 HWC in [0, 1].

    max_side > 0 downscales the longer edge to at most that many pixels
    (bicubic, aspect preserved). Only used for evaluation on the unpaired
    benchmarks, where a handful of images are large enough to exhaust a 4GB
    card; training patches are never touched by it.
    """
    im = Image.open(path).convert("RGB")

    if max_side and max(im.size) > max_side:
        scale = max_side / max(im.size)
        new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        im = im.resize(new_size, Image.BICUBIC)

    return np.asarray(im, dtype=np.float32) / 255.0


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC float32 numpy -> CHW float32 tensor. `.copy()` because the augmentation
    views (np.rot90 / np.flipud) are negative-stride and torch cannot wrap those."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def augment(image: np.ndarray, mode: int) -> np.ndarray:
    """One of the 8 dihedral transforms. `mode` must be the same for both images of a pair."""
    if mode == 0:
        return image
    if mode == 1:
        return np.flipud(image)
    if mode == 2:
        return np.rot90(image)
    if mode == 3:
        return np.flipud(np.rot90(image))
    if mode == 4:
        return np.rot90(image, k=2)
    if mode == 5:
        return np.flipud(np.rot90(image, k=2))
    if mode == 6:
        return np.rot90(image, k=3)
    if mode == 7:
        return np.flipud(np.rot90(image, k=3))
    raise ValueError(f"augmentation mode must be 0..7, got {mode}")


def random_crop_coords(shape: Sequence[int], patch_size: int) -> Tuple[int, int]:
    """
    Top-left corner of a random patch_size x patch_size crop within an HWC image.

    Uses torch's default generator, which the DataLoader seeds deterministically
    per worker from the run seed -- see the reproducibility note in lol.py.
    numpy's `default_rng()` would silently ignore the seed.
    """
    h, w = shape[0], shape[1]
    if h < patch_size or w < patch_size:
        raise ValueError(f"image {h}x{w} is smaller than patch_size {patch_size}")
    return (
        int(torch.randint(0, h - patch_size + 1, (1,)).item()),
        int(torch.randint(0, w - patch_size + 1, (1,)).item()),
    )


class LowLightDataset(Dataset):
    """Base class; subclasses fill in `self.samples` and implement `__getitem__`."""

    def __len__(self) -> int:
        return len(self.samples)
