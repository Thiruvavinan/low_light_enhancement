"""
data/datasets/unpaired.py
-------------------------
Unpaired low-light benchmarks: LIME, MEF, DICM.

These are the three sets the original paper uses for its cross-dataset
generalisation comparison (Section 4.3) -- but the paper reports them only
as visual side-by-side figures, with no metrics. Quantifying that
comparison is this project's contribution over the paper, which is why
these are first-class datasets here rather than a demo folder.

There is no ground truth, so nothing full-reference (PSNR/SSIM/LPIPS) can
be computed on them; evaluation uses the no-reference NIQE only. Samples
therefore carry "low" and "name" and no "high" -- evaluation/engine.py
branches on that, so a missing "high" is a supported state, not an error.

Counts in the bundled Test.zip differ from the paper's text: LIME 10 (as
stated), MEF 79 individual images rather than "17 sequences", DICM 44
rather than 69. The bundle is the one commonly redistributed with
low-light benchmarks; the README records the exact counts actually
evaluated so the numbers are reproducible even though they are not
directly comparable to a differently-sized DICM.
"""

from pathlib import Path
from typing import Dict, List

from .base import LowLightDataset, list_images, load_image, to_tensor


class UnpairedDataset(LowLightDataset):
    """
    Parameters
    ----------
    root     : directory holding the named subdirectories
    subset   : subdirectory to load ("LIME", "MEF", "DICM", ...). Named
               `subset` rather than `name` because `name` is already the
               registry key argument of build_dataset().
    max_side : downscale the longer edge to at most this many pixels before
               inference. 0 disables. DICM in particular contains multi-
               megapixel images; Enhance-Net's full-resolution multi-scale
               concatenation holds 3*C feature maps at input resolution,
               which overruns a 4GB card on those. Applied identically to
               both model variants, so it cannot bias the comparison --
               but it does mean NIQE is measured at the scaled resolution,
               which is recorded in the README.

    __getitem__ -> {"low": [3,h,w], "name": str}
    """

    def __init__(self, root: str = "data/cross_eval", subset: str = "LIME", max_side: int = 0):
        self.root = Path(root)
        self.subset = subset
        self.max_side = max_side
        self.samples: List[Path] = list_images(self.root / subset)

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root / subset}")

    def __repr__(self) -> str:
        return f"UnpairedDataset(subset={self.subset}, images={len(self.samples)}, max_side={self.max_side})"

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.samples[index]
        return {
            "low": to_tensor(load_image(path, max_side=self.max_side)),
            "name": path.stem,
        }
