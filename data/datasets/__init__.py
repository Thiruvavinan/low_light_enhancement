"""
data/datasets/__init__.py
-------------------------
Dataset registry. Configs and scripts name a dataset as a string; nothing
outside this package constructs a dataset class directly, so a new benchmark
means a new file here plus one line in DATASETS.
"""

from .base import LowLightDataset, list_images, load_image, to_tensor
from .lol import LOLDataset, collate_full_size
from .unpaired import UnpairedDataset

DATASETS = {
    "lol": LOLDataset,            # paired, has ground truth
    "unpaired": UnpairedDataset,  # LIME / MEF / DICM, no ground truth
}


def build_dataset(name: str, **kwargs) -> LowLightDataset:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[name](**kwargs)


__all__ = [
    "LowLightDataset", "LOLDataset", "UnpairedDataset",
    "DATASETS", "build_dataset", "collate_full_size",
    "list_images", "load_image", "to_tensor",
]
