"""
evaluation/__init__.py
----------------------
Metric wrappers and the dataset-scoring engine. Kept separate from training/
because evaluation must be able to run against any checkpoint, including ones
produced by a different loss, without importing anything training-specific.
"""

from .denoise import BM3DReflectanceDenoiser
from .engine import evaluate_dataset, save_image
from .metrics import LPIPSMetric, NoReferenceMetric, psnr, ssim, ssim_gray, summarise

__all__ = [
    "evaluate_dataset", "save_image", "BM3DReflectanceDenoiser",
    "psnr", "ssim", "ssim_gray", "LPIPSMetric", "NoReferenceMetric", "summarise",
]
