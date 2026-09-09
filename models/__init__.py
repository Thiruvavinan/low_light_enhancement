"""
models/__init__.py
------------------
Registry of trainable stages. Scripts name a stage in a YAML config; nothing
outside this package imports a network class directly, so adding an
architecture means adding a file here and one line to MODELS -- no changes
to training/, evaluation/ or scripts/.
"""

from .base import BaseModel
from .decom_net import DecomNet
from .enhance_net import EnhanceNet
from .retinexnet import DecomStage, RetinexNet

MODELS = {
    "decom": DecomStage,     # stage 1: shared Decom-Net over a low/normal pair
    "retinex": RetinexNet,   # stage 2 (frozen Decom-Net + Enhance-Net) and inference
}


def build_model(name: str, **kwargs) -> BaseModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = [
    "BaseModel", "DecomNet", "EnhanceNet", "DecomStage", "RetinexNet",
    "MODELS", "build_model",
]
