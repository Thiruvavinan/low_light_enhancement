"""
training/__init__.py
--------------------
Trainer, losses and optimizer factories. Nothing here imports from models/ --
the training loop is architecture-agnostic by construction.
"""

from .losses import DecomLoss, EnhanceLoss, build_loss, smoothness_loss
from .optim import build_optimizer, build_scheduler
from .trainer import Trainer

__all__ = [
    "Trainer", "DecomLoss", "EnhanceLoss", "build_loss", "smoothness_loss",
    "build_optimizer", "build_scheduler",
]
