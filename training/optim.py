"""
training/optim.py
-----------------
Optimizer and LR-schedule factories, so a run is fully described by its YAML
config and nothing in scripts/ hard-codes a hyperparameter.

`multistep` with milestones=[20] and gamma=0.1 reproduces the authors'
schedule exactly: they build an explicit per-epoch LR array,
`lr = 1e-3 * ones(epochs); lr[20:] = 1e-4`. Expressing it as a standard
scheduler keeps that behaviour while letting the milestone move in config.
"""

from typing import Iterable, List, Optional

import torch


def build_optimizer(name: str, params: Iterable, **kwargs) -> torch.optim.Optimizer:
    name = name.lower()
    # Only optimise what actually requires grad -- stage 2 hands us the full
    # RetinexNet including a frozen Decom-Net, and Adam would otherwise
    # allocate (and step) moment buffers for parameters that never move.
    trainable = [p for p in params if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(trainable, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(trainable, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(trainable, **kwargs)
    raise KeyError(f"Unknown optimizer '{name}'. Available: adam, adamw, sgd")


def build_scheduler(
    name: Optional[str], optimizer: torch.optim.Optimizer, **kwargs
):
    if name is None:
        return None

    name = name.lower()
    if name == "multistep":
        milestones: List[int] = kwargs.pop("milestones", [20])
        gamma: float = kwargs.pop("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma, **kwargs
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    raise KeyError(f"Unknown scheduler '{name}'. Available: multistep, step, cosine")
