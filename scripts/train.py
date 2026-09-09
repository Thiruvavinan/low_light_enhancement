#!/usr/bin/env python
"""
scripts/train.py
----------------
Entry point for both training stages. The stage is whatever the config says;
this file contains no Retinex-specific logic.

Usage
-----
    # Stage 1 -- Decom-Net (shared by both arms of the experiment)
    python scripts/train.py --config configs/decom.yaml

    # Stage 2 -- Enhance-Net, baseline (paper's L1 reconstruction loss)
    python scripts/train.py --config configs/enhance_l1.yaml

    # Stage 2 -- Enhance-Net, SSIM-augmented reconstruction loss
    python scripts/train.py --config configs/enhance_ssim.yaml

Stage 2 loads the stage-1 checkpoint named in its config and freezes it, so
run stage 1 first. BOTH arms load the SAME stage-1 checkpoint -- that is what
makes the comparison controlled, and why the decomposition is trained once
rather than once per arm.

Handy overrides (everything else lives in the YAML)
--------------------------------------------------
    --loss {l1,ssim}      force the reconstruction loss, ignoring the config
    --set k.p=v           override any config value, e.g.
                          --set loss.ssim_weight=0.5 --set experiment.output_dir=runs/x
    --epochs N            shorten a run (smoke tests)
    --resume              continue from <output_dir>/last.pth

Ablations that are NOT part of the headline comparison (all default to the
published architecture, and both arms must use the same values):

    --set model.norm=group           normalisation in Enhance-Net
    --set model.padding_mode=symmetric
    --set training.grad_log_every=50 per-module gradient-norm diagnostics
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import build_dataset, collate_full_size
from models import build_model
from training.losses import build_loss
from training.optim import build_optimizer, build_scheduler
from training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Seed every RNG the run touches. Both arms use the same seed, so the two
    Enhance-Nets see identical crops and augmentations in identical order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply `--set a.b=value` pairs in place. Values are parsed as YAML, so
    `0.5`, `true` and `[20]` all arrive with the right type."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument(
        "--loss", choices=["l1", "ssim"], default=None,
        help="Force the Enhance-Net reconstruction loss. 'l1' sets ssim_weight=0 "
             "(the paper's baseline); 'ssim' keeps the config's ssim_weight, or "
             "uses 1.0 if the config has it at 0.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="Override any config value; repeatable")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output_dir>/last.pth if it exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.overrides)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.loss is not None:
        current = float(cfg["loss"].get("ssim_weight", 0.0))
        cfg["loss"]["ssim_weight"] = 0.0 if args.loss == "l1" else (current or 1.0)

    seed = cfg["experiment"].get("seed", 0)
    set_seed(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"Device: {device}   seed: {seed}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ds_cfg = cfg["dataset"]
    train_ds = build_dataset(
        ds_cfg["name"],
        root=ds_cfg["root"],
        split="train",
        splits=ds_cfg.get("splits"),
        patch_size=ds_cfg.get("patch_size", 96),
        augment=ds_cfg.get("augment", True),
        seed=seed,
    )
    val_ds = build_dataset(
        ds_cfg["name"],
        root=ds_cfg["root"],
        split="eval",
        splits=ds_cfg.get("val_splits"),
        augment=False,
        seed=seed,
    )
    print(f"Train: {train_ds}")
    print(f"Val:   {val_ds}")

    dl_cfg = cfg["dataloader"]
    num_workers = dl_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    # Validation runs on full-resolution images of differing sizes, so they
    # cannot be stacked -- batch_size=1, and no workers (15 images is not worth
    # the process spawn cost on Windows).
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_full_size
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg)
    trainable = model.num_parameters(trainable_only=True)
    total = model.num_parameters(trainable_only=False)
    print(f"Model: {model_name}  ({trainable/1e3:.1f}K trainable / {total/1e3:.1f}K total params)")

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    loss_cfg = dict(cfg["loss"])
    loss_name = loss_cfg.pop("name")
    loss_fn = build_loss(loss_name, **loss_cfg)
    print(f"Loss: {loss_name}  {loss_cfg}")

    opt_cfg = dict(cfg["optimizer"])
    opt_name = opt_cfg.pop("name")
    optimizer = build_optimizer(opt_name, model.parameters(), **opt_cfg)

    sched_cfg = dict(cfg.get("scheduler") or {})
    sched_name = sched_cfg.pop("name", None)
    scheduler = build_scheduler(sched_name, optimizer, **sched_cfg) if sched_name else None

    # Freeze the exact config that produced this run's checkpoints next to them.
    with open(output_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_cfg = cfg["training"]
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=str(output_dir),
        grad_clip=train_cfg.get("grad_clip"),
        grad_log_every=train_cfg.get("grad_log_every", 0),
    )

    resume_from = None
    if args.resume:
        candidate = output_dir / "last.pth"
        if candidate.exists():
            resume_from = str(candidate)
        else:
            print(f"--resume passed but no checkpoint at {candidate}; starting fresh")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg["epochs"],
        val_every=train_cfg.get("val_every", 1),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        resume_from=resume_from,
    )

    print(f"\nDone. Checkpoints and history.json in {output_dir}")
    if history:
        print("Final epoch:", json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
