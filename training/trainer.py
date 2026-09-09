"""
training/trainer.py
-------------------
Stage-agnostic training loop.

The Trainer imports nothing from models/ and knows nothing about Retinex
decomposition. It takes a module satisfying the models/base.py contract
(batch dict -> outputs dict) and a loss returning a dict with a "loss" key,
and trains it. Both stages -- Decom-Net on image pairs and Enhance-Net
behind a frozen Decom-Net -- run through this same loop unchanged, which is
what makes the L1-vs-SSIM comparison controlled by construction: the two
arms cannot differ in anything this file does.

Logging
-------
Losses return every component alongside the total, and all of them are
averaged and written to <output_dir>/history.json after each epoch. For the
SSIM arm that means the raw SSIM value is logged next to the L1 term, so
"did the SSIM term actually move anything" is answerable from the history
file rather than from the final metric alone.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .diagnostics import GradientMonitor


def _to_device(batch: Dict[str, object], device: str) -> Dict[str, object]:
    """Move tensors; leave metadata (the "name" list) alone."""
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


class Trainer:
    """
    Parameters
    ----------
    model      : nn.Module -- forward(batch: dict) -> outputs: dict
    loss_fn    : nn.Module -- forward(outputs, batch) -> dict containing "loss"
    optimizer  : torch.optim.Optimizer
    scheduler  : LR scheduler, stepped once per epoch (optional)
    device     : "cuda" | "cpu"
    output_dir : where checkpoints and history.json are written
    grad_clip  : max global grad norm, or None. Off by default -- the
                 reference implementation does not clip and the baseline
                 should not silently differ from it.
    grad_log_every : sample per-module gradient norms every N optimizer steps
                 (see training/diagnostics.py). 0 disables the monitor
                 entirely. The statistics are appended to each epoch's
                 history entry; they never affect the update, so turning
                 this on does not change what the model learns.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        output_dir: str = "runs/experiment",
        grad_clip: Optional[float] = None,
        grad_log_every: int = 0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip
        self.grad_monitor = GradientMonitor(self.model, log_every=grad_log_every)
        self._global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        val_every: int = 1,
        early_stopping_patience: Optional[int] = None,
        resume_from: Optional[str] = None,
    ):
        """
        early_stopping_patience : stop once this many epochs have passed since
            val_loss last improved. None (the default for this project) runs
            the full fixed budget -- both arms of the comparison must see the
            same number of updates, and an arm-dependent stopping point would
            confound the result.
        resume_from : checkpoint written by this Trainer; restores model,
            optimizer and scheduler state and continues from the next epoch.
        """
        start_epoch = 1
        best_val_loss = float("inf")
        epochs_since_improvement = 0

        if resume_from is not None:
            ckpt = self.load_checkpoint(resume_from)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            epochs_since_improvement = ckpt.get("epochs_since_improvement", 0)
            print(
                f"Resumed from {resume_from}: starting at epoch {start_epoch} "
                f"(best_val_loss={best_val_loss:.4f})"
            )

        history = self._load_history() if resume_from is not None else []

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            self.grad_monitor.reset()
            train_stats = self._run_epoch(train_loader, training=True)
            elapsed = time.time() - t0

            log = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"], "time_s": round(elapsed, 1)}
            log.update({f"train_{k}": v for k, v in train_stats.items()})
            log.update(self.grad_monitor.summary())

            if val_loader is not None and epoch % val_every == 0:
                val_stats = self._run_epoch(val_loader, training=False)
                log.update({f"val_{k}": v for k, v in val_stats.items()})

                if val_stats["loss"] < best_val_loss:
                    best_val_loss = val_stats["loss"]
                    epochs_since_improvement = 0
                    self._save_checkpoint("best.pth", epoch, best_val_loss, epochs_since_improvement)
                else:
                    epochs_since_improvement += val_every

            if self.scheduler is not None:
                self.scheduler.step()

            history.append(log)
            self._print_log(log)
            self._save_checkpoint("last.pth", epoch, best_val_loss, epochs_since_improvement)
            self._save_history(history)

            if (
                early_stopping_patience is not None
                and epochs_since_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping: val_loss has not improved in "
                    f"{epochs_since_improvement} epochs (patience={early_stopping_patience})"
                )
                break

        return history

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        totals: Dict[str, float] = {}
        n_batches = 0
        context = torch.enable_grad if training else torch.no_grad

        with context():
            for batch in loader:
                batch = _to_device(batch, self.device)

                outputs = self.model(batch)
                components = self.loss_fn(outputs, batch)
                loss = components["loss"]

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    # After backward, before step: .grad holds the raw gradient,
                    # and before any clipping, so an exploding gradient is still
                    # visible rather than already truncated.
                    self.grad_monitor.record(self._global_step)
                    self._global_step += 1
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.grad_clip,
                        )
                    self.optimizer.step()

                for name, value in components.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _save_checkpoint(
        self, filename: str, epoch: int, best_val_loss: float, epochs_since_improvement: int
    ):
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                # The loss can carry learned parameters (uncertainty weighting).
                # Without this they are lost on save and a resumed run silently
                # restarts them from their initial value.
                "loss_state": self.loss_fn.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_val_loss": best_val_loss,
                "epochs_since_improvement": epochs_since_improvement,
            },
            self.output_dir / filename,
        )

    def _save_history(self, history: list):
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    def _load_history(self) -> list:
        path = self.output_dir / "history.json"
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

    def load_checkpoint(self, path: str) -> dict:
        """Restore model/optimizer/scheduler state; returns the raw checkpoint dict."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("loss_state"):
            self.loss_fn.load_state_dict(ckpt["loss_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and ckpt.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt

    @staticmethod
    def _print_log(log: dict):
        parts = [f"Epoch {log['epoch']:03d}"]
        for key in ("train_loss", "val_loss", "train_l1", "val_ssim"):
            if key in log:
                parts.append(f"{key}={log[key]:.4f}")
        parts.append(f"lr={log['lr']:.2e}")
        parts.append(f"({log['time_s']}s)")
        print("  ".join(parts))
