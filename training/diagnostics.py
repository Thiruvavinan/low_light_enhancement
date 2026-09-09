"""
training/diagnostics.py
-----------------------
Per-module gradient statistics, for telling apart the three ways a training
run goes wrong that all look identical from a loss curve alone:

    loss plateaus high      the network is not learning
    loss plateaus high      gradients vanished in the deep layers
    loss goes to NaN        gradients exploded

A single global grad-norm number cannot distinguish "healthy overall" from
"the encoder is learning and the decoder is dead", because one large norm
hides ten small ones. So gradients are grouped by module and reported per
group, along with the ratio between the largest and smallest group -- which
is the number that actually diagnoses a vanishing-gradient problem.

Why this matters for Retinex-Net specifically
---------------------------------------------
Both networks end in a sigmoid (Decom-Net) or feed a product R * I_delta
(Enhance-Net). Sigmoid saturation is the classic vanishing-gradient source:
once the pre-activation is far from zero the local gradient is ~0 and the
layers behind it stop receiving signal. The illumination smoothness term
makes the opposite failure available too -- it contains exp(-10 * grad R),
which is well-behaved, but its gradient scales with the illumination
gradient magnitude and can spike early in training when I is still noisy.

Neither failure is hypothetical for this architecture, and neither is
visible in the total loss until it is too late to tell which one happened.

Cost
----
One extra pass over the parameter gradients per logged step. Sampling every
N steps (`log_every`) keeps that off the critical path; the default only
samples once per epoch, which is enough to see a trend.

Usage
-----
    monitor = GradientMonitor(model, log_every=50)
    ...
    loss.backward()
    monitor.record(step)             # after backward(), before optimizer.step()
    optimizer.step()
    ...
    monitor.summary()                # -> dict, written into history.json
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# Layer families whose numbered members are reported together: the question
# worth answering is "is the encoder getting gradient", not "is down2".
_COLLAPSED_PREFIXES = ("down", "up", "activated")


def _group_of(param_name: str) -> str:
    """
    Coarse module group for a parameter name. Groups are the units a person
    would reason about ("did the encoder get gradient?"), not individual
    tensors, so the report stays readable at a glance.

        net.shallow.weight           -> decom.shallow     (stage 1)
        net.activated.4.bias         -> decom.activated   (stage 1)
        decom.net.shallow.weight     -> decom.shallow     (stage 2)
        enhance.down2.weight         -> enhance.down
        enhance.up1.weight           -> enhance.up
        enhance.fuse.weight          -> enhance.fuse

    Stage 1 trains a bare DecomStage, whose parameters are named `net.*`;
    stage 2 trains a RetinexNet, where the same tensors are `decom.net.*`.
    Both are normalised to the same `decom.*` labels so a stage-1 and a
    stage-2 history can be read side by side.
    """
    # Drop the trailing "weight"/"bias" and any numeric index (activated.4)
    parts = [p for p in param_name.split(".")[:-1] if not p.isdigit()]
    if not parts:
        return param_name

    # Normalise the two spellings of the Decom-Net wrapper
    if parts[0] == "net":
        parts = ["decom"] + parts[1:]
    elif len(parts) > 1 and parts[0] == "decom" and parts[1] == "net":
        parts = ["decom"] + parts[2:]

    leaf = parts[-1]
    for prefix in _COLLAPSED_PREFIXES:
        if leaf.startswith(prefix) and leaf != prefix:
            parts[-1] = prefix
            break

    return ".".join(parts)


class GradientMonitor:
    """
    Parameters
    ----------
    model     : the module being trained; only parameters with requires_grad
                are tracked, so a frozen Decom-Net simply does not appear
    log_every : record every Nth call to `record`. 0 disables recording
                entirely (the monitor becomes a no-op you can leave wired in).
    explode_threshold / vanish_threshold : group grad-norm bounds that count
                as a warning. The defaults are deliberately wide -- they flag
                "something is structurally wrong", not "this step was large".
    """

    def __init__(
        self,
        model: nn.Module,
        log_every: int = 0,
        explode_threshold: float = 1e3,
        vanish_threshold: float = 1e-7,
    ):
        self.model = model
        self.log_every = log_every
        self.explode_threshold = explode_threshold
        self.vanish_threshold = vanish_threshold

        self.history: List[Dict[str, float]] = []
        self.warnings: List[str] = []
        self._calls = 0

    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.log_every > 0

    @torch.no_grad()
    def record(self, step: int) -> Optional[Dict[str, float]]:
        """
        Call after loss.backward() and before optimizer.step() -- afterwards is
        too late on optimizers that modify .grad in place, and before backward
        there is nothing to read.
        """
        self._calls += 1
        if not self.enabled or (self._calls - 1) % self.log_every:
            return None

        # Sum of squares per group, so the reported number is a true L2 norm
        # over the whole group rather than a mean of per-tensor norms.
        sq: Dict[str, float] = {}
        n_params: Dict[str, int] = {}
        non_finite: List[str] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                non_finite.append(name)
            group = _group_of(name)
            sq[group] = sq.get(group, 0.0) + float(grad.float().pow(2).sum())
            n_params[group] = n_params.get(group, 0) + grad.numel()

        if not sq:
            return None

        norms = {g: math.sqrt(v) for g, v in sq.items()}
        record = {"step": step, **{f"grad_norm/{g}": v for g, v in norms.items()}}
        record["grad_norm/global"] = math.sqrt(sum(sq.values()))

        finite = [v for v in norms.values() if v > 0]
        if finite:
            # The diagnostic number: how unevenly gradient is distributed across
            # the network. A ratio in the thousands means the small end is dead.
            record["grad_norm/spread"] = max(finite) / min(finite)

        self._check(step, norms, non_finite)
        self.history.append(record)
        return record

    # ------------------------------------------------------------------

    def _check(self, step: int, norms: Dict[str, float], non_finite: List[str]) -> None:
        if non_finite:
            self._warn(f"step {step}: non-finite gradients in {non_finite[:3]}"
                       f"{' ...' if len(non_finite) > 3 else ''}")
        for group, value in norms.items():
            if value > self.explode_threshold:
                self._warn(f"step {step}: {group} grad norm {value:.3e} "
                           f"> {self.explode_threshold:.0e} (exploding)")
            elif value < self.vanish_threshold:
                self._warn(f"step {step}: {group} grad norm {value:.3e} "
                           f"< {self.vanish_threshold:.0e} (vanishing)")

    def _warn(self, message: str) -> None:
        # Printed once each; a vanishing gradient repeats every step and would
        # otherwise bury the training log it is meant to annotate.
        if message.split(":", 1)[-1] not in {w.split(":", 1)[-1] for w in self.warnings}:
            print(f"  [grad] {message}")
        self.warnings.append(message)

    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """Mean of each recorded quantity since the last `reset`, for history.json."""
        if not self.history:
            return {}
        keys = [k for k in self.history[-1] if k != "step"]
        out = {k: float(sum(r.get(k, 0.0) for r in self.history) / len(self.history))
               for k in keys}
        out["grad_norm/n_samples"] = len(self.history)
        if self.warnings:
            out["grad_norm/n_warnings"] = len(self.warnings)
        return out

    def reset(self) -> None:
        """Clear the per-epoch buffer; warnings are kept for the whole run."""
        self.history = []
