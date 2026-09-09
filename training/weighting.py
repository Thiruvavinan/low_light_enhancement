"""
training/weighting.py
---------------------
Learned loss-term weighting by homoscedastic uncertainty
(Kendall, Gal & Cipolla, CVPR 2018, "Multi-Task Learning Using Uncertainty to
Weigh Losses for Scene Geometry and Semantics").

The problem it solves
---------------------
A multi-term loss needs relative weights, and hand-tuning them is a grid
search in as many dimensions as there are terms. Retinex-Net's Enhance-Net
loss has three: L1 reconstruction, SSIM, and illumination smoothness, with
the paper/reference using fixed 1.0 / -- / 3.0. Those numbers are asserted,
not derived.

The idea
--------
Treat each term as the negative log-likelihood of an observation model with
its own learned noise scale, and optimise the scales jointly with the
network. For a Gaussian likelihood with variance sigma^2, parameterised by
s = log(sigma^2) for numerical stability:

    L = sum_i [ 0.5 * exp(-s_i) * L_i  +  0.5 * s_i ]

The first term downweights a noisy (high-variance) objective. The second is
what stops the trivial solution: without `+0.5*s_i`, the optimiser would send
every s_i to +infinity, drive every weight to zero, and report a loss of zero
having learned nothing.

The total loss can go NEGATIVE, and that is not a bug
-----------------------------------------------------
`0.5*s_i` is a log-variance, not a magnitude, so it is negative whenever the
optimiser becomes confident about a term (sigma < 1). A well-fitting run with
several learned terms routinely reports a total below zero. It is still a
valid objective -- it is a negative log-likelihood, not a distance -- and
lower is still better, so checkpoint selection by validation loss behaves
normally. What is NOT meaningful is comparing this number against a
fixed-weight run's loss: they are on different scales and measure different
things. Compare arms on evaluation metrics.

Two things worth knowing before using this
------------------------------------------
1. **The Gaussian form is exact only for an L2 term.** `mode="gaussian"`
   reproduces Kendall et al. as published, and it is what almost everyone
   cites -- but an L1 term is the NLL of a *Laplace* likelihood, whose
   correct form is `exp(-s)*L + s` with `s = log b`. `mode="laplace"` gives
   that. Applying either to SSIM is a heuristic in both cases: `1 - SSIM` is
   not a likelihood at all, so its learned "variance" is a free scale
   parameter with a log-barrier, not an uncertainty estimate. That is worth
   saying out loud rather than inheriting the paper's interpretation
   unexamined.

2. **Regularisers are not observations.** A data-fit term is pinned by the
   data: downweight it too far and its loss rises, which raises the total.
   A regulariser has no such counter-pressure, so uncertainty weighting will
   tend to downweight it toward whatever the log-barrier alone permits.
   `fixed_terms` keeps named terms on hand-set weights for this reason, so
   the smoothness prior can be held at the paper's value while only the
   reconstruction terms are learned.

Reuse
-----
Nothing here knows about Retinex. It weights any dict of named scalar
losses:

    weighting = UncertaintyWeighting(["l1", "ssim"], fixed_terms={"smooth": 3.0})
    total, stats = weighting({"l1": l1, "ssim": ssim_term, "smooth": smooth})

`stats` carries the current effective weight and log-variance of every term,
ready to be logged so the learned weights can be read off after training.
Because it holds `nn.Parameter`s, it must be handed to the optimiser --
`training/optim.py` takes an iterable of parameters, and `scripts/train.py`
passes the loss's parameters alongside the model's.
"""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

MODES = ("gaussian", "laplace")


class UncertaintyWeighting(nn.Module):
    """
    Parameters
    ----------
    terms : names of the loss terms whose weights are LEARNED. Order fixes the
        parameter layout; it has no other significance.
    mode : "gaussian" -> 0.5*exp(-s)*L + 0.5*s   (Kendall et al. as published;
               exact for an L2 term)
           "laplace"  -> exp(-s)*L + s           (exact for an L1 term)
    init_log_var : starting value of every s_i. 0.0 means sigma = 1, i.e. every
        learned term starts at effective weight 0.5 (gaussian) or 1.0
        (laplace), so the run begins near the unweighted sum rather than at an
        arbitrary point.
    fixed_terms : name -> constant weight, for terms that should NOT be
        learned (regularisers -- see the module docstring).

    forward
    -------
    losses : dict containing every name in `terms` and in `fixed_terms`
      -> (total, stats) where stats holds detached weights and log-variances
    """

    def __init__(
        self,
        terms: Sequence[str],
        mode: str = "gaussian",
        init_log_var: float = 0.0,
        fixed_terms: Optional[Mapping[str, float]] = None,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not terms:
            raise ValueError("UncertaintyWeighting needs at least one learned term")

        self.terms = list(terms)
        self.mode = mode
        self.fixed_terms = dict(fixed_terms or {})

        overlap = set(self.terms) & set(self.fixed_terms)
        if overlap:
            raise ValueError(f"terms cannot be both learned and fixed: {sorted(overlap)}")

        # One log-variance per learned term. A single Parameter vector rather
        # than a ParameterDict so the whole set moves to a device, is saved,
        # and is handed to the optimiser as one object.
        self.log_var = nn.Parameter(torch.full((len(self.terms),), float(init_log_var)))

    def extra_repr(self) -> str:
        return f"terms={self.terms}, mode={self.mode}, fixed={self.fixed_terms}"

    def weight_of(self, index: int) -> torch.Tensor:
        """Effective multiplier currently applied to learned term `index`."""
        scale = 0.5 if self.mode == "gaussian" else 1.0
        return scale * torch.exp(-self.log_var[index])

    def forward(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        missing = [t for t in self.terms + list(self.fixed_terms) if t not in losses]
        if missing:
            raise KeyError(f"UncertaintyWeighting is missing loss terms: {missing}")

        total = None
        stats: Dict[str, torch.Tensor] = {}

        for index, name in enumerate(self.terms):
            log_var = self.log_var[index]
            weight = self.weight_of(index)
            # The log-barrier: without it every weight collapses to zero.
            barrier = 0.5 * log_var if self.mode == "gaussian" else log_var
            contribution = weight * losses[name] + barrier

            total = contribution if total is None else total + contribution
            stats[f"w_{name}"] = weight.detach()
            stats[f"logvar_{name}"] = log_var.detach()

        for name, weight in self.fixed_terms.items():
            total = total + weight * losses[name]

        return total, stats
