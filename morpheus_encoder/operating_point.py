"""The operating-point token p = (alpha, delta, B, kappa).

Step 9 of the build plan trains a Pareto-conditioned model: one MORPHEUS that
traverses the discovery x utility x artifact surface instead of one model per
tradeoff. That training is emphatically NOT built here. What is built is the
*input path*, so that step 9 is a training change rather than a refactor:

    G_sci = f_theta(X ; p)

p enters in three places, and each of them is a place where the choice actually
has to bite:

  1. the slot queries -- an exploratory operating point should be able to open
     slots that a confirmatory one keeps shut;
  2. the modality router R_rho -- how much cross-modal coupling an object is
     allowed to buy depends on the tolerated artifact risk;
  3. the expert router R_psi and a FiLM on expert outputs -- the complexity
     budget B is exactly a statement about which experts are affordable.

The four coordinates are put on comparable scales before they are embedded.
alpha and delta are risk probabilities living near 0, so they enter as logs; B
is a complexity budget in nats/operators, so it enters as log1p; kappa is
already in [0, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

P_FIELDS = ("alpha", "delta", "B", "kappa")


@dataclass(frozen=True)
class OperatingPoint:
    """alpha  tolerated false-discovery risk
    delta  allowed ordinary-utility degradation
    B      scientific-program complexity budget
    kappa  exploratory (1) versus confirmatory (0) mode
    """
    alpha: float = 0.05
    delta: float = 0.02
    B: float = 8.0
    kappa: float = 0.5

    def as_vector(self) -> torch.Tensor:
        return torch.tensor([
            math.log(max(self.alpha, 1e-6)),
            math.log(max(self.delta, 1e-6)),
            math.log1p(max(self.B, 0.0)),
            float(self.kappa),
        ], dtype=torch.float32)

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(d: dict) -> "OperatingPoint":
        return OperatingPoint(**{k: float(d[k]) for k in P_FIELDS if k in d})

    @staticmethod
    def confirmatory() -> "OperatingPoint":
        return OperatingPoint(alpha=0.01, delta=0.005, B=4.0, kappa=0.0)

    @staticmethod
    def exploratory() -> "OperatingPoint":
        return OperatingPoint(alpha=0.20, delta=0.10, B=16.0, kappa=1.0)


class OperatingPointEncoder(nn.Module):
    """p -> a token in the model's width. Batched or shared across rows."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(len(P_FIELDS), hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, p) -> torch.Tensor:
        """Accepts an OperatingPoint, a (4,) tensor, or a (B, 4) tensor.

        Returns (1, d_model) for the shared case and (B, d_model) per-row. A
        gradient flows back to a (B, 4) tensor input, which is what step 9's
        operating-point sampling will differentiate through.
        """
        if isinstance(p, OperatingPoint):
            v = p.as_vector().to(next(self.parameters()).device).unsqueeze(0)
        else:
            v = torch.as_tensor(p)
            v = v.to(dtype=torch.float32, device=next(self.parameters()).device)
            if v.dim() == 1:
                v = v.unsqueeze(0)
        if v.shape[-1] != len(P_FIELDS):
            raise ValueError("operating point must have %d coordinates, got %s"
                             % (len(P_FIELDS), tuple(v.shape)))
        return self.norm(self.net(v))
