"""Broad, response-free representation arms for the general discovery stack.

This module intentionally separates a representation from scientific-object
proposal.  It knows only observed feature matrices and optional contextual
neighbourhoods: never a response, a benchmark relation label, or oracle truth.
The same identity-preserving encoder is used for the source, SSL, predictive,
and MORPHEUS arms; only the response-free training objective differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


RepresentationArm = Literal["source", "reconstruction", "ssl", "predictive", "morpheus_legacy", "morpheus_factorized", "morpheus"]


@dataclass(frozen=True)
class RepresentationConfig:
    d_input: int
    d_output: int | None = None
    hidden: int | None = None
    temperature: float = 0.2
    mask_probability: float = 0.2
    context_weight: float = 0.25
    variance_weight: float = 0.05
    covariance_weight: float = 0.01
    variance_target_std: float = 1.0

    def __post_init__(self) -> None:
        if self.d_input < 1:
            raise ValueError("d_input must be positive")
        if self.d_output is not None and self.d_output < 1:
            raise ValueError("d_output must be positive when supplied")
        if not 0.0 < self.temperature:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.mask_probability < 1.0:
            raise ValueError("mask_probability must lie in [0, 1)")
        if self.context_weight < 0.0:
            raise ValueError("context_weight must be non-negative")
        if self.variance_weight < 0.0 or self.covariance_weight < 0.0:
            raise ValueError("variance/covariance weights must be non-negative")
        if self.variance_target_std <= 0.0:
            raise ValueError("variance_target_std must be positive")


class IdentityPreservingEncoder(nn.Module):
    """A broad encoder with an explicit source bypass.

    If source and output widths agree, the source path begins as the exact
    identity.  Otherwise it uses the only unavoidable learned projection.  A
    zero-initialised residual branch prevents a newly introduced discovery
    representation from silently discarding a foundation representation before
    response-free training can justify a change.
    """

    def __init__(self, cfg: RepresentationConfig):
        super().__init__()
        self.cfg = cfg
        d_out = cfg.d_output or cfg.d_input
        hidden = cfg.hidden or max(cfg.d_input, d_out)
        self.d_output = int(d_out)
        if self.d_output == cfg.d_input:
            self.source = nn.Identity()
        else:
            self.source = nn.Linear(cfg.d_input, self.d_output, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(cfg.d_input, hidden), nn.GELU(), nn.Linear(hidden, self.d_output)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        # The self and contextual targets answer different questions.  Sharing
        # one head made a context loss able to overwrite the coordinate needed
        # to predict the same observation.  The trunk remains shared; only the
        # two target maps are factorised.
        self.predictor = nn.Sequential(nn.Linear(self.d_output, hidden), nn.GELU(),
                                       nn.Linear(hidden, self.d_output))
        self.context_predictor = nn.Sequential(
            nn.Linear(self.d_output, hidden), nn.GELU(), nn.Linear(hidden, self.d_output)
        )
        self.decoder = nn.Linear(self.d_output, cfg.d_input)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.cfg.d_input:
            raise ValueError("x must have shape (observations, d_input)")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite observed values")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.source(x) + self.residual(x)

    @staticmethod
    def _normal(z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=-1, eps=1e-8)

    def make_masked_view(self, x: torch.Tensor) -> torch.Tensor:
        keep = torch.rand_like(x) >= self.cfg.mask_probability
        # Inverted masking keeps the marginal scale stable without introducing
        # a modality-specific augmentation policy.
        return x * keep.to(x.dtype) / max(1.0 - self.cfg.mask_probability, 1e-6)

    def _infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        a, b = self._normal(z1), self._normal(z2)
        logits = a @ b.T / self.cfg.temperature
        target = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    @staticmethod
    def _mean_neighbours(z: torch.Tensor, neighbours: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor] | None:
        if neighbours is None:
            return None
        index = torch.as_tensor(neighbours, device=z.device, dtype=torch.long)
        if index.ndim != 2 or index.shape[0] != z.shape[0]:
            raise ValueError("neighbours must have shape (observations, degree)")
        valid = index >= 0
        if bool((index[valid] >= z.shape[0]).any()):
            raise ValueError("neighbour index exceeds observation count")
        gathered = z[index.clamp(min=0)]
        weight = valid.to(z.dtype).unsqueeze(-1)
        counts = weight.sum(1)
        valid_rows = counts.squeeze(-1) > 0
        # A declared grid with no actual neighbours has no contextual signal;
        # fall back to self prediction instead of teaching a zero-vector proxy.
        if not bool(valid_rows.any()):
            return None
        return (gathered * weight).sum(1) / counts.clamp_min(1.0), valid_rows

    def _variance_covariance_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """Generic VICReg-style spread penalty, with no relation vocabulary.

        It is applied to the broad observation representation, not to a named
        scientific object.  This only guards against representational collapse;
        it cannot encode a preferred AND/XOR/spatial relation.
        """
        if z.ndim != 2:
            raise ValueError("representation must have shape (observations, features)")
        if z.shape[0] < 2:
            # A one-row minibatch cannot estimate covariance.  Returning an
            # exact zero preserves finite training without inventing a signal.
            return z.new_zeros(())
        centered = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
        variance = F.relu(self.cfg.variance_target_std - std).mean()
        cov = centered.T @ centered / float(z.shape[0] - 1)
        off_diagonal = cov - torch.diag(torch.diagonal(cov))
        covariance = off_diagonal.square().sum() / max(int(z.shape[1]), 1)
        return self.cfg.variance_weight * variance + self.cfg.covariance_weight * covariance

    def _factorized_predictive_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                                    neighbours: torch.Tensor | None) -> torch.Tensor:
        """Self/context latent prediction with a stop-gradient target view."""
        target = self._normal(z2.detach())
        self_prediction = self._normal(self.predictor(z1))
        loss = F.mse_loss(self_prediction, target)
        context = self._mean_neighbours(target, neighbours)
        if context is not None:
            context_target, valid_rows = context
            context_prediction = self._normal(self.context_predictor(z1[valid_rows]))
            loss = loss + self.cfg.context_weight * F.mse_loss(
                context_prediction, self._normal(context_target[valid_rows]))
        return loss

    def _legacy_contextual_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                                neighbours: torch.Tensor | None) -> torch.Tensor:
        """Historical one-head contextual objective, retained as an ablation.

        It deliberately shares ``predictor`` between self and neighbour targets
        and omits the variance/covariance penalty.  This is the exact design
        fault Iteration 1 is intended to diagnose; it is not the MORPHEUS
        method claim.
        """
        target = self._normal(z2.detach())
        prediction = self._normal(self.predictor(z1))
        loss = F.mse_loss(prediction, target)
        context = self._mean_neighbours(target, neighbours)
        if context is not None:
            context_target, valid_rows = context
            loss = loss + self.cfg.context_weight * F.mse_loss(
                prediction[valid_rows], self._normal(context_target[valid_rows]))
        return loss

    def objective(self, x: torch.Tensor, arm: RepresentationArm,
                  *, neighbours: torch.Tensor | None = None) -> torch.Tensor:
        """Return a response-free objective for the selected representation arm."""
        if arm not in {"reconstruction", "ssl", "predictive", "morpheus_legacy", "morpheus_factorized", "morpheus"}:
            raise ValueError("source has no trainable representation objective")
        self._validate_input(x)
        v1, v2 = self.make_masked_view(x), self.make_masked_view(x)
        z1, z2 = self(v1), self(v2)
        if arm == "reconstruction":
            return F.mse_loss(self.decoder(z1), x)
        if arm == "ssl":
            return self._infonce(z1, z2)
        # JEPA-like latent prediction, using a stop-gradient target from an
        # independently masked observed view. No response or object family is
        # supplied to this path.
        predictive = self._factorized_predictive_loss(z1, z2, None)
        if arm == "predictive":
            return predictive
        if arm == "morpheus_legacy":
            return self._legacy_contextual_loss(z1, z2, neighbours)
        factorized = self._factorized_predictive_loss(z1, z2, neighbours)
        if arm == "morpheus_factorized":
            return factorized
        return factorized + self._variance_covariance_penalty(z1)


@torch.no_grad()
def represent(encoder: IdentityPreservingEncoder, x: torch.Tensor,
              arm: RepresentationArm) -> torch.Tensor:
    """Expose a common representation interface, preserving source exactly."""
    if arm == "source":
        return x
    encoder.eval()
    return encoder(x)


def fit_response_free(encoder: IdentityPreservingEncoder, x: torch.Tensor,
                      arm: RepresentationArm, *, neighbours: torch.Tensor | None = None,
                      steps: int = 200, learning_rate: float = 1e-3,
                      generator: torch.Generator | None = None,
                      batch_size: int = 4096) -> list[float]:
    """Train one arm with observed inputs only and return its loss trace.

    When `x` has more rows than `batch_size` and no neighbour tensor is given,
    each step draws a uniform random row subset: the contrastive objective
    forms an n x n logits matrix, which for a real cohort (~20k spots) does
    not fit on the GPU. Rows are indexed, never reordered, so every arm sees
    the same data; with neighbours supplied the full batch is kept because
    neighbour indices refer to full-batch rows.
    """
    if arm == "source":
        return []
    if steps < 1:
        raise ValueError("steps must be positive")
    if generator is not None:
        # The caller controls reproducibility; masking uses the global torch
        # generator, so materialise the caller's deterministic state here.
        torch.random.set_rng_state(generator.get_state())
    encoder.train()
    optimiser = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
    history: list[float] = []
    subsample = neighbours is None and x.shape[0] > batch_size
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        xb = x[torch.randperm(x.shape[0], device=x.device)[:batch_size]] if subsample else x
        loss = encoder.objective(xb, arm, neighbours=neighbours)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite response-free representation loss")
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach().cpu()))
    return history
