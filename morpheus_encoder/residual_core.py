"""Information-preserving structural residual core for MORPHEUS.

This module is deliberately *after* a domain observation model.  Its input is
an already computed feature tensor ``F``--for example tokens emitted by
``FrozenPathologyAdapter`` or a gene-expression adapter--not raw H&E pixels,
RNA counts, or a domain-specific foundation model.  It therefore provides a
small, common structural layer without silently replacing a strong upstream
representation.

The invariant is intentionally stronger than an identity initialisation:
``out.source is F``.  Consumers can always read the untouched source feature
through a direct path, while atomic, relational, and continuous residual
channels are available alongside it.  Slots may consume ``out.combined`` as a
candidate-proposal convenience, but are never an information bottleneck.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ResidualCoreOutput:
    """Outputs of :class:`ResidualStructuralCore`.

    ``source`` is the exact input object, not a projected, normalised, copied,
    or detached version.  All tensors use ``(batch, entity, feature)`` layout.
    ``combined`` is suitable for a downstream proposer/adapter; consumers
    requiring a guarantee against source distortion should use ``source``.
    """

    source: torch.Tensor
    atomic: torch.Tensor
    relational: torch.Tensor
    continuous: torch.Tensor
    combined: torch.Tensor

    def shapes(self) -> dict:
        return {name: tuple(getattr(self, name).shape)
                for name in ("source", "atomic", "relational", "continuous", "combined")}


class _ResidualMLP(nn.Module):
    """Two-layer residual branch with a zero-initialised output projection.

    Zero output at construction means adding the core cannot initially alter a
    source-only consumer.  The final projection still receives a gradient on
    the first optimisation step; later the hidden layer does as well.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def initialise_output_orthogonal(self) -> None:
        """Give a residual channel a small, response-free live start.

        The historical exact-zero output projection preserved an identity
        baseline, but it also made similarity objectives locally blind to a
        channel that was zero in both augmented views.  The caller calibrates
        the resulting scale from input features; this method only chooses a
        deterministic orthogonal orientation and never sees a response.
        """
        projection = self.net[-1]
        if not isinstance(projection, nn.Linear):  # defensive for future refactors
            raise TypeError("residual output projection must be Linear")
        nn.init.orthogonal_(projection.weight)
        nn.init.zeros_(projection.bias)


class ResidualStructuralCore(nn.Module):
    """Add structural channels without compressing away source features.

    Parameters
    ----------
    d_source:
        Width of the frozen/upstream entity feature ``F``.
    d_residual:
        Width contributed by each of atomic, relational, and continuous
        channels.  Defaults to ``d_source``.
    coordinate_dim:
        Number of optional continuous coordinates (e.g. x/y or x/y/time).
        Coordinates are only used by the continuous branch; omitted
        coordinates are represented by zeros, allowing the same core to run on
        non-spatial data.

    Notes
    -----
    ``neighbors`` is an optional integer tensor ``(B, N, K)`` containing
    within-sample entity indices.  ``-1`` denotes padding.  The relation
    branch receives the mean valid neighbour feature and is permutation
    invariant over the K neighbours.  It never constructs a dense N-by-N
    adjacency matrix.  ``lag_indices`` is an optional ``(B, N, L)`` table
    whose entries point from an entity to the same spatial location at the
    preceding predeclared time lags.  It lets the relation branch see
    ``LAG(SPACE(F))`` without estimating a dense space--time graph or using
    the response.  ``-1`` is padding/no predecessor.
    """

    def __init__(self, d_source: int, d_residual: int | None = None,
                 hidden: int | None = None, coordinate_dim: int = 3,
                 temporal_lags: int = 3):
        super().__init__()
        if d_source <= 0:
            raise ValueError("d_source must be positive")
        if coordinate_dim < 0:
            raise ValueError("coordinate_dim must be non-negative")
        self.d_source = int(d_source)
        self.d_residual = int(d_source if d_residual is None else d_residual)
        self.coordinate_dim = int(coordinate_dim)
        self.temporal_lags = int(temporal_lags)
        if self.temporal_lags < 0:
            raise ValueError("temporal_lags must be non-negative")
        hidden = int(hidden or max(self.d_source, self.d_residual))
        self.atomic_branch = _ResidualMLP(self.d_source, self.d_residual, hidden)
        # Source + local neighbourhood + a fixed, coordinate-native lag grid.
        # The branch has one architecture in every arm; a non-temporal packet
        # receives zero lag channels rather than a different model.
        self.relational_branch = _ResidualMLP(
            (2 + self.temporal_lags) * self.d_source, self.d_residual, hidden)
        self.continuous_branch = _ResidualMLP(
            self.d_source + self.coordinate_dim, self.d_residual, hidden)

    @property
    def d_combined(self) -> int:
        return self.d_source + 3 * self.d_residual

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.dim() != 3:
            raise ValueError("features must have shape (B, N, d_source), got %s"
                             % (tuple(features.shape),))
        if features.shape[-1] != self.d_source:
            raise ValueError("features have width %d, expected d_source=%d"
                             % (features.shape[-1], self.d_source))

    def _neighbor_mean(self, features: torch.Tensor,
                       neighbors: torch.Tensor | None) -> torch.Tensor:
        B, N, D = features.shape
        if neighbors is None:
            return torch.zeros_like(features)
        idx = torch.as_tensor(neighbors, device=features.device, dtype=torch.long)
        if idx.dim() != 3 or tuple(idx.shape[:2]) != (B, N):
            raise ValueError("neighbors must have shape (B, N, K) matching features; got %s"
                             % (tuple(idx.shape),))
        valid = idx >= 0
        if bool((idx[valid] >= N).any()):
            raise ValueError("neighbors contains an entity index outside [0, N)")
        safe = idx.clamp(min=0)
        # `features[b, safe[b, n, k]]` without a Python loop or B*N*N matrix.
        batch = torch.arange(B, device=features.device)[:, None, None]
        gathered = features[batch, safe]
        weights = valid.to(dtype=features.dtype).unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (gathered * weights).sum(dim=2) / denom

    def _coordinates(self, features: torch.Tensor,
                     coordinates: torch.Tensor | None) -> torch.Tensor:
        B, N, _ = features.shape
        if self.coordinate_dim == 0:
            return features.new_zeros(B, N, 0)
        if coordinates is None:
            return features.new_zeros(B, N, self.coordinate_dim)
        coords = torch.as_tensor(coordinates, device=features.device,
                                 dtype=features.dtype)
        if tuple(coords.shape) != (B, N, self.coordinate_dim):
            raise ValueError("coordinates must have shape (B, N, %d), got %s"
                             % (self.coordinate_dim, tuple(coords.shape)))
        return coords

    def _lagged_neighborhoods(self, neighbour_mean: torch.Tensor,
                              lag_indices: torch.Tensor | None) -> torch.Tensor:
        """Return predeclared lagged neighbourhood means without response use."""
        B, N, D = neighbour_mean.shape
        if self.temporal_lags == 0:
            return neighbour_mean.new_zeros(B, N, 0)
        if lag_indices is None:
            return neighbour_mean.new_zeros(B, N, self.temporal_lags * D)
        idx = torch.as_tensor(lag_indices, device=neighbour_mean.device,
                              dtype=torch.long)
        if tuple(idx.shape) != (B, N, self.temporal_lags):
            raise ValueError("lag_indices must have shape (B, N, %d), got %s"
                             % (self.temporal_lags, tuple(idx.shape)))
        valid = idx >= 0
        if bool((idx[valid] >= N).any()):
            raise ValueError("lag_indices contains an entity index outside [0, N)")
        safe = idx.clamp(min=0)
        batch = torch.arange(B, device=neighbour_mean.device)[:, None, None]
        gathered = neighbour_mean[batch, safe]
        gathered = gathered * valid.to(neighbour_mean.dtype).unsqueeze(-1)
        return gathered.reshape(B, N, self.temporal_lags * D)

    def forward(self, features: torch.Tensor, *, neighbors: torch.Tensor | None = None,
                coordinates: torch.Tensor | None = None,
                lag_indices: torch.Tensor | None = None) -> ResidualCoreOutput:
        self._validate_features(features)
        source = features  # Direct identity path: no norm, projection, or detach.
        atomic = self.atomic_branch(source)
        neighbor_mean = self._neighbor_mean(source, neighbors)
        lagged = self._lagged_neighborhoods(neighbor_mean, lag_indices)
        relational = self.relational_branch(torch.cat([source, neighbor_mean, lagged], dim=-1))
        coords = self._coordinates(source, coordinates)
        continuous = self.continuous_branch(torch.cat([source, coords], dim=-1))
        combined = torch.cat([source, atomic, relational, continuous], dim=-1)
        return ResidualCoreOutput(source=source, atomic=atomic, relational=relational,
                                  continuous=continuous, combined=combined)

    def initialise_response_free_residuals(
            self, features: torch.Tensor, *, neighbors: torch.Tensor | None = None,
            coordinates: torch.Tensor | None = None,
            lag_indices: torch.Tensor | None = None, target_std: float
    ) -> dict[str, float]:
        """Initialise and calibrate live residual channels from D_P features.

        ``target_std`` is declared by the caller relative to a fixed atomic
        basis.  No response, truth, candidate score, or external environment
        is consulted.  Each output projection is first oriented orthogonally,
        then rescaled so its median D_P coordinate standard deviation matches
        the requested target.  This preserves a small residual contribution
        while preventing zero-channel gradient starvation.
        """
        if target_std <= 0 or not torch.isfinite(torch.tensor(float(target_std))):
            raise ValueError("target_std must be finite and positive")
        self._validate_features(features)
        branches = {
            "atomic": (self.atomic_branch, features),
            "relational": (self.relational_branch, torch.cat([
                features, self._neighbor_mean(features, neighbors),
                self._lagged_neighborhoods(self._neighbor_mean(features, neighbors), lag_indices),
            ], dim=-1)),
            "continuous": (self.continuous_branch, torch.cat([
                features, self._coordinates(features, coordinates),
            ], dim=-1)),
        }
        metadata: dict[str, float] = {}
        with torch.no_grad():
            for name, (branch, branch_input) in branches.items():
                branch.initialise_output_orthogonal()
                output = branch(branch_input)
                scale = output.std(dim=(0, 1), unbiased=False).median().clamp_min(1e-8)
                multiplier = float(target_std) / float(scale)
                projection = branch.net[-1]
                projection.weight.mul_(multiplier)
                projection.bias.mul_(multiplier)
                metadata[f"{name}_precalibration_std"] = float(scale)
                metadata[f"{name}_multiplier"] = float(multiplier)
                metadata[f"{name}_postcalibration_std"] = float(
                    branch(branch_input).std(dim=(0, 1), unbiased=False).median())
        return metadata
