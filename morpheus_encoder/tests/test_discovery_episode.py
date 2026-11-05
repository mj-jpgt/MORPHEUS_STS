"""Focused contracts for the observed-only discovery episode boundary."""
from types import SimpleNamespace
import inspect

import numpy as np
import pytest

from morpheus_encoder.discovery_episode import (
    DiscoveryEpisode, EpisodePartitions, world_to_episode,
    SCALE_FIELDS, validate_scale,
)


def _partitions(n=12):
    return EpisodePartitions(np.arange(0, 4), np.arange(4, 8), np.arange(8, n))


def _episode():
    return DiscoveryEpisode(
        modalities={"rna": np.arange(36, dtype=np.float32).reshape(12, 3),
                    "image": np.ones((12, 2), dtype=np.float32)},
        response=np.linspace(0, 1, 12),
        partitions=_partitions(),
        context={"coords": np.arange(24).reshape(12, 2),
                 "time": np.arange(12),
                 "neighbors": [[(i + 1) % 12] for i in range(12)],
                 "scale": {"microns_per_pixel": 0.5}},
        metadata={"domain": "spatial", "packet": "fixed-01"},
    )


def test_episode_is_deterministic_readable_and_receipt_hashable():
    a, b = _episode(), _episode()
    assert a.content_hash() == b.content_hash()
    assert a.modalities["rna"].flags.writeable is False
    assert a.context["coords"].flags.writeable is False
    assert a.context["neighbors"][0] == (1,)
    receipt = a.receipt_identity()
    assert receipt["content_sha256"] == a.content_hash()
    assert receipt["partition_sizes"] == {"proposal": 4, "replication": 4, "null": 4}


def test_generic_matrix_and_roles_remain_observed_and_disjoint():
    episode = _episode()
    matrix = episode.observed_matrix()
    context = episode.numeric_context_matrix()
    p_x, p_y, p_c = episode.split_arrays("proposal")
    r_x, r_y, r_c = episode.split_arrays("replication")
    assert matrix.shape == (12, 5)
    assert context.shape[0] == 12
    assert p_x.shape == (4, 5) and r_x.shape == (4, 5)
    assert not np.intersect1d(p_y, r_y).size
    assert p_c.shape[0] == len(p_y) and r_c.shape[0] == len(r_y)


def test_content_hash_covers_observed_response_context_and_partitions():
    a = _episode()
    changed = DiscoveryEpisode(
        modalities=a.modalities, response=np.arange(12), partitions=_partitions(),
        context=a.context, metadata=a.metadata,
    )
    assert changed.content_hash() != a.content_hash()
    changed_context = DiscoveryEpisode(
        modalities=a.modalities, response=a.response, partitions=_partitions(),
        context={**a.context, "scale": {"microns_per_pixel": 0.25}}, metadata=a.metadata,
    )
    assert changed_context.content_hash() != a.content_hash()


@pytest.mark.parametrize("partitions", [
    EpisodePartitions(np.array([0]), np.array([1]), np.array([2])),
])
def test_partitions_must_be_disjoint_and_exhaustive(partitions):
    with pytest.raises(ValueError, match="exhaust"):
        DiscoveryEpisode({"x": np.ones((4, 2))}, partitions)
    with pytest.raises(ValueError, match="disjoint"):
        DiscoveryEpisode({"x": np.ones((3, 2))},
                         EpisodePartitions(np.array([0]), np.array([1]), np.array([1])))


class _ObservedWorld:
    """Raises if conversion ever attempts to inspect benchmark oracle fields."""

    name, family, domain = "observed", "real-substrate", "spatial"

    def __init__(self):
        n = 12
        self.modalities = {"rna": np.arange(n * 3, dtype=np.float32).reshape(n, 3)}
        self.target = np.linspace(1, 2, n)
        self.env = SimpleNamespace(role="proposal", labels={"donor": np.array(["a", "b"] * 6)})
        self.structure = SimpleNamespace(
            coords=np.arange(n * 2).reshape(n, 2), time=np.arange(n), unit=None,
            neighbors=[np.array([(i + 1) % n]) for i in range(n)],
        )

    def __getattribute__(self, name):
        if name in {"latents", "truth"}:
            raise AssertionError("episode conversion touched a benchmark oracle field")
        return object.__getattribute__(self, name)


def test_world_conversion_is_observed_only_and_preserves_context():
    episode = world_to_episode(_ObservedWorld(), partitions=_partitions(),
                               scale={"microns_per_pixel": 0.5})
    assert episode.response.shape == (12,)
    assert episode.context["coords"].shape == (12, 2)
    assert episode.context["time"].shape == (12,)
    assert episode.context["environment"].shape == (12, 2)
    assert episode.context["scale"] == {"microns_per_pixel": 0.5}
    source = inspect.getsource(world_to_episode)
    assert ".latents" not in source and ".truth" not in source


def test_empty_null_partition_requires_the_replication_structured_flag():
    partitions = EpisodePartitions(np.arange(0, 6), np.arange(6, 12), np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="null_source"):
        DiscoveryEpisode({"x": np.ones((12, 2))}, partitions,
                         response=np.linspace(0, 1, 12), metadata={"domain": "spatial"})
    episode = DiscoveryEpisode(
        {"x": np.ones((12, 2))}, partitions, response=np.linspace(0, 1, 12),
        metadata={"domain": "spatial", "null_source": "replication_structured"},
    )
    assert episode.partitions.null.size == 0
    assert episode.receipt_identity()["partition_sizes"]["null"] == 0


# ==========================================================================
# E4: `validate_scale` -- advisory, never enforced by `DiscoveryEpisode`
# itself (an episode with no `scale` metadata, or an incomplete one, is
# still a perfectly valid episode; see `_episode()` above, which has no
# `metadata["scale"]` at all and constructs fine).
# ==========================================================================

def _full_scale_record():
    return {"entity_type": "spot", "spatial_support": "55um_visium_spot",
           "temporal_support": "single_timepoint", "modality": "spatial_transcriptomics",
           "biological_scale": "tissue_neighborhood",
           "containment": {"spot": "tissue_section"}}


def test_validate_scale_accepts_a_full_record():
    report = validate_scale(_full_scale_record())
    assert report["valid"] is True
    assert report["missing"] == []
    assert set(report["fields"]) == set(SCALE_FIELDS)


def test_validate_scale_names_the_missing_field():
    incomplete = _full_scale_record()
    del incomplete["biological_scale"]
    report = validate_scale(incomplete)
    assert report["valid"] is False
    assert report["missing"] == ["biological_scale"]


def test_episode_construction_does_not_require_or_enforce_scale():
    # `_episode()` declares `context["scale"]` with only one key and no
    # `metadata["scale"]` at all -- construction must not call, or be
    # gated by, `validate_scale`.
    episode = _episode()
    assert "scale" not in episode.metadata


def test_episode_refuses_nonfinite_or_misaligned_observations():
    with pytest.raises(ValueError, match="non-finite"):
        DiscoveryEpisode({"x": np.array([[1.0], [np.nan], [3.0]])},
                         EpisodePartitions(np.array([0]), np.array([1]), np.array([2])))
    with pytest.raises(ValueError, match="scientific units"):
        DiscoveryEpisode({"x": np.ones((3, 1)), "z": np.ones((4, 1))},
                         EpisodePartitions(np.array([0]), np.array([1]), np.array([2])))
