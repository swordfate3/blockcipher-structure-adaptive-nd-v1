from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_present_linear_subspaces import (
    render_linear_subspace_svg,
)
from blockcipher_nd.tasks.innovation2 import integral_linear_subspace_diversity as linear


def test_rref_canonicalizes_equivalent_bases_and_points_are_unique() -> None:
    basis = tuple(1 << bit for bit in range(4))
    mixed = (basis[3] ^ basis[0], basis[2], basis[1] ^ basis[0], basis[0])
    assert linear.canonical_rref_basis(mixed) == basis
    points = linear.enumerate_subspace_points(basis)
    assert points.dtype == np.uint64
    assert len(points) == 16
    assert len(np.unique(points)) == 16
    assert set(int(value) for value in points) == set(range(16))


def test_random_subspaces_are_full_rank_unique_and_disjoint_from_anchors() -> None:
    structures = linear.make_linear_subspaces(random_subspaces=8, seed=13002)
    assert len(structures) == 12
    assert len({structure.signature for structure in structures}) == 12
    assert [structure.role for structure in structures[:4]] == [
        "coordinate_anchor"
    ] * 4
    assert all(len(structure.basis) == 16 for structure in structures)
    assert all(
        linear.canonical_rref_basis(structure.basis) == structure.basis
        for structure in structures
    )


def test_vectorized_present_matches_scalar_and_known_vector() -> None:
    assert linear.scalar_vectorized_fixture_matches() is True


def test_disk_cache_resumes_without_regenerating_rows(
    monkeypatch, tmp_path: Path
) -> None:
    config = linear.LinearSubspaceAuditConfig(
        run_id="cache-test",
        mode="smoke",
        random_subspaces=1,
        keys=4,
        key_chunk_size=2,
    )
    structure = linear.coordinate_subspaces()[0]
    keys = linear.make_audit_keys(config)
    calls: list[int] = []

    def fake_xor(points, round_keys):
        calls.append(round_keys.shape[1])
        return np.arange(round_keys.shape[1], dtype=np.uint64)

    monkeypatch.setattr(linear, "present_output_xor_words", fake_xor)
    first = linear.run_cached_subspace_parities(
        config,
        structure,
        keys=keys,
        cache_root=tmp_path / "cache",
    )
    resumed = linear.run_cached_subspace_parities(
        config,
        structure,
        keys=keys,
        cache_root=tmp_path / "cache",
    )
    assert first["rows_generated"] == 4
    assert resumed["rows_generated"] == 0
    assert first["completed"].all()
    assert calls == [2, 2]
    assert (tmp_path / "cache" / "points.npy").stat().st_size > 0


def test_gate_separates_smoke_ready_full_ready_and_sparse() -> None:
    readiness = {"protocol": True}
    smoke = linear.adjudicate_linear_subspaces(
        linear.LinearSubspaceAuditConfig(
            run_id="smoke", mode="smoke", random_subspaces=2, keys=4
        ),
        readiness,
        nontrivial_count=0,
        distinct_signatures=0,
        mean_retention=0.0,
    )
    assert smoke["decision"] == (
        "innovation2_present_linear_subspace_readiness_passed"
    )
    audit_config = linear.LinearSubspaceAuditConfig(run_id="audit")
    ready = linear.adjudicate_linear_subspaces(
        audit_config,
        readiness,
        nontrivial_count=8,
        distinct_signatures=4,
        mean_retention=0.5,
    )
    assert ready["decision"] == (
        "innovation2_present_linear_subspace_kernel_family_ready"
    )
    sparse = linear.adjudicate_linear_subspaces(
        audit_config,
        readiness,
        nontrivial_count=7,
        distinct_signatures=7,
        mean_retention=1.0,
    )
    assert sparse["decision"] == (
        "innovation2_present_linear_subspace_kernel_family_too_sparse"
    )


def test_plot_renders_four_anchors_and_random_orientations(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "structure_id": f"coordinate_{index}",
                "role": "coordinate_anchor",
                "discovery_nullity": 4,
                "validation_nullity": 4,
                "joint_nullity": 3,
            }
        )
    for index in range(8):
        rows.append(
            {
                "structure_id": f"random_{index:02d}",
                "role": "random_orientation",
                "discovery_nullity": 2 + index % 3,
                "validation_nullity": 2 + index % 2,
                "joint_nullity": index % 2,
            }
        )
    gate = {
        "decision": "innovation2_present_linear_subspace_kernel_family_ready",
        "metrics": {
            "random_nontrivial_joint_kernel_count": 8,
            "distinct_nonzero_joint_kernel_signatures": 4,
            "mean_half_intersection_retention": 0.5,
        },
    }
    output = tmp_path / "curves.svg"
    render_linear_subspace_svg(rows, gate, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E30" in svg
    assert "64+64经验密钥" in svg
    assert "joint nullity" in svg
