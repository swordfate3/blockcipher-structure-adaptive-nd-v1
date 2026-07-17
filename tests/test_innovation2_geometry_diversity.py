from __future__ import annotations

from blockcipher_nd.tasks.innovation2.integral_geometry_diversity import (
    CyclicGeometryDiversityConfig,
    adjudicate_cyclic_geometry_diversity,
    cyclic_nibble_window,
)


def test_cyclic_window_wraps_at_state_boundary() -> None:
    assert cyclic_nibble_window(0) == tuple(range(16))
    assert cyclic_nibble_window(12) == tuple(range(48, 64))
    wrapped = cyclic_nibble_window(14)
    assert wrapped == tuple(list(range(0, 8)) + list(range(56, 64)))
    assert len(wrapped) == 16


def test_geometry_gate_requires_four_signatures_and_eight_structures() -> None:
    config = CyclicGeometryDiversityConfig(run_id="test")
    readiness = {"ok": True}

    passed = adjudicate_cyclic_geometry_diversity(
        config,
        [],
        readiness,
        distinct_signatures=4,
        nontrivial_structures=8,
    )
    held = adjudicate_cyclic_geometry_diversity(
        config,
        [],
        readiness,
        distinct_signatures=3,
        nontrivial_structures=16,
    )
    invalid = adjudicate_cyclic_geometry_diversity(
        config,
        [],
        {"ok": False},
        distinct_signatures=6,
        nontrivial_structures=16,
    )

    assert passed["decision"] == (
        "innovation2_cyclic_geometry_kernel_diversity_ready"
    )
    assert held["decision"] == (
        "innovation2_cyclic_geometry_kernel_diversity_insufficient"
    )
    assert invalid["decision"] == (
        "innovation2_cyclic_geometry_diversity_protocol_invalid"
    )
