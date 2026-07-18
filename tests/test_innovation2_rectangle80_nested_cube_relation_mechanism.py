from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli.plot_innovation2_rectangle80_nested_cube_relation_mechanism import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.rectangle80_nested_cube_monotonic_readiness import (
    make_nested_chains,
)
from blockcipher_nd.tasks.innovation2.rectangle80_nested_cube_relation_mechanism import (
    CHAIN_COUNT,
    FEATURE_COUNT,
    OUTPUT_BITS,
    PREFIX_DIM,
    NestedCubeRelationConfig,
    adjudicate_relation_checks,
    build_relation_features,
    isotonic_project_triplets,
    make_relation_maps,
    monotonic_violation_count,
    validate_relation_maps,
)


def _chains() -> list[dict[str, object]]:
    rng = random.Random(95)
    seen: set[tuple[int, ...]] = set()
    structures = []
    while len(structures) < CHAIN_COUNT:
        active = tuple(sorted(rng.sample(range(64), 8)))
        if active in seen:
            continue
        seen.add(active)
        structures.append({"active_bits": list(active)})
    return [asdict(chain) for chain in make_nested_chains(structures)]


def test_e95_protocol_is_frozen() -> None:
    assert NestedCubeRelationConfig().ridge_lambda == 1e-3
    with pytest.raises(ValueError, match="frozen"):
        NestedCubeRelationConfig(ridge_lambda=1e-2)


def test_relation_maps_are_split_preserving_and_wrong_map_breaks_all_edges() -> None:
    chains = _chains()

    maps = make_relation_maps(chains)
    checks = validate_relation_maps(chains, maps)

    assert checks["true_map_is_identity"]
    assert checks["shuffled_is_derangement"]
    assert checks["wrong_is_derangement"]
    assert checks["shuffled_preserves_split"]
    assert checks["wrong_preserves_split"]
    assert checks["wrong_all_four_relations_false"]
    assert checks["wrong_relation_violation_rate"] == 1.0
    assert 0.0 <= checks["shuffled_wrong_identical_rate"] <= 1.0


def test_relation_features_keep_equal_width_and_only_context_changes() -> None:
    prefix = np.arange(
        CHAIN_COUNT * 3 * OUTPUT_BITS * PREFIX_DIM, dtype=np.float64
    ).reshape(CHAIN_COUNT, 3, OUTPUT_BITS, PREFIX_DIM)
    identity = {index: index for index in range(CHAIN_COUNT)}

    independent = build_relation_features(prefix, None)
    true = build_relation_features(prefix, identity)

    assert independent.shape == true.shape == (
        CHAIN_COUNT,
        3,
        OUTPUT_BITS,
        FEATURE_COUNT,
    )
    np.testing.assert_array_equal(independent[..., :PREFIX_DIM], prefix)
    assert np.all(independent[..., PREFIX_DIM : 3 * PREFIX_DIM] == 0.0)
    np.testing.assert_array_equal(
        true[0, 1, :, PREFIX_DIM : 2 * PREFIX_DIM], prefix[0, 0]
    )
    np.testing.assert_array_equal(
        true[0, 1, :, 2 * PREFIX_DIM : 3 * PREFIX_DIM], prefix[0, 2]
    )


def test_three_point_isotonic_projection_removes_all_violations() -> None:
    scores = np.zeros((CHAIN_COUNT, 3, OUTPUT_BITS), dtype=np.float64)
    scores[0, :, 0] = (3.0, 1.0, 2.0)
    scores[0, :, 1] = (1.0, 3.0, 2.0)

    projected = isotonic_project_triplets(scores)

    np.testing.assert_allclose(projected[0, :, 0], (2.0, 2.0, 2.0))
    np.testing.assert_allclose(projected[0, :, 1], (1.0, 2.5, 2.5))
    assert monotonic_violation_count(scores) == 2
    assert monotonic_violation_count(projected) == 0


def test_e95_gate_separates_protocol_and_attribution_failures() -> None:
    passed = {"check": True}
    failed = {"check": False}

    assert adjudicate_relation_checks(passed, passed, passed)[:2] == (
        "pass",
        "innovation2_rectangle80_nested_cube_relation_mechanism_ready",
    )
    assert adjudicate_relation_checks(passed, passed, failed)[:2] == (
        "hold",
        "innovation2_rectangle80_nested_cube_relation_not_attributed",
    )
    assert adjudicate_relation_checks(failed, passed, passed)[:2] == (
        "fail",
        "innovation2_rectangle80_nested_cube_relation_protocol_invalid",
    )


def test_plot_writes_clear_chinese_e95_svg(tmp_path: Path) -> None:
    reports = {
        mode: {
            "train_auc": auc + 0.02,
            "validation_auc": auc,
            "raw_monotonic_violations": 100,
            "final_monotonic_violations": 0 if "nesting" in mode or mode == "wrong_superset" else 100,
        }
        for mode, auc in {
            "independent_dimension": 0.70,
            "true_nesting": 0.80,
            "shuffled_nesting": 0.72,
            "wrong_superset": 0.71,
            "true_unconstrained": 0.79,
        }.items()
    }
    summary = {
        "gate": {
            "decision": "innovation2_rectangle80_nested_cube_relation_mechanism_ready",
            "metrics": {
                "reports": reports,
                "margins": {
                    "true_minus_independent": 0.10,
                    "true_minus_shuffled": 0.08,
                    "true_minus_wrong_superset": 0.09,
                    "true_minus_unconstrained": 0.01,
                    "true_train_minus_validation": 0.02,
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E95" in svg
    assert "正确的7/8/9-bit cube嵌套关系" in svg
    assert "不训练神经网络" in svg
