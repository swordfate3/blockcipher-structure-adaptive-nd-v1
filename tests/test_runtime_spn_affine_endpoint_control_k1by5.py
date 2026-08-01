from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_runtime_spn_affine_endpoint_control_k1by5 import (
    render_k1by5_svg,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    materialize_ordered_primitive_payload,
    permute_program_source_endpoints_affine,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_endpoint_control_k1by5 as k1by5,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_permutation_control_identifiability_k1by4 as k1by4,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256


def test_k1by5_config_and_inherited_authority_are_frozen() -> None:
    config = k1by5.load_and_validate_config()
    datasets, _paths, checks, correct, affine = k1by5.load_authority(config)

    assert set(datasets) == {2, 3}
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
    assert affine.control == k1by5.CONTROL
    assert affine.semantic_sha256 != correct.semantic_sha256


def test_k1by5_source_digest_mismatch_fails_closed() -> None:
    config = deepcopy(k1by5.load_and_validate_config())
    config["source"]["digests"]["gate.json"] = "0" * 64

    _datasets, _paths, checks, _correct, _affine = k1by5.load_authority(config)

    assert checks["k1by4_gate.json_digest_exact"] is False


def test_affine_endpoint_control_is_bijective_and_splits_cell_bundles() -> None:
    config = k1by5.load_and_validate_config()
    _datasets, _paths, checks, correct, affine = k1by5.load_authority(config)
    _truth, inverse = materialize_ordered_primitive_payload(affine)
    endpoint_map = tuple((5 * endpoint + 1) % 64 for endpoint in range(64))

    assert all(checks.values())
    assert sorted(endpoint_map) == list(range(64))
    assert all(
        len({endpoint_map[4 * cell + role] // 4 for role in range(4)}) >= 2
        for cell in range(16)
    )
    assert torch.all(inverse.sum(dim=-1) == 1)
    assert torch.all(inverse.sum(dim=-2) == 1)
    assert affine.expert_usage == k1by4.EXPECTED_EXPERT_USAGE

    with pytest.raises(ValueError, match="invertible"):
        permute_program_source_endpoints_affine(
            correct,
            multiplier=4,
            offset=1,
        )


def test_k1by5_small_tap_audit_does_not_modify_source_cache() -> None:
    config = k1by5.load_and_validate_config()
    datasets, paths, checks, correct, affine = k1by5.load_authority(config)
    assert all(checks.values())
    feature_path = paths["k1by4_authority_seed2_features"]
    digest_before = file_sha256(feature_path)

    correct_taps = k1by4.extract_histogram_taps(
        datasets[2][0][:17],
        correct,
        batch_size=6,
    )
    affine_taps = k1by4.extract_histogram_taps(
        datasets[2][0][:17],
        affine,
        batch_size=6,
    )
    rows = k1by4.compare_histogram_taps(correct_taps, affine_taps)

    assert len(rows) == 4
    assert all(row["samples_total"] == 17 for row in rows)
    assert all(row["pairs_per_sample"] == 16 for row in rows)
    assert file_sha256(feature_path) == digest_before


def test_k1by5_gate_routes_are_frozen() -> None:
    config = k1by5.load_and_validate_config()

    passed = k1by5.adjudicate(
        config=config,
        result_rows=_synthetic_rows(equal_rate=0.10, pooled_l1=0.02),
        source_checks={"source": True},
        source_unchanged=True,
    )
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("affine_endpoint_control_ready")
    assert passed["all_taps_identifiable"] is True

    held_rows = _synthetic_rows(equal_rate=0.10, pooled_l1=0.02)
    held_rows[0]["multiset_equal_rate"] = 1.0
    held = k1by5.adjudicate(
        config=config,
        result_rows=held_rows,
        source_checks={"source": True},
        source_unchanged=True,
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("affine_endpoint_control_not_identifiable")

    invalid = k1by5.adjudicate(
        config=config,
        result_rows=_synthetic_rows(equal_rate=0.10, pooled_l1=0.02),
        source_checks={"source": False},
        source_unchanged=True,
    )
    assert invalid["status"] == "invalid"


def test_k1by5_plot_explains_affine_control_in_plain_language(tmp_path: Path) -> None:
    gate = k1by5.adjudicate(
        config=k1by5.load_and_validate_config(),
        result_rows=_synthetic_rows(equal_rate=0.10, pooled_l1=0.02),
        source_checks={"source": True},
        source_unchanged=True,
    )
    output = tmp_path / "curves.svg"

    report = render_k1by5_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["experiment"] == "K1-BY5"
    assert "PRESENT 全局仿射端点控制审计" in svg
    assert "拆散每个 source cell" in svg
    assert "三幅图分别回答无位置、有位置和汇总差异" in svg
    assert "允许进入同预算神经归因" in svg


def _synthetic_rows(*, equal_rate: float, pooled_l1: float) -> list[dict]:
    return [
        {
            "run_id": k1by5.RUN_ID,
            "cipher": "PRESENT-80",
            "rounds": 7,
            "seed": seed,
            "condition": k1by5.CONTROL,
            "execution_step": step,
            "source_stage_index": stage,
            "tap": tap,
            "samples_total": 2048,
            "cells": 16,
            "pairs_per_sample": 16,
            "multiset_equal_rate": equal_rate,
            "pooled_summary_l1": pooled_l1,
            "ordered_histogram_l1": 0.20,
            "neural_training_performed": False,
            "optimizer_steps": 0,
        }
        for seed in (2, 3)
        for step, stage in enumerate((1, 0))
        for tap in k1by4.TAPS
    ]
