from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.cli.plot_runtime_spn_permutation_control_identifiability_k1by4 import (
    render_k1by4_svg,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    materialize_ordered_primitive_payload,
    permute_program_source_roles,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_permutation_control_identifiability_k1by4 as k1by4,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256


def test_k1by4_config_and_source_authority_are_frozen() -> None:
    config = k1by4.load_and_validate_config()
    datasets, _paths, checks, programs = k1by4.load_authority(config)

    assert set(datasets) == {2, 3}
    assert set(programs) == {"correct", *k1by4.CONTROLS}
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]


def test_k1by4_source_digest_mismatch_fails_closed() -> None:
    config = deepcopy(k1by4.load_and_validate_config())
    config["source"]["digests"]["gate.json"] = "0" * 64

    _datasets, _paths, checks, _programs = k1by4.load_authority(config)

    assert checks["gate.json_digest_exact"] is False


def test_source_role_corruption_preserves_permutation_expert_geometry() -> None:
    config = k1by4.load_and_validate_config()
    correct = k1by4.build_programs(config)["correct"]
    corrupted = permute_program_source_roles(
        correct,
        role_permutation=(1, 3, 0, 2),
    )
    _truth, inverse = materialize_ordered_primitive_payload(corrupted)

    assert corrupted.control == k1by4.SOURCE_ROLE_CONTROL
    assert corrupted.semantic_sha256 != correct.semantic_sha256
    assert corrupted.expert_usage == k1by4.EXPECTED_EXPERT_USAGE
    assert torch.all(inverse.sum(dim=-1) == 1)
    assert torch.all(inverse.sum(dim=-2) == 1)

    with pytest.raises(ValueError, match="non-identity"):
        permute_program_source_roles(correct, role_permutation=(0, 1, 2, 3))


def test_k1by4_taps_are_integer_histograms_and_do_not_modify_cache() -> None:
    config = k1by4.load_and_validate_config()
    datasets, paths, checks, programs = k1by4.load_authority(config)
    assert all(checks.values())
    feature_path = paths["seed2_features"]
    digest_before = file_sha256(feature_path)

    taps = k1by4.extract_histogram_taps(
        datasets[2][0][:19],
        programs[k1by4.SOURCE_ROLE_CONTROL],
        batch_size=7,
    )

    assert set(taps) == {
        (0, 1, "inverse_linear"),
        (0, 1, "post_inverse_sbox"),
        (1, 0, "inverse_linear"),
        (1, 0, "post_inverse_sbox"),
    }
    assert all(value.shape == (19, 16, 16) for value in taps.values())
    assert all(value.dtype == np.uint8 for value in taps.values())
    assert all(np.all(value.sum(axis=-1) == 16) for value in taps.values())
    assert file_sha256(feature_path) == digest_before


def test_k1by4_gate_routes_are_frozen() -> None:
    config = k1by4.load_and_validate_config()
    source_checks = {"source": True}

    learned = k1by4.adjudicate(
        config=config,
        result_rows=_synthetic_rows(current=(0.20, 0.05), source=(0.10, 0.08)),
        source_checks=source_checks,
        source_unchanged=True,
    )
    assert learned["status"] == "pass"
    assert learned["decision"].endswith("learned_pooling_audit_required")

    preferred = k1by4.adjudicate(
        config=config,
        result_rows=_synthetic_rows(current=(1.0, 0.0), source=(0.10, 0.08)),
        source_checks=source_checks,
        source_unchanged=True,
    )
    assert preferred["status"] == "pass"
    assert preferred["decision"].endswith("source_role_control_preferred")

    held = k1by4.adjudicate(
        config=config,
        result_rows=_synthetic_rows(current=(1.0, 0.0), source=(1.0, 0.0)),
        source_checks=source_checks,
        source_unchanged=True,
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("permutation_expert_hold")

    invalid = k1by4.adjudicate(
        config=config,
        result_rows=_synthetic_rows(current=(1.0, 0.0), source=(0.10, 0.08)),
        source_checks={"source": False},
        source_unchanged=True,
    )
    assert invalid["status"] == "invalid"


def test_k1by4_plot_uses_plain_language_labels(tmp_path: Path) -> None:
    gate = k1by4.adjudicate(
        config=k1by4.load_and_validate_config(),
        result_rows=_synthetic_rows(current=(1.0, 0.0), source=(0.10, 0.08)),
        source_checks={"source": True},
        source_unchanged=True,
    )
    output = tmp_path / "curves.svg"

    report = render_k1by4_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["experiment"] == "K1-BY4"
    assert "PRESENT 置换控制可识别性审计" in svg
    assert "不训练网络" in svg
    assert "现有错误目标绑定" in svg
    assert "source-role 错接" in svg
    assert "完整 cell 搬移是弱控制" in svg


def _synthetic_rows(
    *,
    current: tuple[float, float],
    source: tuple[float, float],
) -> list[dict]:
    rows = []
    values = {
        k1by4.CURRENT_CONTROL: current,
        k1by4.SOURCE_ROLE_CONTROL: source,
    }
    for seed in (2, 3):
        for condition in k1by4.CONTROLS:
            equal_rate, pooled_l1 = values[condition]
            for execution_step, stage_index in enumerate((1, 0)):
                for tap in k1by4.TAPS:
                    rows.append(
                        {
                            "run_id": k1by4.RUN_ID,
                            "cipher": "PRESENT-80",
                            "rounds": 7,
                            "seed": seed,
                            "condition": condition,
                            "execution_step": execution_step,
                            "source_stage_index": stage_index,
                            "tap": tap,
                            "samples_total": 2048,
                            "cells": 16,
                            "pairs_per_sample": 16,
                            "multiset_equal_rate": equal_rate,
                            "pooled_summary_l1": pooled_l1,
                            "ordered_histogram_l1": 0.1,
                            "neural_training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows
