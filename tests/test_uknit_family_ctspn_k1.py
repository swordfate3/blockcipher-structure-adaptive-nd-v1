from __future__ import annotations

from pathlib import Path

import pytest
import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    K0_DECISION,
    K0_RUN_ID,
    RUN_ID,
    build_ctspn_k1_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1.csv"
)


def _tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


@pytest.fixture(scope="module")
def readiness() -> tuple[list[dict[str, object]], dict[str, object]]:
    return build_ctspn_k1_readiness(
        run_id=RUN_ID,
        tasks=_tasks(),
        k0_gate={
            "run_id": K0_RUN_ID,
            "status": "pass",
            "decision": K0_DECISION,
        },
        k0_validation={"run_id": K0_RUN_ID, "status": "pass"},
        present_gate={
            "run_id": (
                "i1_rtg3b_present80_one_to_one_formal_1000000_seed1_20260727"
            ),
            "status": "pass",
            "decision": "innovation1_rtg3b_present_seed1_formal_adjudicated",
        },
    )


def test_ctspn_k1_real_plan_passes_all_zero_training_readiness_checks(
    readiness: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    manifests, gate = readiness

    assert len(manifests) == 8
    assert gate["status"] == "pass"
    assert gate["implementation_ready"] is True
    assert gate["optimizer_step_authorized"] is True
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert all(gate["protocol_checks"].values())
    assert all(gate["evidence_checks"].values())
    assert {
        row["trainable_parameter_count"]
        for row in manifests
        if row["model"] == "runtime_spn_ct_k1_canonical_true"
    } == {438702}


def test_ctspn_k1_readiness_proves_both_widths_and_controls(
    readiness: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    _, gate = readiness
    evidence = gate["structural_evidence"]

    assert evidence["cross_width_strict_state_dict_load"] is True
    assert evidence["no_cipher_identity_tensor"] is True
    for cipher in ("uknit64", "dialga128"):
        row = evidence["per_cipher"][cipher]
        assert row["canonical_unit_inverse_exact"] is True
        assert row["canonical_edge_relation_exact"] is True
        assert row["cell_relabel_max_logit_error"] <= 1e-6
        assert row["controls"]["fingerprints_deterministic_and_distinct"] is True
        assert row["controls"]["strict_state_dict_load"] is True
        assert row["controls"]["factors_survive_checkpoint_load"] is True
        assert row["controls"]["control_logits_noncollapsed"] is True


def test_ctspn_model_factory_preserves_geometry_across_block_widths() -> None:
    common = {
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "temporal_hidden_dim": 76,
        "dropout": 0.0,
    }
    uknit = build_model(
        "runtime_spn_ct_k1_canonical_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options={
            **common,
            "runtime_structure_path": "configs/runtime/spn/uknit64.json",
            "runtime_round_start": 3,
        },
    )
    dialga = build_model(
        "runtime_spn_ct_k1_canonical_true",
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options={
            **common,
            "runtime_structure_path": "configs/runtime/spn/dialga128.json",
            "runtime_round_start": 2,
        },
    )

    assert model_metadata(uknit)["trainable_parameter_count"] == 438702
    assert model_metadata(dialga)["trainable_parameter_count"] == 438702
    assert [
        (name, tuple(value.shape)) for name, value in uknit.state_dict().items()
    ] == [(name, tuple(value.shape)) for name, value in dialga.state_dict().items()]
    dialga_factor = dialga.canonical_factor_manifest_sha256
    dialga.load_state_dict(uknit.state_dict(), strict=True)
    assert dialga.canonical_factor_manifest_sha256 == dialga_factor
    assert uknit(torch.zeros(2, 512)).shape == (2, 1)
    assert dialga(torch.zeros(2, 1024)).shape == (2, 1)


def test_ctspn_readiness_fails_closed_without_k0_evidence() -> None:
    _, gate = build_ctspn_k1_readiness(
        run_id="invalid",
        tasks=[],
        k0_gate={},
        k0_validation={},
    )

    assert gate["status"] == "fail"
    assert gate["implementation_ready"] is False
    assert gate["optimizer_step_authorized"] is False
    assert gate["protocol_checks"]["k0_gate_and_validation_pass"] is False
    assert gate["protocol_checks"]["eight_row_frozen_panel"] is False
