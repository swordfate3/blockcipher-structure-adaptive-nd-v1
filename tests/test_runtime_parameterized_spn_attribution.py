from __future__ import annotations

from copy import deepcopy

from blockcipher_nd.tasks.innovation1.runtime_parameterized_spn_attribution import (
    adjudicate_runtime_spn_r1,
    adjudicate_runtime_spn_r1a_cell_token,
    adjudicate_runtime_spn_r1b_position,
    adjudicate_runtime_spn_r1c_view_encoder,
    adjudicate_runtime_spn_r1d_cell_mixer,
    adjudicate_runtime_spn_r2a_e4_attribution,
    adjudicate_runtime_spn_r2d_sbox_scale,
    adjudicate_runtime_spn_r2e_sbox_location,
)


def _row(model: str, auc: float) -> dict:
    return {
        "model": model,
        "cipher": "GIFT-64",
        "rounds": 6,
        "seed": 0,
        "samples_per_class": 8192,
        "dataset_label_mode": "balanced_per_class",
        "pairs_per_sample": 4,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "difference_profile": "gift64_shen2024_spn_screen",
        "difference_member": 0,
        "train_key": 0,
        "validation_key": 1,
        "parameter_count": 100,
        "metrics": {"auc": auc},
        "training": {
            "epochs": 10,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "selected_checkpoint": "best",
            "train_rows": 16384,
            "validation_rows": 8192,
            "train_dataset_storage": "disk",
            "validation_dataset_storage": "disk",
        },
    }


def _inputs() -> tuple[list[dict], list[dict], dict]:
    runtime = [
        _row("gift64_runtime_spn_true", 0.56),
        _row("gift64_runtime_spn_corrupted", 0.54),
        _row("gift64_runtime_spn_independent", 0.53),
    ]
    anchor = [_row("gift_cross_spn_typed_cell_true", 0.55)]
    r0 = {
        "status": "pass",
        "decision": "innovation1_runtime_spn_r0_readiness_passed",
    }
    return runtime, anchor, r0


def test_runtime_spn_r1_gate_passes_only_attributed_same_protocol_result() -> None:
    runtime, anchor, r0 = _inputs()
    gate = adjudicate_runtime_spn_r1(
        run_id="pass",
        cipher="GIFT-64",
        runtime_rows=runtime,
        anchor_rows=anchor,
        r0_gate=r0,
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_runtime_spn_r1_gate_holds_random_or_misordered_candidate() -> None:
    runtime, anchor, r0 = _inputs()
    runtime[0]["metrics"]["auc"] = 0.506
    runtime[1]["metrics"]["auc"] = 0.508
    runtime[2]["metrics"]["auc"] = 0.507
    gate = adjudicate_runtime_spn_r1(
        run_id="hold",
        cipher="GIFT-64",
        runtime_rows=runtime,
        anchor_rows=anchor,
        r0_gate=r0,
    )

    assert gate["status"] == "hold"
    assert not any(gate["research_checks"].values())
    assert "cell-token" in gate["next_action"]


def test_runtime_spn_r1_gate_fails_protocol_mismatch() -> None:
    runtime, anchor, r0 = _inputs()
    broken = deepcopy(runtime)
    broken[1]["negative_mode"] = "random_ciphertext"
    gate = adjudicate_runtime_spn_r1(
        run_id="invalid",
        cipher="GIFT-64",
        runtime_rows=broken,
        anchor_rows=anchor,
        r0_gate=r0,
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["same_data_protocol_as_anchor"] is False
    assert gate["protocol_checks"]["encrypted_random_plaintext_negatives"] is False


def test_cell_token_calibration_gate_rejects_random_misordered_candidate() -> None:
    current = _row("gift64_runtime_spn_true", 0.5024)
    cell_true = _row("gift64_runtime_cell_token_true", 0.5019)
    cell_corrupted = _row("gift64_runtime_cell_token_corrupted", 0.5095)
    for row in (current, cell_true, cell_corrupted):
        row["samples_per_class"] = 2048
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    cell_true["parameter_count"] = 200
    cell_corrupted["parameter_count"] = 200

    gate = adjudicate_runtime_spn_r1a_cell_token(
        run_id="r1a",
        rows=[current, cell_true, cell_corrupted],
        r1_gate={
            "status": "hold",
            "decision": "innovation1_runtime_spn_r1_seed0_not_supported",
        },
    )

    assert gate["status"] == "hold"
    assert all(gate["protocol_checks"].values())
    assert not any(gate["research_checks"].values())
    assert "position-identifiability" in gate["next_action"]


def test_position_audit_supports_coordinate_redesign_only_with_matched_gap() -> None:
    learned = _row("gift_cross_spn_typed_cell_true", 0.536)
    zero = _row("gift_cross_spn_typed_cell_no_position", 0.521)
    for row in (learned, zero):
        row["samples_per_class"] = 2048
        row["parameter_count"] = 200
        row["trainable_parameter_count"] = 200
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    gate = adjudicate_runtime_spn_r1b_position(
        run_id="r1b",
        rows=[learned, zero],
        r1a_gate={
            "status": "hold",
            "decision": (
                "innovation1_runtime_spn_cell_token_calibration_not_supported"
            ),
        },
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert "coordinate" in gate["next_action"]


def test_position_audit_holds_when_position_gap_is_small() -> None:
    learned = _row("gift_cross_spn_typed_cell_true", 0.523)
    zero = _row("gift_cross_spn_typed_cell_no_position", 0.519)
    for row in (learned, zero):
        row["samples_per_class"] = 2048
        row["parameter_count"] = 200
        row["trainable_parameter_count"] = 200
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    gate = adjudicate_runtime_spn_r1b_position(
        run_id="r1b",
        rows=[learned, zero],
        r1a_gate={
            "status": "hold",
            "decision": (
                "innovation1_runtime_spn_cell_token_calibration_not_supported"
            ),
        },
    )

    assert gate["status"] == "hold"
    assert gate["research_checks"]["learned_position_auc_at_least_0p520"]
    assert not gate["research_checks"][
        "learned_position_exceeds_zero_by_0p010"
    ]
    assert "typed fusion" in gate["next_action"]


def test_view_encoder_audit_requires_matched_separate_view_gain() -> None:
    separate = _row("gift_cross_spn_typed_cell_no_position", 0.535)
    shared = _row("gift_cross_spn_typed_cell_shared_view_encoder", 0.520)
    for row in (separate, shared):
        row["samples_per_class"] = 2048
        row["parameter_count"] = 200
        row["trainable_parameter_count"] = 200
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    gate = adjudicate_runtime_spn_r1c_view_encoder(
        run_id="r1c",
        rows=[separate, shared],
        r1b_gate={
            "status": "hold",
            "decision": "innovation1_runtime_spn_position_identity_not_supported",
        },
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert "view-role" in gate["next_action"]


def test_view_encoder_audit_holds_without_separate_view_gain() -> None:
    separate = _row("gift_cross_spn_typed_cell_no_position", 0.526)
    shared = _row("gift_cross_spn_typed_cell_shared_view_encoder", 0.524)
    for row in (separate, shared):
        row["samples_per_class"] = 2048
        row["parameter_count"] = 200
        row["trainable_parameter_count"] = 200
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    gate = adjudicate_runtime_spn_r1c_view_encoder(
        run_id="r1c",
        rows=[separate, shared],
        r1b_gate={
            "status": "hold",
            "decision": "innovation1_runtime_spn_position_identity_not_supported",
        },
    )

    assert gate["status"] == "hold"
    assert not gate["research_checks"][
        "separate_view_exceeds_shared_by_0p010"
    ]
    assert "Token-Mixer" in gate["next_action"]


def _r1d_rows(fixed_auc: float, equivariant_auc: float) -> list[dict]:
    fixed = _row("gift_cross_spn_typed_cell_shared_view_encoder", fixed_auc)
    equivariant = _row(
        "gift_cross_spn_typed_cell_equivariant_mixer", equivariant_auc
    )
    for row in (fixed, equivariant):
        row["samples_per_class"] = 2048
        row["parameter_count"] = 200
        row["trainable_parameter_count"] = 200
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    return [fixed, equivariant]


def _r1c_hold() -> dict:
    return {
        "status": "hold",
        "decision": "innovation1_runtime_spn_typed_view_identity_not_supported",
    }


def test_cell_mixer_audit_supports_fixed_mixer_dependency() -> None:
    gate = adjudicate_runtime_spn_r1d_cell_mixer(
        run_id="r1d", rows=_r1d_rows(0.540, 0.525), r1c_gate=_r1c_hold()
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("fixed_cell_mixer_dependency_supported")
    assert all(gate["protocol_checks"].values())


def test_cell_mixer_audit_promotes_strong_equivariant_backbone() -> None:
    gate = adjudicate_runtime_spn_r1d_cell_mixer(
        run_id="r1d", rows=_r1d_rows(0.536, 0.534), r1c_gate=_r1c_hold()
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("equivariant_e4_backbone_supported")
    assert "true/corrupted/independent" in gate["next_action"]


def test_cell_mixer_audit_holds_when_both_mixers_lack_signal() -> None:
    gate = adjudicate_runtime_spn_r1d_cell_mixer(
        run_id="r1d", rows=_r1d_rows(0.511, 0.508), r1c_gate=_r1c_hold()
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("cell_mixer_calibration_not_supported")


def _r2a_rows(true_auc: float, corrupted_auc: float, independent_auc: float) -> list[dict]:
    rows = [
        _row("gift64_runtime_e4_equivariant_true", true_auc),
        _row("gift64_runtime_e4_equivariant_corrupted", corrupted_auc),
        _row("gift64_runtime_e4_equivariant_independent", independent_auc),
    ]
    for row in rows:
        row["samples_per_class"] = 2048
        row["parameter_count"] = 200
        row["trainable_parameter_count"] = 200
        row["input_bit_order"] = "project_msb_to_runtime_lsb"
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    return rows


def _r1d_equivariant_pass() -> dict:
    return {
        "status": "pass",
        "decision": "innovation1_runtime_spn_equivariant_e4_backbone_supported",
        "aucs": {"equivariant": 0.540},
    }


def test_r2a_gate_passes_only_signal_preserving_attributed_result() -> None:
    gate = adjudicate_runtime_spn_r2a_e4_attribution(
        run_id="r2a",
        rows=_r2a_rows(0.542, 0.532, 0.530),
        r1d_gate=_r1d_equivariant_pass(),
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_r2a_gate_holds_when_true_topology_is_not_attributed() -> None:
    gate = adjudicate_runtime_spn_r2a_e4_attribution(
        run_id="r2a",
        rows=_r2a_rows(0.539, 0.540, 0.537),
        r1d_gate=_r1d_equivariant_pass(),
    )

    assert gate["status"] == "hold"
    assert gate["research_checks"]["true_auc_at_least_0p520"]
    assert not gate["research_checks"]["true_exceeds_corrupted_by_0p005"]
    assert "do not scale" in gate["next_action"]


def test_r2a_gate_rejects_protocol_mismatch() -> None:
    rows = _r2a_rows(0.542, 0.532, 0.530)
    rows[2]["negative_mode"] = "random_ciphertext"
    gate = adjudicate_runtime_spn_r2a_e4_attribution(
        run_id="r2a",
        rows=rows,
        r1d_gate=_r1d_equivariant_pass(),
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["same_data_protocol"] is False
    assert gate["protocol_checks"]["encrypted_random_plaintext_negatives"] is False


def test_r2a_gate_rejects_unrecorded_runtime_bit_order() -> None:
    rows = _r2a_rows(0.542, 0.532, 0.530)
    rows[0].pop("input_bit_order")
    gate = adjudicate_runtime_spn_r2a_e4_attribution(
        run_id="r2a",
        rows=rows,
        r1d_gate=_r1d_equivariant_pass(),
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["runtime_bit_order_adapter_recorded"] is False


def _r2d_inputs(candidate_auc: float) -> tuple[list[dict], list[dict], dict, dict]:
    candidate = _r2a_rows(candidate_auc, 0.0, 0.0)[0]
    baseline_rows = _r2a_rows(0.534461, 0.493224, 0.496503)
    common_options = {
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
    }
    candidate["training"]["model_options"] = {
        **common_options,
        "sbox_context_scale": 0.1,
    }
    for row in baseline_rows:
        row["training"]["model_options"] = dict(common_options)
    r2c_gate = {
        "status": "hold",
        "decision": "innovation1_runtime_spn_r2a_topology_attribution_not_supported",
        "protocol_checks": {"valid": True},
    }
    r1d_gate = _r1d_equivariant_pass()
    return [candidate], baseline_rows, r2c_gate, r1d_gate


def test_r2d_sbox_scale_gate_supports_anchor_recovery() -> None:
    candidate, baseline, r2c_gate, r1d_gate = _r2d_inputs(0.536)

    gate = adjudicate_runtime_spn_r2d_sbox_scale(
        run_id="r2d",
        candidate_rows=candidate,
        r2c_rows=baseline,
        r2c_gate=r2c_gate,
        r1d_gate=r1d_gate,
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert "true/corrupted/no-topology" in gate["next_action"]


def test_r2d_sbox_scale_gate_holds_below_anchor_tolerance() -> None:
    candidate, baseline, r2c_gate, r1d_gate = _r2d_inputs(0.534)

    gate = adjudicate_runtime_spn_r2d_sbox_scale(
        run_id="r2d",
        candidate_rows=candidate,
        r2c_rows=baseline,
        r2c_gate=r2c_gate,
        r1d_gate=r1d_gate,
    )

    assert gate["status"] == "hold"
    assert gate["research_checks"]["candidate_auc_at_least_0p520"]
    assert not gate["research_checks"][
        "candidate_within_r1d_anchor_tolerance"
    ]
    assert "do not scale" in gate["next_action"]


def test_r2d_sbox_scale_gate_rejects_post_hoc_scale() -> None:
    candidate, baseline, r2c_gate, r1d_gate = _r2d_inputs(0.536)
    candidate[0]["training"]["model_options"]["sbox_context_scale"] = 0.2

    gate = adjudicate_runtime_spn_r2d_sbox_scale(
        run_id="r2d",
        candidate_rows=candidate,
        r2c_rows=baseline,
        r2c_gate=r2c_gate,
        r1d_gate=r1d_gate,
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["preregistered_nonzero_scale"] is False


def _r2e_inputs(candidate_auc: float) -> tuple[list[dict], list[dict], dict, dict, dict]:
    candidate, baseline, r2c_gate, r1d_gate = _r2d_inputs(candidate_auc)
    candidate[0]["training"]["model_options"].pop("sbox_context_scale")
    candidate[0]["training"]["model_options"]["sbox_context_mode"] = "late_pair"
    r2d_gate = {
        "status": "hold",
        "decision": "innovation1_runtime_spn_sbox_scale_calibration_not_supported",
        "protocol_checks": {"valid": True},
    }
    return candidate, baseline, r2c_gate, r2d_gate, r1d_gate


def test_r2e_sbox_location_gate_supports_anchor_recovery() -> None:
    candidate, baseline, r2c_gate, r2d_gate, r1d_gate = _r2e_inputs(0.536)

    gate = adjudicate_runtime_spn_r2e_sbox_location(
        run_id="r2e",
        candidate_rows=candidate,
        r2c_rows=baseline,
        r2c_gate=r2c_gate,
        r2d_gate=r2d_gate,
        r1d_gate=r1d_gate,
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert "true/corrupted/no-topology" in gate["next_action"]


def test_r2e_sbox_location_gate_holds_below_anchor_tolerance() -> None:
    candidate, baseline, r2c_gate, r2d_gate, r1d_gate = _r2e_inputs(0.534)

    gate = adjudicate_runtime_spn_r2e_sbox_location(
        run_id="r2e",
        candidate_rows=candidate,
        r2c_rows=baseline,
        r2c_gate=r2c_gate,
        r2d_gate=r2d_gate,
        r1d_gate=r1d_gate,
    )

    assert gate["status"] == "hold"
    assert gate["research_checks"]["candidate_auc_at_least_0p520"]
    assert not gate["research_checks"][
        "candidate_within_r1d_anchor_tolerance"
    ]


def test_r2e_sbox_location_gate_rejects_scale_change() -> None:
    candidate, baseline, r2c_gate, r2d_gate, r1d_gate = _r2e_inputs(0.536)
    candidate[0]["training"]["model_options"]["sbox_context_scale"] = 0.1

    gate = adjudicate_runtime_spn_r2e_sbox_location(
        run_id="r2e",
        candidate_rows=candidate,
        r2c_rows=baseline,
        r2c_gate=r2c_gate,
        r2d_gate=r2d_gate,
        r1d_gate=r1d_gate,
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["only_sbox_location_option_changed"] is False
