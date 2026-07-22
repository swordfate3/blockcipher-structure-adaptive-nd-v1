from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_position_bound_spn_rescnn import (
    MODEL_SPECS,
    OPD1_GATE_SHA256,
    OPD1_PLAINTEXTS_SHA256,
    OPD1_RELEASE_DECISION,
    OPD1_RUN_ID,
    PositionBoundSpnResCnnConfig,
    adjudicate_position_bound,
    authorize_round_extension_from_opd1_gate,
    prepare_position_bound_data,
    validate_position_bound_contract,
)


def _opd1_gate() -> dict[str, object]:
    return {
        "run_id": OPD1_RUN_ID,
        "status": "hold",
        "decision": OPD1_RELEASE_DECISION,
        "protocol_checks": {"valid": True},
        "execution_checks": {"valid": True},
        "metrics": {
            "mean_auc_by_model": {
                "selected8_position_head_spn_rescnn_exact_p_true_output": (
                    0.999996158392159
                )
            }
        },
    }


def _training(
    config: PositionBoundSpnResCnnConfig,
    aucs: tuple[float, float, float, float, float],
) -> dict[str, object]:
    auc_by_model = {
        model: auc for (model, _, _), auc in zip(MODEL_SPECS, aucs, strict=True)
    }
    rows = [
        {
            "model": model,
            "msb_index": bit,
            "threshold_accuracy": 0.70,
            "majority_accuracy": 0.50,
            "accuracy_minus_majority": 0.20,
            "auc": auc_by_model[model],
            "mse": 0.20,
        }
        for model, _, _ in MODEL_SPECS
        for bit in config.selected_msb_indices
    ]
    return {
        "rows": rows,
        "summaries": [{"model": model} for model in auc_by_model],
        "history": [
            {"model": model, "epoch": epoch}
            for model in auc_by_model
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(5)],
    }


def _tiny_r4_config() -> PositionBoundSpnResCnnConfig:
    return PositionBoundSpnResCnnConfig(
        run_id="i2_output_prediction_opf1_present_r4_position_bound_spn_rescnn_test",
        mode="round_extension_smoke",
        rounds=4,
        train_rows=8,
        test_rows=8,
        rescnn_channels=4,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
        maximum_parameter_gap=1.0,
    )


def test_round_extension_config_changes_only_rounds() -> None:
    r3 = PositionBoundSpnResCnnConfig.formal(device="cpu")
    r4 = PositionBoundSpnResCnnConfig.round_extension(device="cpu")

    ignored = {"run_id", "mode", "rounds"}
    assert {
        key: value for key, value in r3.__dict__.items() if key not in ignored
    } == {key: value for key, value in r4.__dict__.items() if key not in ignored}
    assert (r3.rounds, r4.rounds) == (3, 4)
    with pytest.raises(ValueError, match="mode-matched PRESENT round count"):
        PositionBoundSpnResCnnConfig(mode="round_extension", rounds=3)


def test_round_extension_reuses_opd1_plaintexts_but_changes_targets(
    tmp_path: Path,
) -> None:
    r3 = PositionBoundSpnResCnnConfig(
        **{
            **_tiny_r4_config().__dict__,
            "run_id": "r3",
            "mode": "smoke",
            "rounds": 3,
        }
    )
    r4 = _tiny_r4_config()
    r3_data = prepare_position_bound_data(r3, tmp_path / "r3")
    r4_data = prepare_position_bound_data(r4, tmp_path / "r4")

    assert np.array_equal(r3_data["plaintexts"], r4_data["plaintexts"])
    assert np.array_equal(r3_data["features"], r4_data["features"])
    assert not np.array_equal(r3_data["full_targets"], r4_data["full_targets"])
    assert all(validate_position_bound_contract(r4, r4_data).values())


def test_round_extension_authority_is_frozen_to_completed_opd1() -> None:
    authorize_round_extension_from_opd1_gate(_opd1_gate())
    wrong = _opd1_gate()
    wrong["decision"] = "wrong"
    with pytest.raises(ValueError, match="requires the frozen completed OPD1 gate"):
        authorize_round_extension_from_opd1_gate(wrong)


def test_r4_output_gate_can_pass_when_exact_and_wrong_p_are_tied() -> None:
    config = PositionBoundSpnResCnnConfig.round_extension(device="cpu")
    gate = adjudicate_position_bound(
        config,
        {"valid": True},
        _training(config, (0.60, 0.60, 0.70, 0.70, 0.50)),
        reference_gate=_opd1_gate(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_position_bound_r4_output_supported"
    assert all(gate["metrics"]["round_extension_checks"].values())
    assert gate["metrics"]["formal_checks"][
        "candidate_minus_wrong_mean_auc_at_least_0_020"
    ] is False
    assert gate["metrics"]["r3_reference_exact_p_mean_auc"] == pytest.approx(
        0.999996158392159
    )


def test_r4_output_gate_records_boundary_when_candidate_is_near_chance() -> None:
    config = PositionBoundSpnResCnnConfig.round_extension(device="cpu")
    gate = adjudicate_position_bound(
        config,
        {"valid": True},
        _training(config, (0.51, 0.51, 0.52, 0.52, 0.50)),
        reference_gate=_opd1_gate(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_position_bound_r4_boundary_observed"
    assert gate["metrics"]["passed_output_bit_count"] == 0


def test_opf1_result_index_name_is_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opf1_present_r4_position_bound_spn_rescnn_"
        "smoke_seed7_20260722"
    )
    assert name == "创新2 OPF1：PRESENT四轮位置绑定网络同协议真实密文输出预测"


def test_opf1_remote_package_is_matched_cached_and_windows_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (
            root
            / "configs/experiment/innovation2/innovation2_output_prediction_"
            "opf1_present_r4_position_bound_spn_rescnn_key7.json"
        ).read_text(encoding="utf-8")
    )
    remote = json.loads(
        (
            root
            / "configs/remote/innovation2_output_prediction_opf1_present_r4_"
            "position_bound_spn_rescnn_key7_gpu0_20260722.json"
        ).read_text(encoding="utf-8")
    )
    run = (
        root
        / "configs/remote/generated/run_i2_output_prediction_opf1_present_r4_"
        "position_bound_spn_rescnn_key7_gpu0_20260722.cmd"
    ).read_text(encoding="utf-8")
    launch = (
        root
        / "configs/remote/generated/launch_i2_output_prediction_opf1_present_r4_"
        "position_bound_spn_rescnn_key7_gpu0_20260722.cmd"
    ).read_text(encoding="utf-8")
    monitor = (
        root
        / "configs/remote/generated/monitor_i2_output_prediction_opf1_present_r4_"
        "position_bound_spn_rescnn_key7_gpu0_20260722.sh"
    ).read_text(encoding="utf-8")
    windows_scripts = run + launch

    assert plan["only_changed_variable"] == {
        "field": "rounds",
        "opd1_value": 3,
        "opf1_value": 4,
    }
    assert plan["source_authorities"]["opd1"]["gate_sha256"] == OPD1_GATE_SHA256
    assert (
        plan["source_authorities"]["opd1"]["plaintexts_sha256"]
        == OPD1_PLAINTEXTS_SHA256
    )
    assert remote["rounds"] == 4
    assert remote["train_total_rows"] == 131072
    assert remote["test_total_rows"] == 65536
    assert remote["expected_cache_rows"] == 196608
    assert remote["opd1_plaintexts_sha256"] == OPD1_PLAINTEXTS_SHA256
    assert "--mode round_extension" in run
    assert "--opd1-gate" in run
    assert "plaintext_file_sha256_matches_opd1" in run
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in windows_scripts.lower()
    assert "EnableDelayedExpansion" not in windows_scripts
    assert "!" not in windows_scripts
    assert "G:\\lxy" in windows_scripts
    assert "git status --porcelain" in windows_scripts
    assert "source_expected_commit.txt" in windows_scripts
    assert OPD1_GATE_SHA256 in run and OPD1_GATE_SHA256 in monitor
    assert OPD1_PLAINTEXTS_SHA256 in run and OPD1_PLAINTEXTS_SHA256 in monitor
    assert "outputs/remote_results/" in monitor
