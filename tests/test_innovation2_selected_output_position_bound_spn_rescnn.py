from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_selected_output_position_bound_spn_rescnn import (
    render_position_bound_spn_rescnn,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputPositionBoundResidualCnn,
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_position_bound_spn_rescnn import (
    MODEL_SPECS,
    OPC1_RELEASE_DECISION,
    OPC1_RUN_ID,
    OPN1_RELEASE_DECISION,
    OPN1_RUN_ID,
    PositionBoundSpnResCnnConfig,
    adjudicate_position_bound,
    authorize_from_source_gates,
    position_bound_parameter_counts,
    prepare_position_bound_data,
    train_position_bound_matrix,
    validate_position_bound_contract,
)


def _tiny_config() -> PositionBoundSpnResCnnConfig:
    return PositionBoundSpnResCnnConfig(
        train_rows=8,
        test_rows=8,
        rescnn_channels=4,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
        maximum_parameter_gap=1.0,
    )


def _source_gates() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "run_id": OPC1_RUN_ID,
            "status": "hold",
            "decision": OPC1_RELEASE_DECISION,
            "protocol_checks": {"valid": True},
            "execution_checks": {"valid": True},
        },
        {
            "run_id": OPN1_RUN_ID,
            "status": "pass",
            "decision": OPN1_RELEASE_DECISION,
            "protocol_checks": {"valid": True},
            "execution_checks": {"valid": True},
        },
    )


def _formal_training(
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


def test_position_bound_model_has_local_heads_and_eight_outputs() -> None:
    model = SelectedOutputPositionBoundResidualCnn(
        channels=4,
        stage_blocks=(1, 1, 1),
        source_for_destination=_present_topology_mapping("exact"),
    )

    assert len(model.head.heads) == 8
    assert all(head[0].in_features == 4 for head in model.head.heads)
    assert model(torch.zeros((2, 64))).shape == (2, 8)


def test_formal_position_heads_are_parameter_matched_and_topology_identifiable() -> (
    None
):
    config = PositionBoundSpnResCnnConfig()
    counts = position_bound_parameter_counts(config)
    local = {
        counts["position_head_rescnn_no_p"],
        counts["position_head_spn_rescnn_exact_p"],
        counts["position_head_spn_rescnn_wrong_p"],
    }

    assert local == {3_956_928}
    assert counts["rescnn"] == 3_955_904
    assert abs(next(iter(local)) - counts["rescnn"]) / counts["rescnn"] < 0.001
    selected = torch.tensor(config.selected_msb_indices)
    exact_sources = _present_topology_mapping("exact").index_select(0, selected)
    wrong_sources = _present_topology_mapping("wrong").index_select(0, selected)
    assert torch.all(exact_sources != wrong_sources)


def test_formal_mode_requires_opc1_hold_and_opn1_pass() -> None:
    authorize_from_source_gates(*_source_gates())
    opc1, opn1 = _source_gates()
    opn1["decision"] = "wrong"
    with pytest.raises(ValueError, match="requires frozen OPC1 hold and OPN1 pass"):
        authorize_from_source_gates(opc1, opn1)


def test_tiny_position_bound_matrix_is_complete_and_replayable(tmp_path: Path) -> None:
    config = _tiny_config()
    data = prepare_position_bound_data(config, tmp_path)
    checks = validate_position_bound_contract(config, data)
    training = train_position_bound_matrix(config, data, tmp_path)
    gate = adjudicate_position_bound(config, checks, training)

    assert all(checks.values())
    assert len(training["rows"]) == 40
    assert len(training["history"]) == 5
    assert len(training["checkpoints"]) == 5
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_position_bound_spn_rescnn_local_readiness_passed"
    )


def test_formal_gate_requires_all_anchors_and_controls() -> None:
    config = PositionBoundSpnResCnnConfig.formal(device="cpu")
    training = _formal_training(config, (0.60, 0.60, 0.70, 0.60, 0.50))

    gate = adjudicate_position_bound(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_position_bound_spn_rescnn_requires_confirmation"
    )


@pytest.mark.parametrize(
    "aucs",
    [
        (0.54, 0.53, 0.549, 0.52, 0.50),
        (0.695, 0.60, 0.70, 0.60, 0.50),
        (0.60, 0.695, 0.70, 0.60, 0.50),
        (0.60, 0.60, 0.70, 0.69, 0.50),
        (0.60, 0.60, 0.70, 0.60, 0.68),
    ],
)
def test_formal_gate_holds_when_any_mean_gate_fails(
    aucs: tuple[float, float, float, float, float],
) -> None:
    config = PositionBoundSpnResCnnConfig.formal(device="cpu")
    gate = adjudicate_position_bound(
        config,
        {"valid": True},
        _formal_training(config, aucs),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_position_bound_spn_rescnn_not_supported"


def test_position_bound_plot_has_plain_chinese_scope(tmp_path: Path) -> None:
    config = _tiny_config()
    rows = [
        {"model": model, "msb_index": bit, "auc": 0.50 + 0.01 * index}
        for index, (model, _, _) in enumerate(MODEL_SPECS)
        for bit in config.selected_msb_indices
    ]
    summary = {
        "metadata": {
            "mode": "smoke",
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "bit_rows": rows,
        "gate": {
            "decision": "innovation2_position_bound_spn_rescnn_local_readiness_passed"
        },
    }
    output = tmp_path / "curves.svg"

    render_position_bound_spn_rescnn(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "位置绑定" in svg
    assert "真实P位置头 - 全局头ResCNN" in svg
    assert "真实P位置头 - 无P位置头" in svg
    assert "不是四轮" in svg


def test_training_cli_import_does_not_require_matplotlib() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import blockcipher_nd.cli.run_innovation2_selected_output_position_bound_spn_rescnn
print('import=pass')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import=pass"


def test_result_index_names_opd1_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opd1_present_r3_position_bound_spn_rescnn_smoke_seed7_20260722"
    )
    assert name == "创新2 OPD1：PRESENT三轮位置绑定SPN-ResCNN真实密文输出预测"


def test_formal_remote_package_is_gate_owned_cached_and_windows_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    plan_path = root / (
        "configs/experiment/innovation2/"
        "innovation2_output_prediction_opd1_present_r3_position_bound_"
        "spn_rescnn_key7.json"
    )
    authority_path = root / (
        "configs/experiment/innovation2/authorities/"
        "innovation2_output_prediction_opn1_gate_20260722.json"
    )
    remote_config_path = root / (
        "configs/remote/innovation2_output_prediction_opd1_present_r3_"
        "position_bound_spn_rescnn_key7_gpu0_20260722.json"
    )
    run_path = root / (
        "configs/remote/generated/run_i2_output_prediction_opd1_present_r3_"
        "position_bound_spn_rescnn_key7_gpu0_20260722.cmd"
    )
    launch_path = root / (
        "configs/remote/generated/launch_i2_output_prediction_opd1_present_r3_"
        "position_bound_spn_rescnn_key7_gpu0_20260722.cmd"
    )
    monitor_path = root / (
        "configs/remote/generated/monitor_i2_output_prediction_opd1_present_r3_"
        "position_bound_spn_rescnn_key7_gpu0_20260722.sh"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    remote_config = json.loads(remote_config_path.read_text(encoding="utf-8"))
    run = run_path.read_text(encoding="utf-8")
    launch = launch_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")
    windows_scripts = run + launch
    opc1_sha = "ebb86a9feab6d2d9993937f5c0a7f4afe1bfe3597c8c1dff083956381e0310b4"
    opn1_sha = "887a7db3643e73bdda67958bcaae470881a09db25ab0ba5ff6c3d6bb0a2503d7"

    assert plan["source_authorities"]["opc1"]["status"] == "hold"
    assert plan["source_authorities"]["opc1"]["decision"] == OPC1_RELEASE_DECISION
    assert plan["source_authorities"]["opc1"]["gate_sha256"] == opc1_sha
    assert plan["source_authorities"]["opn1"]["status"] == "pass"
    assert plan["source_authorities"]["opn1"]["decision"] == OPN1_RELEASE_DECISION
    assert plan["source_authorities"]["opn1"]["gate_sha256"] == opn1_sha
    assert authority["run_id"] == OPN1_RUN_ID
    assert authority["decision"] == OPN1_RELEASE_DECISION
    assert plan["common"]["seed"] == 7
    assert plan["common"]["train_total_rows"] == 131072
    assert plan["common"]["test_total_rows"] == 65536
    assert plan["common"]["epochs"] == 100
    assert len(plan["rows"]) == 5
    assert len({row["parameters"] for row in plan["rows"][1:]}) == 1
    assert plan["final_gate"] == {
        "minimum_candidate_mean_auc": 0.55,
        "minimum_candidate_minus_global_mean_auc": 0.01,
        "minimum_candidate_minus_no_p_mean_auc": 0.01,
        "minimum_candidate_minus_wrong_mean_auc": 0.02,
        "minimum_candidate_minus_shuffle_mean_auc": 0.03,
        "minimum_joint_passed_bits": 4,
        "minimum_per_bit_candidate_auc": 0.55,
        "minimum_per_bit_candidate_minus_global_auc": 0.005,
        "minimum_per_bit_candidate_minus_no_p_auc": 0.005,
        "minimum_per_bit_candidate_minus_each_control_auc": 0.015,
        "minimum_per_bit_accuracy_minus_majority": 0.005,
    }
    assert {row["model"] for row in plan["rows"]} == set(remote_config["models"])
    assert remote_config["opc1_gate_sha256"] == opc1_sha
    assert remote_config["opn1_gate_sha256"] == opn1_sha
    assert remote_config["expected_result_rows"] == 40
    assert remote_config["expected_history_rows"] == 500
    assert remote_config["expected_checkpoints"] == 5
    assert remote_config["expected_cache_rows"] == 196608
    assert remote_config["remote_directory"] == "i2_opd1_poshead_k7_20260722"
    assert remote_config["archive_directory"] == "i2_opd1_poshead_k7_20260722"
    assert remote_config["dataset_cache_root"].startswith("G:\\lxy\\")
    assert remote_config["checkpoint_root"].startswith("G:\\lxy\\")
    assert "--mode position_bound_head" in run
    assert "--opc1-gate" in run and "--opn1-gate" in run
    assert "selected8_position_head_spn_rescnn_wrong_p_true_output_final.pt" in run
    assert "cache['completed_rows']==196608" in run
    assert "meta['config']['seed']==7" in run
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in windows_scripts.lower()
    assert "EnableDelayedExpansion" not in windows_scripts
    assert "!" not in windows_scripts
    assert "G:\\lxy" in windows_scripts
    assert "source_expected_commit.txt" in windows_scripts
    assert "git status --porcelain" in windows_scripts
    assert opc1_sha in run and opc1_sha in monitor
    assert opn1_sha in run and opn1_sha in monitor
    assert "plot_innovation2_selected_output_position_bound_spn_rescnn" in monitor
    assert "outputs/remote_results/" in monitor
    assert "opc1_gate.json" in monitor and "opn1_gate.json" in monitor
