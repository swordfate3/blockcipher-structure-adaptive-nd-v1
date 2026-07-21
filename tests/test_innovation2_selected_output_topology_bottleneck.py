from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_selected_output_topology_bottleneck import (
    render_topology_bottleneck,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputTopologyBottleneckSpn,
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_topology_bottleneck import (
    MODEL_SPECS,
    OPA3_DECISION,
    OPA3_REMOTE_RUN_ID,
    TopologyBottleneckConfig,
    adjudicate_bottleneck,
    authorize_from_opa3_gate,
    bottleneck_parameter_counts,
    prepare_bottleneck_data,
    train_bottleneck_matrix,
    validate_bottleneck_contract,
)


def _opa3_gate() -> dict[str, object]:
    return {
        "run_id": OPA3_REMOTE_RUN_ID,
        "status": "hold",
        "decision": OPA3_DECISION,
        "protocol_checks": {"valid": True},
        "execution_checks": {"complete": True},
        "metrics": {
            "priority_passed": False,
            "attributed_bit_count": 0,
            "mean_auc_by_model": {
                "present_spn_exact_p_true_output": 1.0,
                "present_spn_identity_p_true_output": 0.532,
                "present_spn_wrong_p_true_output": 1.0,
            },
        },
    }


def _tiny_config() -> TopologyBottleneckConfig:
    return TopologyBottleneckConfig(
        run_id="test-opb1",
        train_rows=8,
        test_rows=8,
        mlp_hidden_dim=16,
        lstm_hidden_dim=4,
        lstm_layers=1,
        present_spn_dim=32,
        present_spn_blocks=1,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
        maximum_parameter_gap=0.20,
    )


def test_topology_bottleneck_has_low_rank_position_condition_and_output_shape() -> None:
    model = SelectedOutputTopologyBottleneckSpn(
        token_dim=12,
        blocks=3,
        source_for_destination=_present_topology_mapping("exact"),
    )

    assert not hasattr(model, "position_embedding")
    assert [tuple(item.shape) for item in model.key_strengths] == [
        (1, 64, 1),
        (1, 64, 1),
        (1, 64, 1),
    ]
    assert [tuple(item.shape) for item in model.key_directions] == [
        (1, 1, 12),
        (1, 1, 12),
        (1, 1, 12),
    ]
    assert model(torch.zeros((2, 64))).shape == (2, 8)


def test_exact_and_wrong_bottleneck_models_match_parameters_but_change_mapping() -> None:
    features = torch.arange(128, dtype=torch.float32).reshape(2, 64) % 2
    models = []
    for mode in ("exact", "wrong"):
        torch.manual_seed(20260722)
        models.append(
            SelectedOutputTopologyBottleneckSpn(
                token_dim=12,
                blocks=1,
                source_for_destination=_present_topology_mapping(mode),
            )
        )

    reference = dict(models[0].named_parameters())
    assert all(
        torch.equal(reference[name], parameter)
        for name, parameter in models[1].named_parameters()
    )
    assert not torch.equal(models[0](features), models[1](features))


def test_formal_candidate_is_parameter_matched_to_original_spn_anchor() -> None:
    config = TopologyBottleneckConfig()
    counts = bottleneck_parameter_counts(config)
    anchor = counts["present_spn_exact_p"]
    exact = counts["topology_bottleneck_exact_p"]
    wrong = counts["topology_bottleneck_wrong_p"]

    assert exact == wrong
    assert abs(exact - anchor) / anchor <= config.maximum_parameter_gap


def test_opa3_verified_hold_is_the_only_authority() -> None:
    assert authorize_from_opa3_gate(_opa3_gate()) == {
        "exact_mean_auc": 1.0,
        "wrong_mean_auc": 1.0,
    }

    invalid = _opa3_gate()
    invalid["status"] = "pass"
    with pytest.raises(ValueError, match="verified topology hold"):
        authorize_from_opa3_gate(invalid)
    invalid = _opa3_gate()
    invalid["metrics"]["attributed_bit_count"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="must remain zero"):
        authorize_from_opa3_gate(invalid)


def test_tiny_bottleneck_matrix_is_complete_and_replayable(tmp_path: Path) -> None:
    config = _tiny_config()
    opa3_gate = _opa3_gate()
    data = prepare_bottleneck_data(config, tmp_path)
    checks = validate_bottleneck_contract(config, data, opa3_gate)
    training = train_bottleneck_matrix(config, data, tmp_path)
    gate = adjudicate_bottleneck(config, checks, training)

    assert all(checks.values())
    assert len(training["rows"]) == 32
    assert len(training["history"]) == 4
    assert len(training["checkpoints"]) == 4
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_topology_bottleneck_local_smoke_passed"
    assert gate["next_action"]["reopens_r4"] is False


def _formal_training(
    config: TopologyBottleneckConfig,
    aucs: dict[str, float],
) -> dict[str, object]:
    rows = []
    for model, architecture, _ in MODEL_SPECS:
        for bit in config.selected_msb_indices:
            rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "msb_index": bit,
                    "threshold_accuracy": 0.95,
                    "majority_accuracy": 0.5,
                    "accuracy_minus_majority": 0.45,
                    "auc": aucs[model],
                    "mse": 0.05,
                    "invalid_numpy_rint_rate": 0.0,
                    "test_target_identity": "true_selected_ciphertext_targets",
                }
            )
    return {
        "rows": rows,
        "summaries": [{"model": model} for model in aucs],
        "history": [
            {"model": model, "epoch": epoch}
            for model in aucs
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(4)],
    }


def test_formal_gate_requires_utility_topology_and_shuffle_gains() -> None:
    config = TopologyBottleneckConfig.formal(device="cpu")
    anchor, exact, wrong, shuffle = [model for model, _, _ in MODEL_SPECS]
    training = _formal_training(
        config,
        {
            anchor: 0.98,
            exact: 0.96,
            wrong: 0.70,
            shuffle: 0.50,
        },
    )

    gate = adjudicate_bottleneck(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_topology_bottleneck_ready_for_independent_confirmation"
    )
    assert gate["metrics"]["attributed_bit_count"] == 8
    assert gate["metrics"]["priority_passed"] is True
    assert gate["next_action"]["next_adjudication"] == (
        "opb2_seed5_independent_confirmation"
    )


def test_attribution_without_anchor_utility_is_held() -> None:
    config = TopologyBottleneckConfig.formal(device="cpu")
    anchor, exact, wrong, shuffle = [model for model, _, _ in MODEL_SPECS]
    training = _formal_training(
        config,
        {
            anchor: 0.99,
            exact: 0.80,
            wrong: 0.60,
            shuffle: 0.50,
        },
    )

    gate = adjudicate_bottleneck(config, {"valid": True}, training)

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_topology_bottleneck_attributed_with_performance_cost"
    )
    assert gate["metrics"]["attribution_passed"] is True
    assert gate["metrics"]["utility_passed"] is False


def test_bottleneck_plot_has_plain_chinese_scope_and_controls(tmp_path: Path) -> None:
    config = _tiny_config()
    rows = []
    for model_index, (model, architecture, _) in enumerate(MODEL_SPECS):
        for bit_index, bit in enumerate(config.selected_msb_indices):
            rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "msb_index": bit,
                    "auc": 0.58 - 0.02 * model_index + 0.001 * bit_index,
                }
            )
    summary = {
        "metadata": {
            "mode": "smoke",
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "bit_rows": rows,
        "gate": {
            "decision": "innovation2_topology_bottleneck_local_smoke_passed",
            "metrics": {
                "candidate_minus_wrong_mean_auc": 0.02,
                "candidate_minus_shuffle_mean_auc": 0.04,
            },
        },
    }
    output = tmp_path / "curves.svg"

    render_topology_bottleneck(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "拓扑瓶颈真实输出预测" in svg
    assert "候选错误P" in svg
    assert "标签打乱" in svg
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
import blockcipher_nd.cli.run_innovation2_selected_output_topology_bottleneck
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


def test_result_index_names_opb1_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opb1_present_r3_topology_bottleneck_smoke_20260722"
    )

    assert "OPB1" in name
    assert "拓扑瓶颈" in name
    assert "真实密文输出预测" in name


def test_formal_remote_package_is_frozen_and_windows_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    plan_path = root / (
        "configs/experiment/innovation2/"
        "innovation2_output_prediction_opb1_present_r3_topology_bottleneck_key4.json"
    )
    remote_config_path = root / (
        "configs/remote/"
        "innovation2_output_prediction_opb1_present_r3_topology_bottleneck_"
        "key4_gpu0_20260722.json"
    )
    run_path = root / (
        "configs/remote/generated/"
        "run_i2_output_prediction_opb1_present_r3_topology_bottleneck_"
        "key4_gpu0_20260722.cmd"
    )
    launch_path = root / (
        "configs/remote/generated/"
        "launch_i2_output_prediction_opb1_present_r3_topology_bottleneck_"
        "key4_gpu0_20260722.cmd"
    )
    monitor_path = root / (
        "configs/remote/generated/"
        "monitor_i2_output_prediction_opb1_present_r3_topology_bottleneck_"
        "key4_gpu0_20260722.sh"
    )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    remote_config = json.loads(remote_config_path.read_text(encoding="utf-8"))
    run = run_path.read_text(encoding="utf-8")
    launch = launch_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")
    windows_scripts = run + launch

    expected_gate_sha = (
        "def55214d46acf0e199f465fda66e6ca394f094ceec78d419354357df1c50943"
    )
    assert plan["opa3_authority"]["status"] == "hold"
    assert plan["opa3_authority"]["attributed_bits"] == 0
    assert plan["opa3_authority"]["gate_sha256"] == expected_gate_sha
    assert plan["common"]["seed"] == 4
    assert plan["common"]["train_total_rows"] == 131072
    assert plan["common"]["test_total_rows"] == 65536
    assert plan["common"]["epochs"] == 100
    assert len(plan["rows"]) == 4
    assert {row["model"] for row in plan["rows"]} == set(remote_config["models"])
    assert remote_config["opa3_gate_sha256"] == expected_gate_sha
    assert remote_config["expected_result_rows"] == 32
    assert remote_config["expected_history_rows"] == 400
    assert remote_config["expected_checkpoints"] == 4
    assert remote_config["remote_directory"] == "i2_opb1_tbneck_k4_20260722"
    assert remote_config["archive_directory"] == "i2_opb1_tbneck_k4_20260722"
    assert remote_config["dataset_cache_root"].startswith("G:\\lxy\\")
    assert remote_config["checkpoint_root"].startswith("G:\\lxy\\")

    assert "set REMOTE_DIR=i2_opb1_tbneck_k4_20260722" in windows_scripts
    assert "set ARCHIVE_NAME=i2_opb1_tbneck_k4_20260722" in run
    assert "set SCHEDULE_CMD=%SCHEDULE_ROOT%\\i2_opb1_tbneck_k4.cmd" in launch
    assert '/TR "cmd.exe /c %SCHEDULE_CMD%"' in launch
    assert len("cmd.exe /c G:\\lxy\\scheduled-runs\\i2_opb1_tbneck_k4.cmd") < 261
    assert "cmd.exe /k" not in windows_scripts.lower()
    assert "EnableDelayedExpansion" not in windows_scripts
    assert "!" not in windows_scripts

    assert "opa3_gate.json" in run
    assert "progress.jsonl" in run
    assert "cache_metadata.json" in run
    assert "OPA3_GATE_SHA256" in monitor
    assert "sha256sum -c" in monitor
    assert "visual_qa_pending.marker" in monitor
    assert "scripts/index-results" in monitor
    assert "retrieved_from_verified_result_branch.marker" in monitor

    allowed_user_path = (
        "C:/Users/1304Lijinlin/.ssh/"
        "github_blockcipher_20260612_result_pusher_ed25519"
    )
    for text in (launch, run):
        assert text.count("C:/Users") == 1
        assert allowed_user_path in text
    assert "C:/Users" not in monitor
