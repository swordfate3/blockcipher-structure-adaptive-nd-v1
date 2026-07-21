from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_selected_output_spn_rescnn_hybrid import (
    render_spn_rescnn_hybrid,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputSpnResidualCnn,
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_spn_rescnn_hybrid import (
    MODEL_SPECS,
    OPB1_RELEASE_DECISION,
    OPB1_RUN_ID,
    SpnResCnnHybridConfig,
    adjudicate_hybrid,
    authorize_from_opb1_gate,
    hybrid_parameter_counts,
    prepare_hybrid_data,
    train_hybrid_matrix,
    validate_hybrid_contract,
)


def _tiny_config() -> SpnResCnnHybridConfig:
    return SpnResCnnHybridConfig(
        train_rows=8,
        test_rows=8,
        rescnn_channels=4,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
    )


def _opb1_negative_gate() -> dict[str, object]:
    return {
        "run_id": OPB1_RUN_ID,
        "status": "hold",
        "decision": OPB1_RELEASE_DECISION,
        "protocol_checks": {"valid": True},
        "execution_checks": {"valid": True},
        "metrics": {"attribution_passed": False},
    }


def test_hybrid_has_three_stages_and_eight_outputs() -> None:
    model = SelectedOutputSpnResidualCnn(
        channels=4,
        stage_blocks=(1, 1, 1),
        source_for_destination=_present_topology_mapping("exact"),
    )

    assert len(model.stages) == 3
    assert model(torch.zeros((2, 64))).shape == (2, 8)


def test_hybrid_variants_match_anchor_parameter_count_and_initialization() -> None:
    config = SpnResCnnHybridConfig()
    counts = hybrid_parameter_counts(config)
    assert (
        counts["rescnn"] == counts["spn_rescnn_exact_p"] == counts["spn_rescnn_wrong_p"]
    )

    models = []
    for mode in ("exact", "wrong"):
        torch.manual_seed(20260722)
        models.append(
            SelectedOutputSpnResidualCnn(
                channels=4,
                stage_blocks=(1, 1, 1),
                source_for_destination=_present_topology_mapping(mode),
            )
        )
    assert all(
        torch.equal(dict(models[0].named_parameters())[name], parameter)
        for name, parameter in models[1].named_parameters()
    )


def test_formal_mode_requires_valid_opb1_non_attribution() -> None:
    authorize_from_opb1_gate(_opb1_negative_gate())
    invalid = _opb1_negative_gate()
    invalid["decision"] = (
        "innovation2_topology_bottleneck_ready_for_independent_confirmation"
    )
    with pytest.raises(
        ValueError, match="requires OPB1 topology-bottleneck non-attribution"
    ):
        authorize_from_opb1_gate(invalid)


def test_tiny_hybrid_matrix_is_complete_and_replayable(tmp_path: Path) -> None:
    config = _tiny_config()
    data = prepare_hybrid_data(config, tmp_path)
    checks = validate_hybrid_contract(config, data)
    training = train_hybrid_matrix(config, data, tmp_path)
    gate = adjudicate_hybrid(config, checks, training)

    assert all(checks.values())
    assert len(training["rows"]) == 32
    assert len(training["history"]) == 4
    assert len(training["checkpoints"]) == 4
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_spn_rescnn_hybrid_local_smoke_passed"


def test_formal_gate_requires_anchor_and_control_gains() -> None:
    config = SpnResCnnHybridConfig.formal(device="cpu")
    aucs = {
        MODEL_SPECS[0][0]: 0.60,
        MODEL_SPECS[1][0]: 0.70,
        MODEL_SPECS[2][0]: 0.60,
        MODEL_SPECS[3][0]: 0.50,
    }
    rows = [
        {
            "model": model,
            "msb_index": bit,
            "threshold_accuracy": 0.70,
            "majority_accuracy": 0.50,
            "accuracy_minus_majority": 0.20,
            "auc": aucs[model],
            "mse": 0.20,
        }
        for model, _, _ in MODEL_SPECS
        for bit in config.selected_msb_indices
    ]
    training = {
        "rows": rows,
        "summaries": [{"model": model} for model in aucs],
        "history": [
            {"model": model, "epoch": epoch}
            for model in aucs
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(4)],
    }

    gate = adjudicate_hybrid(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert (
        gate["decision"]
        == "innovation2_spn_rescnn_hybrid_candidate_requires_confirmation"
    )


@pytest.mark.parametrize(
    "aucs,failed_check",
    [
        (
            (0.50, 0.549, 0.50, 0.50),
            "candidate_mean_auc_at_least_0_550",
        ),
        (
            (0.695, 0.70, 0.60, 0.50),
            "candidate_minus_anchor_mean_auc_at_least_0_010",
        ),
        (
            (0.60, 0.70, 0.69, 0.50),
            "candidate_minus_wrong_mean_auc_at_least_0_020",
        ),
        (
            (0.60, 0.70, 0.60, 0.68),
            "candidate_minus_shuffle_mean_auc_at_least_0_030",
        ),
    ],
)
def test_formal_gate_holds_when_a_mean_gate_fails(
    aucs: tuple[float, float, float, float],
    failed_check: str,
) -> None:
    config = SpnResCnnHybridConfig.formal(device="cpu")
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
    training = {
        "rows": rows,
        "summaries": [{"model": model} for model in auc_by_model],
        "history": [
            {"model": model, "epoch": epoch}
            for model in auc_by_model
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(4)],
    }

    gate = adjudicate_hybrid(config, {"valid": True}, training)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_spn_rescnn_hybrid_not_supported"
    assert gate["metrics"]["formal_checks"][failed_check] is False


def test_formal_gate_requires_at_least_four_jointly_passing_bits() -> None:
    config = SpnResCnnHybridConfig.formal(device="cpu")
    anchor_name, exact_name, wrong_name, shuffle_name = [
        model for model, _, _ in MODEL_SPECS
    ]
    rows = []
    for bit_index, bit in enumerate(config.selected_msb_indices):
        control_auc = 0.60 if bit_index < 3 else 0.70
        auc_by_model = {
            anchor_name: control_auc,
            exact_name: 0.70,
            wrong_name: control_auc,
            shuffle_name: 0.50,
        }
        rows.extend(
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
        )
    training = {
        "rows": rows,
        "summaries": [{"model": model} for model, _, _ in MODEL_SPECS],
        "history": [
            {"model": model, "epoch": epoch}
            for model, _, _ in MODEL_SPECS
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(4)],
    }

    gate = adjudicate_hybrid(config, {"valid": True}, training)

    formal_checks = gate["metrics"]["formal_checks"]
    assert gate["status"] == "hold"
    assert gate["metrics"]["passed_bit_count"] == 3
    assert formal_checks["at_least_four_bits_pass"] is False
    assert all(
        passed for name, passed in formal_checks.items() if name != "at_least_four_bits_pass"
    )


def test_hybrid_plot_has_plain_chinese_scope(tmp_path: Path) -> None:
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
        "gate": {"decision": "innovation2_spn_rescnn_hybrid_local_smoke_passed"},
    }
    output = tmp_path / "curves.svg"

    render_spn_rescnn_hybrid(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "SPN-ResCNN" in svg
    assert "真实密文输出bit" in svg
    assert "不是四轮" in svg
    assert "真实P - 错误P" in svg
    assert "逐bit门" in svg


def test_training_cli_import_does_not_require_matplotlib() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import blockcipher_nd.cli.run_innovation2_selected_output_spn_rescnn_hybrid
print('import=pass')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import=pass"


def test_result_index_names_opc1_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_smoke_seed6_20260722"
    )
    assert name == "创新2 OPC1：PRESENT三轮SPN-ResCNN混合真实密文输出预测"


def test_formal_remote_package_is_gate_owned_cached_and_windows_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    plan_path = root / (
        "configs/experiment/innovation2/"
        "innovation2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_key6.json"
    )
    remote_config_path = root / (
        "configs/remote/"
        "innovation2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_"
        "key6_gpu0_20260722.json"
    )
    run_path = root / (
        "configs/remote/generated/"
        "run_i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_"
        "key6_gpu0_20260722.cmd"
    )
    launch_path = root / (
        "configs/remote/generated/"
        "launch_i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_"
        "key6_gpu0_20260722.cmd"
    )
    monitor_path = root / (
        "configs/remote/generated/"
        "monitor_i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_"
        "key6_gpu0_20260722.sh"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    remote_config = json.loads(remote_config_path.read_text(encoding="utf-8"))
    run = run_path.read_text(encoding="utf-8")
    launch = launch_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")
    windows_scripts = run + launch
    expected_gate_sha = (
        "776a43a7e0b13e9db17d825ec20f83fc6ce54ca8a36408849d7007a8ec46a549"
    )

    assert plan["opb1_authority"]["status"] == "hold"
    assert plan["opb1_authority"]["decision"] == (
        "innovation2_topology_bottleneck_not_attributed"
    )
    assert plan["opb1_authority"]["attributed_bits"] == 0
    assert plan["opb1_authority"]["gate_sha256"] == expected_gate_sha
    assert plan["common"]["seed"] == 6
    assert plan["common"]["train_total_rows"] == 131072
    assert plan["common"]["test_total_rows"] == 65536
    assert plan["common"]["epochs"] == 100
    assert len(plan["rows"]) == 4
    assert len({row["parameters"] for row in plan["rows"]}) == 1
    assert plan["final_gate"] == {
        "minimum_candidate_mean_auc": 0.55,
        "minimum_candidate_minus_anchor_mean_auc": 0.01,
        "minimum_candidate_minus_wrong_mean_auc": 0.02,
        "minimum_candidate_minus_shuffle_mean_auc": 0.03,
        "minimum_joint_passed_bits": 4,
        "minimum_per_bit_candidate_auc": 0.55,
        "minimum_per_bit_candidate_minus_anchor_auc": 0.005,
        "minimum_per_bit_candidate_minus_each_control_auc": 0.015,
        "minimum_per_bit_accuracy_minus_majority": 0.005,
    }
    assert {row["model"] for row in plan["rows"]} == set(remote_config["models"])
    assert remote_config["opb1_gate_sha256"] == expected_gate_sha
    assert remote_config["expected_result_rows"] == 32
    assert remote_config["expected_history_rows"] == 400
    assert remote_config["expected_checkpoints"] == 4
    assert remote_config["remote_directory"] == "i2_opc1_hybrid_k6_20260722"
    assert remote_config["archive_directory"] == "i2_opc1_hybrid_k6_20260722"
    assert remote_config["dataset_cache_root"].startswith("G:\\lxy\\")
    assert remote_config["checkpoint_root"].startswith("G:\\lxy\\")
    assert "--mode spn_rescnn_hybrid" in run
    assert "--opb1-gate" in run
    assert "selected8_spn_rescnn_wrong_p_true_output_final.pt" in run
    assert "cache['completed_rows']==196608" in run
    assert "meta['config']['seed']==6" in run
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in windows_scripts.lower()
    assert "EnableDelayedExpansion" not in windows_scripts
    assert "!" not in windows_scripts
    assert "G:\\lxy" in windows_scripts
    assert "source_expected_commit.txt" in windows_scripts
    assert "git status --porcelain" in windows_scripts
    assert expected_gate_sha in run
    assert expected_gate_sha in monitor
    assert "plot_innovation2_selected_output_spn_rescnn_hybrid" in monitor
    assert "outputs/remote_results/" in monitor
    assert "opb1_gate.json" in monitor
