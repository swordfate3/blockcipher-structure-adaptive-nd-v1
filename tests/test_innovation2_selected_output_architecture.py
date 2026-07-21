from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.cli.plot_innovation2_selected_output_architecture import (
    render_selected_output_architecture,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    ARCHITECTURES,
    MODEL_SPECS,
    SelectedOutputArchitectureConfig,
    SelectedOutputLstm,
    SelectedOutputPresentSpn,
    SelectedOutputResidualCnn,
    SelectedOutputTransformer,
    _present_source_for_destination,
    adjudicate_architecture_gate,
    parameter_counts,
    prepare_architecture_data,
    validate_architecture_contract,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import SelectedOutputMlp


def test_five_models_emit_only_the_preregistered_eight_outputs() -> None:
    features = torch.zeros((2, 64))
    models = (
        SelectedOutputMlp(16, output_bits=8),
        SelectedOutputLstm(hidden_dim=4, layers=1),
        SelectedOutputResidualCnn(channels=4, blocks=1),
        SelectedOutputTransformer(token_dim=8, heads=2, layers=1, feedforward_dim=16),
        SelectedOutputPresentSpn(token_dim=4, blocks=1),
    )

    assert ARCHITECTURES == ("mlp", "lstm", "rescnn", "transformer", "present_spn")
    assert all(model(features).shape == (2, 8) for model in models)
    assert len(MODEL_SPECS) == 5
    assert all(not shuffled for _, _, shuffled in MODEL_SPECS)


def test_all_phase_a_models_are_within_three_percent_of_mlp_budget() -> None:
    counts = parameter_counts(SelectedOutputArchitectureConfig())

    assert set(counts) == set(ARCHITECTURES)
    assert all(abs(count - counts["mlp"]) / counts["mlp"] <= 0.03 for count in counts.values())


def test_rescnn_readout_preserves_absolute_bit_positions() -> None:
    model = SelectedOutputResidualCnn(channels=4, blocks=1)

    assert isinstance(model.head[0], torch.nn.Flatten)
    assert isinstance(model.head[1], torch.nn.Linear)
    assert model.head[1].in_features == 64 * 4


def test_present_spn_uses_the_exact_msb_first_p_layer_mapping() -> None:
    source_for_destination = _present_source_for_destination().tolist()

    assert sorted(source_for_destination) == list(range(64))
    for destination_msb, source_msb in enumerate(source_for_destination):
        source_integer = 63 - source_msb
        destination_integer = 63 - destination_msb
        assert Present80.permutation_layer(1 << source_integer) == 1 << destination_integer


def test_seed2_contract_replays_true_outputs_and_parameter_budget(tmp_path: Path) -> None:
    config = SelectedOutputArchitectureConfig(
        train_rows=8,
        test_rows=8,
        data_chunk_rows=4,
    )
    data = prepare_architecture_data(config, tmp_path)

    checks = validate_architecture_contract(config, data)

    assert all(checks.values())
    assert config.seed == 2
    assert int(data["metadata"]["seed"]) == 2


def test_phase_a_can_only_select_a_candidate_for_independent_confirmation() -> None:
    config = SelectedOutputArchitectureConfig(mode="phase_a_screen")
    aucs = {
        "mlp": 0.520,
        "lstm": 0.521,
        "rescnn": 0.525,
        "transformer": 0.519,
        "present_spn": 0.523,
    }
    rows = []
    for architecture, auc in aucs.items():
        for bit in config.selected_msb_indices:
            rows.append(
                {
                    "model": f"selected8_{architecture}_true_output",
                    "architecture": architecture,
                    "msb_index": bit,
                    "threshold_accuracy": 0.515,
                    "majority_accuracy": 0.5,
                    "accuracy_minus_majority": 0.015,
                    "auc": auc,
                    "mse": 0.24,
                    "invalid_numpy_rint_rate": 0.0,
                }
            )
    training = {
        "rows": rows,
        "summaries": [{"architecture": item} for item in ARCHITECTURES],
        "history": [
            {"architecture": item, "epoch": epoch}
            for item in ARCHITECTURES
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in ARCHITECTURES],
    }

    gate = adjudicate_architecture_gate(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_selected8_architecture_candidate_requires_confirmation"
    assert gate["metrics"]["selected_candidate_for_phase_b"] == "rescnn"
    assert gate["next_action"]["phase_b_requires_matched_shuffle"] is True
    assert "final architecture claim" in gate["claim_scope"]


def test_plot_explains_models_and_discovery_boundary(tmp_path: Path) -> None:
    config = SelectedOutputArchitectureConfig()
    rows = []
    for model_index, architecture in enumerate(ARCHITECTURES):
        for bit_index, bit in enumerate(config.selected_msb_indices):
            rows.append(
                {
                    "architecture": architecture,
                    "msb_index": bit,
                    "auc": 0.5 + 0.001 * (model_index + bit_index),
                    "accuracy_minus_majority": 0.001 * model_index,
                }
            )
    summary = {
        "metadata": {
            "mode": "smoke",
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "bit_rows": rows,
        "gate": {
            "decision": "innovation2_selected8_architecture_screen_local_smoke_passed",
            "metrics": {"selected_candidate_for_phase_b": None},
        },
    }
    output = tmp_path / "curves.svg"

    render_selected_output_architecture(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "多架构发现屏" in svg
    assert "PRESENT结构网络" in svg
    assert "不作最终架构结论" in svg
    assert "不是真假样本分类" in svg


def test_result_index_names_opa1_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opa1_present_r3_selected8_architecture_screen_smoke_20260721"
    )

    assert "OPA1" in name
    assert "PRESENT三轮" in name
    assert "五模型架构发现屏" in name


def test_remote_package_freezes_five_model_phase_a_and_windows_hygiene() -> None:
    root = Path(__file__).resolve().parents[1]
    stem = "innovation2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2"
    config = json.loads(
        (root / f"configs/remote/{stem}_gpu0_20260721.json").read_text(
            encoding="utf-8"
        )
    )
    plan = json.loads(
        (root / f"configs/experiment/innovation2/{stem}.json").read_text(
            encoding="utf-8"
        )
    )
    launch = (
        root / f"configs/remote/generated/launch_i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721.cmd"
    ).read_text(encoding="utf-8")
    run = (
        root / f"configs/remote/generated/run_i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721.cmd"
    ).read_text(encoding="utf-8")
    monitor = (
        root / f"configs/remote/generated/monitor_i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721.sh"
    ).read_text(encoding="utf-8")

    assert config["train_total_rows"] == 1 << 17
    assert config["test_total_rows"] == 1 << 16
    assert config["expected_result_rows"] == 40
    assert config["expected_history_rows"] == 500
    assert config["seed"] == 2
    assert config["sample_classification"] is False
    assert len(config["models"]) == 5
    assert len(plan["rows"]) == 5
    assert all(row["parameters"] for row in plan["rows"])
    assert "samples_per_class" not in config
    assert config["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )
    assert "cmd.exe /c" in launch
    assert "set SCHEDULE_ROOT=G:\\lxy\\scheduled-runs" in launch
    assert "set SCHEDULE_CMD=%SCHEDULE_ROOT%\\i2_opa1_key2.cmd" in launch
    assert '>>"%SCHEDULE_CMD%" echo call "%RUN_CMD%" 0' in launch
    assert '/TR "cmd.exe /c %SCHEDULE_CMD%"' in launch
    assert len("cmd.exe /c G:\\lxy\\scheduled-runs\\i2_opa1_key2.cmd") < 261
    assert "cmd.exe /k" not in launch + run
    assert "EnableDelayedExpansion" not in launch + run
    assert "!" not in launch + run
    assert "source_expected_commit.txt" in launch + run
    assert "--mode phase_a_screen" in run
    assert "expected_rows=40" in run
    assert "cache_metadata.json" in run
    assert "progress.jsonl" in monitor
    assert "plot-innovation2-selected-output-architecture" in monitor
    assert "visual_qa_pending.marker" in monitor
    assert "scripts/index-results" in monitor
    assert "_result_branch_pushed.marker" in run + monitor
    assert "C:\\Users" not in run.replace(
        "C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519",
        "",
    )
