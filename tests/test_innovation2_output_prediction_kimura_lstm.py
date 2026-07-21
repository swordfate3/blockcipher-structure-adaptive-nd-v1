from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_output_prediction_kimura_lstm import (
    render_kimura_lstm_output_prediction,
)
from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraCiphertextLstm,
    KimuraOutputPredictionConfig,
    ParameterMatchedOutputMlp,
    adjudicate_kimura_output_prediction,
    full_output_metrics,
    parameter_counts,
    prepare_disk_output_prediction_data,
    train_kimura_output_matrix,
    validate_kimura_output_contract,
)


def _tiny_config() -> KimuraOutputPredictionConfig:
    return KimuraOutputPredictionConfig(
        run_id="test-op9",
        train_rows=8,
        test_rows=8,
        hidden_dim=4,
        layers=1,
        mlp_hidden_dim=8,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
    )


def test_paper_profile_is_exactly_frozen() -> None:
    config = KimuraOutputPredictionConfig.paper_calibration()

    assert config.rounds == 3
    assert config.train_rows == 1 << 17
    assert config.test_rows == 1 << 16
    assert config.hidden_dim == 300
    assert config.layers == 6
    assert config.epochs == 100
    assert config.batch_size == 250
    assert config.learning_rate == 1e-3


def test_models_accept_msb_bit_sequences_and_are_parameter_matched() -> None:
    config = KimuraOutputPredictionConfig()
    features = torch.zeros((3, 64), dtype=torch.float32)

    assert KimuraCiphertextLstm()(features).shape == (3, 64)
    assert ParameterMatchedOutputMlp()(features).shape == (3, 64)
    counts = parameter_counts(config)
    relative_gap = abs(counts["kimura_lstm"] - counts["matched_mlp"]) / counts[
        "kimura_lstm"
    ]
    assert relative_gap < 0.01


def test_disk_cache_is_chunked_replayable_and_parameter_matched(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    events: list[tuple[str, dict[str, object]]] = []
    data = prepare_disk_output_prediction_data(
        config,
        tmp_path,
        progress=lambda event, payload: events.append((event, payload)),
    )
    checks = validate_kimura_output_contract(config, data)
    first_hash = hashlib.sha256(
        (tmp_path / "data" / "full_targets.npy").read_bytes()
    ).hexdigest()
    reused = prepare_disk_output_prediction_data(config, tmp_path)
    second_hash = hashlib.sha256(
        (tmp_path / "data" / "full_targets.npy").read_bytes()
    ).hexdigest()

    assert all(checks.values())
    assert data["features"].shape == (16, 64)
    assert data["full_targets"].shape == (16, 64)
    assert sum(event == "cache_chunk" for event, _ in events) == 4
    assert reused["cache_reused"] is True
    assert first_hash == second_hash
    with pytest.raises(ValueError, match="parameters do not match"):
        prepare_disk_output_prediction_data(
            KimuraOutputPredictionConfig(
                **{**config.__dict__, "rounds": 2}
            ),
            tmp_path,
        )


def test_small_training_matrix_remains_full_output_prediction(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    data = prepare_disk_output_prediction_data(config, tmp_path)
    checks = validate_kimura_output_contract(config, data)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training = train_kimura_output_matrix(config, data, tmp_path)
    finally:
        torch.set_num_threads(previous_threads)
    gate = adjudicate_kimura_output_prediction(config, checks, training)

    assert len(training["rows"]) == 3
    assert all(row["target"] == "full_64_bit_true_ciphertext_output" for row in training["rows"])
    assert all(row["sample_classification"] is False for row in training["rows"])
    assert all(row["test_target_identity"] == "true_full_ciphertext_targets" for row in training["rows"])
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("local_smoke_passed")
    assert gate["next_action"]["sample_classification"] is False
    assert all((tmp_path / item["path"]).exists() for item in training["checkpoints"])


def test_raw_mse_metrics_use_rounding_for_paper_exact_match() -> None:
    targets = np.zeros((2, 64), dtype=np.float32)
    targets[1] = 1.0
    scores = targets.copy()
    scores[0, 0] = 1.6

    metrics = full_output_metrics(scores, targets)

    assert metrics["test_exact_match_count"] == 1
    assert metrics["test_exact_match"] == 0.5
    assert metrics["test_bit_match"] == 127 / 128
    assert metrics["invalid_rounded_cell_rate"] == 1 / 128


def test_plot_names_true_output_and_paper_metric(tmp_path: Path) -> None:
    rows = []
    for name, bit_match, auc, exact, parameters in (
        ("kimura_lstm_true_output", 0.52, 0.54, 2, 4_000_000),
        ("matched_mlp_true_output", 0.51, 0.52, 1, 4_010_000),
        ("kimura_lstm_label_shuffle", 0.50, 0.50, 0, 4_000_000),
    ):
        rows.append(
            {
                "model": name,
                "test_bit_match": bit_match,
                "test_macro_auc": auc,
                "test_exact_match_count": exact,
                "parameters": parameters,
            }
        )
    summary = {
        "metadata": {"mode": "smoke"},
        "trained_rows": rows,
        "gate": {
            "status": "pass",
            "decision": "innovation2_output_prediction_kimura_lstm_local_smoke_passed",
            "metrics": {
                "lstm_minus_shuffled_bit_match": 0.02,
                "lstm_minus_shuffled_macro_auc": 0.04,
                "lstm_minus_matched_mlp_bit_match": 0.01,
            },
        },
    }
    output = tmp_path / "curves.svg"

    render_kimura_lstm_output_prediction(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "完整64-bit真实密文输出预测" in svg
    assert "不存在真假样本类别" in svg
    assert "论文主指标" in svg
    assert "AUC只是支持项" in svg


def test_remote_package_freezes_total_rows_and_windows_hygiene() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / (
        "configs/remote/innovation2_output_prediction_op9_present_r3_"
        "kimura_lstm_2p17_seed0_gpu0_20260721.json"
    )
    plan_path = root / (
        "configs/experiment/innovation2/innovation2_output_prediction_op9_"
        "present_r3_kimura_lstm_2p17_seed0.json"
    )
    launch_path = root / (
        "configs/remote/generated/launch_i2_output_prediction_op9_present_r3_"
        "kimura_lstm_2p17_seed0_gpu0_20260721.cmd"
    )
    run_path = root / (
        "configs/remote/generated/run_i2_output_prediction_op9_present_r3_"
        "kimura_lstm_2p17_seed0_gpu0_20260721.cmd"
    )
    monitor_path = root / (
        "configs/remote/generated/monitor_i2_output_prediction_op9_present_r3_"
        "kimura_lstm_2p17_seed0_gpu0_20260721.sh"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch = launch_path.read_text(encoding="utf-8")
    run = run_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")

    assert config["train_total_rows"] == 1 << 17
    assert config["test_total_rows"] == 1 << 16
    assert "samples_per_class" not in config
    assert config["sample_classification"] is False
    assert config["dataset_cache"] is True
    assert config["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )
    assert plan["common"]["target"] == "64_msb_first_true_ciphertext_bits"
    assert plan["common"]["sample_classification"] is False
    assert len(plan["rows"]) == 3
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in launch + run
    assert "source_expected_commit.txt" in launch + run
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run
    assert "--mode paper_calibration" in run
    assert '"%RESULTS_DIR%\\data\\%%F"' in run
    assert '"%RESULTS_DIR%\\models\\%%F"' in run
    assert "_result_branch_pushed.marker" in run + monitor
    assert "${RUN_ID}_failed.marker" in monitor
    assert "sleep 120" in monitor
    assert "scripts/index-results" in monitor
    assert "C:\\Users" not in run.replace(
        "C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519",
        "",
    )
