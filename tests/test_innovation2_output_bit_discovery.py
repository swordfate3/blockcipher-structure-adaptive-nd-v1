from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_output_bit_discovery import (
    render_output_bit_discovery,
)
from blockcipher_nd.cli.run_innovation2_output_bit_discovery import main
from blockcipher_nd.tasks.innovation2.output_bit_discovery import (
    OutputBitDiscoveryConfig,
    adjudicate_output_bit_discovery,
    per_bit_metric_rows,
    select_discovery_candidates,
)
from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraOutputPredictionConfig,
    prepare_disk_output_prediction_data,
    train_kimura_output_matrix,
)


def test_per_bit_metrics_use_explicit_msb_mapping_and_true_output_values() -> None:
    rng = np.random.default_rng(4)
    targets = rng.integers(0, 2, size=(256, 64)).astype(np.float32)
    scores = np.full((256, 64), 0.5, dtype=np.float32)
    scores[:, 0] = targets[:, 0] * 0.9 + 0.05

    rows = per_bit_metric_rows("kimura_lstm_true_output", "discovery", scores, targets)

    assert len(rows) == 64
    assert rows[0]["msb_index"] == 0
    assert rows[0]["integer_bit"] == 63
    assert rows[0]["nibble_msb_index"] == 0
    assert rows[0]["bit_in_nibble_msb"] == 0
    assert rows[0]["threshold_accuracy"] == 1.0
    assert rows[0]["auc"] == 1.0
    assert rows[0]["sample_classification"] is False
    assert rows[63]["integer_bit"] == 0


def test_candidate_selection_and_fresh_gate_are_bitwise_not_full_exact() -> None:
    config = OutputBitDiscoveryConfig()
    discovery_rows = _synthetic_metric_rows("discovery", strong_bit=7)
    candidates = select_discovery_candidates(
        config, discovery_rows, source_run_id="source"
    )

    assert candidates["candidate_msb_indices"] == [7]
    assert candidates["candidates"][0]["integer_bit"] == 56

    fresh_rows = _synthetic_metric_rows("fresh_confirmation", strong_bit=7)
    gate = adjudicate_output_bit_discovery(
        OutputBitDiscoveryConfig(mode="fresh_confirmation"),
        {"source_valid": True},
        {"fresh_valid": True},
        discovery_rows,
        fresh_rows,
        candidates,
    )

    assert gate["status"] == "pass"
    assert gate["metrics"]["fresh_confirmed_msb_indices"] == [7]
    assert gate["next_action"]["target"] == "selected_true_ciphertext_output_bits"
    assert "full-ciphertext recovery" in gate["claim_scope"]


def test_candidate_selection_can_freeze_mlp_for_an_easy_bit() -> None:
    config = OutputBitDiscoveryConfig()
    discovery_rows = _synthetic_metric_rows(
        "discovery", strong_bit=11, strong_model="matched_mlp_true_output"
    )

    candidates = select_discovery_candidates(
        config, discovery_rows, source_run_id="source"
    )

    assert candidates["candidate_msb_indices"] == [11]
    assert candidates["candidates"][0]["selector_model"] == (
        "matched_mlp_true_output"
    )
    assert candidates["candidates"][0]["shuffle_control_scope"] == (
        "cross_architecture_negative_control"
    )
    fresh_rows = _synthetic_metric_rows(
        "fresh_confirmation",
        strong_bit=11,
        strong_model="matched_mlp_true_output",
    )
    gate = adjudicate_output_bit_discovery(
        OutputBitDiscoveryConfig(mode="fresh_confirmation"),
        {"source_valid": True},
        {"fresh_valid": True},
        discovery_rows,
        fresh_rows,
        candidates,
    )
    assert gate["metrics"]["fresh_confirmed_msb_indices"] == [11]
    assert gate["candidate_confirmation"][0]["fresh_selector_model"] == (
        "matched_mlp_true_output"
    )


def test_end_to_end_smoke_freezes_candidates_before_fresh_holdout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_config = KimuraOutputPredictionConfig(
        run_id="source_smoke",
        train_rows=16,
        test_rows=16,
        hidden_dim=4,
        layers=1,
        mlp_hidden_dim=4,
        epochs=1,
        batch_size=8,
        data_chunk_rows=8,
        device="cpu",
    )
    data = prepare_disk_output_prediction_data(source_config, source_root)
    training = train_kimura_output_matrix(source_config, data, source_root)
    metadata = {
        "run_id": source_config.run_id,
        "cipher": "PRESENT-80",
        "config": {
            "rounds": source_config.rounds,
            "train_rows": source_config.train_rows,
            "test_rows": source_config.test_rows,
            "hidden_dim": source_config.hidden_dim,
            "layers": source_config.layers,
            "mlp_hidden_dim": source_config.mlp_hidden_dim,
        },
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "target": "64 MSB-first true ciphertext bits",
        "sample_classification": False,
    }
    _write_json(source_root / "metadata.json", metadata)
    _write_json(source_root / "checkpoint_manifest.json", training["checkpoints"])
    output_root = tmp_path / "op10"

    exit_code = main(
        [
            "--mode",
            "smoke",
            "--source-output-root",
            str(source_root),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    results = (output_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    candidates_bytes = (output_root / "candidates.json").read_bytes()
    candidate_sum = (output_root / "candidates.sha256").read_text(encoding="ascii")
    source_plaintexts = set(np.load(source_root / "data" / "plaintexts.npy"))
    fresh_plaintexts = set(np.load(output_root / "fresh_data" / "plaintexts.npy"))

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert len(results) == 64 * 3 * 2
    assert hashlib.sha256(candidates_bytes).hexdigest() in candidate_sum
    assert source_plaintexts.isdisjoint(fresh_plaintexts)
    events = [
        json.loads(line)["event"]
        for line in (output_root / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events.index("candidates_frozen_before_fresh_evaluation") < events.index(
        "fresh_cache_chunk"
    )


def test_plot_explains_bit_order_and_fresh_confirmation(tmp_path: Path) -> None:
    summary = {
        "ranking": [
            {
                "msb_index": bit,
                "discovery_lstm_auc": 0.5 + (0.02 if bit == 7 else 0.0),
                "discovery_shuffle_auc": 0.5,
                "discovery_mlp_auc": 0.5,
                "fresh_lstm_auc": 0.5 + (0.018 if bit == 7 else 0.0),
                "fresh_shuffle_auc": 0.5,
                "fresh_mlp_auc": 0.5,
                "discovery_lstm_accuracy_margin": 0.006 if bit == 7 else 0.0,
                "fresh_lstm_accuracy_margin": 0.006 if bit == 7 else 0.0,
            }
            for bit in range(64)
        ],
        "candidates": {"candidate_msb_indices": [7]},
        "gate": {
            "decision": "innovation2_true_output_bits_fresh_confirmed",
            "metrics": {"fresh_confirmed_msb_indices": [7]},
            "candidate_confirmation": [
                {
                    "msb_index": 7,
                    "selector_model": "kimura_lstm_true_output",
                    "fresh_auc": 0.518,
                    "fresh_shuffle_auc": 0.5,
                }
            ],
        },
    }
    output = tmp_path / "curves.svg"

    render_output_bit_discovery(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "哪些真实密文输出bit容易预测" in svg
    assert "0是密文最高位" in svg
    assert "不要求64-bit完整密文同时命中" in svg
    assert "全新明文确认集" in svg


def test_remote_package_waits_for_op9_and_freezes_fresh_scale() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / (
        "configs/remote/innovation2_output_prediction_op10_present_r3_"
        "easy_bit_confirm_gpu0_20260721.json"
    )
    plan_path = root / (
        "configs/experiment/innovation2/innovation2_output_prediction_op10_"
        "present_r3_easy_bit_confirm_seed0.json"
    )
    launch_path = root / (
        "configs/remote/generated/launch_i2_output_prediction_op10_present_"
        "r3_easy_bit_confirm_gpu0_20260721.cmd"
    )
    run_path = root / (
        "configs/remote/generated/run_i2_output_prediction_op10_present_r3_"
        "easy_bit_confirm_gpu0_20260721.cmd"
    )
    monitor_path = root / (
        "configs/remote/generated/monitor_i2_output_prediction_op10_present_"
        "r3_easy_bit_confirm_gpu0_20260721.sh"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch = launch_path.read_text(encoding="utf-8")
    run = run_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")

    assert config["discovery_total_rows"] == 1 << 16
    assert config["fresh_confirmation_total_rows"] == 1 << 16
    assert config["expected_result_rows"] == 384
    assert config["training"] is False
    assert config["sample_classification"] is False
    assert plan["target"] == "selected_true_ciphertext_output_bits"
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in launch + run
    assert "EnableDelayedExpansion" not in launch + run
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run
    assert "_result_branch_pushed.marker" in launch + run + monitor
    assert "source_checkpoint_manifest.json" in run
    assert "expected_rows=384" in run
    assert "_plot_deferred.marker" in run
    assert "plot-innovation2-output-bit-discovery" not in run
    assert "plot-innovation2-output-bit-discovery" in monitor
    assert "waiting_for_verified_op9_retrieval" in monitor
    assert "sleep 120" in monitor


def _synthetic_metric_rows(
    split: str,
    *,
    strong_bit: int,
    strong_model: str = "kimura_lstm_true_output",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in (
        "kimura_lstm_true_output",
        "matched_mlp_true_output",
        "kimura_lstm_label_shuffle",
    ):
        for bit in range(64):
            strong = model == strong_model and bit == strong_bit
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "msb_index": bit,
                    "threshold_accuracy": 0.515 if strong else 0.5,
                    "majority_accuracy": 0.5,
                    "accuracy_minus_majority": 0.015 if strong else 0.0,
                    "auc": 0.52 if strong else 0.5,
                    "mse": 0.24 if strong else 0.25,
                    "invalid_numpy_rint_rate": 0.0,
                }
            )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
