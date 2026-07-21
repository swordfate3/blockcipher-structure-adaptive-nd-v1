from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_innovation2_selected_output_bit_head import (
    render_selected_output_bit_head,
)
from blockcipher_nd.cli.run_innovation2_selected_output_bit_head import main
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    SelectedOutputMlp,
    adjudicate_selected_output_head,
    parameter_counts,
)


def test_selected_output_head_emits_only_preregistered_eight_bits() -> None:
    model = SelectedOutputMlp(hidden_dim=16)

    output = model(torch.zeros((3, 64)))

    assert output.shape == (3, 8)
    assert SELECTED_MSB_INDICES == (0, 2, 8, 10, 32, 34, 40, 42)


def test_selected_output_head_is_close_to_full64_parameter_budget() -> None:
    counts = parameter_counts(SelectedOutputBitHeadConfig())

    relative_gap = abs(counts["selected8_mlp"] - counts["full64_mlp"]) / counts[
        "full64_mlp"
    ]
    assert relative_gap < 0.03


def test_cross_key_gate_separates_position_confirmation_from_head_gain() -> None:
    config = SelectedOutputBitHeadConfig(mode="independent_key_confirmation")
    rows = _synthetic_rows(selected_auc=0.53, full_auc=0.529, shuffle_auc=0.5)
    training = {
        "rows": rows,
        "summaries": [{"model": str(index)} for index in range(3)],
        "history": [
            {"model": model, "epoch": 1}
            for model in range(3)
            for _ in range(config.epochs)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(3)],
    }

    gate = adjudicate_selected_output_head(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert gate["metrics"]["confirmed_count"] == 8
    assert gate["metrics"]["dedicated_head_supported"] is False
    assert gate["decision"].endswith("without_head_gain")


def test_end_to_end_smoke_uses_seed1_true_outputs_and_three_rows(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "op11"

    exit_code = main(["--mode", "smoke", "--output-root", str(output_root)])

    assert exit_code == 0
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (output_root / "metadata.json").read_text(encoding="utf-8")
    )
    results = (output_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    checkpoints = json.loads(
        (output_root / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["execution_checks"].values())
    assert metadata["config"]["seed"] == 1
    assert metadata["selected_msb_indices"] == list(SELECTED_MSB_INDICES)
    assert metadata["sample_classification"] is False
    assert len(results) == 24
    assert len(checkpoints) == 3


def test_plot_names_independent_key_and_matched_shuffle(tmp_path: Path) -> None:
    selected_bits = list(SELECTED_MSB_INDICES)
    bit_rows = _synthetic_rows(selected_auc=0.53, full_auc=0.52, shuffle_auc=0.5)
    gate = adjudicate_selected_output_head(
        SelectedOutputBitHeadConfig(mode="independent_key_confirmation"),
        {"valid": True},
        {
            "rows": bit_rows,
            "summaries": [{"model": str(index)} for index in range(3)],
            "history": [
                {"epoch": 1}
                for _ in range(3)
                for _ in range(SelectedOutputBitHeadConfig().epochs)
            ],
            "checkpoints": [{"sha256": "hash"} for _ in range(3)],
        },
    )
    summary = {
        "metadata": {
            "mode": "independent_key_confirmation",
            "selected_msb_indices": selected_bits,
        },
        "bit_rows": bit_rows,
        "gate": gate,
    }
    output = tmp_path / "curves.svg"

    render_selected_output_bit_head(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "固定八输出bit的独立密钥确认" in svg
    assert "架构匹配标签打乱" in svg
    assert "不在本次结果中重新选位置" in svg
    assert "不是完整密文恢复" in svg


def test_result_index_names_op11_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721"
    )

    assert "OP11" in name
    assert "PRESENT三轮" in name
    assert "八个真实密文输出bit" in name


def test_remote_package_freezes_independent_key_selected8_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / (
        "configs/remote/innovation2_output_prediction_op11_present_r3_"
        "selected8_key1_gpu0_20260721.json"
    )
    plan_path = root / (
        "configs/experiment/innovation2/innovation2_output_prediction_op11_"
        "present_r3_selected8_key1.json"
    )
    launch_path = root / (
        "configs/remote/generated/launch_i2_output_prediction_op11_present_r3_"
        "selected8_key1_gpu0_20260721.cmd"
    )
    run_path = root / (
        "configs/remote/generated/run_i2_output_prediction_op11_present_r3_"
        "selected8_key1_gpu0_20260721.cmd"
    )
    monitor_path = root / (
        "configs/remote/generated/monitor_i2_output_prediction_op11_present_r3_"
        "selected8_key1_gpu0_20260721.sh"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch = launch_path.read_text(encoding="utf-8")
    run = run_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")

    assert config["train_total_rows"] == 1 << 17
    assert config["test_total_rows"] == 1 << 16
    assert "samples_per_class" not in config
    assert config["expected_result_rows"] == 24
    assert config["seed"] == 1
    assert config["sample_classification"] is False
    assert config["selected_msb_indices"] == list(SELECTED_MSB_INDICES)
    assert config["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )
    assert plan["common"]["target"] == (
        "eight_preregistered_msb_first_true_ciphertext_bits"
    )
    assert plan["common"]["sample_classification"] is False
    assert len(plan["rows"]) == 3
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in launch + run
    assert "EnableDelayedExpansion" not in launch + run
    assert "source_expected_commit.txt" in launch + run
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run
    assert "--mode independent_key_confirmation" in run
    assert "expected_rows=24" in run
    assert '"%RESULTS_DIR%\\data\\%%F"' in run
    assert '"%RESULTS_DIR%\\models\\%%F"' in run
    assert "_plot_deferred.marker" in run
    assert "plot-innovation2-selected-output-bit-head" not in run
    assert "plot-innovation2-selected-output-bit-head" in monitor
    assert "visual_qa_pending.marker" in monitor
    assert "_result_branch_pushed.marker" in run + monitor
    assert '${RUN_ID}_failed.marker' in monitor
    assert 'rm -rf "${MONITOR_ROOT}/${RUN_ID}/logs"' in monitor
    assert "sleep 120" in monitor
    assert "scripts/index-results" in monitor
    assert "C:\\Users" not in run.replace(
        "C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519",
        "",
    )


def _synthetic_rows(
    *, selected_auc: float, full_auc: float, shuffle_auc: float
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, auc, margin in (
        ("full64_mlp_true_output", full_auc, 0.012),
        ("selected8_mlp_true_output", selected_auc, 0.015),
        ("selected8_mlp_label_shuffle", shuffle_auc, 0.0),
    ):
        for bit in SELECTED_MSB_INDICES:
            rows.append(
                {
                    "model": model,
                    "msb_index": bit,
                    "threshold_accuracy": 0.515 if margin else 0.5,
                    "majority_accuracy": 0.5,
                    "accuracy_minus_majority": margin,
                    "auc": auc,
                    "mse": 0.24 if margin else 0.25,
                    "invalid_numpy_rint_rate": 0.0,
                    "test_target_identity": "true_selected_ciphertext_targets",
                }
            )
    return rows
