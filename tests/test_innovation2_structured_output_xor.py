from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_structured_output_xor import (
    render_structured_output_xor,
)
from blockcipher_nd.cli.run_innovation2_structured_output_xor import main
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.structured_output_xor_round_extension import (
    GEOMETRY_CONTROL_MASKS,
    STRUCTURED_MASKS,
    StructuredOutputXorConfig,
    adjudicate_structured_xor,
    validate_structured_xor_contract,
    xor_targets,
)


def test_xor_targets_equal_manual_true_output_xors() -> None:
    full_targets = np.zeros((3, 64), dtype=np.float32)
    full_targets[0, [0, 32]] = 1
    full_targets[1, [0, 2, 8]] = 1
    full_targets[2, [32, 34, 40, 42]] = 1

    targets = xor_targets(full_targets, STRUCTURED_MASKS)

    assert targets.shape == (3, 6)
    for row in range(3):
        for column, (_, bits, _) in enumerate(STRUCTURED_MASKS):
            expected = 0
            for bit in bits:
                expected ^= int(full_targets[row, bit])
            assert int(targets[row, column]) == expected


def test_mask_families_freeze_inverse_p_geometry_and_frequency() -> None:
    config = StructuredOutputXorConfig()
    total_rows = config.train_rows + config.test_rows
    plaintexts = np.arange(total_rows, dtype=np.uint64)
    features = np.stack(
        [
            ((plaintexts >> shift) & 1).astype(np.float32)
            for shift in range(63, -1, -1)
        ],
        axis=1,
    )
    from blockcipher_nd.ciphers.spn.present import Present80

    cipher = Present80(rounds=4, key=1)
    ciphertexts = np.asarray(
        [cipher.encrypt(int(value)) for value in plaintexts], dtype=np.uint64
    )
    targets = np.stack(
        [
            ((ciphertexts >> shift) & 1).astype(np.float32)
            for shift in range(63, -1, -1)
        ],
        axis=1,
    )
    data = {
        "plaintexts": plaintexts,
        "features": features,
        "full_targets": targets,
        "secret_key": 1,
        "metadata": {"status": "complete", "completed_rows": total_rows},
    }

    checks = validate_structured_xor_contract(config, data)

    assert all(checks.values())
    assert [len(bits) for _, bits, _ in STRUCTURED_MASKS] == [2, 2, 2, 2, 4, 4]
    assert [len(bits) for _, bits, _ in GEOMETRY_CONTROL_MASKS] == [
        2,
        2,
        2,
        2,
        4,
        4,
    ]


def test_formal_gate_requires_structured_gain_over_all_controls() -> None:
    config = StructuredOutputXorConfig(mode="round_extension")
    training = _synthetic_training(
        structured_auc=0.54,
        geometry_auc=0.51,
        shuffle_auc=0.5,
        derived_auc=0.51,
        component_auc=0.52,
    )

    gate = adjudicate_structured_xor(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert gate["metrics"]["passed_mask_count"] == 6
    assert gate["metrics"]["pair_family_passed"] is True
    assert gate["metrics"]["role4_family_passed"] is True


def test_end_to_end_smoke_emits_four_models_and_true_xor_rows(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "op12"

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
    assert metadata["config"]["rounds"] == 4
    assert metadata["config"]["seed"] == 1
    assert metadata["sample_classification"] is False
    assert len(results) == 26
    assert len(checkpoints) == 4


def test_plot_explains_true_output_xor_and_strong_controls(tmp_path: Path) -> None:
    training = _synthetic_training(
        structured_auc=0.54,
        geometry_auc=0.51,
        shuffle_auc=0.5,
        derived_auc=0.51,
        component_auc=0.52,
    )
    gate = adjudicate_structured_xor(
        StructuredOutputXorConfig(mode="round_extension"),
        {"valid": True},
        training,
    )
    summary = {
        "metadata": {"mode": "round_extension"},
        "result_rows": training["rows"],
        "gate": gate,
    }
    output = tmp_path / "curves.svg"

    render_structured_output_xor(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "PRESENT四轮多输出bit结构化XOR预测" in svg
    assert "不是样本分类" in svg
    assert "同重量几何控制" in svg
    assert "单bit派生parity" in svg
    assert "最佳组成bit" in svg


def test_result_index_names_op12_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_op12_present_r4_structured_xor_key1_gpu0_20260721"
    )

    assert "OP12" in name
    assert "PRESENT四轮" in name
    assert "结构化XOR真实值预测" in name


def test_remote_package_freezes_r4_xor_protocol_and_local_plotting() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / (
        "configs/remote/innovation2_output_prediction_op12_present_r4_"
        "structured_xor_key1_gpu0_20260721.json"
    )
    plan_path = root / (
        "configs/experiment/innovation2/innovation2_output_prediction_op12_"
        "present_r4_structured_xor_key1.json"
    )
    launch_path = root / (
        "configs/remote/generated/launch_i2_output_prediction_op12_present_r4_"
        "structured_xor_key1_gpu0_20260721.cmd"
    )
    run_path = root / (
        "configs/remote/generated/run_i2_output_prediction_op12_present_r4_"
        "structured_xor_key1_gpu0_20260721.cmd"
    )
    monitor_path = root / (
        "configs/remote/generated/monitor_i2_output_prediction_op12_present_r4_"
        "structured_xor_key1_gpu0_20260721.sh"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch = launch_path.read_text(encoding="utf-8")
    run = run_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")

    assert config["rounds"] == 4
    assert config["seed"] == 1
    assert config["train_total_rows"] == 1 << 17
    assert config["test_total_rows"] == 1 << 16
    assert "samples_per_class" not in config
    assert config["expected_result_rows"] == 26
    assert config["expected_history_rows"] == 400
    assert config["sample_classification"] is False
    assert config["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )
    assert plan["common"]["target"] == (
        "six_preregistered_true_ciphertext_output_xor_values"
    )
    assert len(plan["structured_masks"]) == 6
    assert len(plan["geometry_control_masks"]) == 6
    assert len(plan["rows"]) == 4
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in launch + run
    assert "EnableDelayedExpansion" not in launch + run
    assert "source_expected_commit.txt" in launch + run
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run
    assert "--mode round_extension" in run
    assert "expected_rows=26" in run
    assert "len(history)==400" in run
    assert '"%RESULTS_DIR%\\data\\%%F"' in run
    assert '"%RESULTS_DIR%\\models\\%%F"' in run
    assert "_plot_deferred.marker" in run
    assert "plot-innovation2-structured-output-xor" not in run
    assert "plot-innovation2-structured-output-xor" in monitor
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


def _synthetic_training(
    *,
    structured_auc: float,
    geometry_auc: float,
    shuffle_auc: float,
    derived_auc: float,
    component_auc: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for bit in (0, 2, 8, 10, 32, 34, 40, 42):
        rows.append(
            {
                "model": "selected8_mlp_true_output",
                "msb_index": bit,
                "threshold_accuracy": 0.52,
                "majority_accuracy": 0.5,
                "accuracy_minus_majority": 0.02,
                "auc": component_auc,
                "mse": 0.24,
            }
        )
    for model, masks, auc in (
        ("structured6_mlp_true_xor", STRUCTURED_MASKS, structured_auc),
        ("geometry6_mlp_true_xor", GEOMETRY_CONTROL_MASKS, geometry_auc),
        ("structured6_mlp_label_shuffle", STRUCTURED_MASKS, shuffle_auc),
    ):
        for index, (name, bits, family) in enumerate(masks):
            row: dict[str, object] = {
                "model": model,
                "mask_name": name,
                "mask_bits": list(bits),
                "family": family,
                "threshold_accuracy": 0.52 if model.startswith("structured6_mlp_true") else 0.5,
                "majority_accuracy": 0.5,
                "accuracy_minus_majority": 0.02 if model.startswith("structured6_mlp_true") else 0.0,
                "auc": auc,
                "mse": 0.24,
            }
            if model == "structured6_mlp_true_xor":
                row.update(
                    {
                        "geometry_auc": geometry_auc,
                        "shuffle_auc": shuffle_auc,
                        "derived_auc": derived_auc,
                        "best_component_auc": component_auc,
                        "auc_minus_geometry": structured_auc - geometry_auc,
                        "auc_minus_shuffle": structured_auc - shuffle_auc,
                        "auc_minus_derived": structured_auc - derived_auc,
                        "auc_minus_best_component": structured_auc - component_auc,
                        "paired_geometry_control": GEOMETRY_CONTROL_MASKS[index][0],
                    }
                )
            rows.append(row)
    return {
        "rows": rows,
        "summaries": [{"model": index} for index in range(4)],
        "history": [
            {"epoch": 1, "model": index}
            for index in range(4)
            for _ in range(StructuredOutputXorConfig().epochs)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(4)],
    }
