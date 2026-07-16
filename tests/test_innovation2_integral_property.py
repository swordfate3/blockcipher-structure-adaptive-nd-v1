from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.cli.run_innovation2_integral_property import main
from blockcipher_nd.cli.evaluate_innovation2_integral_ranking import (
    main as ranking_main,
    render_ranking_svg,
)
from blockcipher_nd.cli.summarize_innovation2_integral_ranking import (
    main as ranking_summary_main,
)
from blockcipher_nd.tasks.innovation2.integral_property_calibration import (
    IntegralCalibrationConfig,
    adjudicate_calibration,
    binary_log_loss,
    fit_monotone_affine_logit_calibration,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    INPUT_BITS,
    IntegralExperimentConfig,
    IntegralStructure,
    adjudicate,
    build_integral_split,
    integral_mask_parity,
    make_structure_splits,
    summarize_splits,
)
from blockcipher_nd.tasks.innovation2.integral_property_ranking import (
    adjudicate_joint_integral_ranking,
    evaluate_integral_ranking,
    spearman_correlation,
)


def test_integral_mask_parity_matches_manual_masked_output_xor() -> None:
    structure = IntegralStructure(
        structure_id="manual",
        active_nibble=13,
        output_nibble=8,
        output_mask=0b1111,
        fixed_plaintext=0x120456789ABCDEF0,
    )
    cipher = Present80(rounds=5, key=0x00010203040506070809)

    manual = 0
    for value in range(16):
        ciphertext = cipher.encrypt(structure.plaintext(value))
        nibble = (ciphertext >> (4 * structure.output_nibble)) & 0xF
        manual ^= (nibble & structure.output_mask).bit_count() & 1

    assert integral_mask_parity(cipher, structure) == manual == 1
    features = structure.feature_vector()
    assert features.shape == (INPUT_BITS,)
    assert int(features.sum()) >= 3


def test_integral_splits_repeat_structure_over_keys_and_remain_disjoint() -> None:
    train = build_integral_split(
        name="train",
        rounds=5,
        structure_count=8,
        key_count=4,
        structure_seed=101,
        key_seed=201,
    )
    validation = build_integral_split(
        name="validation",
        rounds=5,
        structure_count=8,
        key_count=4,
        structure_seed=301,
        key_seed=401,
    )
    test = build_integral_split(
        name="test",
        rounds=5,
        structure_count=8,
        key_count=4,
        structure_seed=501,
        key_seed=601,
    )
    summary = summarize_splits(
        {"train": train, "validation": validation, "test": test}
    )

    assert train.dataset.features.shape == (32, INPUT_BITS)
    assert train.dataset.labels.shape == (32,)
    np.testing.assert_array_equal(
        train.dataset.features[0],
        train.dataset.features[3],
    )
    assert summary["status"] == "pass"
    assert summary["structure_splits_disjoint"] is True
    assert summary["key_splits_disjoint"] is True


def test_geometry_disjoint_structure_splits_withhold_all_geometry_ids() -> None:
    counts = {"train": 32, "validation": 8, "calibration": 8, "test": 8}
    structures = make_structure_splits(
        split_counts=counts,
        seed=7,
        structure_split_mode="geometry-disjoint",
        random_seed_offsets={
            "train": 101,
            "validation": 301,
            "calibration": 701,
            "test": 501,
        },
    )

    geometry_sets = {
        name: {structure.geometry_id for structure in split}
        for name, split in structures.items()
    }
    assert {name: len(split) for name, split in structures.items()} == counts
    assert all(len(geometry_sets[name]) == counts[name] for name in counts)
    assert all(
        geometry_sets[left].isdisjoint(geometry_sets[right])
        for index, left in enumerate(counts)
        for right in tuple(counts)[index + 1 :]
    )
    assert all(
        structure.fixed_plaintext & (0xF << (4 * structure.active_nibble)) == 0
        for split in structures.values()
        for structure in split
    )


def test_diagnostic_gate_requires_candidate_to_beat_linear_and_shuffle() -> None:
    config = IntegralExperimentConfig(
        run_id="gate",
        train_structures=1,
        validation_structures=1,
        test_structures=1,
        train_keys=1,
        validation_keys=1,
        test_keys=1,
        epochs=1,
    )
    rows = [
        {
            "role": "anchor",
            "test_auc": 0.58,
            "test_structure_rate_mae": 0.24,
        },
        {
            "role": "candidate",
            "test_auc": 0.63,
            "test_structure_rate_mae": 0.20,
        },
        {
            "role": "control",
            "test_auc": 0.50,
            "test_structure_rate_mae": 0.26,
        },
    ]
    dataset_summary = {
        "structure_splits_disjoint": True,
        "key_splits_disjoint": True,
        "splits": {
            name: {"q0_rows": 1, "q1_rows": 1}
            for name in ("train", "validation", "test")
        },
    }

    gate = adjudicate(config, rows, dataset_summary)

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_integral_property_advance_multiseed"


def test_cli_writes_complete_smoke_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    status = main(
        [
            "--run-id",
            "i2_test_smoke",
            "--output-root",
            str(output),
            "--train-structures",
            "32",
            "--validation-structures",
            "16",
            "--test-structures",
            "16",
            "--train-keys",
            "4",
            "--validation-keys",
            "8",
            "--test-keys",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--hidden-bits",
            "16",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--gate-mode",
            "smoke",
        ]
    )

    assert status == 0
    assert (output / "results.jsonl").is_file()
    assert (output / "progress.jsonl").is_file()
    assert (output / "dataset_summary.json").is_file()
    assert (output / "structure_rates.csv").is_file()
    assert (output / "gate.json").is_file()
    assert (output / "curves.svg").is_file()
    assert (output / "history.csv").is_file()
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    svg = (output / "curves.svg").read_text(encoding="utf-8")
    assert [row["role"] for row in rows] == ["anchor", "candidate", "control"]
    assert gate["status"] == "pass"
    assert gate["readiness_checks"]["all_splits_have_both_labels"] is True
    assert "训练 32 个结构" in svg
    assert "每个积分集合 16 个明文" in svg
    assert "同输入线性基线" in svg
    assert "训练标签打乱 MLP 控制" in svg
    assert "训练 0/类" not in svg


def test_monotone_logit_calibration_preserves_order_and_improves_fit() -> None:
    probabilities = np.repeat(np.array([0.10, 0.30, 0.60, 0.80]), 4)
    labels = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        dtype=np.uint8,
    )

    calibration = fit_monotone_affine_logit_calibration(probabilities, labels)
    calibrated = calibration.transform(probabilities)

    assert calibration.slope > 0.0
    assert np.all(np.diff(calibrated.reshape(4, 4)[:, 0]) > 0.0)
    assert binary_log_loss(labels, calibrated) < binary_log_loss(
        labels, probabilities
    )


def test_calibration_gate_requires_probability_and_stability_margins() -> None:
    config = IntegralCalibrationConfig(
        run_id="calibration-gate",
        train_structures=1,
        validation_structures=1,
        calibration_structures=1,
        test_structures=1,
        train_keys=1,
        validation_keys=1,
        calibration_keys=1,
        test_keys=1,
        stability_test_keys=1,
        epochs=1,
    )
    rows = [
        {
            "role": "anchor",
            "test_auc": 0.61,
            "calibration_slope": 1.0,
            "calibration_intercept": 0.0,
            "stability_calibrated_structure_rate_mae_256key": 0.105,
            "observed_rate_32_256_mae": 0.04,
        },
        {
            "role": "candidate",
            "test_auc": 0.66,
            "calibration_slope": 1.0,
            "calibration_intercept": 0.0,
            "stability_calibrated_structure_rate_mae_256key": 0.085,
            "observed_rate_32_256_mae": 0.04,
        },
        {
            "role": "control",
            "test_auc": 0.51,
            "calibration_slope": 1.0,
            "calibration_intercept": 0.0,
            "stability_calibrated_structure_rate_mae_256key": 0.20,
            "observed_rate_32_256_mae": 0.04,
        },
    ]
    summary = {
        "structure_splits_disjoint": True,
        "key_splits_disjoint": True,
        "stability_structures_match_test": True,
        "splits": {
            name: {"q0_rows": 1, "q1_rows": 1}
            for name in (
                "train",
                "validation",
                "calibration",
                "test",
                "stability",
            )
        },
    }

    gate = adjudicate_calibration(config, rows, summary)

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_integral_calibration_advance_seed1_geometry"
    )


def test_cli_writes_complete_calibration_smoke_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "calibration-smoke"
    status = main(
        [
            "--run-id",
            "i2_test_calibration_smoke",
            "--output-root",
            str(output),
            "--train-structures",
            "32",
            "--validation-structures",
            "16",
            "--calibration-structures",
            "16",
            "--test-structures",
            "16",
            "--train-keys",
            "4",
            "--validation-keys",
            "8",
            "--calibration-keys",
            "8",
            "--test-keys",
            "8",
            "--stability-test-keys",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--hidden-bits",
            "16",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--gate-mode",
            "calibration-smoke",
            "--structure-split-mode",
            "geometry-disjoint",
        ]
    )

    assert status == 0
    assert (output / "results.jsonl").is_file()
    assert (output / "progress.jsonl").is_file()
    assert (output / "dataset_summary.json").is_file()
    assert (output / "structure_rates.csv").is_file()
    assert (output / "observation_predictions.csv").is_file()
    assert (output / "gate.json").is_file()
    assert (output / "curves.svg").is_file()
    assert (output / "history.csv").is_file()
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (output / "dataset_summary.json").read_text(encoding="utf-8")
    )
    prediction_lines = (output / "observation_predictions.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_integral_geometry_holdout_implementation_ready"
    )
    assert summary["structure_splits_disjoint"] is True
    assert summary["key_splits_disjoint"] is True
    assert summary["stability_structures_match_test"] is True
    assert summary["structure_split_mode"] == "geometry-disjoint"
    assert summary["geometry_splits_disjoint"] is True
    assert summary["one_structure_per_geometry"] is True
    assert gate["structure_split_mode"] == "geometry-disjoint"
    assert gate["readiness_checks"][
        "geometry_splits_disjoint_when_required"
    ] is True
    assert len(prediction_lines) == 1 + 3 * (16 * (8 + 8 + 8 + 16))


def test_spearman_correlation_uses_average_ranks_for_ties() -> None:
    values = [0.1, 0.1, 0.4, 0.8]

    assert spearman_correlation(values, values) == 1.0
    assert spearman_correlation(values, list(reversed(values))) < -0.80


def test_integral_ranking_gate_requires_attributed_top16_utility() -> None:
    source_rows = _ranking_source_rows()
    source_gate = _ranking_source_gate()

    result = evaluate_integral_ranking(
        run_id="i2-ranking-test",
        source_rows=source_rows,
        source_gate=source_gate,
    )

    gate = result["gate"]
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_integral_ranking_utility_advance_independent_confirmation"
    )
    assert gate["training_performed"] is False
    assert all(gate["checks"].values())
    assert len(result["rows"]) == 3
    assert len(result["ranking_rows"]) == 128
    assert sum(
        row["candidate_selected_top16"] for row in result["ranking_rows"]
    ) == 16


def test_integral_ranking_preserves_source_seed_and_confirmation_decision() -> None:
    result = evaluate_integral_ranking(
        run_id="i2-ranking-seed1-test",
        source_rows=_ranking_source_rows(),
        source_gate=_ranking_source_gate(seed=1),
    )

    assert {row["seed"] for row in result["rows"]} == {1}
    assert result["gate"]["decision"] == (
        "innovation2_integral_ranking_utility_independent_confirmation_passed"
    )


def test_geometry_holdout_ranking_uses_geometry_specific_decision(
    tmp_path: Path,
) -> None:
    source_rows = _ranking_source_rows()
    for index, row in enumerate(source_rows):
        row["geometry_id"] = f"geometry-{index:03d}"
        row["structure_split_mode"] = "geometry-disjoint"
    source_gate = _ranking_source_gate()
    source_gate["structure_split_mode"] = "geometry-disjoint"

    result = evaluate_integral_ranking(
        run_id="i2-geometry-ranking-test",
        source_rows=source_rows,
        source_gate=source_gate,
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_integral_geometry_holdout_passed"
    )
    assert result["gate"]["structure_split_mode"] == "geometry-disjoint"
    curves_path = tmp_path / "geometry-ranking.svg"
    render_ranking_svg(result["rows"], result["gate"], curves_path)
    svg = curves_path.read_text(encoding="utf-8")
    assert "创新2 E4" in svg
    assert "未见位置与掩码组合" in svg


def test_joint_integral_ranking_requires_seed0_and_seed1_passes() -> None:
    seed0 = evaluate_integral_ranking(
        run_id="ranking-seed0",
        source_rows=_ranking_source_rows(),
        source_gate=_ranking_source_gate(seed=0),
    )["gate"]
    seed1 = evaluate_integral_ranking(
        run_id="ranking-seed1",
        source_rows=_ranking_source_rows(),
        source_gate=_ranking_source_gate(seed=1),
    )["gate"]
    seed0["run_id"] = "i2_ranking_seed0"
    seed1["run_id"] = "i2_ranking_seed1"

    result = adjudicate_joint_integral_ranking(
        run_id="i2-ranking-joint-test",
        source_gates=[seed1, seed0],
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_integral_ranking_utility_two_seed_confirmed"
    )
    assert result["gate"]["checks"] == {
        "exact_seed0_seed1_pair": True,
        "frozen_thresholds_match": True,
        "both_seed_gates_pass": True,
    }
    assert [row["seed"] for row in result["rows"]] == [0, 1]


def test_integral_ranking_cli_writes_read_only_e2_artifacts(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    source_rows = _ranking_source_rows()
    with (source_root / "structure_rates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    (source_root / "gate.json").write_text(
        json.dumps(_ranking_source_gate()),
        encoding="utf-8",
    )

    status = ranking_main(
        [
            "--run-id",
            "i2-ranking-cli-test",
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "ranking.csv",
        "gate.json",
        "curves.svg",
        "progress.jsonl",
    ):
        assert (output_root / name).is_file()
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line)
        for line in (output_root / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    svg = (output_root / "curves.svg").read_text(encoding="utf-8")
    assert gate["training_performed"] is False
    assert [item["event"] for item in progress] == ["run_start", "run_done"]
    assert "积分输出平衡候选排序审判" in svg
    assert "top-16" in svg
    assert "训练轮次" not in svg


def test_joint_integral_ranking_cli_writes_complete_artifacts(
    tmp_path: Path,
) -> None:
    gate_paths: list[Path] = []
    for seed in (0, 1):
        source_gate = _ranking_source_gate(seed=seed)
        gate = evaluate_integral_ranking(
            run_id=f"i2-ranking-seed{seed}",
            source_rows=_ranking_source_rows(),
            source_gate=source_gate,
        )["gate"]
        gate["run_id"] = f"i2_ranking_seed{seed}"
        gate_path = tmp_path / f"seed{seed}-gate.json"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        gate_paths.append(gate_path)
    output_root = tmp_path / "joint"

    status = ranking_summary_main(
        [
            "--run-id",
            "i2-ranking-joint-cli-test",
            "--source-gates",
            str(gate_paths[0]),
            str(gate_paths[1]),
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "seed_metrics.csv",
        "gate.json",
        "curves.svg",
        "progress.jsonl",
    ):
        assert (output_root / name).is_file()
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    svg = (output_root / "curves.svg").read_text(encoding="utf-8")
    assert gate["status"] == "pass"
    assert gate["training_performed"] is False
    assert "双 seed 联合裁决" in svg


def _ranking_source_gate(seed: int = 0) -> dict[str, object]:
    return {
        "status": "hold",
        "decision": "innovation2_integral_rate_target_unstable",
        "run_id": f"i2_present_r5_integral_parity_calibration_seed{seed}",
    }


def _ranking_source_rows() -> list[dict[str, str]]:
    structure_count = 128
    observed = np.linspace(0.0, 1.0, structure_count)
    candidate = observed.copy()
    linear = observed[::-1].copy()
    control_selection_order: list[int] = []
    for index in range(8):
        control_selection_order.extend((index, structure_count - 1 - index))
    control_selection_order.extend(
        index
        for index in range(structure_count)
        if index not in set(control_selection_order)
    )
    control = np.empty(structure_count, dtype=np.float64)
    for rank, index in enumerate(control_selection_order):
        control[index] = rank / float(structure_count - 1)

    rows: list[dict[str, str]] = []
    for index in range(structure_count):
        rows.append(
            {
                "structure_id": f"test-{index:06d}",
                "signature": f"a{index % 16:02d}-o{(index // 2) % 16:02d}",
                "active_nibble": str(index % 16),
                "output_nibble": str((index // 2) % 16),
                "output_mask": f"{1 + index % 15:04b}",
                "observed_q1_rate_256key": str(observed[index]),
                "linear_same_input_calibrated_predicted_q1_rate": str(
                    linear[index]
                ),
                "structure_mlp_calibrated_predicted_q1_rate": str(
                    candidate[index]
                ),
                "structure_mlp_shuffled_labels_calibrated_predicted_q1_rate": str(
                    control[index]
                ),
            }
        )
    return rows
