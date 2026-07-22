from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli.plot_innovation2_selected_output_position_bound_spn_rescnn import (
    render_position_bound_spn_rescnn,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2 import (
    selected_output_position_bound_spn_rescnn as task,
)
from blockcipher_nd.tasks.innovation2.selected_output_position_bound_spn_rescnn import (
    MODEL_SPECS,
    OPF1_GATE_SHA256,
    OPF1_RELEASE_DECISION,
    OPF1_RUN_ID,
    OPF1_TEST_PLAINTEXTS_RAW_SHA256,
    OPF1_TRAIN_PLAINTEXTS_RAW_SHA256,
    PositionBoundSpnResCnnConfig,
    adjudicate_position_bound,
    authorize_scale_extension_from_opf1_gate,
    prepare_position_bound_data,
    train_position_bound_matrix,
    validate_position_bound_contract,
)


def _opf1_gate() -> dict[str, object]:
    return {
        "run_id": OPF1_RUN_ID,
        "status": "hold",
        "decision": OPF1_RELEASE_DECISION,
        "protocol_checks": {"valid": True},
        "execution_checks": {"valid": True},
        "metrics": {
            "mean_auc_by_model": {
                "selected8_position_head_spn_rescnn_exact_p_true_output": (
                    0.5137553581211971
                ),
                "selected8_position_head_spn_rescnn_exact_p_label_shuffle": (
                    0.5009341432724128
                ),
            }
        },
    }


def _smoke_config() -> PositionBoundSpnResCnnConfig:
    return PositionBoundSpnResCnnConfig(
        run_id="i2_output_prediction_opf2_scale_test",
        mode="scale_extension_smoke",
        rounds=4,
        train_rows=64,
        test_rows=64,
        rescnn_channels=4,
        epochs=1,
        batch_size=16,
        data_chunk_rows=16,
        maximum_parameter_gap=1.0,
    )


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


def _words(features: np.ndarray) -> np.ndarray:
    weights = np.left_shift(np.uint64(1), np.arange(63, -1, -1, dtype=np.uint64))
    return (features.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def test_scale_extension_changes_only_training_rows() -> None:
    opf1 = PositionBoundSpnResCnnConfig.round_extension(device="cpu")
    opf2 = PositionBoundSpnResCnnConfig.scale_extension(device="cpu")

    ignored = {"run_id", "mode", "train_rows"}
    assert {
        key: value for key, value in opf1.__dict__.items() if key not in ignored
    } == {key: value for key, value in opf2.__dict__.items() if key not in ignored}
    assert (opf1.train_rows, opf2.train_rows) == (1 << 17, 1 << 20)
    with pytest.raises(ValueError, match="mode-matched PRESENT round count"):
        PositionBoundSpnResCnnConfig(mode="scale_extension", rounds=3)


def test_scale_extension_authority_is_frozen_to_completed_opf1() -> None:
    authorize_scale_extension_from_opf1_gate(_opf1_gate())
    wrong = _opf1_gate()
    wrong["decision"] = "wrong"
    with pytest.raises(ValueError, match="requires the frozen completed OPF1 gate"):
        authorize_scale_extension_from_opf1_gate(wrong)


def test_scale_smoke_split_excludes_reserved_test_from_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _smoke_config()
    data = prepare_position_bound_data(config, tmp_path)
    captured: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_train(
        _config: object,
        *,
        model_name: str,
        architecture: str,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        test_features: np.ndarray,
        test_targets: np.ndarray,
        output_root: Path,
        progress: object,
    ) -> dict[str, object]:
        del architecture, train_targets, test_targets, output_root, progress
        captured.append((train_features.copy(), test_features.copy()))
        return {
            "rows": [
                {
                    "model": model_name,
                    "msb_index": bit,
                    "threshold_accuracy": 0.5,
                    "majority_accuracy": 0.5,
                    "accuracy_minus_majority": 0.0,
                    "auc": 0.5,
                    "mse": 0.25,
                }
                for bit in config.selected_msb_indices
            ],
            "summary": {"model": model_name},
            "history": [{"model": model_name, "epoch": 1}],
            "checkpoint": {"sha256": "hash"},
        }

    monkeypatch.setattr(task, "_train_one_model", fake_train)
    checks = validate_position_bound_contract(config, data)
    training = train_position_bound_matrix(config, data, tmp_path)

    assert all(checks.values())
    assert data["split_layout"]["train_index_segments"] == [[0, 32], [96, 128]]
    assert data["split_layout"]["test_index_segment"] == [32, 96]
    expected_train = np.concatenate((data["plaintexts"][:32], data["plaintexts"][96:]))
    expected_test = np.asarray(data["plaintexts"][32:96])
    assert len(captured) == 5
    assert np.array_equal(_words(captured[0][0]), expected_train)
    assert np.array_equal(_words(captured[0][1]), expected_test)
    assert set(map(int, expected_train)).isdisjoint(set(map(int, expected_test)))
    assert len(training["rows"]) == 40
    assert len(training["history"]) == 5
    assert len(training["checkpoints"]) == 5


def test_scale_gate_can_support_or_reject_training_scale() -> None:
    config = PositionBoundSpnResCnnConfig.scale_extension(device="cpu")
    supported = adjudicate_position_bound(
        config,
        {"valid": True},
        _training(config, (0.55, 0.55, 0.60, 0.59, 0.50)),
        reference_gate=_opf1_gate(),
    )
    held = adjudicate_position_bound(
        config,
        {"valid": True},
        _training(config, (0.51, 0.51, 0.52, 0.52, 0.50)),
        reference_gate=_opf1_gate(),
    )

    assert supported["status"] == "pass"
    assert supported["decision"] == "innovation2_position_bound_r4_scale_supported"
    assert not supported["metrics"]["formal_checks"][
        "candidate_minus_wrong_mean_auc_at_least_0_020"
    ]
    assert all(supported["metrics"]["round_extension_checks"].values())
    assert supported["metrics"]["opf1_reference_exact_p_mean_auc"] == pytest.approx(
        0.5137553581211971
    )
    assert held["status"] == "hold"
    assert held["decision"] == "innovation2_position_bound_r4_scale_not_supported"


def test_scale_gate_requires_accuracy_margin_and_four_output_bits() -> None:
    config = PositionBoundSpnResCnnConfig.scale_extension(device="cpu")
    exact_name = MODEL_SPECS[2][0]

    low_accuracy = _training(config, (0.51, 0.51, 0.60, 0.59, 0.50))
    for row in low_accuracy["rows"]:
        if row["model"] == exact_name:
            row["accuracy_minus_majority"] = 0.004
    low_accuracy_gate = adjudicate_position_bound(
        config,
        {"valid": True},
        low_accuracy,
        reference_gate=_opf1_gate(),
    )

    three_bits = _training(config, (0.51, 0.51, 0.60, 0.59, 0.50))
    for row in three_bits["rows"]:
        if row["model"] == exact_name and row["msb_index"] not in config.selected_msb_indices[:3]:
            row["auc"] = 0.54
    three_bits_gate = adjudicate_position_bound(
        config,
        {"valid": True},
        three_bits,
        reference_gate=_opf1_gate(),
    )

    assert low_accuracy_gate["status"] == "hold"
    assert not low_accuracy_gate["metrics"]["round_extension_checks"][
        "candidate_mean_accuracy_margin_at_least_0_005"
    ]
    assert three_bits_gate["status"] == "hold"
    assert three_bits_gate["metrics"]["passed_output_bit_count"] == 3
    assert not three_bits_gate["metrics"]["round_extension_checks"][
        "at_least_four_output_bits_pass"
    ]


def test_scale_gate_treats_protocol_failure_as_invalid_not_negative() -> None:
    config = PositionBoundSpnResCnnConfig.scale_extension(device="cpu")

    gate = adjudicate_position_bound(
        config,
        {"data_contract": True, "source_contract": False},
        _training(config, (0.51, 0.51, 0.60, 0.59, 0.50)),
        reference_gate=_opf1_gate(),
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation2_position_bound_spn_rescnn_protocol_invalid"


def test_scale_plot_and_result_index_are_plain_chinese(tmp_path: Path) -> None:
    config = _smoke_config()
    summary = {
        "metadata": {
            "mode": config.mode,
            "selected_msb_indices": list(config.selected_msb_indices),
            "config": {
                "rounds": 4,
                "train_rows": 64,
                "test_rows": 64,
                "epochs": 1,
            },
        },
        "bit_rows": _training(config, (0.51, 0.51, 0.52, 0.51, 0.50))["rows"],
        "gate": {
            "decision": "innovation2_position_bound_r4_scale_local_readiness_passed"
        },
    }
    output = tmp_path / "curves.svg"

    render_position_bound_spn_rescnn(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "OPF2" in svg
    assert "训练规模审判" in svg
    assert "OPF1原测试集" in svg
    assert display_name_for_run(
        "i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_test"
    ) == "创新2 OPF2：PRESENT四轮位置绑定网络2^20训练规模真实密文输出预测"


def test_opf2_remote_package_is_cached_matched_and_windows_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (
            root
            / "configs/experiment/innovation2/innovation2_output_prediction_"
            "opf2_present_r4_position_bound_spn_rescnn_2p20_key7.json"
        ).read_text(encoding="utf-8")
    )
    remote = json.loads(
        (
            root
            / "configs/remote/innovation2_output_prediction_opf2_present_r4_"
            "position_bound_spn_rescnn_2p20_key7_gpu0_20260722.json"
        ).read_text(encoding="utf-8")
    )
    run = (
        root
        / "configs/remote/generated/run_i2_output_prediction_opf2_present_r4_"
        "position_bound_spn_rescnn_2p20_key7_gpu0_20260722.cmd"
    ).read_text(encoding="utf-8")
    launch = (
        root
        / "configs/remote/generated/launch_i2_output_prediction_opf2_present_r4_"
        "position_bound_spn_rescnn_2p20_key7_gpu0_20260722.cmd"
    ).read_text(encoding="utf-8")
    monitor = (
        root
        / "configs/remote/generated/monitor_i2_output_prediction_opf2_present_r4_"
        "position_bound_spn_rescnn_2p20_key7_gpu0_20260722.sh"
    ).read_text(encoding="utf-8")
    windows_scripts = run + launch

    assert plan["only_changed_variable"] == {
        "field": "train_total_rows",
        "opf1_value": 131072,
        "opf2_value": 1048576,
    }
    assert plan["split_layout"]["test_index_segment"] == [131072, 196608]
    assert remote["train_total_rows"] == 1 << 20
    assert remote["test_total_rows"] == 1 << 16
    assert remote["expected_cache_rows"] == 1114112
    assert remote["opf1_gate_sha256"] == OPF1_GATE_SHA256
    assert remote["opf1_train_plaintexts_raw_sha256"] == (
        OPF1_TRAIN_PLAINTEXTS_RAW_SHA256
    )
    assert remote["opf1_test_plaintexts_raw_sha256"] == (
        OPF1_TEST_PLAINTEXTS_RAW_SHA256
    )
    assert "--mode scale_extension" in run
    assert "--opf1-gate" in run
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in windows_scripts.lower()
    assert "EnableDelayedExpansion" not in windows_scripts
    assert "!" not in windows_scripts
    assert "G:\\lxy" in windows_scripts
    assert "split_layout" in run and "split_layout" in monitor
    assert 'RESULT_REF="refs/remotes/origin/results/${RUN_ID}"' in monitor
    assert 'git fetch origin "refs/heads/results/${RUN_ID}:${RESULT_REF}"' in monitor
    assert 'git archive --format=tar "${RESULT_REF}"' in monitor
    assert "retrieve_verified_branch || exit 2" in monitor
    assert "source/results_archive" not in monitor
    assert "layout['train_index_segments']==[[0,131072],[196608,1114112]]" in monitor
    assert "len(results)==40 and len(history)==500 and len(checkpoints)==5" in monitor
    assert "git status --porcelain" in windows_scripts
    assert "source_expected_commit.txt" in windows_scripts
