from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.run_innovation2_high_round_integral import (
    _write_deferred_history_csv,
    _write_deferred_svg,
    main,
    write_training_artifacts,
)
from blockcipher_nd.models.structure.spn.present_integral_multiset import (
    PresentIntegralPaperMbconvAnchor,
    PresentIntegralStructuredResidualCandidate,
    integral_input_bits,
)
from blockcipher_nd.tasks.innovation2.high_round_integral_data import (
    NEGATIVE_MODE,
    IntegralMultisetCacheConfig,
    build_integral_multiset_sample,
    cache_directory,
    make_integral_multiset_cache,
)
from blockcipher_nd.tasks.innovation2.high_round_integral_experiment import (
    HighRoundIntegralExperimentConfig,
    adjudicate_high_round_integral,
)


def test_positive_and_negative_plaintext_protocols_are_distinct() -> None:
    positive = build_integral_multiset_sample(
        rounds=5,
        multiset_count=2,
        label=1,
        seed=0,
        split="train",
        row_index=1,
    )
    negative = build_integral_multiset_sample(
        rounds=5,
        multiset_count=2,
        label=0,
        seed=0,
        split="train",
        row_index=0,
    )

    for multiset in positive.plaintexts:
        np.testing.assert_array_equal(
            multiset & np.uint64(0xF),
            np.arange(16, dtype=np.uint64),
        )
        assert np.unique(multiset & np.uint64(0xFFFFFFFFFFFFFFF0)).size == 1
    assert any(
        np.unique(multiset & np.uint64(0xFFFFFFFFFFFFFFF0)).size > 1
        for multiset in negative.plaintexts
    )
    assert np.all(positive.views[:, 0, 0] == 0)
    assert np.all(negative.views[:, 0, 0] == 0)
    assert np.all(positive.views[:, 1, 0] == Present80.inverse_sbox_layer(0))
    assert np.all(negative.views[:, 1, 0] == Present80.inverse_sbox_layer(0))
    assert positive.features.shape == (integral_input_bits(2),)
    assert negative.features.shape == (integral_input_bits(2),)


def test_invp_invs_fixture_and_split_key_domain_are_stable() -> None:
    train = build_integral_multiset_sample(
        rounds=5,
        multiset_count=1,
        label=1,
        seed=0,
        split="train",
        row_index=1,
    )
    validation = build_integral_multiset_sample(
        rounds=5,
        multiset_count=1,
        label=1,
        seed=0,
        split="validation",
        row_index=1,
    )
    test = build_integral_multiset_sample(
        rounds=5,
        multiset_count=1,
        label=1,
        seed=0,
        split="test",
        row_index=1,
    )

    assert len({train.key, validation.key, test.key}) == 3
    assert train.key == 0xC9D9E42D90987377F824
    assert int(train.plaintexts[0, 0]) == 0x88B3BC8DE7F66820
    assert int(train.ciphertexts[0, 0]) == 0x86EFECC6BFB23D75
    assert int(train.ciphertexts[0, 1]) == 0x7FA25BEA781856BF
    difference = int(train.ciphertexts[0, 1] ^ train.ciphertexts[0, 0])
    expected_invp = Present80.inverse_permutation_layer(difference)
    expected_invs = Present80.inverse_sbox_layer(expected_invp)
    assert expected_invp == 0xEBDC967F3960FC38
    assert expected_invs == 0x937042DA8425A08B
    assert int(train.views[0, 1, 0]) == 0x5555555555555555
    assert int(train.views[0, 0, 1]) == expected_invp
    assert int(train.views[0, 1, 1]) == expected_invs
    ciphertext_xor = 0
    for ciphertext in train.ciphertexts[0]:
        ciphertext_xor ^= int(ciphertext)
    assert ciphertext_xor == 0x0808900808080000
    assert ciphertext_xor & 0xF == 0
    np.testing.assert_array_equal(
        train.features.reshape(1, 2, 16, 64)[0, 0, 1],
        np.array([(expected_invp >> bit) & 1 for bit in range(64)], dtype=np.uint8),
    )


def test_models_expose_paper_and_semantic_tensor_shapes() -> None:
    features = torch.zeros((3, integral_input_bits(2)))
    paper = PresentIntegralPaperMbconvAnchor(
        multiset_count=2,
        base_channels=4,
        head_bits=8,
    )
    candidate = PresentIntegralStructuredResidualCandidate(
        multiset_count=2,
        base_channels=4,
        head_bits=8,
        block_count=1,
    )

    assert paper.paper_tensor_view(features).shape == (3, 8, 16, 32)
    assert candidate.structured_tensor_view(features).shape == (3, 16, 16, 16)
    assert paper(features).shape == (3, 1)
    assert candidate(features).shape == (3, 1)


def test_paper_tensor_joins_individually_reshaped_multisets() -> None:
    paper = PresentIntegralPaperMbconvAnchor(
        multiset_count=2,
        base_channels=4,
        head_bits=8,
    )
    indexed = torch.arange(integral_input_bits(2)).reshape(1, -1)
    view = paper.paper_tensor_view(indexed)

    assert view[0, 0, 0, 0].item() == 0
    assert view[0, 0, 0, 15].item() == 120
    assert view[0, 0, 0, 16].item() == 2048
    assert view[0, 0, 0, 31].item() == 2168


def test_disk_cache_is_parameter_matched_reusable_and_resumable(
    tmp_path: Path,
) -> None:
    config = IntegralMultisetCacheConfig(
        split="train",
        rounds=5,
        total_rows=8,
        multiset_count=1,
        seed=7,
        cache_root=tmp_path,
        chunk_size=4,
    )
    created = make_integral_multiset_cache(config)
    original_features = np.asarray(created.features).copy()
    original_labels = np.asarray(created.labels).copy()
    reused = make_integral_multiset_cache(config)

    assert reused.metadata["cache_status"] == "reused"
    np.testing.assert_array_equal(reused.features, original_features)
    np.testing.assert_array_equal(reused.labels, original_labels)

    cache_dir = cache_directory(config)
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "in_progress",
            "rows_generated": 4,
            "positive_rows": 2,
            "negative_rows": 2,
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial_features = np.load(cache_dir / "features.npy", mmap_mode="r+")
    partial_labels = np.load(cache_dir / "labels.npy", mmap_mode="r+")
    partial_features[4:] = 0
    partial_labels[4:] = 0
    partial_features.flush()
    partial_labels.flush()
    del partial_features
    del partial_labels

    resumed = make_integral_multiset_cache(config)
    assert resumed.metadata["cache_status"] == "resumed"
    np.testing.assert_array_equal(resumed.features, original_features)
    np.testing.assert_array_equal(resumed.labels, original_labels)


def test_readiness_cli_writes_complete_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    status = main(
        [
            "--run-id",
            "i2_high_round_test_readiness",
            "--output-root",
            str(output),
            "--cache-root",
            str(tmp_path / "cache"),
            "--rounds",
            "5",
            "--train-rows",
            "32",
            "--validation-rows",
            "16",
            "--test-rows",
            "16",
            "--multiset-count",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--base-channels",
            "4",
            "--head-bits",
            "8",
            "--block-count",
            "1",
            "--cache-chunk-size",
            "8",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--gate-mode",
            "readiness",
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "progress.jsonl",
        "dataset_summary.json",
        "fixed_baselines.json",
        "gate.json",
        "validation.json",
        "curves.svg",
        "history.csv",
    ):
        assert (output / name).is_file()
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    fixed_baselines = json.loads(
        (output / "fixed_baselines.json").read_text(encoding="utf-8")
    )
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    progress = (output / "progress.jsonl").read_text(encoding="utf-8")
    assert [row["role"] for row in rows] == [
        "anchor",
        "candidate",
        "linear",
        "control",
    ]
    assert rows[1]["model_seed"] == rows[3]["model_seed"]
    assert rows[3]["fit_train_labels_shuffled"] is True
    assert rows[3]["fit_validation_labels_shuffled"] is True
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_high_round_integral_readiness_passed"
    assert "untrained_structured_candidate" in fixed_baselines
    assert validation["status"] == "pass"
    assert "integral_cache_chunk" in progress
    assert "high_round_model_done" in progress


def test_deferred_plot_artifacts_are_explicit_and_valid(tmp_path: Path) -> None:
    svg_path = tmp_path / "curves.svg"
    history_path = tmp_path / "history.csv"
    rows = [
        {
            "run_id": "bridge",
            "role": "candidate",
            "model": "structured",
            "history": [{"epoch": 1}, {"epoch": 2}],
        }
    ]

    _write_deferred_svg(svg_path, missing_module="matplotlib")
    _write_deferred_history_csv(history_path, rows)

    assert "<svg" in svg_path.read_text(encoding="utf-8")
    assert "Remote plotting deferred" in svg_path.read_text(encoding="utf-8")
    history = history_path.read_text(encoding="utf-8")
    assert "plot_deferred_to_local_retrieval" in history
    assert "bridge,candidate,structured,2" in history


def test_training_artifacts_defer_when_optional_plot_import_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "run_id": "bridge",
                "role": "candidate",
                "model": "structured",
                "history": [{"epoch": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    real_import = builtins.__import__

    def missing_plot_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "blockcipher_nd.evaluation.plots":
            raise ModuleNotFoundError(
                "No module named 'matplotlib'",
                name="matplotlib",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_plot_import)
    report = write_training_artifacts(results_path, tmp_path, title="bridge")

    assert report["status"] == "deferred_to_local_retrieval"
    assert report["missing_module"] == "matplotlib"
    assert (tmp_path / "plot_deferred.marker").is_file()
    assert "<svg" in (tmp_path / "curves.svg").read_text(encoding="utf-8")


def test_bridge_gate_enforces_frozen_plan_and_advance_thresholds(
    tmp_path: Path,
) -> None:
    gate = adjudicate_high_round_integral(
        _bridge_config(tmp_path),
        rows=_bridge_rows(candidate_auc=0.56, anchor_auc=0.531),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_high_round_integral_bridge_advance"
    assert all(gate["bridge_plan_checks"].values())
    assert all(gate["bridge_signal_checks"].values())
    assert gate["metrics"]["strongest_oriented_fixed_parity_name"] == (
        "negative_invp_parity_weight"
    )
    assert gate["metrics"]["candidate_strongest_fixed_parity_auc_delta"] == (
        pytest.approx(0.02)
    )


def test_bridge_gate_stops_weak_signal_and_rejects_plan_mismatch(
    tmp_path: Path,
) -> None:
    weak_gate = adjudicate_high_round_integral(
        _bridge_config(tmp_path),
        rows=_bridge_rows(candidate_auc=0.521, anchor_auc=0.529),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )
    mismatched_gate = adjudicate_high_round_integral(
        _bridge_config(tmp_path, train_rows=131072),
        rows=_bridge_rows(candidate_auc=0.56, anchor_auc=0.531),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert weak_gate["status"] == "hold"
    assert weak_gate["decision"] == "innovation2_high_round_integral_bridge_stop"
    assert mismatched_gate["status"] == "fail"
    assert mismatched_gate["decision"] == (
        "innovation2_high_round_integral_bridge_plan_mismatch"
    )
    assert not mismatched_gate["bridge_plan_checks"][
        "train_total_rows_is_262144"
    ]


def test_bridge_gate_does_not_advance_from_anchor_only_signal(
    tmp_path: Path,
) -> None:
    gate = adjudicate_high_round_integral(
        _bridge_config(tmp_path),
        rows=_bridge_rows(candidate_auc=0.529, anchor_auc=0.56),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_high_round_integral_bridge_stop"
    assert not gate["bridge_signal_checks"]["candidate_test_auc_at_least_0_53"]


def test_bridge_gate_rejects_invalid_shuffled_fit_control(tmp_path: Path) -> None:
    rows = _bridge_rows(candidate_auc=0.56, anchor_auc=0.531)
    rows[-1]["fit_validation_auc"] = 0.54

    gate = adjudicate_high_round_integral(
        _bridge_config(tmp_path),
        rows=rows,
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation2_high_round_integral_bridge_invalid_control"
    )


def test_remote_bridge_package_is_plan_aligned_and_fail_closed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        project_root
        / (
            "configs/remote/innovation2_present_r8_high_round_integral_bridge_"
            "262144_seed0_gpu0_20260716.json"
        )
    )
    plan_path = (
        project_root
        / (
            "configs/experiment/innovation2/"
            "innovation2_present_r8_high_round_integral_bridge_262144_seed0.json"
        )
    )
    run_script = (
        project_root
        / (
            "configs/remote/generated/run_i2_present_r8_high_round_integral_bridge_"
            "262144_seed0_gpu0_20260716.cmd"
        )
    ).read_text(encoding="utf-8")
    cli_wrapper = (
        project_root / "scripts/run-innovation2-high-round-integral"
    ).read_text(encoding="utf-8")
    launch_script = (
        project_root
        / (
            "configs/remote/generated/"
            "launch_i2_present_r8_high_round_integral_bridge_"
            "262144_seed0_gpu0_20260716.cmd"
        )
    ).read_text(encoding="utf-8")
    monitor_script = (
        project_root
        / (
            "configs/remote/generated/"
            "monitor_i2_present_r8_high_round_integral_bridge_"
            "262144_seed0_gpu0_20260716.sh"
        )
    ).read_text(encoding="utf-8")

    readiness = remote_readiness_report(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert readiness["status"] == "pass"
    assert readiness["errors"] == []
    assert readiness["plan_rows"] == 4
    assert readiness["max_samples_per_class"] == 131072
    assert [row["role"] for row in plan["rows"]] == [
        "anchor",
        "candidate",
        "linear",
        "control",
    ]
    assert config["experiment"] == {
        "batch_size": 128,
        "base_channels": 16,
        "block_count": 2,
        "cipher": "PRESENT-80",
        "dropout": 0.1,
        "epochs": 5,
        "gate_mode": "bridge",
        "head_bits": 256,
        "input_bits": 4096,
        "key_sampling": "one_unique_present80_master_key_per_sample",
        "learning_rate": 0.001,
        "loss": "mse",
        "multisets_per_sample": 2,
        "negative_definition": NEGATIVE_MODE,
        "optimizer": "adam",
        "paper_tensor_concat_assumption": "spatial_axis_1",
        "rounds": 8,
        "seed": 0,
        "test_samples_per_class": 32768,
        "test_total_rows": 65536,
        "texts_per_multiset": 16,
        "train_samples_per_class": 131072,
        "train_total_rows": 262144,
        "validation_samples_per_class": 16384,
        "validation_total_rows": 32768,
        "weight_decay": 0.00001,
    }
    assert "cmd.exe /k" not in (run_script + launch_script + monitor_script).lower()
    assert "cmd.exe /c" in launch_script.lower()
    for argument in (
        "--rounds 8",
        "--train-rows 262144",
        "--validation-rows 32768",
        "--test-rows 65536",
        "--multiset-count 2",
        "--epochs 5",
        "--batch-size 128",
        "--base-channels 16",
        "--head-bits 256",
        "--block-count 2",
        "--seed 0",
        "--device cuda",
        "--gate-mode bridge",
    ):
        assert argument in run_script
    assert "_started.marker" in run_script
    assert "_done.marker" in run_script
    assert "_failed.marker" in run_script
    assert "_done.marker\" goto already_complete" in run_script
    assert "sys.path.insert" in cli_wrapper
    assert "cache_metadata.json" in run_script
    assert "result_branch_pushed.marker" in monitor_script
    assert "scripts/plot-results" in monitor_script
    assert "scripts/index-results" in monitor_script


def _bridge_config(
    tmp_path: Path,
    *,
    train_rows: int = 262144,
) -> HighRoundIntegralExperimentConfig:
    return HighRoundIntegralExperimentConfig(
        run_id="i2_present_r8_bridge_test",
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        rounds=8,
        train_rows=train_rows,
        validation_rows=32768,
        test_rows=65536,
        multiset_count=2,
        epochs=5,
        batch_size=128,
        seed=0,
        base_channels=16,
        head_bits=256,
        block_count=2,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cuda",
        gate_mode="bridge",
    )


def _bridge_rows(*, candidate_auc: float, anchor_auc: float) -> list[dict[str, Any]]:
    def row(
        role: str,
        auc: float,
        *,
        shuffled: bool = False,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "test_accuracy": auc,
            "test_auc": auc,
            "test_calibrated_accuracy": auc,
            "fit_validation_auc": 0.51 if shuffled else auc,
            "fit_train_labels_shuffled": shuffled,
            "fit_validation_labels_shuffled": shuffled,
        }

    return [
        row("anchor", anchor_auc),
        row("candidate", candidate_auc),
        row("linear", 0.52),
        row("control", 0.501, shuffled=True),
    ]


def _valid_dataset_summary() -> dict[str, Any]:
    return {
        "status": "pass",
        "negative_definition": NEGATIVE_MODE,
        "all_caches_complete": True,
        "all_splits_disk_backed": True,
        "all_reference_c0_views_expected": True,
        "key_splits_disjoint_by_construction": True,
        "protocol_fixture": {
            "known_values_match": True,
            "feature_length_matches_protocol": True,
        },
    }


def _bridge_fixed_baselines() -> dict[str, dict[str, float]]:
    return {
        "negative_total_parity_weight": {"auc": 0.52, "best_accuracy": 0.53},
        "negative_invp_parity_weight": {"auc": 0.54, "best_accuracy": 0.55},
        "negative_invs_parity_weight": {"auc": 0.49, "best_accuracy": 0.52},
        "untrained_structured_candidate": {
            "auc": 0.51,
            "best_accuracy": 0.52,
        },
    }
