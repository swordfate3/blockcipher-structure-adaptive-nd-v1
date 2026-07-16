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
from blockcipher_nd.cli.gate_innovation2_high_round_integral_joint import (
    main as joint_gate_main,
)
from blockcipher_nd.cli.run_innovation2_high_round_integral import (
    _write_deferred_history_csv,
    _write_deferred_svg,
    main,
    write_training_artifacts,
)
from blockcipher_nd.cli.readjudicate_innovation2_high_round_integral import (
    POLICY_VERSION,
    readjudicate_retrieved_artifacts,
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
    run_cuda_memory_preflight,
)
from blockcipher_nd.tasks.innovation2.high_round_integral_joint import (
    adjudicate_joint_high_round_integral,
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

    independent_seed_gate = adjudicate_high_round_integral(
        _bridge_config(tmp_path, seed=1),
        rows=_bridge_rows(candidate_auc=0.56, anchor_auc=0.531),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )
    unsupported_seed_gate = adjudicate_high_round_integral(
        _bridge_config(tmp_path, seed=2),
        rows=_bridge_rows(candidate_auc=0.56, anchor_auc=0.531),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert independent_seed_gate["status"] == "pass"
    assert independent_seed_gate["bridge_plan_checks"][
        "seed_is_frozen_bridge_seed"
    ]
    assert unsupported_seed_gate["status"] == "fail"
    assert not unsupported_seed_gate["bridge_plan_checks"][
        "seed_is_frozen_bridge_seed"
    ]


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


def test_retrieved_bridge_readjudication_excludes_invalid_anchor_layout(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = _bridge_rows(candidate_auc=0.529, anchor_auc=0.56)
    (artifacts / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (artifacts / "dataset_summary.json").write_text(
        json.dumps(_valid_dataset_summary()),
        encoding="utf-8",
    )
    (artifacts / "fixed_baselines.json").write_text(
        json.dumps(_bridge_fixed_baselines()),
        encoding="utf-8",
    )
    (artifacts / "gate.json").write_text(
        json.dumps({"status": "pass", "decision": "remote_old_gate"}),
        encoding="utf-8",
    )
    source_commit = "4b3a2c33cc323b5586533f0fffb78edbe70e0adf"
    (artifacts / "git_revision.txt").write_text(
        source_commit + "\n",
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    remote_config = project_root / (
        "configs/remote/innovation2_present_r8_high_round_integral_bridge_"
        "262144_seed0_gpu0_20260716.json"
    )

    gate = readjudicate_retrieved_artifacts(
        artifacts,
        remote_config,
        invalidate_anchor_layout=True,
        expected_source_commit=source_commit,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_high_round_integral_bridge_stop"
    assert gate["readjudication"]["policy_version"] == POLICY_VERSION
    assert gate["readjudication"]["anchor_layout_invalidated"] is True
    assert gate["readjudication"]["source_revision_matches_expected"] is True
    assert gate["readjudication"]["evidence_exclusions"][0]["role"] == "anchor"

    mismatch = readjudicate_retrieved_artifacts(
        artifacts,
        remote_config,
        invalidate_anchor_layout=True,
        expected_source_commit="0" * 40,
    )
    assert mismatch["status"] == "fail"
    assert mismatch["decision"] == (
        "innovation2_high_round_integral_readjudication_source_mismatch"
    )


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


def test_paper_reference_gate_freezes_scale_and_confirms_candidate_advantage(
    tmp_path: Path,
) -> None:
    gate = adjudicate_high_round_integral(
        _paper_reference_config(tmp_path),
        rows=_bridge_rows(candidate_auc=0.56, anchor_auc=0.54),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_high_round_integral_paper_reference_candidate_advantage"
    )
    assert all(gate["paper_reference_plan_checks"].values())
    assert gate["paper_reference_signal_checks"][
        "at_least_one_neural_model_confirms_r8_round_reach"
    ]
    assert gate["paper_reference_signal_checks"][
        "candidate_beats_anchor_by_0_005_accuracy_or_auc"
    ]
    assert gate["metrics"][
        "paper_reference_candidate_accuracy_95pct_lower_bound"
    ] > 0.5
    approximation = gate["paper_alignment"][
        "project_paper_reference_approximation"
    ]
    assert approximation["per_epoch_full_train_evaluation"] is False
    assert approximation["final_full_train_evaluation"] is True
    assert approximation["cuda_memory_preflight_required_before_cache"] is True
    assert "not exact reproduction" in gate["claim_scope"]


def test_paper_reference_gate_separates_round_reach_from_architecture_gain(
    tmp_path: Path,
) -> None:
    gate = adjudicate_high_round_integral(
        _paper_reference_config(tmp_path),
        rows=_bridge_rows(candidate_auc=0.535, anchor_auc=0.55),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_high_round_integral_paper_reference_round_reach_only"
    )
    assert not gate["paper_reference_signal_checks"][
        "candidate_beats_anchor_by_0_005_accuracy_or_auc"
    ]


def test_paper_reference_gate_holds_weak_signal_and_rejects_plan_mismatch(
    tmp_path: Path,
) -> None:
    weak_gate = adjudicate_high_round_integral(
        _paper_reference_config(tmp_path),
        rows=_bridge_rows(candidate_auc=0.52, anchor_auc=0.52),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )
    mismatched_gate = adjudicate_high_round_integral(
        _paper_reference_config(tmp_path, batch_size=128),
        rows=_bridge_rows(candidate_auc=0.56, anchor_auc=0.54),
        dataset_summary=_valid_dataset_summary(),
        fixed_baselines=_bridge_fixed_baselines(),
    )

    assert weak_gate["status"] == "hold"
    assert weak_gate["decision"] == (
        "innovation2_high_round_integral_paper_reference_not_confirmed"
    )
    assert mismatched_gate["status"] == "fail"
    assert mismatched_gate["decision"] == (
        "innovation2_high_round_integral_paper_reference_plan_mismatch"
    )
    assert not mismatched_gate["paper_reference_plan_checks"][
        "batch_size_is_2000"
    ]


def test_paper_reference_seed0_plan_is_frozen_after_joint_confirmation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (
            project_root
            / "configs/experiment/innovation2/"
            "innovation2_present_r8_high_round_integral_"
            "paper_reference_2pow21_seed0.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["launch_state"] == "ready_for_remote_launch_after_commit_push"
    assert plan["required_precondition"] == {
        "run_id": (
            "i2_present_r8_high_round_integral_bridge_262144_"
            "joint_seed0_seed1_20260716"
        ),
        "status": "pass",
        "decision": "innovation2_high_round_integral_two_seed_bridge_confirmed",
    }
    assert plan["precondition_evidence"].endswith(
        "i2_present_r8_high_round_integral_bridge_262144_"
        "joint_seed0_seed1_20260716/gate.json"
    )
    common = plan["common"]
    assert common["train_total_rows"] == 1 << 21
    assert common["validation_total_rows"] == common["test_total_rows"] == 1 << 17
    assert common["epochs"] == 50
    assert common["batch_size"] == 2000
    assert common["gate_mode"] == "paper_reference"
    approximation = plan["paper_reference_approximation"]
    assert approximation["head_bits"] == 2048
    assert approximation["fc_layer_count"] == 3
    assert approximation["depicted_mbconv_block_count"] == 1
    assert approximation["learning_rate_scheduler"] == "none"
    assert approximation["independent_training_repetitions"] == 1
    assert approximation["per_epoch_full_train_evaluation"] is False
    assert approximation["final_full_train_evaluation"] is True
    assert approximation["cuda_memory_preflight_before_cache"] is True
    assert approximation["exact_reproduction"] is False
    assert [row["role"] for row in plan["rows"]] == [
        "anchor",
        "candidate",
        "linear",
        "control",
    ]
    assert plan["forbidden_during_paper_reference_run"] == [
        "r9_probe",
        "gift_extension",
        "aes_extension",
    ]


def test_cuda_memory_preflight_fails_closed_without_cuda_request(
    tmp_path: Path,
) -> None:
    config = HighRoundIntegralExperimentConfig(
        run_id="memory_preflight_cpu_rejected",
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        rounds=8,
        train_rows=2,
        validation_rows=2,
        test_rows=2,
        multiset_count=2,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    with pytest.raises(ValueError, match="requires a CUDA device"):
        run_cuda_memory_preflight(config)


def test_paper_reference_remote_package_is_plan_aligned_and_fail_closed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_id = (
        "i2_present_r8_high_round_integral_paper_reference_"
        "2pow21_seed0_gpu0_20260716"
    )
    config_path = project_root / (
        "configs/remote/innovation2_present_r8_high_round_integral_"
        "paper_reference_2pow21_seed0_gpu0_20260716.json"
    )
    plan_path = project_root / (
        "configs/experiment/innovation2/innovation2_present_r8_high_round_"
        "integral_paper_reference_2pow21_seed0.json"
    )
    run_script = (
        project_root
        / (
            "configs/remote/generated/run_i2_present_r8_high_round_integral_"
            "paper_reference_2pow21_seed0_gpu0_20260716.cmd"
        )
    ).read_text(encoding="utf-8")
    launch_script = (
        project_root
        / (
            "configs/remote/generated/launch_i2_present_r8_high_round_integral_"
            "paper_reference_2pow21_seed0_gpu0_20260716.cmd"
        )
    ).read_text(encoding="utf-8")
    monitor_script = (
        project_root
        / (
            "configs/remote/generated/monitor_i2_present_r8_high_round_integral_"
            "paper_reference_2pow21_seed0_gpu0_20260716.sh"
        )
    ).read_text(encoding="utf-8")

    readiness = remote_readiness_report(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert readiness["status"] == "pass"
    assert readiness["errors"] == []
    assert readiness["plan_rows"] == 4
    assert readiness["max_samples_per_class"] == 1 << 20
    assert config["run_id"] == config["task_name"] == run_id
    assert config["launch_enabled"] is True
    assert config["dataset_cache"] is True
    assert config["cuda_memory_preflight"] is True
    assert config["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )
    experiment = config["experiment"]
    assert experiment["train_total_rows"] == 1 << 21
    assert experiment["validation_total_rows"] == 1 << 17
    assert experiment["test_total_rows"] == 1 << 17
    assert experiment["epochs"] == 50
    assert experiment["batch_size"] == 2000
    assert experiment["base_channels"] == 16
    assert experiment["head_bits"] == 2048
    assert experiment["fc_layer_count"] == 3
    assert experiment["block_count"] == 1
    assert experiment["learning_rate_scheduler"] == "none"
    assert experiment["train_eval_interval"] == 0
    assert experiment["gate_mode"] == "paper_reference"
    assert experiment["exact_reproduction"] is False
    assert config["expected_storage"]["feature_cache_bytes"] == 9 * (1 << 30)
    assert plan["launch_state"] == "ready_for_remote_launch_after_commit_push"

    for fragment in (
        "--train-rows 2097152",
        "--validation-rows 131072",
        "--test-rows 131072",
        "--epochs 50",
        "--batch-size 2000",
        "--head-bits 2048",
        "--block-count 1",
        "--cuda-memory-preflight",
        "--gate-mode paper_reference",
        "paper_reference_plan_checks",
        "memory_preflight.json",
    ):
        assert fragment in run_script
    assert "cmd.exe /k" not in run_script + launch_script
    assert "EnableDelayedExpansion" not in run_script + launch_script
    assert "!" not in run_script + launch_script
    assert 'cmd.exe /c %RUN_CMD% 0' in launch_script
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in (
        run_script + launch_script
    )
    assert "C:\\Users" not in run_script + launch_script
    assert "sleep 300" in monitor_script
    assert "paper_reference_plan_checks" in monitor_script
    assert "memory_preflight.json" in monitor_script
    assert "scripts/index-results" in monitor_script


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
    recovery_launcher = (
        project_root
        / (
            "configs/remote/generated/"
            "launch_recover_i2_present_r8_high_round_integral_bridge_"
            "262144_seed0_gpu0_20260716.cmd"
        )
    ).read_text(encoding="utf-8")
    recovery_script = (
        project_root
        / (
            "configs/remote/generated/"
            "recover_i2_present_r8_high_round_integral_bridge_"
            "262144_seed0_gpu0_20260716.cmd"
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
    assert "!" not in run_script
    assert "not p.name == 'SHA256SUMS'" in run_script
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
    assert "DisableDelayedExpansion" in recovery_launcher
    assert "DisableDelayedExpansion" in recovery_script
    assert '"%ACTUAL_COMMIT%"' in recovery_launcher
    assert "scripts\\run-innovation2-high-round-integral" not in recovery_script
    assert "scripts\\readjudicate-innovation2-high-round-integral" in recovery_script
    assert "validation.recovery.json" in recovery_script
    assert "gate.recovery.json" in recovery_script
    assert "--invalidate-anchor-layout" in recovery_script
    assert "4b3a2c33cc323b5586533f0fffb78edbe70e0adf" in recovery_script
    assert "not p.name == 'SHA256SUMS'" in recovery_script
    assert "_recovery_done.marker" in recovery_script
    assert "_result_branch_pushed.marker" in recovery_script
    assert 'if exist "%LOG_DIR%\\%RUN_ID%_recovery_failed.marker" del /Q' in recovery_script
    assert "!" not in recovery_launcher + recovery_script


def test_remote_bridge_seed1_package_changes_only_the_frozen_seed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / (
        "configs/remote/innovation2_present_r8_high_round_integral_bridge_"
        "262144_seed1_gpu0_20260716.json"
    )
    plan_path = project_root / (
        "configs/experiment/innovation2/"
        "innovation2_present_r8_high_round_integral_bridge_262144_seed1.json"
    )
    run_script = (
        project_root
        / (
            "configs/remote/generated/run_i2_present_r8_high_round_integral_"
            "bridge_262144_seed1_gpu0_20260716.cmd"
        )
    ).read_text(encoding="utf-8")
    launch_script = (
        project_root
        / (
            "configs/remote/generated/launch_i2_present_r8_high_round_integral_"
            "bridge_262144_seed1_gpu0_20260716.cmd"
        )
    ).read_text(encoding="utf-8")
    monitor_script = (
        project_root
        / (
            "configs/remote/generated/monitor_i2_present_r8_high_round_integral_"
            "bridge_262144_seed1_gpu0_20260716.sh"
        )
    ).read_text(encoding="utf-8")

    readiness = remote_readiness_report(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert readiness["status"] == "pass"
    assert readiness["errors"] == []
    assert readiness["plan_rows"] == 4
    assert readiness["max_samples_per_class"] == 131072
    assert plan["common"]["seed"] == 1
    assert config["experiment"]["seed"] == 1
    assert config["experiment"]["train_total_rows"] == 262144
    assert config["experiment"]["validation_total_rows"] == 32768
    assert config["experiment"]["test_total_rows"] == 65536
    assert config["experiment"]["multisets_per_sample"] == 2
    assert config["experiment"]["negative_definition"] == NEGATIVE_MODE
    assert [row["role"] for row in plan["rows"]] == [
        "anchor",
        "candidate",
        "linear",
        "control",
    ]
    assert "--seed 1" in run_script
    assert "source_expected_commit.txt" in run_script
    assert "not p.name == 'SHA256SUMS'" in run_script
    assert "!" not in run_script
    assert "cmd.exe /c" in launch_script.lower()
    assert "cmd.exe /k" not in (run_script + launch_script + monitor_script).lower()
    assert "source_expected_commit.txt" in monitor_script
    assert "--expected-source-commit" in monitor_script
    assert "--invalidate-anchor-layout" not in monitor_script
    assert "anchor_layout_invalidated" in monitor_script
    assert "scripts/index-results" in monitor_script


def test_joint_bridge_gate_confirms_two_valid_independent_seeds() -> None:
    result = adjudicate_joint_high_round_integral(
        run_id="joint_bridge",
        sources=[_joint_bridge_source(seed=0), _joint_bridge_source(seed=1)],
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_high_round_integral_two_seed_bridge_confirmed"
    )
    assert all(result["gate"]["validity_checks"].values())
    assert all(result["gate"]["signal_checks"].values())
    assert [row["seed"] for row in result["rows"]] == [0, 1]
    assert result["gate"]["metrics"]["candidate_test_auc_min"] == pytest.approx(
        0.55
    )


def test_joint_bridge_gate_holds_when_seed1_signal_misses() -> None:
    result = adjudicate_joint_high_round_integral(
        run_id="joint_bridge",
        sources=[
            _joint_bridge_source(seed=0),
            _joint_bridge_source(seed=1, candidate_auc=0.529, source_pass=False),
        ],
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_high_round_integral_two_seed_bridge_not_confirmed"
    )
    assert all(result["gate"]["validity_checks"].values())
    assert not result["gate"]["signal_checks"][
        "both_candidate_test_auc_at_least_0_53"
    ]
    assert not result["gate"]["signal_checks"]["both_source_gates_pass"]


@pytest.mark.parametrize("invalid_case", ["duplicate_seed", "protocol", "revision"])
def test_joint_bridge_gate_rejects_invalid_sources(invalid_case: str) -> None:
    seed0 = _joint_bridge_source(seed=0)
    seed1 = _joint_bridge_source(seed=1)
    if invalid_case == "duplicate_seed":
        seed1 = _joint_bridge_source(seed=0)
    elif invalid_case == "protocol":
        for row in seed1["rows"]:
            row["batch_size"] = 256
    else:
        seed1["gate"]["readjudication"][
            "source_revision_matches_expected"
        ] = False

    result = adjudicate_joint_high_round_integral(
        run_id="joint_bridge",
        sources=[seed0, seed1],
    )

    assert result["gate"]["status"] == "fail"
    assert result["gate"]["decision"] == (
        "innovation2_high_round_integral_two_seed_bridge_invalid"
    )
    assert not all(result["gate"]["validity_checks"].values())


def test_joint_bridge_cli_writes_complete_artifacts(tmp_path: Path) -> None:
    source_roots = [tmp_path / "seed0", tmp_path / "seed1"]
    for seed, source_root in enumerate(source_roots):
        source = _joint_bridge_source(seed=seed, artifact_root=str(source_root))
        source_root.mkdir()
        (source_root / "gate.local.json").write_text(
            json.dumps(source["gate"]),
            encoding="utf-8",
        )
        (source_root / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in source["rows"]),
            encoding="utf-8",
        )
    output_root = tmp_path / "joint"

    status = joint_gate_main(
        [
            "--run-id",
            "joint_bridge",
            "--source-artifacts",
            *(str(path) for path in source_roots),
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "seed_metrics.csv",
        "curves.svg",
        "gate.json",
        "progress.jsonl",
    ):
        assert (output_root / name).is_file()
        assert (output_root / name).stat().st_size > 0
    assert "<svg" in (output_root / "curves.svg").read_text(encoding="utf-8")
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert "run_start" in (output_root / "progress.jsonl").read_text(
        encoding="utf-8"
    )
    assert "run_done" in (output_root / "progress.jsonl").read_text(
        encoding="utf-8"
    )


def _bridge_config(
    tmp_path: Path,
    *,
    train_rows: int = 262144,
    seed: int = 0,
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
        seed=seed,
        base_channels=16,
        head_bits=256,
        block_count=2,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cuda",
        gate_mode="bridge",
    )


def _paper_reference_config(
    tmp_path: Path,
    *,
    batch_size: int = 2000,
) -> HighRoundIntegralExperimentConfig:
    return HighRoundIntegralExperimentConfig(
        run_id="i2_present_r8_paper_reference_test",
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        rounds=8,
        train_rows=1 << 21,
        validation_rows=1 << 17,
        test_rows=1 << 17,
        multiset_count=2,
        epochs=50,
        batch_size=batch_size,
        seed=0,
        base_channels=16,
        head_bits=2048,
        block_count=1,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cuda",
        gate_mode="paper_reference",
    )


def _joint_bridge_source(
    *,
    seed: int,
    candidate_auc: float = 0.55,
    source_pass: bool = True,
    artifact_root: str | None = None,
) -> dict[str, Any]:
    run_id = f"i2_present_r8_bridge_seed{seed}"
    common = {
        "run_id": run_id,
        "cipher": "PRESENT-80",
        "task": "innovation2_present_high_round_integral_multiset",
        "rounds": 8,
        "seed": seed,
        "train_total_rows": 262144,
        "validation": {"total_rows": 32768, "samples_per_class": 16384},
        "test": {"total_rows": 65536, "samples_per_class": 32768},
        "multisets_per_sample": 2,
        "texts_per_multiset": 16,
        "input_bits": 4096,
        "input_view": "wu_guo_invp_invs_cj_xor_c0",
        "negative_mode": NEGATIVE_MODE,
        "key_sampling": "one unique PRESENT-80 master key per sample",
        "epochs": 5,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "loss": "mse",
        "optimizer": "adam",
        "paper_tensor_concat_assumption": "spatial_axis_1",
    }

    def row(role: str, test_auc: float, fit_validation_auc: float) -> dict[str, Any]:
        return {
            **common,
            "role": role,
            "test_accuracy": test_auc,
            "test_auc": test_auc,
            "fit_validation_auc": fit_validation_auc,
        }

    return {
        "artifact_root": artifact_root or f"/artifacts/{run_id}",
        "rows": [
            row("anchor", 0.535, 0.535),
            row("candidate", candidate_auc, candidate_auc),
            row("linear", 0.52, 0.52),
            row("control", 0.501, 0.507),
        ],
        "gate": {
            "status": "pass" if source_pass else "hold",
            "decision": (
                "innovation2_high_round_integral_bridge_advance"
                if source_pass
                else "innovation2_high_round_integral_bridge_stop"
            ),
            "gate_mode": "bridge",
            "rounds": 8,
            "run_id": run_id,
            "metrics": {
                "architecture_prior_oriented_auc": 0.501,
                "strongest_oriented_fixed_parity_auc": 0.505,
            },
            "bridge_plan_checks": {"frozen_protocol": True},
            "readiness_checks": {"artifacts_complete": True},
            "bridge_signal_checks": {"candidate_signal": source_pass},
            "readjudication": {
                "source_revision_matches_expected": True,
                "anchor_layout_invalidated": seed == 0,
            },
        },
    }


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
