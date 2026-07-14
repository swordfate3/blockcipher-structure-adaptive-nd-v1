from __future__ import annotations

import copy
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import test_cross_spn_typed_transfer_gate as transfer_fixtures
from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    write_score_artifact,
)
from blockcipher_nd.planning.cross_spn_target_adaptation_gate import (
    ADAPTATION_MODEL_ROLES,
    gate_cross_spn_target_adaptation,
    gate_cross_spn_target_adaptation_joint,
    paired_stratified_bootstrap_auc_differences,
)
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.training.metrics import binary_auc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _score_arrays() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = np.array([0.0] * 32 + [1.0] * 32, dtype=np.float32)
    index = np.arange(64, dtype=np.float32)
    noise = ((index * 17.0) % 23.0) / 100.0
    scores = {
        "typed_scratch": 0.35 + 0.17 * labels + noise,
        "true_to_true": 0.28 + 0.43 * labels + noise / 2.0,
        "shuffled_to_true": 0.34 + 0.19 * labels + noise,
        "true_to_shuffled": 0.40 + 0.04 * labels + noise,
    }
    return labels, {role: values.astype(np.float32) for role, values in scores.items()}


def _write_adaptation_fixture(tmp_path: Path) -> dict[str, Any]:
    results = tmp_path / "results.jsonl"
    progress = tmp_path / "progress.jsonl"
    labels, scores = _score_arrays()
    aucs = {role: binary_auc(labels, values) for role, values in scores.items()}
    transfer_fixtures._write_transfer_run(
        results,
        {
            "gift_anchor": 0.5,
            "gift_typed_scratch": aucs["typed_scratch"],
            "true_to_true": aucs["true_to_true"],
            "shuffled_to_true": aucs["shuffled_to_true"],
            "true_to_shuffled": aucs["true_to_shuffled"],
        },
        samples_per_class=64,
        epochs=1,
        seed=2,
        device="cpu",
    )
    source_progress = results.with_name(f"{results.stem}.progress.jsonl")
    expected_models = set(ADAPTATION_MODEL_ROLES.values())
    result_rows = [
        copy.deepcopy(row)
        for row in _read_jsonl(results)
        if row["selected_model"] in expected_models
    ]
    by_model = {row["selected_model"]: row for row in result_rows}
    score_paths: dict[str, Path] = {}
    for role, model in ADAPTATION_MODEL_ROLES.items():
        checkpoint = tmp_path / "checkpoints" / f"{role}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(role.encode("ascii"))
        row = by_model[model]
        row["training"]["checkpoint_output"] = str(checkpoint)
        score_path = tmp_path / "scores" / role
        write_score_artifact(
            score_path,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=scores[role],
                logits=scores[role],
                sample_ids=np.array([str(index) for index in range(64)]),
                metadata={
                    "cipher": "GIFT-64",
                    "cipher_key": "gift64",
                    "rounds": 6,
                    "seed": 2,
                    "samples_per_class": 64,
                    "pairs_per_sample": 4,
                    "feature_encoding": "ciphertext_pair_bits",
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "difference_profile": "gift64_shen2024_spn_screen",
                    "difference_member": 0,
                    "model_key": model,
                    "score_split": "validation",
                    "score_samples_per_class": 32,
                    "validation_samples_per_class": 32,
                    "dataset_cache_enabled": True,
                    "dataset_cache_root": row["training"]["dataset_cache_root"],
                    "train_key": row["train_key"],
                    "validation_key": row["validation_key"],
                    "model_options": row["training"]["model_options"],
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": f"{len(role):064x}",
                    "checkpoint_metadata": {
                        "checkpoint_output": str(checkpoint),
                        "seed": 2,
                        "epochs": 1,
                        "selected_checkpoint": "best",
                        "restore_best_checkpoint": True,
                    },
                },
            ),
        )
        score_paths[role] = score_path
    _write_jsonl(results, result_rows)

    progress_rows = []
    scratch_model = ADAPTATION_MODEL_ROLES["typed_scratch"]
    for row in _read_jsonl(source_progress):
        model = row.get("model")
        if model is not None and model not in expected_models:
            continue
        if row.get("event") in {"cache_done", "cache_reuse"}:
            row["event"] = "cache_done" if model == scratch_model else "cache_reuse"
        if row.get("event") == "run_done":
            row["output"] = str(results)
            row["total"] = 4
        if "total" in row and row.get("event") == "initialization_ready":
            row["total"] = 4
        progress_rows.append(row)
    _write_jsonl(progress, progress_rows)

    plan = tmp_path / "plan.csv"
    fieldnames = (
        "cipher",
        "structure",
        "rounds",
        "seed",
        "model_key",
        "samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "integral_active_nibble",
        "key_rotation_interval",
        "difference_profile",
        "difference_member",
        "model_options",
    )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(
                {
                    "cipher": row["cipher"],
                    "structure": row["structure"],
                    "rounds": row["rounds"],
                    "seed": row["seed"],
                    "model_key": row["selected_model"],
                    "samples_per_class": row["samples_per_class"],
                    "pairs_per_sample": row["pairs_per_sample"],
                    "feature_encoding": row["feature_encoding"],
                    "negative_mode": row["negative_mode"],
                    "sample_structure": row["sample_structure"],
                    "integral_active_nibble": row["integral_active_nibble"],
                    "key_rotation_interval": row["key_rotation_interval"],
                    "difference_profile": row["difference_profile"],
                    "difference_member": row["difference_member"],
                    "model_options": json.dumps(
                        row["training"]["model_options"],
                        sort_keys=True,
                    ),
                }
            )
    return {
        "plan": plan,
        "results": results,
        "progress": progress,
        "scores": score_paths,
    }


def _build_plan(path: Path) -> list[dict[str, Any]]:
    return build_tasks(
        SimpleNamespace(
            plan=str(path),
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=1,
            difference_profile=None,
            difference_member=0,
            key_rotation_interval=0,
            sample_structure="independent_pairs",
            integral_active_nibble=0,
        )
    )


def test_e4_r4_plans_build_exact_four_role_new_seed_matrices() -> None:
    root = Path("configs/experiment/innovation1")
    cases = (
        ("innovation1_spn_gift64_cross_spn_target_adaptation_smoke_seed2.csv", 2, 64),
        ("innovation1_spn_gift64_cross_spn_target_adaptation_smoke_seed3.csv", 3, 64),
        ("innovation1_spn_gift64_cross_spn_target_adaptation_65536_seed2.csv", 2, 65536),
        ("innovation1_spn_gift64_cross_spn_target_adaptation_65536_seed3.csv", 3, 65536),
    )

    for name, seed, samples_per_class in cases:
        tasks = _build_plan(root / name)
        assert len(tasks) == 4
        assert [task["model_key"] for task in tasks] == list(
            ADAPTATION_MODEL_ROLES.values()
        )
        assert {task["seed"] for task in tasks} == {seed}
        assert {task["samples_per_class"] for task in tasks} == {
            samples_per_class
        }
        assert {task["negative_mode"] for task in tasks} == {
            "encrypted_random_plaintexts"
        }
        assert {task["pairs_per_sample"] for task in tasks} == {4}


def test_e4_r4_remote_configs_pass_fail_closed_readiness() -> None:
    root = Path("configs/remote")
    for name in (
        "innovation1_gift64_cross_spn_target_adaptation_r4_65536_seed2_gpu0_20260715.json",
        "innovation1_gift64_cross_spn_target_adaptation_r4_65536_seed3_gpu1_20260715.json",
    ):
        report = remote_readiness_report(root / name)
        assert report["status"] == "pass", report["errors"]
        assert "e4_r4_target_adaptation_protocol_lock" in report["checked_invariants"]


def test_e4_r4_remote_assets_lock_scoring_bootstrap_and_unattended_launch() -> None:
    generated = Path("configs/remote/generated")
    run_script = (
        generated
        / "run_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.cmd"
    ).read_text(encoding="utf-8")
    launcher = (
        generated
        / "launch_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.cmd"
    ).read_text(encoding="utf-8")
    monitor = (
        generated
        / "monitor_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.sh"
    ).read_text(encoding="utf-8")

    assert "--epochs 1" in run_script
    assert run_script.count("call :export_score") == 4
    assert "scripts\\export-checkpoint-scores" in run_script
    assert "--samples-per-class 32768" in run_script
    assert "scripts\\gate-cross-spn-target-adaptation" in run_script
    assert "--bootstrap-replicates 10000" in run_script
    assert "--bootstrap-seed 20260715" in run_script
    assert "paired_scores.csv.gz" in run_script
    assert run_script.index("scripts\\gate-cross-spn-target-adaptation") < (
        run_script.index("scripts\\plot-results")
    )
    assert "plot_deferred_to_local" in run_script
    assert "SHA256SUMS" in run_script
    assert "git add \"results_archive\\%RUN_ID%\"" in run_script
    assert "git add ." not in run_script
    assert "cmd.exe /k" not in run_script.lower()
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run_script

    assert launcher.count("cmd.exe /c") == 2
    assert launcher.count("/RU SYSTEM") == 2
    assert launcher.count("/RL HIGHEST") == 2
    assert launcher.count("schtasks /Run /I") == 2
    assert "cmd.exe /k" not in launcher.lower()
    assert "set SOURCE_COMMIT=%~1" in launcher
    assert 'git checkout --detach "%EXPECTED_COMMIT%"' in launcher
    assert 'if /I not "%ACTUAL_COMMIT%"=="%EXPECTED_COMMIT%" exit /b 1' in launcher

    assert "outputs/remote_results_incomplete" in monitor
    assert "outputs/remote_results" in monitor
    assert "retrieved_from_verified_result_branch.marker" in monitor
    assert "scripts/plot-results" in monitor
    assert "scripts/index-results" in monitor
    assert "sleep 300" in monitor


def test_paired_stratified_bootstrap_is_deterministic_and_directional() -> None:
    labels, scores = _score_arrays()

    first = paired_stratified_bootstrap_auc_differences(
        labels,
        scores,
        candidate_role="true_to_true",
        control_roles=("typed_scratch", "shuffled_to_true", "true_to_shuffled"),
        replicates=256,
        seed=17,
        chunk_size=16,
    )
    second = paired_stratified_bootstrap_auc_differences(
        labels,
        scores,
        candidate_role="true_to_true",
        control_roles=("typed_scratch", "shuffled_to_true", "true_to_shuffled"),
        replicates=256,
        seed=17,
        chunk_size=16,
    )

    assert first == second
    assert first["comparisons"]["typed_scratch"]["point_difference"] > 0.0
    assert first["comparisons"]["shuffled_to_true"]["point_difference"] > 0.0
    assert first["comparisons"]["true_to_shuffled"]["point_difference"] > 0.0


def test_e4_r4_readiness_validates_paired_scores_and_writes_csv(tmp_path: Path) -> None:
    fixture = _write_adaptation_fixture(tmp_path)
    paired_scores = tmp_path / "paired_scores.csv.gz"

    report = gate_cross_spn_target_adaptation(
        plan_path=fixture["plan"],
        results_path=fixture["results"],
        progress_path=fixture["progress"],
        score_artifact_paths=fixture["scores"],
        expected_seed=2,
        samples_per_class=64,
        epochs=1,
        readiness_only=True,
        bootstrap_replicates=64,
        paired_scores_output=paired_scores,
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "implementation_ready"
    assert report["research_decision_applied"] is False
    assert report["score_rows"] == 64
    with gzip.open(paired_scores, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64 * 4
    assert set(rows[0]) == {
        "sample_index",
        "label",
        "score",
        "target_seed",
        "model_role",
        "checkpoint_sha256",
    }


def test_e4_r4_readiness_rejects_misaligned_sample_ids(tmp_path: Path) -> None:
    fixture = _write_adaptation_fixture(tmp_path)
    path = fixture["scores"]["shuffled_to_true"] / "sample_ids.npy"
    sample_ids = np.load(path).astype(str)
    sample_ids[0] = "different"
    np.save(path, sample_ids)

    report = gate_cross_spn_target_adaptation(
        plan_path=fixture["plan"],
        results_path=fixture["results"],
        progress_path=fixture["progress"],
        score_artifact_paths=fixture["scores"],
        expected_seed=2,
        samples_per_class=64,
        epochs=1,
        readiness_only=True,
        bootstrap_replicates=64,
    )

    assert report["status"] == "fail"
    assert "score sample_ids differ for role=shuffled_to_true" in report["errors"]


def test_e4_r4_readiness_rejects_score_cache_mismatch(tmp_path: Path) -> None:
    fixture = _write_adaptation_fixture(tmp_path)
    path = fixture["scores"]["true_to_true"] / "models.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["dataset_cache_root"] = "/different/cache"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    report = gate_cross_spn_target_adaptation(
        plan_path=fixture["plan"],
        results_path=fixture["results"],
        progress_path=fixture["progress"],
        score_artifact_paths=fixture["scores"],
        expected_seed=2,
        samples_per_class=64,
        epochs=1,
        readiness_only=True,
        bootstrap_replicates=64,
    )

    assert report["status"] == "fail"
    assert any(
        "score metadata role=true_to_true field=dataset_cache_root" in error
        for error in report["errors"]
    )


def _seed_report(seed: int, decision: str) -> dict[str, Any]:
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seed": seed,
        "samples_per_class": 65536,
        "epochs": 1,
        "experiment_stage": "e4_r4",
        "research_decision_applied": True,
    }


def test_e4_r4_joint_gate_requires_both_new_seeds_to_confirm() -> None:
    confirmed = "e4_r4_target_adaptation_efficiency_confirmed"
    report = gate_cross_spn_target_adaptation_joint(
        [_seed_report(2, confirmed), _seed_report(3, confirmed)]
    )

    assert report["status"] == "pass"
    assert report["decision"] == (
        "e4_r4_two_seed_target_adaptation_efficiency_confirmed"
    )

    unstable = gate_cross_spn_target_adaptation_joint(
        [
            _seed_report(2, confirmed),
            _seed_report(3, "e4_r4_target_adaptation_signal_unstable"),
        ]
    )
    assert unstable["decision"] == "e4_r4_two_seed_target_adaptation_signal_unstable"
