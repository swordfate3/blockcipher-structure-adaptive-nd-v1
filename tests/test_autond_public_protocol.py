from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path
import pytest

from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.engine.matrix_runner import parse_args
from blockcipher_nd.engine.datasets import dataset_key_for_split
from blockcipher_nd.engine.progress import task_progress_payload
from blockcipher_nd.engine.task_runner import run_task
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.planning.next_action_readiness import launch_artifacts
from blockcipher_nd.registry.cipher_factory import build_cipher


def public_dataset_config(*, total_samples: int = 31) -> DifferentialDatasetConfig:
    return DifferentialDatasetConfig(
        cipher=build_cipher("present80", 5, key=0),
        input_difference=0x000000000D000000,
        samples_per_class=total_samples // 2,
        samples_total=total_samples,
        dataset_label_mode="random_labels_total",
        seed=17,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        negative_mode="random_ciphertext",
        key_rotation_interval=1,
        sample_structure="independent_pairs",
    )


def write_public_plan(
    path: Path,
    *,
    train_samples_total: int = 12,
    validation_samples_total: int = 10,
    final_test_samples_total: int = 12,
    final_test_repeats: int = 5,
) -> None:
    row = {
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "network": "AutoND-Public-Code-Test",
        "model_key": "autond_dbitnet2023",
        "architecture_rank": 0,
        "score": 100,
        "rounds": 9,
        "seed": 0,
        "samples_per_class": train_samples_total // 2,
        "pairs_per_sample": 1,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "random_ciphertext",
        "key_rotation_interval": 1,
        "sample_structure": "independent_pairs",
        "difference_profile": "present_autond_dbitnet2023_highround",
        "difference_member": 0,
        "loss": "mse",
        "learning_rate": 0.001,
        "optimizer": "adam",
        "optimizer_state_transition": "carry_across_stages",
        "checkpoint_metric": "val_loss",
        "restore_best_checkpoint": "true",
        "pretrain_round_sequence": "[5]",
        "pretrain_epochs": 1,
        "dataset_label_mode": "random_labels_total",
        "train_samples_total": train_samples_total,
        "validation_samples_total": validation_samples_total,
        "final_test_samples_total": final_test_samples_total,
        "final_test_repeats": final_test_repeats,
        "final_test_key": "0x22222222222222222222",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_random_label_dataset_uses_exact_total_and_is_deterministic() -> None:
    config = public_dataset_config(total_samples=31)

    first = make_differential_dataset(config)
    second = make_differential_dataset(config)

    assert first.features.shape == (31, 128)
    assert first.labels.shape == (31,)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)
    assert 0 < int(first.labels.sum()) < 31
    assert first.metadata["samples_total"] == 31
    assert first.metadata["positive_rows"] == int(first.labels.sum())
    assert first.metadata["negative_rows"] == 31 - int(first.labels.sum())
    assert first.metadata["dataset_label_mode"] == "random_labels_total"
    assert first.metadata["key_schedule"] == "per_row_random"


def test_balanced_dataset_keeps_historical_per_class_semantics() -> None:
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=build_cipher("present80", 5, key=0),
            input_difference=0x000000000D000000,
            samples_per_class=7,
            seed=3,
        )
    )

    assert dataset.features.shape[0] == 14
    assert int(dataset.labels.sum()) == 7
    assert dataset.metadata["samples_total"] == 14
    assert dataset.metadata["positive_rows"] == 7
    assert dataset.metadata["negative_rows"] == 7
    assert dataset.metadata["dataset_label_mode"] == "balanced_per_class"


def test_random_label_disk_cache_uses_exact_total_and_reuses(tmp_path) -> None:
    config = public_dataset_config(total_samples=31)
    cache_dir = tmp_path / "public-cache"

    created = make_chunked_differential_dataset(
        config,
        cache_dir=cache_dir,
        chunk_size=7,
        workers=2,
    )
    reused = make_chunked_differential_dataset(
        config,
        cache_dir=cache_dir,
        chunk_size=7,
        workers=2,
    )

    assert created.features.shape == (31, 128)
    assert created.labels.shape == (31,)
    assert np.array_equal(created.features, reused.features)
    assert np.array_equal(created.labels, reused.labels)
    assert created.metadata["total_rows"] == 31
    assert created.metadata["positive_rows"] == int(created.labels.sum())
    assert created.metadata["negative_rows"] == 31 - int(created.labels.sum())
    assert created.metadata["cache_status"] == "created"
    assert reused.metadata["cache_status"] == "reused"


def test_cli_and_plan_parse_public_code_split_sizes(tmp_path) -> None:
    args = parse_args(
        [
            "--train-samples-total",
            "32",
            "--validation-samples-total",
            "16",
            "--final-test-samples-total",
            "12",
            "--final-test-repeats",
            "5",
            "--dataset-label-mode",
            "random_labels_total",
        ]
    )
    assert args.train_samples_total == 32
    assert args.validation_samples_total == 16
    assert args.final_test_samples_total == 12
    assert args.final_test_repeats == 5
    assert args.dataset_label_mode == "random_labels_total"

    plan = tmp_path / "public.csv"
    write_public_plan(plan)
    task = build_tasks(parse_args(["--plan", str(plan)]))[0]
    assert task["train_samples_total"] == 12
    assert task["validation_samples_total"] == 10
    assert task["final_test_samples_total"] == 12
    assert task["final_test_repeats"] == 5
    assert task["final_test_key"] == 0x22222222222222222222
    assert task["dataset_label_mode"] == "random_labels_total"


def test_final_test_key_fallback_is_consistent_in_progress_and_cache(tmp_path) -> None:
    plan = tmp_path / "public.csv"
    write_public_plan(plan)
    task = build_tasks(parse_args(["--plan", str(plan)]))[0]
    task["train_key"] = 101
    task["validation_key"] = 202
    task["final_test_key"] = None

    progress = task_progress_payload(task)

    assert progress["train_key"] == 101
    assert progress["validation_key"] == 202
    assert progress["final_test_key"] == 202
    assert dataset_key_for_split(task, "train") == 101
    assert dataset_key_for_split(task, "validation") == 202
    assert dataset_key_for_split(task, "final_test_1") == 202

    task["validation_key"] = None

    progress = task_progress_payload(task)
    assert progress["validation_key"] == 101
    assert progress["final_test_key"] == 101
    assert dataset_key_for_split(task, "validation") == 101
    assert dataset_key_for_split(task, "final_test_1") == 101


def test_public_code_task_uses_exact_splits_and_five_fresh_tests(tmp_path) -> None:
    plan = tmp_path / "public.csv"
    write_public_plan(plan)
    args = parse_args(
        [
            "--plan",
            str(plan),
            "--epochs",
            "1",
            "--pretrain-epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-bits",
            "8",
            "--device",
            "cpu",
            "--amsgrad",
            "--dataset-cache-root",
            str(tmp_path / "cache"),
            "--dataset-cache-chunk-size",
            "4",
        ]
    )

    row = run_task(build_tasks(args)[0], args)

    assert row["training"]["train_rows"] == 12
    assert row["training"]["validation_rows"] == 10
    assert row["training"]["dataset_label_mode"] == "random_labels_total"
    assert row["validation"]["samples_total"] == 10
    stage = row["training"]["pretraining"]["curriculum_stages"][0]
    assert stage["train_rows"] == 12
    assert stage["validation_rows"] == 10
    final = row["final_evaluation"]
    assert row["validation_key"] != row["final_test_key"]
    assert row["final_test_key"] == 0x22222222222222222222
    assert final["final_test_key"] == 0x22222222222222222222
    assert final["repeats"] == 5
    assert final["samples_total_per_repeat"] == 12
    assert final["seeds"] == [50_000, 50_001, 50_002, 50_003, 50_004]
    assert [item["samples_total"] for item in final["metrics_by_repeat"]] == [12] * 5
    assert {
        item["final_test_key"] for item in final["metrics_by_repeat"]
    } == {0x22222222222222222222}
    accuracies = [item["accuracy"] for item in final["metrics_by_repeat"]]
    aucs = [item["auc"] for item in final["metrics_by_repeat"]]
    assert final["accuracy_mean"] == pytest.approx(float(np.mean(accuracies)))
    assert final["accuracy_std"] == pytest.approx(float(np.std(accuracies)))
    assert final["auc_mean"] == pytest.approx(float(np.mean(aucs)))
    assert final["auc_std"] == pytest.approx(float(np.std(aucs)))


@pytest.mark.parametrize(
    (
        "plan_name",
        "samples_per_class",
        "train_total",
        "validation_total",
        "final_total",
        "final_repeats",
        "pretrain_epochs",
    ),
    [
        (
            "innovation1_spn_present_autond_public_code_readiness_smoke_seed0.csv",
            16,
            32,
            16,
            12,
            5,
            1,
        ),
        (
            "innovation1_spn_present_autond_public_code_paperscale_seed0.csv",
            5_000_000,
            10_000_000,
            1_000_000,
            1_000_000,
            5,
            40,
        ),
    ],
)
def test_public_code_plans_lock_exact_total_protocol(
    plan_name: str,
    samples_per_class: int,
    train_total: int,
    validation_total: int,
    final_total: int,
    final_repeats: int,
    pretrain_epochs: int,
) -> None:
    plan = Path("configs/experiment/innovation1") / plan_name
    task = build_tasks(parse_args(["--plan", str(plan)]))[0]

    assert task["model_key"] == "autond_dbitnet2023"
    assert task["rounds"] == 9
    assert task["samples_per_class"] == samples_per_class
    assert task["train_samples_total"] == train_total
    assert task["validation_samples_total"] == validation_total
    assert task["final_test_samples_total"] == final_total
    assert task["final_test_repeats"] == final_repeats
    assert task["dataset_label_mode"] == "random_labels_total"
    assert task["negative_mode"] == "random_ciphertext"
    assert task["key_rotation_interval"] == 1
    assert task["pretrain_round_sequence"] == (5, 6, 7, 8)
    assert task["pretrain_epochs"] == pretrain_epochs
    assert task["checkpoint_metric"] == "val_loss"
    assert task["optimizer_state_transition"] == "carry_across_stages"


def test_public_code_paperscale_remote_readiness_locks_exact_totals(tmp_path) -> None:
    run_id = "i1_present_autond_public_code_paperscale_seed0_gpu1_20260710"
    config = {
        "run_id": run_id,
        "task_name": run_id,
        "archive_work_id": run_id,
        "plan": (
            "configs\\experiment\\innovation1\\"
            "innovation1_spn_present_autond_public_code_paperscale_seed0.csv"
        ),
        "runner_script": "scripts/train",
        "expected_rows": 1,
        "device": "cuda:1",
        "epochs": 40,
        "batch_size": 5000,
        "learning_rate": 0.001,
        "loss": "mse",
        "optimizer": "adam",
        "amsgrad": True,
        "optimizer_state_transition": "carry_across_stages",
        "weight_decay": 0.0,
        "lr_scheduler": "none",
        "max_learning_rate": None,
        "checkpoint_metric": "val_loss",
        "restore_best_checkpoint": True,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "pretrain_rounds": None,
        "pretrain_round_sequence": [5, 6, 7, 8],
        "pretrain_epochs": 40,
        "pairs_per_sample": 1,
        "negative_mode": "random_ciphertext",
        "sample_structure": "independent_pairs",
        "key_rotation_interval": 1,
        "dataset_label_mode": "random_labels_total",
        "train_samples_total": 10_000_000,
        "validation_samples_total": 1_000_000,
        "final_test_samples_total": 1_000_000,
        "final_test_repeats": 5,
        "dataset_cache": True,
        "dataset_cache_root": f"G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\{run_id}\\dataset_cache",
        "dataset_cache_chunk_size": 8192,
        "dataset_cache_workers": 4,
        "monitor_script_name": f"monitor_{run_id}.sh",
        "branch": "main",
        "project_id": "blockcipher-structure-adaptive-nd-v1",
        "clone_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "source_commit": "recorded_in_remote_run_script_git_revision",
        "claim_scope": "public-code-aligned paper-scale AutoND reproduction",
        "launch_policy": "cmd.exe /c; all project artifacts under G:\\lxy",
    }
    config_path = tmp_path / "public_paperscale.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    readiness = remote_readiness_report(config_path)
    assert readiness["status"] == "pass", readiness["errors"]
    assert "autond_public_code_paperscale_lock" in readiness["checked_invariants"]

    config["train_samples_total"] = 9_999_999
    config_path.write_text(json.dumps(config), encoding="utf-8")
    invalid = remote_readiness_report(config_path)
    assert invalid["status"] == "fail"
    assert any("train_samples_total=9999999 expected=10000000" in error for error in invalid["errors"])


def test_public_code_paperscale_remote_package_locks_protocol() -> None:
    config_path = Path(
        "configs/remote/"
        "innovation1_spn_present_autond_public_code_paperscale_seed0_gpu1_20260710.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    readiness = remote_readiness_report(config_path)
    artifacts = launch_artifacts(config_path)
    launcher = Path(artifacts["launcher"]).read_text(encoding="utf-8")
    monitor = Path(artifacts["monitor"]).read_text(encoding="utf-8")

    assert readiness["status"] == "pass", readiness["errors"]
    assert "autond_public_code_paperscale_lock" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass", artifacts["errors"]
    assert config["epochs"] == 40
    assert config["pretrain_epochs"] == 40
    assert config["train_samples_total"] == 10_000_000
    assert config["validation_samples_total"] == 1_000_000
    assert config["final_test_samples_total"] == 1_000_000
    assert config["final_test_repeats"] == 5
    assert config["dataset_label_mode"] == "random_labels_total"
    assert config["negative_mode"] == "random_ciphertext"
    assert config["key_rotation_interval"] == 1
    assert "--epochs 40" in launcher
    assert "--pretrain-epochs 40" in launcher
    assert "--train-samples-total 10000000" in launcher
    assert "--validation-samples-total 1000000" in launcher
    assert "--final-test-samples-total 1000000" in launcher
    assert "--final-test-repeats 5" in launcher
    assert "--dataset-label-mode random_labels_total" in launcher
    assert "--negative-mode random_ciphertext" in launcher
    assert "--key-rotation-interval 1" in launcher
    assert (
        "set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/"
        "github_blockcipher_20260612_result_pusher_ed25519"
    ) in launcher
    assert (
        "set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% "
        "-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    ) in launcher
    assert launcher.index("set GIT_SSH_COMMAND=") < launcher.index("git fetch origin")
    assert "cmd.exe /k" not in launcher
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher
    assert "C:\\Users" not in launcher
    assert '"exact_split_rows"' in monitor
    assert '"optimizer_step_continuity"' in monitor
    assert '"final_test_aggregation"' in monitor
    assert '"paper_r9_accuracy"' in monitor
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor


def test_public_fields_do_not_add_warnings_to_strict_autond_readiness() -> None:
    config_path = Path(
        "configs/remote/"
        "innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0_gpu1_20260710.json"
    )

    readiness = remote_readiness_report(config_path)

    assert not any(
        field in warning
        for warning in readiness["warnings"]
        for field in (
            "dataset_label_mode",
            "train_samples_total",
            "validation_samples_total",
            "final_test_samples_total",
            "final_test_repeats",
        )
    )


def test_typed_invp_local_gate_plan_locks_four_row_protocol() -> None:
    plan = Path(
        "configs/experiment/innovation1/"
        "innovation1_spn_present_autond_typed_invp_local_gate_seed0.csv"
    )

    tasks = build_tasks(parse_args(["--plan", str(plan)]))

    assert [task["model_key"] for task in tasks] == [
        "autond_dbitnet2023",
        "present_nibble_invp_only_spn_only",
        "present_nibble_shuffled_paligned_spn_only",
        "present_nibble_delta_only_spn_only",
    ]
    assert len(tasks) == 4
    assert all(task["rounds"] == 9 for task in tasks)
    assert all(task["seed"] == 0 for task in tasks)
    assert all(task["samples_per_class"] == 8_192 for task in tasks)
    assert all(task["train_samples_total"] == 16_384 for task in tasks)
    assert all(task["validation_samples_total"] == 4_096 for task in tasks)
    assert all(task["final_test_samples_total"] == 4_096 for task in tasks)
    assert all(task["final_test_repeats"] == 3 for task in tasks)
    assert all(task["dataset_label_mode"] == "random_labels_total" for task in tasks)
    assert all(task["negative_mode"] == "random_ciphertext" for task in tasks)
    assert all(task["key_rotation_interval"] == 1 for task in tasks)
    assert all(task["pretrain_round_sequence"] == (5, 6, 7, 8) for task in tasks)
    assert all(task["pretrain_epochs"] == 3 for task in tasks)
    assert all(task["checkpoint_metric"] == "val_loss" for task in tasks)
    assert all(task["optimizer_state_transition"] == "carry_across_stages" for task in tasks)
