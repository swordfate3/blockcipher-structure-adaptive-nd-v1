from __future__ import annotations

import csv
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import test_cross_spn_typed_cell_gate as r1_fixtures
from blockcipher_nd.engine.checkpoint_initialization import (
    initialize_model_from_manifest,
)
from blockcipher_nd.engine.matrix_runner import parse_args as parse_matrix_args
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.planning.cross_spn_typed_transfer_gate import (
    TRANSFER_MODEL_ROLES,
    gate_cross_spn_typed_transfer,
    gate_cross_spn_typed_transfer_joint,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report


ROLES = {
    "gift_anchor": "gift_cross_spn_aligned_token_mixer_raw_anchor",
    "gift_typed_scratch": "gift_cross_spn_typed_cell_true",
    "true_to_true": "gift_cross_spn_typed_cell_true_from_present_true",
    "shuffled_to_true": "gift_cross_spn_typed_cell_true_from_present_shuffled",
    "true_to_shuffled": "gift_cross_spn_typed_cell_shuffled_from_present_true",
}
MANIFEST = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_cross_spn_typed_transfer_seed0_sources.json"
)


def test_cross_spn_transfer_manifest_locks_all_five_roles() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["version"] == 1
    assert set(manifest["targets"]) == set(ROLES.values())
    assert manifest["targets"][ROLES["gift_anchor"]] == {
        "kind": "scratch",
        "target_mapping": "aligned",
    }
    assert manifest["targets"][ROLES["gift_typed_scratch"]] == {
        "kind": "scratch",
        "target_mapping": "true",
    }
    assert (
        manifest["targets"][ROLES["true_to_true"]]["source_checkpoint_sha256"]
        == "eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1"
    )
    assert (
        manifest["targets"][ROLES["shuffled_to_true"]][
            "source_checkpoint_sha256"
        ]
        == "fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22"
    )


@pytest.mark.parametrize(
    ("role", "target_mapping", "source_mapping"),
    [
        ("true_to_true", "true", "true"),
        ("shuffled_to_true", "true", "shuffled"),
        ("true_to_shuffled", "shuffled", "true"),
    ],
)
def test_cross_spn_transfer_manifest_strictly_loads_real_source_checkpoints(
    role: str,
    target_mapping: str,
    source_mapping: str,
) -> None:
    model_key = ROLES[role]
    model = build_model(
        model_key,
        input_bits=4 * 128,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options={
            "mixer_depth": 2,
            "token_mlp_ratio": 2,
            "activation": "relu",
            "norm": "layernorm",
            "pooling": "attention_mean_max",
            "dropout": 0.0,
        },
    )

    report = initialize_model_from_manifest(
        model,
        target_model=model_key,
        target_mapping=target_mapping,
        manifest_path=MANIFEST,
    )

    assert report["kind"] == "checkpoint"
    assert report["source_mapping"] == source_mapping
    assert report["target_mapping"] == target_mapping
    assert report["strict_state_dict_load"] is True


@pytest.mark.parametrize(
    (
        "filename",
        "family",
        "samples_per_class",
        "epochs",
        "batch_size",
        "seed",
        "device",
    ),
    [
        (
            "innovation1_spn_gift64_cross_spn_typed_transfer_smoke_seed0.csv",
            "gift64_cross_spn_typed_transfer_smoke",
            64,
            1,
            32,
            0,
            "cpu",
        ),
        (
            "innovation1_spn_gift64_cross_spn_typed_transfer_smoke_seed1.csv",
            "gift64_cross_spn_typed_transfer_smoke_seed1",
            64,
            1,
            32,
            1,
            "cpu",
        ),
        (
            "innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed0.csv",
            "gift64_cross_spn_typed_transfer_8192",
            8192,
            10,
            256,
            0,
            "cpu",
        ),
        (
            "innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed1.csv",
            "gift64_cross_spn_typed_transfer_8192_seed1",
            8192,
            10,
            256,
            1,
            "cpu",
        ),
        (
            "innovation1_spn_gift64_cross_spn_typed_transfer_65536_seed0.csv",
            "gift64_cross_spn_typed_transfer_65536_seed0",
            65536,
            10,
            256,
            0,
            "cuda",
        ),
        (
            "innovation1_spn_gift64_cross_spn_typed_transfer_65536_seed1.csv",
            "gift64_cross_spn_typed_transfer_65536_seed1",
            65536,
            10,
            256,
            1,
            "cuda",
        ),
    ],
)
def test_cross_spn_transfer_matrices_build_exact_frozen_tasks(
    filename: str,
    family: str,
    samples_per_class: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    plan = Path("configs/experiment/innovation1") / filename
    args = parse_matrix_args(
        [
            "--plan",
            str(plan),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--hidden-bits",
            "32",
            "--device",
            device,
            "--initialization-manifest",
            str(MANIFEST),
        ]
    )
    tasks = build_tasks(args)
    with plan.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(tasks) == len(rows) == 5
    assert [task["model_key"] for task in tasks] == list(ROLES.values())
    assert [int(row["architecture_rank"]) for row in rows] == [0, 1, 2, 3, 4]
    assert {row["family"] for row in rows} == {family}
    assert {task["cipher_key"] for task in tasks} == {"gift64"}
    assert {task["rounds"] for task in tasks} == {6}
    assert {task["seed"] for task in tasks} == {seed}
    assert {task["samples_per_class"] for task in tasks} == {samples_per_class}
    assert {task["pairs_per_sample"] for task in tasks} == {4}
    assert {task["feature_encoding"] for task in tasks} == {"ciphertext_pair_bits"}
    assert {task["negative_mode"] for task in tasks} == {
        "encrypted_random_plaintexts"
    }
    assert {task["sample_structure"] for task in tasks} == {"independent_pairs"}
    assert {task["difference_profile"] for task in tasks} == {
        "gift64_shen2024_spn_screen"
    }
    assert {task["loss"] for task in tasks} == {"mse"}
    assert {task["optimizer"] for task in tasks} == {"adam"}
    assert {task["learning_rate"] for task in tasks} == {0.0001}
    assert {task["weight_decay"] for task in tasks} == {0.00001}
    assert {task["lr_scheduler"] for task in tasks} == {"none"}
    assert {task["checkpoint_metric"] for task in tasks} == {"val_auc"}
    assert all(task["restore_best_checkpoint"] for task in tasks)
    assert args.epochs == epochs
    assert args.batch_size == batch_size
    assert args.device == device


def test_cross_spn_transfer_r3_matrices_differ_only_by_target_seed() -> None:
    plan_root = Path("configs/experiment/innovation1")
    seed0_plan = (
        plan_root
        / "innovation1_spn_gift64_cross_spn_typed_transfer_65536_seed0.csv"
    )
    seed1_plan = (
        plan_root
        / "innovation1_spn_gift64_cross_spn_typed_transfer_65536_seed1.csv"
    )
    with seed0_plan.open(newline="", encoding="utf-8") as handle:
        seed0_rows = list(csv.DictReader(handle))
    with seed1_plan.open(newline="", encoding="utf-8") as handle:
        seed1_rows = list(csv.DictReader(handle))

    assert len(seed0_rows) == len(seed1_rows) == 5
    for seed0, seed1 in zip(seed0_rows, seed1_rows, strict=True):
        assert seed0["seed"] == "0"
        assert seed1["seed"] == "1"
        assert seed0["family"].endswith("seed0")
        assert seed1["family"].endswith("seed1")
        for field in ("network", "family", "seed", "evidence"):
            seed0.pop(field)
            seed1.pop(field)
        assert seed0 == seed1


@pytest.mark.parametrize(
    ("seed", "gpu"),
    [(0, 0), (1, 1)],
)
def test_cross_spn_transfer_r3_remote_configs_pass_readiness(
    seed: int,
    gpu: int,
) -> None:
    config = Path(
        "configs/remote/"
        "innovation1_gift64_cross_spn_typed_transfer_r3_65536_"
        f"seed{seed}_gpu{gpu}_20260714.json"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    report = remote_readiness_report(config)

    assert report["status"] == "pass", report["errors"]
    assert report["expected_rows"] == report["plan_rows"] == 5
    assert report["max_samples_per_class"] == 65536
    assert payload["device"] == "cuda"
    assert payload["physical_gpu"] == gpu
    assert payload["dataset_cache"] is True
    assert payload["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )


def test_cross_spn_transfer_r3_remote_assets_are_fail_closed() -> None:
    run_script = Path(
        "configs/remote/generated/"
        "run_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.cmd"
    ).read_text(encoding="utf-8")
    launcher = Path(
        "configs/remote/generated/"
        "launch_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.cmd"
    ).read_text(encoding="utf-8")
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.sh"
    ).read_text(encoding="utf-8")

    assert "set CUDA_VISIBLE_DEVICES=%PHYSICAL_GPU%" in run_script
    assert "source_expected_commit.txt" in run_script
    assert "goto source_revision_mismatch" in run_script
    assert "--device cuda" in run_script
    assert "--initialization-manifest" in run_script
    assert "--dataset-cache-root" in run_script
    assert "--progress-output" in run_script
    assert "--experiment-stage e4_r3" in run_script
    assert "git add \"results_archive\\%RUN_ID%\"" in run_script
    assert "git add ." not in run_script
    assert "mkdir \"%JOINT_LOCK%\" > nul 2>&1 || exit /b 0" in run_script
    assert "rmdir \"%JOINT_LOCK%\"" in run_script
    assert "cmd.exe /k" not in run_script.lower()
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run_script

    assert launcher.count("cmd.exe /c") == 2
    assert "cmd.exe /k" not in launcher.lower()
    assert launcher.count("schtasks /Run") == 2
    assert "set SOURCE_COMMIT=%~1" in launcher
    assert 'git checkout --detach "%EXPECTED_COMMIT%"' in launcher
    assert 'if /I not "%ACTUAL_COMMIT%"=="%EXPECTED_COMMIT%" exit /b 1' in launcher
    assert "findstr /x" not in launcher
    assert "source_expected_commit.txt" in launcher
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher

    assert "outputs/remote_results_incomplete" in monitor
    assert "retrieved_from_verified_result_branch.marker" in monitor
    assert "scripts/index-results" in monitor
    assert "results_archive/${run_id}" in monitor


def test_cross_spn_transfer_source_assets_match_manifest_hashes() -> None:
    source_root = Path(
        "outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0"
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    targets = manifest["targets"]
    expected = {
        "gift_cross_spn_typed_cell_true_from_present_true": (
            source_root
            / "checkpoints/row0002_present_cross_spn_typed_cell_true_seed0.pt"
        ),
        "gift_cross_spn_typed_cell_true_from_present_shuffled": (
            source_root
            / "checkpoints/row0003_present_cross_spn_typed_cell_shuffled_seed0.pt"
        ),
    }

    assert (source_root / "results.jsonl").is_file()
    for target, checkpoint in expected.items():
        assert checkpoint.is_file()
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == targets[target]["source_checkpoint_sha256"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _progress_path(results: Path) -> Path:
    return results.with_name(f"{results.stem}.progress.jsonl")


def _set_auc(row: dict[str, Any], auc: float) -> None:
    epochs = len(row["history"])
    for index, history in enumerate(row["history"], start=1):
        history["val_auc"] = auc - 0.001 * (epochs - index)
    row["metrics"]["auc"] = auc
    row["training"]["best_epoch"] = epochs
    row["training"]["best_checkpoint_metric"] = auc


def _initialization(role: str) -> dict[str, Any]:
    if role == "gift_anchor":
        return {
            "kind": "scratch",
            "target_model": ROLES[role],
            "target_mapping": "aligned",
            "strict_state_dict_load": False,
            "state_dict_key_count": 20,
            "initial_state_sha256": "a" * 64,
        }
    if role == "gift_typed_scratch":
        return {
            "kind": "scratch",
            "target_model": ROLES[role],
            "target_mapping": "true",
            "strict_state_dict_load": False,
            "state_dict_key_count": 40,
            "initial_state_sha256": "b" * 64,
        }
    true_source = role in {"true_to_true", "true_to_shuffled"}
    target_mapping = "shuffled" if role == "true_to_shuffled" else "true"
    return {
        "kind": "checkpoint",
        "source_checkpoint": (
            "outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/"
            "checkpoints/row0002_present_cross_spn_typed_cell_true_seed0.pt"
            if true_source
            else "outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/"
            "checkpoints/row0003_present_cross_spn_typed_cell_shuffled_seed0.pt"
        ),
        "source_checkpoint_sha256": (
            "eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1"
            if true_source
            else "fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22"
        ),
        "source_results": (
            "outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/"
            "results.jsonl"
        ),
        "source_model": (
            "present_cross_spn_typed_cell_true"
            if true_source
            else "present_cross_spn_typed_cell_shuffled"
        ),
        "source_cipher": "PRESENT-80",
        "source_rounds": 7,
        "source_seed": 0,
        "source_samples_per_class": 8192,
        "source_epochs": 10,
        "source_mapping": "true" if true_source else "shuffled",
        "target_model": ROLES[role],
        "target_mapping": target_mapping,
        "strict_state_dict_load": True,
        "state_dict_key_count": 40,
        "initial_state_sha256": "3" * 64 if true_source else "4" * 64,
    }


def _write_transfer_run(
    results: Path,
    aucs: dict[str, float],
    *,
    samples_per_class: int = 8192,
    epochs: int = 10,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    base = results.with_name("base.jsonl")
    r1_fixtures._write_cross_run(
        base,
        {
            "anchor": aucs["gift_anchor"],
            "candidate": aucs["gift_typed_scratch"],
            "shuffled_p": aucs["true_to_shuffled"],
            "raw_delta": 0.5,
        },
        cipher_key="gift64",
        samples_per_class=samples_per_class,
        epochs=epochs,
    )
    base_rows = {
        row["selected_model"]: row for row in _read_jsonl(base)
    }
    templates = {
        "gift_anchor": base_rows[r1_fixtures.GIFT_ROLES["anchor"]],
        "gift_typed_scratch": base_rows[r1_fixtures.GIFT_ROLES["candidate"]],
        "true_to_true": base_rows[r1_fixtures.GIFT_ROLES["candidate"]],
        "shuffled_to_true": base_rows[r1_fixtures.GIFT_ROLES["candidate"]],
        "true_to_shuffled": base_rows[r1_fixtures.GIFT_ROLES["shuffled_p"]],
    }
    rows = []
    for role, model in ROLES.items():
        row = copy.deepcopy(templates[role])
        row["model"] = row["selected_model"] = model
        row["seed"] = seed
        row["training"]["device"] = device
        row["initialization"] = _initialization(role)
        row["parameter_count"] = 219970 if role == "gift_anchor" else 187426
        row["trainable_parameter_count"] = row["parameter_count"]
        _set_auc(row, aucs[role])
        rows.append(row)
    _write_jsonl(results, rows)

    cache_root = rows[0]["training"]["dataset_cache_root"]
    progress_rows: list[dict[str, Any]] = []
    for index, (role, model) in enumerate(ROLES.items(), start=1):
        initialization = _initialization(role)
        progress_rows.append(
            {
                "event": "initialization_ready",
                "index": index,
                "total": 5,
                "kind": initialization["kind"],
                "model": model,
                "source_model": initialization.get("source_model"),
                "source_checkpoint_sha256": initialization.get(
                    "source_checkpoint_sha256"
                ),
                "strict_state_dict_load": initialization["strict_state_dict_load"],
                "target_model": model,
                "target_mapping": initialization["target_mapping"],
                "seed": seed,
            }
        )
        for split in ("train", "validation"):
            progress_rows.append(
                {
                    "event": "cache_done" if role == "gift_anchor" else "cache_reuse",
                    "model": model,
                    "seed": seed,
                    "split": split,
                    "cache_path": str(Path(cache_root) / split),
                    "cipher_key": "gift64",
                    "rounds": 6,
                    "dataset_label_mode": "balanced_per_class",
                    "feature_encoding": "ciphertext_pair_bits",
                    "negative_mode": "encrypted_random_plaintexts",
                    "pairs_per_sample": 4,
                    "key_rotation_interval": 0,
                    "sample_structure": "independent_pairs",
                    "difference_profile": "gift64_shen2024_spn_screen",
                    "difference_member": 0,
                    "input_bits": 512,
                    "optimizer_state_transition": "reset_each_stage",
                    "loss": "mse",
                    "samples_per_class": samples_per_class,
                    "total_rows": (
                        2 * samples_per_class
                        if split == "train"
                        else samples_per_class
                    ),
                }
            )
    progress_rows.append(
        {"event": "run_done", "output": str(results), "total": 5}
    )
    _write_jsonl(_progress_path(results), progress_rows)


def _gate(results: Path, **kwargs: Any) -> dict[str, Any]:
    return gate_cross_spn_typed_transfer(
        [results],
        progress_paths=[_progress_path(results)],
        **kwargs,
    )


def test_cross_spn_transfer_roles_are_exact() -> None:
    assert TRANSFER_MODEL_ROLES == ROLES


def test_cross_spn_transfer_readiness_is_neutral(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_transfer_run(
        results,
        {
            "gift_anchor": 0.9,
            "gift_typed_scratch": 0.8,
            "true_to_true": 0.1,
            "shuffled_to_true": 0.2,
            "true_to_shuffled": 0.3,
        },
        samples_per_class=64,
        epochs=1,
    )

    report = _gate(
        results,
        samples_per_class=64,
        epochs=1,
        readiness_only=True,
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "implementation_ready"
    assert report["research_decision_applied"] is False


@pytest.mark.parametrize(
    ("aucs", "decision"),
    [
        (
            {
                "gift_anchor": 0.500,
                "gift_typed_scratch": 0.540,
                "true_to_true": 0.560,
                "shuffled_to_true": 0.550,
                "true_to_shuffled": 0.545,
            },
            "promote_e4_transfer_seed1",
        ),
        (
            {
                "gift_anchor": 0.500,
                "gift_typed_scratch": 0.557,
                "true_to_true": 0.560,
                "shuffled_to_true": 0.550,
                "true_to_shuffled": 0.545,
            },
            "weak_transfer_no_scale",
        ),
        (
            {
                "gift_anchor": 0.500,
                "gift_typed_scratch": 0.540,
                "true_to_true": 0.560,
                "shuffled_to_true": 0.561,
                "true_to_shuffled": 0.545,
            },
            "generic_pretraining_not_typed_transfer",
        ),
        (
            {
                "gift_anchor": 0.500,
                "gift_typed_scratch": 0.540,
                "true_to_true": 0.560,
                "shuffled_to_true": 0.550,
                "true_to_shuffled": 0.561,
            },
            "target_topology_not_attributed",
        ),
        (
            {
                "gift_anchor": 0.500,
                "gift_typed_scratch": 0.561,
                "true_to_true": 0.560,
                "shuffled_to_true": 0.550,
                "true_to_shuffled": 0.545,
            },
            "reject_e4_transfer",
        ),
    ],
)
def test_cross_spn_transfer_r2_decisions(
    tmp_path: Path,
    aucs: dict[str, float],
    decision: str,
) -> None:
    results = tmp_path / "results.jsonl"
    _write_transfer_run(results, aucs)

    report = _gate(results)

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == decision


def test_cross_spn_transfer_gate_rejects_provenance_mismatch(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_transfer_run(
        results,
        {
            "gift_anchor": 0.500,
            "gift_typed_scratch": 0.540,
            "true_to_true": 0.560,
            "shuffled_to_true": 0.550,
            "true_to_shuffled": 0.545,
        },
    )
    rows = _read_jsonl(results)
    rows[2]["initialization"]["source_checkpoint_sha256"] = "9" * 64
    _write_jsonl(results, rows)

    report = _gate(results)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid_e4_protocol"
    assert any("source_checkpoint_sha256" in error for error in report["errors"])


def test_cross_spn_transfer_gate_cli_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from blockcipher_nd.cli.gate_cross_spn_typed_transfer import main

    results = tmp_path / "results.jsonl"
    output = tmp_path / "gate" / "gate.json"
    _write_transfer_run(
        results,
        {
            "gift_anchor": 0.500,
            "gift_typed_scratch": 0.540,
            "true_to_true": 0.560,
            "shuffled_to_true": 0.550,
            "true_to_shuffled": 0.545,
        },
    )

    exit_code = main(
        [
            "--results",
            str(results),
            "--progress",
            str(_progress_path(results)),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["decision"] == "promote_e4_transfer_seed1"
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert capsys.readouterr().out == json.dumps(report, sort_keys=True) + "\n"


def test_cross_spn_transfer_seed1_promotes_only_joint_gate(tmp_path: Path) -> None:
    results = tmp_path / "seed1.jsonl"
    _write_transfer_run(
        results,
        {
            "gift_anchor": 0.500,
            "gift_typed_scratch": 0.540,
            "true_to_true": 0.560,
            "shuffled_to_true": 0.550,
            "true_to_shuffled": 0.545,
        },
        seed=1,
    )

    report = _gate(results, expected_seeds=(1,))

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "promote_e4_transfer_joint_gate"
    assert report["next_action"] == "run_frozen_e4_r2_joint_gate"
    assert report["expected_seeds"] == [1]


def test_cross_spn_transfer_joint_confirms_two_passing_seeds(tmp_path: Path) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    aucs = {
        "gift_anchor": 0.500,
        "gift_typed_scratch": 0.540,
        "true_to_true": 0.560,
        "shuffled_to_true": 0.550,
        "true_to_shuffled": 0.545,
    }
    _write_transfer_run(seed0, aucs, seed=0)
    _write_transfer_run(seed1, aucs, seed=1)

    report = gate_cross_spn_typed_transfer_joint(
        [seed0, seed1],
        progress_paths=[_progress_path(seed0), _progress_path(seed1)],
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "two_seed_transfer_signal_confirmed"
    assert report["next_action"] == "design_e4_r3_same_protocol_medium_diagnostic"
    assert report["expected_seeds"] == [0, 1]


def test_cross_spn_transfer_joint_stops_after_seed1_control_failure(
    tmp_path: Path,
) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    passing = {
        "gift_anchor": 0.500,
        "gift_typed_scratch": 0.540,
        "true_to_true": 0.560,
        "shuffled_to_true": 0.550,
        "true_to_shuffled": 0.545,
    }
    seed1_failure = {**passing, "shuffled_to_true": 0.561}
    _write_transfer_run(seed0, passing, seed=0)
    _write_transfer_run(seed1, seed1_failure, seed=1)

    report = gate_cross_spn_typed_transfer_joint(
        [seed0, seed1],
        progress_paths=[_progress_path(seed0), _progress_path(seed1)],
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "two_seed_transfer_unstable_no_scale"
    assert report["next_action"] == "stop_e4_transfer_scale_after_two_seed_variance"
    assert {item["action"] for item in report["stopped_actions"]} >= {
        "remote_scale",
        "sample_scale",
        "formal_claim",
    }


def test_cross_spn_transfer_r3_accepts_only_frozen_medium_budget(
    tmp_path: Path,
) -> None:
    results = tmp_path / "seed0.jsonl"
    aucs = {
        "gift_anchor": 0.500,
        "gift_typed_scratch": 0.540,
        "true_to_true": 0.560,
        "shuffled_to_true": 0.550,
        "true_to_shuffled": 0.545,
    }
    _write_transfer_run(
        results,
        aucs,
        samples_per_class=65536,
        device="cuda",
    )

    report = _gate(
        results,
        samples_per_class=65536,
        experiment_stage="e4_r3",
    )
    wrong_budget = _gate(
        results,
        samples_per_class=8192,
        experiment_stage="e4_r3",
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "e4_r3_seed_signal_preserved"
    assert report["next_action"] == "run_e4_r3_joint_gate"
    assert report["experiment_stage"] == "e4_r3"
    assert wrong_budget["status"] == "fail"
    assert any("65536/class" in error for error in wrong_budget["errors"])


def test_cross_spn_transfer_r3_joint_confirms_two_medium_seeds(
    tmp_path: Path,
) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    aucs = {
        "gift_anchor": 0.500,
        "gift_typed_scratch": 0.540,
        "true_to_true": 0.560,
        "shuffled_to_true": 0.550,
        "true_to_shuffled": 0.545,
    }
    _write_transfer_run(
        seed0,
        aucs,
        samples_per_class=65536,
        seed=0,
        device="cuda",
    )
    _write_transfer_run(
        seed1,
        aucs,
        samples_per_class=65536,
        seed=1,
        device="cuda",
    )

    report = gate_cross_spn_typed_transfer_joint(
        [seed0, seed1],
        progress_paths=[_progress_path(seed0), _progress_path(seed1)],
        samples_per_class=65536,
        experiment_stage="e4_r3",
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "e4_r3_two_seed_medium_signal_confirmed"
    assert report["next_action"] == (
        "design_e4_r4_262144_class_diagnostic_with_remote_readiness"
    )
    assert {item["action"] for item in report["stopped_actions"]} == {
        "remote_launch",
        "sample_scale",
        "formal_claim",
    }


def test_cross_spn_transfer_r3_joint_stops_after_margin_miss(
    tmp_path: Path,
) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    passing = {
        "gift_anchor": 0.500,
        "gift_typed_scratch": 0.540,
        "true_to_true": 0.560,
        "shuffled_to_true": 0.550,
        "true_to_shuffled": 0.545,
    }
    seed1_margin_miss = {**passing, "gift_typed_scratch": 0.557}
    _write_transfer_run(
        seed0,
        passing,
        samples_per_class=65536,
        seed=0,
        device="cuda",
    )
    _write_transfer_run(
        seed1,
        seed1_margin_miss,
        samples_per_class=65536,
        seed=1,
        device="cuda",
    )

    report = gate_cross_spn_typed_transfer_joint(
        [seed0, seed1],
        progress_paths=[_progress_path(seed0), _progress_path(seed1)],
        samples_per_class=65536,
        experiment_stage="e4_r3",
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "e4_r3_two_seed_medium_signal_unstable"
    assert report["next_action"] == (
        "stop_mechanical_scale_and_audit_seed_variance"
    )


def test_cross_spn_transfer_gate_cli_forwards_seed1(
    tmp_path: Path,
) -> None:
    from blockcipher_nd.cli.gate_cross_spn_typed_transfer import main

    results = tmp_path / "seed1.jsonl"
    output = tmp_path / "seed1-gate.json"
    _write_transfer_run(
        results,
        {
            "gift_anchor": 0.500,
            "gift_typed_scratch": 0.540,
            "true_to_true": 0.560,
            "shuffled_to_true": 0.550,
            "true_to_shuffled": 0.545,
        },
        seed=1,
    )

    assert main(
        [
            "--results",
            str(results),
            "--progress",
            str(_progress_path(results)),
            "--expected-seed",
            "1",
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == (
        "promote_e4_transfer_joint_gate"
    )


def test_cross_spn_transfer_gate_cli_forwards_r3_stage(tmp_path: Path) -> None:
    from blockcipher_nd.cli.gate_cross_spn_typed_transfer import main

    results = tmp_path / "seed1.jsonl"
    output = tmp_path / "seed1-r3-gate.json"
    _write_transfer_run(
        results,
        {
            "gift_anchor": 0.500,
            "gift_typed_scratch": 0.540,
            "true_to_true": 0.560,
            "shuffled_to_true": 0.550,
            "true_to_shuffled": 0.545,
        },
        samples_per_class=65536,
        seed=1,
        device="cuda",
    )

    assert main(
        [
            "--results",
            str(results),
            "--progress",
            str(_progress_path(results)),
            "--expected-seed",
            "1",
            "--samples-per-class",
            "65536",
            "--experiment-stage",
            "e4_r3",
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "e4_r3_seed_signal_preserved"
    assert report["experiment_stage"] == "e4_r3"


def test_cross_spn_transfer_joint_cli_writes_two_seed_report(
    tmp_path: Path,
) -> None:
    from blockcipher_nd.cli.gate_cross_spn_typed_transfer_joint import main

    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    output = tmp_path / "joint-gate.json"
    aucs = {
        "gift_anchor": 0.500,
        "gift_typed_scratch": 0.540,
        "true_to_true": 0.560,
        "shuffled_to_true": 0.550,
        "true_to_shuffled": 0.545,
    }
    _write_transfer_run(seed0, aucs, seed=0)
    _write_transfer_run(seed1, aucs, seed=1)

    assert main(
        [
            "--seed0-results",
            str(seed0),
            "--seed0-progress",
            str(_progress_path(seed0)),
            "--seed1-results",
            str(seed1),
            "--seed1-progress",
            str(_progress_path(seed1)),
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "two_seed_transfer_signal_confirmed"
    assert output.read_text(encoding="utf-8").endswith("\n")
