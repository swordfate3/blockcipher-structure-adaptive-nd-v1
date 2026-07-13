from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import test_invp_topology_residual_gate as h1_fixtures
from blockcipher_nd.engine.matrix_runner import parse_args as parse_matrix_args
from blockcipher_nd.planning.cross_spn_typed_cell_gate import (
    GIFT_CROSS_SPN_MODEL_ROLES,
    PRESENT_CROSS_SPN_MODEL_ROLES,
    gate_cross_spn_typed_cell,
)
from blockcipher_nd.planning.matrix import build_tasks


PRESENT_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_cross_spn_typed_cell_true",
    "shuffled_p": "present_cross_spn_typed_cell_shuffled",
    "raw_delta": "present_cross_spn_typed_cell_raw",
}

GIFT_ROLES = {
    "anchor": "gift_cross_spn_aligned_token_mixer_raw_anchor",
    "candidate": "gift_cross_spn_typed_cell_true",
    "shuffled_p": "gift_cross_spn_typed_cell_shuffled",
    "raw_delta": "gift_cross_spn_typed_cell_raw",
}

TYPED_OPTIONS = {
    "mixer_depth": 2,
    "token_mlp_ratio": 2,
    "activation": "relu",
    "norm": "layernorm",
    "pooling": "attention_mean_max",
    "dropout": 0.0,
}
GIFT_ANCHOR_OPTIONS = {
    "mixer_depth": 1,
    "token_mlp_ratio": 2,
    "activation": "relu",
    "norm": "layernorm",
    "pooling": "topk_logsumexp",
    "dropout": 0.0,
    "top_k": 2,
    "lse_temperature": 1.0,
}


@pytest.mark.parametrize(
    (
        "filename",
        "family",
        "cipher_key",
        "roles",
        "rounds",
        "pairs_per_sample",
        "sample_structure",
        "difference_profile",
        "samples_per_class",
        "epochs",
        "batch_size",
    ),
    [
        (
            "innovation1_spn_present_cross_spn_typed_cell_smoke_seed0.csv",
            "present_cross_spn_typed_cell_smoke",
            "present80",
            PRESENT_ROLES,
            7,
            16,
            "zhang_wang_case2_official_mcnd",
            "present_zhang_wang2022_mcnd",
            64,
            1,
            32,
        ),
        (
            "innovation1_spn_gift64_cross_spn_typed_cell_smoke_seed0.csv",
            "gift64_cross_spn_typed_cell_smoke",
            "gift64",
            GIFT_ROLES,
            6,
            4,
            "independent_pairs",
            "gift64_shen2024_spn_screen",
            64,
            1,
            32,
        ),
        (
            "innovation1_spn_present_cross_spn_typed_cell_8192_seed0.csv",
            "present_cross_spn_typed_cell_8192",
            "present80",
            PRESENT_ROLES,
            7,
            16,
            "zhang_wang_case2_official_mcnd",
            "present_zhang_wang2022_mcnd",
            8192,
            10,
            256,
        ),
        (
            "innovation1_spn_gift64_cross_spn_typed_cell_8192_seed0.csv",
            "gift64_cross_spn_typed_cell_8192",
            "gift64",
            GIFT_ROLES,
            6,
            4,
            "independent_pairs",
            "gift64_shen2024_spn_screen",
            8192,
            10,
            256,
        ),
    ],
)
def test_cross_spn_matrices_build_exact_frozen_tasks(
    filename: str,
    family: str,
    cipher_key: str,
    roles: dict[str, str],
    rounds: int,
    pairs_per_sample: int,
    sample_structure: str,
    difference_profile: str,
    samples_per_class: int,
    epochs: int,
    batch_size: int,
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
            "cpu",
            "--dataset-cache-root",
            str(Path("outputs/local_cache") / family),
            "--dataset-cache-chunk-size",
            "64" if samples_per_class == 64 else "512",
            "--dataset-cache-workers",
            "1" if samples_per_class == 64 else "4",
            "--train-eval-interval",
            "1",
        ]
    )
    tasks = build_tasks(args)
    with plan.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(tasks) == len(rows) == 4
    assert [task["model_key"] for task in tasks] == list(roles.values())
    assert [int(row["architecture_rank"]) for row in rows] == [0, 1, 2, 3]
    assert {row["family"] for row in rows} == {family}
    assert {task["cipher_key"] for task in tasks} == {cipher_key}
    assert {task["rounds"] for task in tasks} == {rounds}
    assert {task["seed"] for task in tasks} == {0}
    assert {task["samples_per_class"] for task in tasks} == {samples_per_class}
    assert {task["pairs_per_sample"] for task in tasks} == {pairs_per_sample}
    assert {task["feature_encoding"] for task in tasks} == {
        "ciphertext_pair_bits"
    }
    assert {task["negative_mode"] for task in tasks} == {
        "encrypted_random_plaintexts"
    }
    assert {task["sample_structure"] for task in tasks} == {sample_structure}
    assert {task["difference_profile"] for task in tasks} == {difference_profile}
    assert {task["difference_member"] for task in tasks} == {0}
    assert {task["loss"] for task in tasks} == {"mse"}
    assert {task["optimizer"] for task in tasks} == {"adam"}
    assert {task["learning_rate"] for task in tasks} == {0.0001}
    assert {task["weight_decay"] for task in tasks} == {0.00001}
    assert {task["lr_scheduler"] for task in tasks} == {"none"}
    assert {task["checkpoint_metric"] for task in tasks} == {"val_auc"}
    assert all(task["restore_best_checkpoint"] for task in tasks)
    assert {task["early_stopping_patience"] for task in tasks} == {0}
    assert args.epochs == epochs
    assert args.batch_size == batch_size
    assert args.train_eval_interval == 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _progress_path(results_path: Path) -> Path:
    return results_path.with_name(f"{results_path.stem}.progress.jsonl")


def _write_cross_run(
    results_path: Path,
    aucs: dict[str, float],
    *,
    cipher_key: str,
    samples_per_class: int = 8192,
    epochs: int = 10,
) -> None:
    roles = PRESENT_ROLES if cipher_key == "present80" else GIFT_ROLES
    h1_fixtures._write_h1_run(
        results_path,
        {
            "anchor": aucs["anchor"],
            "candidate": aucs["candidate"],
            "shuffled_p": aucs["shuffled_p"],
            "delta_only": aucs["raw_delta"],
        },
        samples_per_class=samples_per_class,
        epochs=epochs,
    )
    role_by_old_model = {
        model: role for role, model in h1_fixtures.H1_ROLES.items()
    }
    rows = _read_jsonl(results_path)
    for row in rows:
        role = role_by_old_model[row["model"]]
        model = roles["raw_delta" if role == "delta_only" else role]
        row["model"] = row["selected_model"] = model
        row["training"].update(
            {
                "lr_scheduler": "none",
                "max_learning_rate": None,
                "early_stopping_patience": 0,
                "early_stopping_min_delta": 0.0,
                "model_options": (
                    {"spn_mixer_depth": 2, "activation": "relu", "norm": "layernorm"}
                    if cipher_key == "present80" and role == "anchor"
                    else GIFT_ANCHOR_OPTIONS
                    if cipher_key == "gift64" and role == "anchor"
                    else TYPED_OPTIONS
                ),
            }
        )
        if role != "anchor":
            row["parameter_count"] = 2222
            row["trainable_parameter_count"] = 2222
        if cipher_key == "gift64":
            row.update(
                {
                    "cipher": "GIFT-64",
                    "cipher_key": "gift64",
                    "rounds": 6,
                    "input_difference": 0x40,
                    "pairs_per_sample": 4,
                    "train_key": 0,
                    "validation_key": int("11" * 16, 16),
                    "sample_structure": "independent_pairs",
                    "difference_profile": "gift64_shen2024_spn_screen",
                }
            )
            row["training"].update(
                {
                    "key_schedule": "fixed",
                    "input_bits": 512,
                    "pairs_per_sample": 4,
                    "sample_structure": "independent_pairs",
                }
            )
            row["validation"].update(
                {
                    "cipher": "GIFT-64",
                    "rounds": 6,
                    "key_schedule": "fixed",
                    "pairs_per_sample": 4,
                    "sample_structure": "independent_pairs",
                }
            )
    _write_jsonl(results_path, rows)

    progress_rows = _read_jsonl(_progress_path(results_path))
    for row in progress_rows:
        old_model = row.get("model")
        if old_model in role_by_old_model:
            role = role_by_old_model[old_model]
            row["model"] = roles["raw_delta" if role == "delta_only" else role]
        if cipher_key == "gift64" and row.get("event") in {
            "cache_done",
            "cache_reuse",
        }:
            row.update(
                {
                    "cipher_key": "gift64",
                    "rounds": 6,
                    "pairs_per_sample": 4,
                    "sample_structure": "independent_pairs",
                    "difference_profile": "gift64_shen2024_spn_screen",
                    "input_bits": 512,
                }
            )
    _write_jsonl(_progress_path(results_path), progress_rows)


def _gate(
    present_results: Path,
    gift_results: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return gate_cross_spn_typed_cell(
        [present_results],
        present_progress_paths=[_progress_path(present_results)],
        gift_results_paths=[gift_results],
        gift_progress_paths=[_progress_path(gift_results)],
        **kwargs,
    )


def test_cross_spn_gate_roles_are_exact() -> None:
    assert PRESENT_CROSS_SPN_MODEL_ROLES == PRESENT_ROLES
    assert GIFT_CROSS_SPN_MODEL_ROLES == GIFT_ROLES


def test_cross_spn_readiness_is_neutral(tmp_path: Path) -> None:
    present = tmp_path / "present.jsonl"
    gift = tmp_path / "gift.jsonl"
    _write_cross_run(
        present,
        {"anchor": 0.9, "candidate": 0.1, "shuffled_p": 0.2, "raw_delta": 0.3},
        cipher_key="present80",
        samples_per_class=64,
        epochs=1,
    )
    _write_cross_run(
        gift,
        {"anchor": 0.1, "candidate": 0.9, "shuffled_p": 0.8, "raw_delta": 0.7},
        cipher_key="gift64",
        samples_per_class=64,
        epochs=1,
    )

    report = _gate(
        present,
        gift,
        samples_per_class=64,
        epochs=1,
        readiness_only=True,
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "implementation_ready"
    assert report["research_decision_applied"] is False


@pytest.mark.parametrize(
    ("present_aucs", "decision", "next_action"),
    [
        (
            {"anchor": 0.660, "candidate": 0.670, "shuffled_p": 0.660, "raw_delta": 0.660},
            "promote_e4_r2",
            "freeze_and_implement_e4_r2_checkpoint_transfer",
        ),
        (
            {"anchor": 0.660, "candidate": 0.651, "shuffled_p": 0.649, "raw_delta": 0.648},
            "run_present_seed1_fragility",
            "run_only_same_budget_present_seed1_fragility_gate",
        ),
        (
            {"anchor": 0.630, "candidate": 0.640, "shuffled_p": 0.620, "raw_delta": 0.610},
            "reject_e4_shared_operator",
            "stop_e4_transfer_and_consolidate_invp_method_evidence",
        ),
    ],
)
def test_cross_spn_r1_source_decisions_ignore_weak_gift_scratch(
    tmp_path: Path,
    present_aucs: dict[str, float],
    decision: str,
    next_action: str,
) -> None:
    present = tmp_path / "present.jsonl"
    gift = tmp_path / "gift.jsonl"
    _write_cross_run(present, present_aucs, cipher_key="present80")
    _write_cross_run(
        gift,
        {"anchor": 0.501, "candidate": 0.499, "shuffled_p": 0.502, "raw_delta": 0.500},
        cipher_key="gift64",
    )

    report = _gate(present, gift)

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == decision
    assert report["next_action"] == next_action
    assert report["research_decision_applied"] is True


def test_cross_spn_gate_rejects_invalid_gift_protocol(tmp_path: Path) -> None:
    present = tmp_path / "present.jsonl"
    gift = tmp_path / "gift.jsonl"
    _write_cross_run(
        present,
        {"anchor": 0.660, "candidate": 0.670, "shuffled_p": 0.660, "raw_delta": 0.660},
        cipher_key="present80",
    )
    _write_cross_run(
        gift,
        {"anchor": 0.501, "candidate": 0.499, "shuffled_p": 0.502, "raw_delta": 0.500},
        cipher_key="gift64",
    )
    rows = _read_jsonl(gift)
    rows[1]["training"]["key_schedule"] = "per_pair_random"
    _write_jsonl(gift, rows)

    report = _gate(present, gift)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid_e4_protocol"
    assert any("gift:" in error and "key_schedule" in error for error in report["errors"])


def test_cross_spn_gate_cli_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from blockcipher_nd.cli.gate_cross_spn_typed_cell import main

    present = tmp_path / "present.jsonl"
    gift = tmp_path / "gift.jsonl"
    output = tmp_path / "nested" / "gate.json"
    _write_cross_run(
        present,
        {"anchor": 0.660, "candidate": 0.670, "shuffled_p": 0.660, "raw_delta": 0.660},
        cipher_key="present80",
    )
    _write_cross_run(
        gift,
        {"anchor": 0.501, "candidate": 0.499, "shuffled_p": 0.502, "raw_delta": 0.500},
        cipher_key="gift64",
    )

    exit_code = main(
        [
            "--present-results",
            str(present),
            "--present-progress",
            str(_progress_path(present)),
            "--gift-results",
            str(gift),
            "--gift-progress",
            str(_progress_path(gift)),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["decision"] == "promote_e4_r2"
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert capsys.readouterr().out == json.dumps(report, sort_keys=True) + "\n"
