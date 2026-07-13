import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pytest

import test_invp_topology_residual_gate as h1_fixtures
from blockcipher_nd.engine.matrix_runner import parse_args as parse_matrix_args
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.planning.present_case3_topology_residual_gate import (
    CASE3_MODEL_ROLES,
    gate_present_case3_topology_residual,
)


CASE3_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_case3_invp_topology_residual_spn_only",
    "shuffled_p": "present_nibble_case3_shuffled_p_topology_residual_spn_only",
    "raw_triple": "present_nibble_case3_raw_topology_residual_spn_only",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _progress_path(results_path: Path) -> Path:
    return results_path.with_name(f"{results_path.stem}.progress.jsonl")


def _write_case3_run(
    results_path: Path,
    aucs: dict[str, float],
    *,
    seed: int = 0,
    samples_per_class: int = 8192,
    epochs: int = 10,
) -> None:
    h1_fixtures._write_h1_run(
        results_path,
        {
            "anchor": aucs["anchor"],
            "candidate": aucs["candidate"],
            "shuffled_p": aucs["shuffled_p"],
            "delta_only": aucs["raw_triple"],
        },
        seed=seed,
        samples_per_class=samples_per_class,
        epochs=epochs,
    )
    h1_to_case3 = {
        h1_fixtures.H1_ROLES["anchor"]: CASE3_ROLES["anchor"],
        h1_fixtures.H1_ROLES["candidate"]: CASE3_ROLES["candidate"],
        h1_fixtures.H1_ROLES["shuffled_p"]: CASE3_ROLES["shuffled_p"],
        h1_fixtures.H1_ROLES["delta_only"]: CASE3_ROLES["raw_triple"],
    }
    result_rows = _read_jsonl(results_path)
    for row in result_rows:
        row["model"] = h1_to_case3[row["model"]]
        row["selected_model"] = h1_to_case3[row["selected_model"]]
    _write_jsonl(results_path, result_rows)

    progress_rows = _read_jsonl(_progress_path(results_path))
    for row in progress_rows:
        if row.get("model") in h1_to_case3:
            row["model"] = h1_to_case3[row["model"]]
    _write_jsonl(_progress_path(results_path), progress_rows)


def _gate(results_path: Path, **kwargs: Any) -> dict[str, Any]:
    return gate_present_case3_topology_residual(
        [results_path],
        progress_paths=[_progress_path(results_path)],
        **kwargs,
    )


def test_case3_roles_are_exact_and_use_raw_triple() -> None:
    assert CASE3_MODEL_ROLES == CASE3_ROLES
    assert "delta_only" not in CASE3_MODEL_ROLES


@pytest.mark.parametrize(
    ("aucs", "decision", "next_action"),
    [
        (
            {
                "anchor": 0.600,
                "candidate": 0.604,
                "shuffled_p": 0.600,
                "raw_triple": 0.600,
            },
            "promote_seed1",
            "run_identical_h2_seed1_local_gate",
        ),
        (
            {
                "anchor": 0.600,
                "candidate": 0.602,
                "shuffled_p": 0.599,
                "raw_triple": 0.598,
            },
            "weak_or_fragile_no_scale",
            "inspect_histories_once_and_stop_h2_scaling",
        ),
        (
            {
                "anchor": 0.604,
                "candidate": 0.604,
                "shuffled_p": 0.600,
                "raw_triple": 0.600,
            },
            "reject_h2",
            "stop_h2_and_keep_token_mixer_anchor",
        ),
        (
            {
                "anchor": 0.600,
                "candidate": 0.604,
                "shuffled_p": 0.604,
                "raw_triple": 0.600,
            },
            "reject_h2",
            "stop_h2_and_keep_token_mixer_anchor",
        ),
        (
            {
                "anchor": 0.600,
                "candidate": 0.604,
                "shuffled_p": 0.600,
                "raw_triple": 0.604,
            },
            "reject_h2",
            "stop_h2_and_keep_token_mixer_anchor",
        ),
    ],
)
def test_case3_seed0_decisions(
    tmp_path: Path,
    aucs: dict[str, float],
    decision: str,
    next_action: str,
) -> None:
    results = tmp_path / "results.jsonl"
    _write_case3_run(results, aucs)

    report = _gate(results)

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == decision
    assert report["next_action"] == next_action
    assert set(report["seeds"]["0"]["aucs"]) == set(CASE3_ROLES)
    assert report["seeds"]["0"]["representation_margin"] == pytest.approx(
        aucs["candidate"] - aucs["raw_triple"]
    )


def test_case3_readiness_is_neutral(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_case3_run(
        results,
        {
            "anchor": 0.9,
            "candidate": 0.1,
            "shuffled_p": 0.2,
            "raw_triple": 0.3,
        },
        samples_per_class=64,
        epochs=1,
    )

    report = _gate(results, samples_per_class=64, epochs=1, readiness_only=True)

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "implementation_ready"
    assert report["research_decision_applied"] is False
    assert report["next_action"] == "run_frozen_h2_seed0_local_diagnostic"


def test_case3_gate_rejects_wrong_effective_key_schedule(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_case3_run(
        results,
        {
            "anchor": 0.6,
            "candidate": 0.61,
            "shuffled_p": 0.59,
            "raw_triple": 0.58,
        },
    )
    rows = _read_jsonl(results)
    rows[1]["training"]["key_schedule"] = "fixed"
    _write_jsonl(results, rows)

    report = _gate(results)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid_protocol"
    assert any("training key_schedule" in error for error in report["errors"])


def test_case3_gate_rejects_missing_raw_triple(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_case3_run(
        results,
        {
            "anchor": 0.6,
            "candidate": 0.61,
            "shuffled_p": 0.59,
            "raw_triple": 0.58,
        },
    )
    _write_jsonl(results, _read_jsonl(results)[:-1])

    report = _gate(results)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid_protocol"
    assert any("raw_triple" in error and "rows=0" in error for error in report["errors"])


def test_case3_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from blockcipher_nd.cli.gate_present_case3_topology_residual import main

    results = tmp_path / "results.jsonl"
    output = tmp_path / "nested" / "gate.json"
    _write_case3_run(
        results,
        {
            "anchor": 0.6,
            "candidate": 0.604,
            "shuffled_p": 0.6,
            "raw_triple": 0.6,
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
    assert report["decision"] == "promote_seed1"
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert capsys.readouterr().out == json.dumps(report, sort_keys=True) + "\n"


@pytest.mark.parametrize("value", ["0,bad", "0,", ",0", ""])
def test_case3_cli_rejects_invalid_expected_seeds(value: str) -> None:
    from blockcipher_nd.cli.gate_present_case3_topology_residual import (
        expected_seeds_arg,
    )

    with pytest.raises(argparse.ArgumentTypeError):
        expected_seeds_arg(value)


@pytest.mark.parametrize(
    ("filename", "family", "samples_per_class", "epochs", "batch_size"),
    [
        (
            "innovation1_spn_present_case3_topology_residual_smoke_seed0.csv",
            "present_case3_topology_residual_smoke",
            64,
            1,
            32,
        ),
        (
            "innovation1_spn_present_case3_topology_residual_8192_seed0.csv",
            "present_case3_topology_residual_8192",
            8192,
            10,
            256,
        ),
    ],
)
def test_case3_matrices_build_exact_frozen_tasks(
    filename: str,
    family: str,
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
            "--device",
            "cpu",
            "--dataset-cache-root",
            str(Path("outputs/local_cache") / family),
            "--dataset-cache-chunk-size",
            "64" if samples_per_class == 64 else "512",
            "--dataset-cache-workers",
            "1" if samples_per_class == 64 else "4",
        ]
    )
    tasks = build_tasks(args)
    with plan.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(tasks) == len(rows) == 4
    assert [task["model_key"] for task in tasks] == list(CASE3_ROLES.values())
    assert [int(row["architecture_rank"]) for row in rows] == [0, 1, 2, 3]
    assert {row["family"] for row in rows} == {family}
    assert {task["samples_per_class"] for task in tasks} == {samples_per_class}
    assert {task["seed"] for task in tasks} == {0}
    assert {task["rounds"] for task in tasks} == {7}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["feature_encoding"] for task in tasks} == {
        "ciphertext_pair_bits"
    }
    assert {task["negative_mode"] for task in tasks} == {
        "encrypted_random_plaintexts"
    }
    assert {task["sample_structure"] for task in tasks} == {
        "zhang_wang_case2_official_mcnd"
    }
    assert {task["difference_profile"] for task in tasks} == {
        "present_zhang_wang2022_mcnd"
    }
    assert {task["loss"] for task in tasks} == {"mse"}
    assert {task["optimizer"] for task in tasks} == {"adam"}
    assert {task["checkpoint_metric"] for task in tasks} == {"val_auc"}
    assert all(task["restore_best_checkpoint"] for task in tasks)
    assert all("effective per_pair_random" in row["evidence"] for row in rows)
    assert args.epochs == epochs
    assert args.batch_size == batch_size
