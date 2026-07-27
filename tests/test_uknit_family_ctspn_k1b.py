from __future__ import annotations

from pathlib import Path

import pytest

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1 import render_ctspn_k1b_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1b import main as run_k1b
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_endpoint_alignment import (
    RUN_ID as K1A_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    RUN_ID as K1_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    K1A_DECISION,
    K1_DECISION,
    adjudicate_k1b,
    build_k1b_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1.csv"
)


def _tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def _k1_gate() -> dict[str, object]:
    return {
        "run_id": K1_RUN_ID,
        "status": "hold",
        "decision": K1_DECISION,
        "protocol_checks": {"complete": True},
        "seed_results": {
            "uknit64": {
                str(seed): {"anchor_auc": 0.526, "candidate_auc": 0.500}
                for seed in (0, 1)
            },
            "dialga128": {
                str(seed): {"anchor_auc": 0.961, "candidate_auc": 0.963}
                for seed in (0, 1)
            },
        },
    }


def _k1a_gate() -> dict[str, object]:
    return {
        "run_id": K1A_RUN_ID,
        "status": "pass",
        "decision": K1A_DECISION,
        "protocol_checks": {"complete": True},
    }


@pytest.fixture(scope="module")
def readiness() -> tuple[list[dict[str, object]], dict[str, object]]:
    return build_k1b_readiness(
        tasks=_tasks(),
        k1_gate=_k1_gate(),
        k1a_gate=_k1a_gate(),
    )


def test_k1b_real_plan_passes_zero_training_readiness(
    readiness: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    manifests, gate = readiness

    assert len(manifests) == 4
    assert gate["status"] == "pass"
    assert gate["optimizer_step_authorized"] is True
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert all(gate["protocol_checks"].values())
    assert all(gate["evidence_checks"].values())
    assert {row["trainable_parameter_count"] for row in manifests} == {439982}
    assert {row["edge_input_values"] for row in manifests} == {22}
    assert {row["endpoint_identity_mode"] for row in manifests} == {
        "native_cell_role"
    }


def test_k1b_gate_requires_every_cipher_seed_and_control() -> None:
    training_rows, control_rows = _passing_rows()
    gate = adjudicate_k1b(
        tasks=_tasks(),
        training_rows=training_rows,
        control_rows=control_rows,
        k1_gate=_k1_gate(),
        k1a_gate=_k1a_gate(),
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = [dict(row) for row in control_rows]
    target = next(
        row
        for row in failed
        if row["cipher_key"] == "uknit64"
        and row["seed"] == 1
        and row["condition"] == "rotated"
    )
    target["auc"] = 0.539
    held = adjudicate_k1b(
        tasks=_tasks(),
        training_rows=training_rows,
        control_rows=failed,
        k1_gate=_k1_gate(),
        k1a_gate=_k1a_gate(),
    )

    assert held["status"] == "hold"
    assert held["research_checks"]["uknit64_seed1_beats_rotated"] is False


def test_k1b_gate_requires_validation_dataset_sha_to_match_k1() -> None:
    training_rows, control_rows = _passing_rows()
    mismatched = [dict(row) for row in control_rows]
    target = next(
        row
        for row in mismatched
        if row["cipher_key"] == "dialga128"
        and row["seed"] == 0
        and row["condition"] == "correct_ordered"
    )
    target["prior_k1_dataset_sha256"] = "different-k1-dataset"

    gate = adjudicate_k1b(
        tasks=_tasks(),
        training_rows=training_rows,
        control_rows=mismatched,
        k1_gate=_k1_gate(),
        k1a_gate=_k1a_gate(),
    )

    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["validation_dataset_matches_k1"] is False


def test_k1b_runner_does_not_create_output_when_not_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "run"
    monkeypatch.setattr(
        "blockcipher_nd.cli.run_uknit_family_ctspn_k1b.build_k1b_readiness",
        lambda **_kwargs: (
            [],
            {
                "optimizer_step_authorized": False,
                "status": "fail",
                "decision": "not_authorized",
            },
        ),
    )

    exit_code = run_k1b(
        [
            "--plan",
            str(PLAN),
            "--k1-gate",
            str(source),
            "--k1a-gate",
            str(source),
            "--k1-controls",
            str(source),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 4
    assert not output_root.exists()


def test_k1b_chinese_plot_explains_endpoint_candidate_and_anchor(
    tmp_path: Path,
) -> None:
    training_rows, control_rows = _passing_rows()
    gate = adjudicate_k1b(
        tasks=_tasks(),
        training_rows=training_rows,
        control_rows=control_rows,
        k1_gate=_k1_gate(),
        k1a_gate=_k1a_gate(),
    )
    output = tmp_path / "curves.svg"

    render_ctspn_k1b_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "保留原生端点身份能否恢复 uKNIT 类 SPN 结构归因" in svg
    assert "K1-B 原生端点" in svg
    assert "最强旧锚点" in svg
    assert "重复末层" in svg
    assert "错误拓扑" in svg


def _passing_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    training_rows = []
    control_rows = []
    for task in _tasks():
        cipher = str(task["cipher_key"])
        seed = int(task["seed"])
        candidate_auc = 0.540 if cipher == "uknit64" else 0.960
        dataset_sha = f"dataset-{cipher}-{seed}"
        state_sha = f"state-{cipher}-{seed}"
        training_rows.append(
            {
                "cipher_key": cipher,
                "seed": seed,
                "model": CANDIDATE_MODEL,
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "trainable_parameter_count": 439982,
                "training": {
                    "batch_size": 64,
                    "epochs": 10,
                    "checkpoint_metric": "val_auc",
                    "selected_checkpoint": "best",
                    "train_rows": 4096,
                    "validation_rows": 2048,
                },
            }
        )
        for condition in CONTROL_CONDITIONS:
            auc = candidate_auc if condition == "correct_ordered" else candidate_auc - 0.02
            control_rows.append(
                {
                    "cipher_key": cipher,
                    "seed": seed,
                    "condition": condition,
                    "auc": auc,
                    "correct_minus_condition_auc": 0.0
                    if condition == "correct_ordered"
                    else 0.02,
                    "max_abs_probability_delta_from_correct": 0.0
                    if condition == "correct_ordered"
                    else 0.1,
                    "mean_abs_probability_delta_from_correct": 0.0
                    if condition == "correct_ordered"
                    else 0.01,
                    "dataset_sha256": dataset_sha,
                    "prior_k1_dataset_sha256": dataset_sha,
                    "state_dict_sha256": state_sha,
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return training_rows, control_rows
