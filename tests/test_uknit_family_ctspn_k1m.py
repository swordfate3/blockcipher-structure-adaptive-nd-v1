from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    expected_task_keys,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import (
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    build_k1k_control,
    candidate_task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1m import (
    ANCHOR_CONDITION,
    EXPECTED_EVALUATION_ROWS,
    INITIAL_EFFECTIVE_GATE,
    READINESS_RUN_ID,
    adjudicate_k1m,
    build_k1m_readiness,
    candidate_protocol_frozen,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_gate_opening_k1m_2048_seed0_seed1.csv"
)


def tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def test_k1m_changes_only_initial_effective_gate_and_preserves_geometry() -> None:
    mapped = candidate_task_map(tasks())
    task = mapped[("uknit64", 0)]
    candidate = build_k1k_control(
        task=task,
        condition="exact_ordered",
        input_bits=512,
    )
    default_task = dict(task)
    default_options = dict(task["model_options"])
    default_options.pop("residual_gate_initial_effective")
    default_task["model_options"] = default_options
    default = build_k1k_control(
        task=default_task,
        condition="exact_ordered",
        input_bits=512,
    )

    assert abs(
        float(torch.tanh(candidate.backbone.residual_gate.detach()))
        - INITIAL_EFFECTIVE_GATE
    ) <= 1e-7
    assert float(default.backbone.residual_gate.detach()) == 0.0
    assert [
        (name, tuple(value.shape)) for name, value in candidate.state_dict().items()
    ] == [(name, tuple(value.shape)) for name, value in default.state_dict().items()]
    assert candidate_protocol_frozen(mapped)


def test_k1m_real_plan_passes_zero_training_readiness() -> None:
    manifests, gate = build_k1m_readiness(
        tasks=tasks(),
        datasets=synthetic_datasets(),
        source_checks={"source_binding": True},
    )

    assert len(manifests) == 4
    assert gate["status"] == "pass"
    assert gate["optimizer_step_authorized"] is True
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert all(gate["protocol_checks"].values())
    assert all(gate["evidence_checks"].values())
    assert all(
        abs(row["initial_effective_gate"] - INITIAL_EFFECTIVE_GATE) <= 1e-7
        for row in manifests
    )


def test_k1m_gate_requires_every_fresh_anchor_control_and_active_gate() -> None:
    training_rows = synthetic_training_rows()
    evaluation_rows = synthetic_evaluation_rows()
    gate = adjudicate_k1m(
        tasks=tasks(),
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        readiness_gate=readiness_gate(),
    )

    assert len(evaluation_rows) == EXPECTED_EVALUATION_ROWS
    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = [dict(row) for row in evaluation_rows]
    for row in failed:
        if (
            row["cipher_key"] == "uknit64"
            and row["seed"] == 1
            and row["split"] == "same_key_fresh"
            and row["condition"] == "exact_ordered"
        ):
            row["auc"] = 0.505
    held = adjudicate_k1m(
        tasks=tasks(),
        training_rows=training_rows,
        evaluation_rows=failed,
        readiness_gate=readiness_gate(),
    )

    assert held["status"] == "hold"
    assert held["decision"].endswith("gate_opened_uknit_signal_not_supported")


def synthetic_datasets() -> dict[tuple[str, int, str], DifferentialDataset]:
    datasets: dict[tuple[str, int, str], DifferentialDataset] = {}
    for cipher, seed in expected_task_keys():
        width = 512 if cipher == "uknit64" else 1024
        for split_index, split in enumerate(
            ("train_seen", "same_key_fresh", "cross_key_validation")
        ):
            generator = np.random.default_rng(20260728 + seed + split_index)
            datasets[(cipher, seed, split)] = DifferentialDataset(
                features=generator.integers(
                    0,
                    2,
                    size=(64, width),
                    dtype=np.uint8,
                ),
                labels=np.tile(np.array([0, 1], dtype=np.uint8), 32),
                metadata={},
            )
    return datasets


def synthetic_training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks():
        rows.append(
            {
                "cipher_key": task["cipher_key"],
                "seed": task["seed"],
                "model": CANDIDATE_MODEL,
                "trainable_parameter_count": 128707,
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "training": {
                    "batch_size": 64,
                    "epochs": 10,
                    "checkpoint_metric": "val_auc",
                    "selected_checkpoint": "best",
                },
            }
        )
    return rows


def synthetic_evaluation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cipher, seed in expected_task_keys():
        candidate_auc = 0.60 if cipher == "uknit64" else 0.96
        anchor_auc = candidate_auc - 0.01 if cipher == "uknit64" else 0.955
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            dataset_sha = f"dataset-{cipher}-{seed}-{split}"
            state_sha = f"candidate-state-{cipher}-{seed}"
            for index, condition in enumerate(CONTROL_CONDITIONS):
                auc = candidate_auc if condition == "exact_ordered" else candidate_auc - 0.01
                rows.append(
                    {
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "rows": 4096 if split == "train_seen" else 2048,
                        "auc": auc,
                        "effective_gate": 0.03,
                        "dataset_sha256": dataset_sha,
                        "state_dict_sha256": state_sha,
                        "topology_edge_sha256": f"edge-{condition}",
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
            rows.append(
                {
                    "cipher_key": cipher,
                    "seed": seed,
                    "split": split,
                    "condition": ANCHOR_CONDITION,
                    "rows": 4096 if split == "train_seen" else 2048,
                    "auc": anchor_auc,
                    "dataset_sha256": dataset_sha,
                    "state_dict_sha256": f"anchor-state-{cipher}-{seed}",
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return rows


def readiness_gate() -> dict[str, object]:
    return {
        "run_id": READINESS_RUN_ID,
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"protocol": True},
        "evidence_checks": {"evidence": True},
    }
