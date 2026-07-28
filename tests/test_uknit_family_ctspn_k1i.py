from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.gf2_boolean_view import (
    VIEW_NAMES,
    apply_gf2_operator,
    gf2_boolean_views,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    ANCHOR_CONDITION,
    CONTROL_CONDITIONS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1i import (
    CANDIDATE_MODEL,
    EXPECTED_PARAMETER_COUNT,
    READINESS_RUN_ID,
    RUN_ID,
    adjudicate_k1i,
    build_k1i_control,
    build_k1i_readiness,
    candidate_task_map,
    cell_relabel_logit_delta,
    composition_order_exact,
    expected_task_keys,
    input_geometry,
    project_features,
    scalar_gf2_operator,
    scalar_vectorized_exact,
    structural_readiness,
    transformed_difference_consistent,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_gf2_boolean_view_k1i_2048_seed0_seed1.csv"
)


def tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def test_k1i_real_plan_passes_zero_training_structural_readiness() -> None:
    manifests, gate = build_k1i_readiness(
        tasks=tasks(),
        k1h_gate=k1h_gate(),
        source_checks=source_checks(),
    )

    assert len(manifests) == 4
    assert gate["status"] == "pass"
    assert gate["optimizer_step_authorized"] is True
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert all(gate["protocol_checks"].values())
    assert all(gate["evidence_checks"].values())
    assert {row["trainable_parameter_count"] for row in manifests} == {
        EXPECTED_PARAMETER_COUNT
    }
    json.dumps(gate)


def test_k1i_vectorized_gf2_matches_hand_xor_and_cancellation() -> None:
    operator = torch.tensor(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=torch.uint8,
    )
    values = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )

    observed = apply_gf2_operator(values, operator)
    expected = scalar_gf2_operator(values, operator)

    assert torch.equal(observed, expected)
    assert observed[0].tolist() == [0.0, 0.0, 0.0]
    assert observed[1].tolist() == [1.0, 1.0, 0.0]


def test_k1i_views_freeze_composition_order_and_difference_identity() -> None:
    task = candidate_task_map(tasks())[("uknit64", 0)]
    model = build_k1i_control(
        task=task,
        condition="exact_ordered",
        input_bits=512,
    )
    features = torch.randint(
        0,
        2,
        (4, 512),
        generator=torch.Generator().manual_seed(20260728),
    ).float()
    runtime = project_features(features, model.runtime_structure)
    views = gf2_boolean_views(runtime, model.runtime_structure)

    assert views.shape == (4, 4, 64, 12)
    assert tuple(model.boolean_view_names) == VIEW_NAMES
    assert scalar_vectorized_exact(model.runtime_structure)
    assert transformed_difference_consistent(views)
    assert composition_order_exact(runtime, model.runtime_structure, views)
    assert not torch.equal(views[..., 9:12], views[..., 3:6])
    assert not torch.equal(views[..., 9:12], views[..., 6:9])
    assert cell_relabel_logit_delta(model, features) <= 1e-6


def test_k1i_controls_strict_load_same_cross_width_geometry() -> None:
    task_map = candidate_task_map(tasks())
    models = {
        cipher: build_k1i_control(
            task=task_map[(cipher, 0)],
            condition="exact_ordered",
            input_bits=input_geometry(cipher)[0],
        )
        for cipher in ("uknit64", "dialga128")
    }
    state = models["uknit64"].state_dict()

    assert model_metadata(models["uknit64"])["trainable_parameter_count"] == 94754
    assert model_metadata(models["dialga128"])["trainable_parameter_count"] == 94754
    assert [(name, tuple(value.shape)) for name, value in state.items()] == [
        (name, tuple(value.shape))
        for name, value in models["dialga128"].state_dict().items()
    ]
    models["dialga128"].load_state_dict(state, strict=True)
    assert not any("operator" in name or "matrix" in name for name in state)


def test_k1i_structural_readiness_proves_both_operators_and_controls() -> None:
    _, checks, metrics = structural_readiness(candidate_task_map(tasks()))

    assert all(checks.values())
    for cipher in ("uknit64", "dialga128"):
        assert all(
            value > 1e-7
            for value in metrics[cipher]["individual_operator_logit_deltas"]
        )
        controls = metrics[cipher]["controls"]
        assert set(controls) == set(CONTROL_CONDITIONS[1:])
        assert all(row["logit_max_abs_delta"] > 1e-7 for row in controls.values())


def test_k1i_gate_requires_every_seed_split_anchor_and_control() -> None:
    training_rows, evaluation_rows = passing_evidence()
    gate = adjudicate_k1i(
        tasks=tasks(),
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        readiness_gate=readiness_gate(),
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = [dict(row) for row in evaluation_rows]
    target = next(
        row
        for row in failed
        if row["cipher_key"] == "uknit64"
        and row["seed"] == 1
        and row["split"] == "same_key_fresh"
        and row["condition"] == "operator_corrupted"
    )
    target["auc"] = 0.54
    held = adjudicate_k1i(
        tasks=tasks(),
        training_rows=training_rows,
        evaluation_rows=failed,
        readiness_gate=readiness_gate(),
    )

    assert held["status"] == "hold"
    assert (
        held["research_checks"]["uknit64_seed1_same_key_fresh_beats_controls"] is False
    )


def test_k1i_readiness_fails_closed_without_cache_and_anchor_sources() -> None:
    manifests, gate = build_k1i_readiness(
        tasks=tasks(),
        k1h_gate=k1h_gate(),
        source_checks={"twelve_caches_reused": False},
    )

    assert manifests == []
    assert gate["status"] == "fail"
    assert gate["optimizer_step_authorized"] is False
    assert gate["protocol_checks"]["twelve_caches_reused"] is False


def k1h_gate() -> dict[str, object]:
    return {
        "run_id": (
            "i1_uknit_family_ctspn_operator_tied_latent_k1h_2048_seed0_seed1_20260728"
        ),
        "status": "hold",
        "decision": "innovation1_uknit_family_ctspn_k1h_operator_tied_latent_not_supported",
        "protocol_checks": {"source": True},
    }


def source_checks() -> dict[str, bool]:
    return {
        "twelve_caches_reused_by_digest": True,
        "four_runtime_e4_anchors_bound": True,
        "candidate_anchor_protocol_aligned": True,
    }


def readiness_gate() -> dict[str, object]:
    return {
        "run_id": READINESS_RUN_ID,
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"source": True},
        "evidence_checks": {"structure": True},
    }


def passing_evidence() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    training_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    for cipher, seed in sorted(expected_task_keys()):
        training_rows.append(
            {
                "model": CANDIDATE_MODEL,
                "cipher_key": cipher,
                "seed": seed,
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                "training": {
                    "batch_size": 64,
                    "epochs": 10,
                    "checkpoint_metric": "val_auc",
                    "selected_checkpoint": "best",
                },
            }
        )
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            candidate_auc = 0.54 if cipher == "uknit64" else 0.96
            anchor_auc = 0.53 if cipher == "uknit64" else 0.957
            rows = 4096 if split == "train_seen" else 2048
            dataset = f"dataset-{cipher}-{seed}-{split}"
            state = f"state-{cipher}-{seed}"
            for condition in (*CONTROL_CONDITIONS, ANCHOR_CONDITION):
                if condition == "exact_ordered":
                    auc = candidate_auc
                    role = "candidate"
                elif condition == ANCHOR_CONDITION:
                    auc = anchor_auc
                    role = "anchor"
                else:
                    auc = candidate_auc - 0.01
                    role = "candidate"
                evaluation_rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "source_role": role,
                        "condition": condition,
                        "rows": rows,
                        "auc": auc,
                        "dataset_sha256": dataset,
                        "checkpoint_sha256": f"checkpoint-{role}-{cipher}-{seed}",
                        "state_dict_sha256": (
                            state if role == "candidate" else f"anchor-{cipher}-{seed}"
                        ),
                        "operator_routing_sha256": (
                            None
                            if role == "anchor"
                            else f"operator-{cipher}-{condition}"
                        ),
                        "boolean_view_sha256": (
                            None if role == "anchor" else f"view-{cipher}-{condition}"
                        ),
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return training_rows, evaluation_rows
