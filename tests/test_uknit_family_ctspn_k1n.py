from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    COMPOSITION_STAGE_NAMES,
    exact_operator_composition_views,
)
from blockcipher_nd.models.structure.spn.runtime_structure import apply_gf2
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    expected_task_keys,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    ANCHOR_CONDITION,
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    EXPECTED_EVALUATION_ROWS,
    READINESS_RUN_ID,
    adjudicate_k1n,
    build_k1n_readiness,
    candidate_protocol_frozen,
    candidate_task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_exact_operator_composition_k1n_2048_seed0_seed1.csv"
)
MODEL_KEYS = (
    "runtime_spn_ct_k1n_exact_composition_true",
    "runtime_spn_ct_k1n_exact_composition_wrong_sbox",
    "runtime_spn_ct_k1n_exact_composition_reversed_linear",
    "runtime_spn_ct_k1n_exact_composition_corrupted_linear",
    "runtime_spn_ct_k1n_exact_composition_no_sbox",
    "runtime_spn_ct_k1n_exact_composition_none",
)


def test_k1n_exact_views_follow_inverse_operator_order_and_round_trip() -> None:
    for cipher in ("uknit64", "dialga128"):
        model = build_control(cipher, MODEL_KEYS[0])
        structure = model.runtime_structure
        runtime = torch.randint(
            0,
            2,
            (3, 4, 2, structure.block_bits),
            generator=torch.Generator().manual_seed(20260728),
        ).float()

        views = exact_operator_composition_views(runtime, structure)

        assert COMPOSITION_STAGE_NAMES == (
            "ciphertext",
            "inverse_linear_1",
            "inverse_sbox_1",
            "inverse_linear_0",
            "inverse_sbox_0",
        )
        assert views.shape == (3, 4, structure.block_bits, 15)
        assert torch.all((views == 0) | (views == 1))
        stages = [views[..., offset : offset + 3] for offset in range(0, 15, 3)]
        assert torch.equal(
            stages[1],
            apply_gf2(structure.inverse_linear_matrices[1], stages[0].movedim(-1, -2)).movedim(-2, -1),
        )
        assert torch.equal(
            stages[3],
            apply_gf2(structure.inverse_linear_matrices[0], stages[2].movedim(-1, -2)).movedim(-2, -1),
        )
        assert_triplet_matches_inverse_sbox(stages[1], stages[2], structure, 1)
        assert_triplet_matches_inverse_sbox(stages[3], stages[4], structure, 0)

        recovered = stages[4][..., :2]
        for slot in (0, 1):
            left = structure.apply_sboxes(recovered[..., 0], slot)
            right = structure.apply_sboxes(recovered[..., 1], slot)
            recovered = torch.stack(
                (
                    apply_gf2(structure.linear_matrices[slot], left),
                    apply_gf2(structure.linear_matrices[slot], right),
                ),
                dim=-1,
            )
        assert torch.equal(recovered.movedim(-1, -2), runtime)


def test_k1n_controls_strict_load_same_geometry_and_change_declared_semantics() -> None:
    for cipher in ("uknit64", "dialga128"):
        models = [build_control(cipher, key) for key in MODEL_KEYS]
        candidate, wrong_sbox, reversed_linear, corrupted, no_sbox, none = models
        state = candidate.state_dict()
        geometry = [(name, tuple(value.shape)) for name, value in state.items()]

        assert sum(parameter.numel() for parameter in candidate.parameters()) == 131875
        assert len({model.composition_sha256 for model in models}) == len(models)
        for model in models[1:]:
            assert [(name, tuple(value.shape)) for name, value in model.state_dict().items()] == geometry
            model.load_state_dict(state, strict=True)

        assert torch.equal(
            wrong_sbox.runtime_structure.linear_matrices,
            candidate.runtime_structure.linear_matrices,
        )
        assert not torch.equal(
            wrong_sbox.runtime_structure.sbox_truth_bits,
            candidate.runtime_structure.sbox_truth_bits,
        )
        assert torch.equal(
            reversed_linear.runtime_structure.sbox_truth_bits,
            candidate.runtime_structure.sbox_truth_bits,
        )
        assert torch.equal(
            reversed_linear.runtime_structure.linear_matrices,
            candidate.runtime_structure.linear_matrices.flip(0),
        )
        assert torch.equal(
            corrupted.runtime_structure.sbox_truth_bits,
            candidate.runtime_structure.sbox_truth_bits,
        )
        assert not torch.equal(
            corrupted.runtime_structure.linear_matrices,
            candidate.runtime_structure.linear_matrices,
        )
        assert no_sbox.apply_sboxes is False
        assert torch.equal(
            no_sbox.runtime_structure.linear_matrices,
            candidate.runtime_structure.linear_matrices,
        )
        assert none.apply_sboxes is False
        identity = torch.eye(candidate.runtime_structure.block_bits, dtype=torch.uint8)
        assert all(
            torch.equal(matrix, identity)
            for matrix in none.runtime_structure.linear_matrices
        )


def test_k1n_real_plan_passes_zero_training_readiness() -> None:
    manifests, gate = build_k1n_readiness(
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
    assert candidate_protocol_frozen(candidate_task_map(tasks()))


def test_k1n_gate_requires_every_fresh_anchor_and_semantic_control() -> None:
    evaluation_rows = synthetic_evaluation_rows()
    gate = adjudicate_k1n(
        tasks=tasks(),
        training_rows=synthetic_training_rows(),
        evaluation_rows=evaluation_rows,
        readiness_gate=readiness_gate(),
    )

    assert len(evaluation_rows) == EXPECTED_EVALUATION_ROWS == 84
    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = [dict(row) for row in evaluation_rows]
    for row in failed:
        if (
            row["cipher_key"] == "uknit64"
            and row["seed"] == 1
            and row["split"] == "same_key_fresh"
            and row["condition"] == "wrong_sbox_semantics"
        ):
            row["auc"] = 0.599
    held = adjudicate_k1n(
        tasks=tasks(),
        training_rows=synthetic_training_rows(),
        evaluation_rows=failed,
        readiness_gate=readiness_gate(),
    )

    assert held["status"] == "hold"
    assert held["decision"].endswith("semantic_attribution_not_supported")


def assert_triplet_matches_inverse_sbox(
    before: torch.Tensor,
    after: torch.Tensor,
    structure: object,
    slot: int,
) -> None:
    left = structure.apply_inverse_sboxes(before[..., 0], slot)
    right = structure.apply_inverse_sboxes(before[..., 1], slot)
    assert torch.equal(after[..., 0], left)
    assert torch.equal(after[..., 1], right)
    assert torch.equal(after[..., 2], torch.remainder(left + right, 2.0))


def build_control(cipher: str, model_key: str) -> torch.nn.Module:
    if cipher == "uknit64":
        descriptor = ROOT / "configs/runtime/spn/uknit64.json"
        round_start = 3
        input_bits = 512
        pair_bits = 128
    else:
        descriptor = ROOT / "configs/runtime/spn/dialga128.json"
        round_start = 2
        input_bits = 1024
        pair_bits = 256
    return build_model(
        model_key,
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "runtime_structure_path": str(descriptor),
            "runtime_round_start": round_start,
            "runtime_rounds": 2,
            "pair_embedding_dim": 128,
            "dropout": 0.0,
            "residual_gate_initial_effective": 0.05,
        },
    )


def tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def synthetic_datasets() -> dict[tuple[str, int, str], DifferentialDataset]:
    datasets: dict[tuple[str, int, str], DifferentialDataset] = {}
    for cipher, seed in expected_task_keys():
        width = 512 if cipher == "uknit64" else 1024
        for split_index, split in enumerate(
            ("train_seen", "same_key_fresh", "cross_key_validation")
        ):
            generator = np.random.default_rng(20260728 + seed + split_index)
            datasets[(cipher, seed, split)] = DifferentialDataset(
                features=generator.integers(0, 2, size=(64, width), dtype=np.uint8),
                labels=np.tile(np.array([0, 1], dtype=np.uint8), 32),
                metadata={},
            )
    return datasets


def synthetic_training_rows() -> list[dict[str, object]]:
    return [
        {
            "cipher_key": task["cipher_key"],
            "seed": task["seed"],
            "model": CANDIDATE_MODEL,
            "trainable_parameter_count": 131875,
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
        for task in tasks()
    ]


def synthetic_evaluation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cipher, seed in expected_task_keys():
        candidate_auc = 0.60 if cipher == "uknit64" else 0.96
        anchor_auc = candidate_auc - 0.01 if cipher == "uknit64" else 0.955
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            dataset_sha = f"dataset-{cipher}-{seed}-{split}"
            state_sha = f"candidate-state-{cipher}-{seed}"
            for index, condition in enumerate(CONTROL_CONDITIONS):
                auc = candidate_auc if index == 0 else candidate_auc - 0.01
                rows.append(
                    {
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "rows": 4096 if split == "train_seen" else 2048,
                        "auc": auc,
                        "effective_gate": 0.05,
                        "dataset_sha256": dataset_sha,
                        "state_dict_sha256": state_sha,
                        "composition_sha256": f"composition-{condition}",
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
        "protocol_checks": {"ready": True},
        "evidence_checks": {"ready": True},
    }
