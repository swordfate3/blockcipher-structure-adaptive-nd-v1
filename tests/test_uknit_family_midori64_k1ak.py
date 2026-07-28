from __future__ import annotations

import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.position_histogram_residual import (
    deterministic_sbox_transition_histogram,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_sbox_transition_k1ak import (
    CONTROL_CONDITIONS,
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    adjudicate_k1ak,
)


OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/midori64.json",
    "runtime_round_start": 0,
    "runtime_rounds": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "residual_gate_initial_effective": 0.05,
    "transition_gate_initial_effective": 0.05,
    "transition_value_dim": 20,
    "virtual_projection_slots": 16,
    "topology_corruption_seed": 20260729,
}


def build(condition: str) -> torch.nn.Module:
    return build_model(
        f"runtime_spn_ct_k1ak_sbox_transition_{condition}",
        input_bits=512,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options=OPTIONS,
    )


def test_k1ak_transition_histogram_is_normalized_and_sbox_sensitive() -> None:
    correct = build("true")
    wrong = build("wrong_sbox")
    fixture = torch.randint(0, 2, (7, 4, 2, 64), dtype=torch.float32)

    exact = deterministic_sbox_transition_histogram(
        fixture,
        correct.runtime_structure,
    )
    changed = deterministic_sbox_transition_histogram(
        fixture,
        wrong.runtime_structure,
    )

    assert exact.shape == (7, 2, 16, 256)
    assert torch.allclose(exact.sum(dim=-1), torch.ones(7, 2, 16))
    assert not torch.equal(exact, changed)


def test_k1ak_transition_embedding_is_invariant_to_consistent_cell_relabel() -> None:
    model = build("true")
    fixture = torch.randint(0, 2, (5, 4, 2, 64), dtype=torch.float32)
    permutation = tuple(reversed(range(16)))
    relabeled_structure, bit_permutation = model.runtime_structure.relabel_cells(
        permutation
    )
    relabeled_fixture = torch.empty_like(fixture)
    relabeled_fixture[..., bit_permutation] = fixture

    model.eval()
    with torch.no_grad():
        original = model.backbone.transition_embedding(
            fixture,
            model.runtime_structure,
        )
        relabeled = model.backbone.transition_embedding(
            relabeled_fixture,
            relabeled_structure,
        )

    assert torch.allclose(original, relabeled, atol=1e-6, rtol=0.0)


def test_k1ak_controls_share_geometry_and_only_runtime_semantics_change() -> None:
    models = [
        build(name) for name in ("true", "wrong_sbox", "corrupted_linear", "none")
    ]
    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models
    }
    counts = {model_metadata(model)["trainable_parameter_count"] for model in models}

    assert len(geometries) == 1
    assert len(counts) == 1
    assert max(counts) <= int(214_316 * 1.025)
    assert all(model.uses_cipher_identity is False for model in models)
    assert all(model.uses_absolute_cell_or_bit_identity is False for model in models)
    assert all(model.uses_runtime_native_cell_slots is False for model in models)
    assert models[-1].apply_sboxes is False


def test_k1ak_forward_uses_fixed_input_geometry() -> None:
    model = build("true")
    output = model(torch.randint(0, 2, (5, 512), dtype=torch.float32))

    assert output.shape == (5, 1)
    assert torch.isfinite(output).all()
    assert model.virtual_projection_effective_weight_shape == (128, 40)


def test_k1ak_gate_passes_only_when_all_fresh_semantic_margins_hold() -> None:
    gate = synthetic_gate()

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("sbox_transition_residual_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert gate["remote_scale"] == "no"


def test_k1ak_gate_isolates_remaining_sbox_failure() -> None:
    rows = synthetic_evaluation_rows()
    for row in rows:
        if (
            row["seed"] == 6
            and row["split"] == "same_key_fresh"
            and row["condition"] == "wrong_sbox"
        ):
            row["auc"] = 0.648
    gate = synthetic_gate(evaluation_rows=rows)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("sbox_transition_discrimination_failed")
    assert gate["research_checks"]["seed6_same_key_fresh_beats_wrong_sbox"] is False
    assert all(
        passed
        for name, passed in gate["research_checks"].items()
        if "beats_corrupted_linear" in name or "beats_no_structure" in name
    )


def synthetic_gate(
    *,
    evaluation_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows = synthetic_evaluation_rows() if evaluation_rows is None else evaluation_rows
    return adjudicate_k1ak(
        tasks=synthetic_tasks(),
        training_rows=synthetic_training_rows(),
        evaluation_rows=rows,
        checkpoint_manifest={
            "entries": [
                {"seed": seed, "condition": condition}
                for seed in EXPECTED_SEEDS
                for condition in CONTROL_CONDITIONS
            ]
        },
        anchor_rows=[
            {
                "seed": seed,
                "split": split,
                "condition": "correct_structure",
                "auc": 0.64,
            }
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ],
        source_checks={"source": True},
        control_checks={"controls": True},
        cache_checks={"cache": True},
    )


def synthetic_tasks() -> list[dict[str, object]]:
    options = {
        **OPTIONS,
        "cipher_round_window_start": 2,
        "active_cell": 8,
        "active_bit_role": 1,
        "input_difference_hex": "0x0000000400000000",
    }
    return [
        {
            "cipher_key": "midori64",
            "rounds": 4,
            "seed": seed,
            "model_key": CONTROL_MODELS[condition],
            "samples_per_class": 2048,
            "validation_samples_total": None,
            "pairs_per_sample": 4,
            "input_difference": 0x0000000400000000,
            "difference_profile": "midori64_k1ah_cell8_r4",
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "key_rotation_interval": 0,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "lr_scheduler": "none",
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "target_epochs": 10,
            "model_options": options,
        }
        for seed in EXPECTED_SEEDS
        for condition in CONTROL_CONDITIONS
    ]


def synthetic_training_rows() -> list[dict[str, object]]:
    aucs = {
        "correct_structure": 0.65,
        "wrong_sbox": 0.63,
        "corrupted_linear": 0.58,
        "no_structure": 0.50,
    }
    return [
        {
            "model": CONTROL_MODELS[condition],
            "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
            "rounds": 4,
            "seed": seed,
            "input_difference": 0x0000000400000000,
            "difference_profile": "midori64_k1ah_cell8_r4",
            "samples_per_class": 2048,
            "pairs_per_sample": 4,
            "negative_mode": "encrypted_random_plaintexts",
            "metrics": {"auc": aucs[condition]},
            "training": {
                "batch_size": 64,
                "epochs": 10,
                "epochs_ran": 10,
                "checkpoint_metric": "val_auc",
                "selected_checkpoint": "best",
                "samples_total": 4096,
            },
            "validation": {"samples_total": 2048},
        }
        for seed in EXPECTED_SEEDS
        for condition in CONTROL_CONDITIONS
    ]


def synthetic_evaluation_rows() -> list[dict[str, object]]:
    aucs = {
        "correct_structure": 0.65,
        "wrong_sbox": 0.63,
        "corrupted_linear": 0.58,
        "no_structure": 0.50,
    }
    return [
        {
            "seed": seed,
            "split": split,
            "condition": condition,
            "auc": aucs[condition],
            "rows": 4096 if split == "train_seen" else 2048,
            "dataset_sha256": f"dataset-{seed}-{split}",
            "sbox_transition_semantics_sha256": f"semantic-{condition}",
            "residual_gate": 0.05,
            "transition_gate": 0.05,
            "training_performed": False,
            "optimizer_steps": 0,
            "strict_state_dict_load": True,
        }
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in CONTROL_CONDITIONS
    ]
