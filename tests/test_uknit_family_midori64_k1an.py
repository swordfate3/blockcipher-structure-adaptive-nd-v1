from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_midori64_k1an import render_k1an_svg
from blockcipher_nd.cli.run_uknit_family_midori64_k1an import (
    build_initialization_manifest,
)
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.position_histogram_residual import (
    canonical_walsh_fingerprint,
    canonical_walsh_mask_pairs,
    deterministic_sbox_transition_walsh_features,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_canonical_walsh_k1an import (
    CONTROL_CONDITIONS,
    CONTROL_MODELS,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    adjudicate_k1an,
    build_control_checks,
)


OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/midori64.json",
    "runtime_round_start": 0,
    "runtime_rounds": 2,
    "cipher_round_window_start": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "residual_gate_initial_effective": 0.05,
    "transition_gate_initial_effective": 0.05,
    "canonical_walsh_features": 64,
    "active_cell": 8,
    "active_bit_role": 1,
    "input_difference_hex": "0x0000000400000000",
    "topology_corruption_seed": 20260729,
}


def build(condition: str) -> torch.nn.Module:
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=512,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options=OPTIONS,
    )


def test_k1an_canonical_walsh_basis_is_frozen_and_stable() -> None:
    pairs = canonical_walsh_mask_pairs(64)

    assert len(pairs) == 64
    assert len(set(pairs)) == 64
    assert (0, 0) not in pairs
    assert pairs == canonical_walsh_mask_pairs(64)
    assert canonical_walsh_fingerprint(64) == (
        "7c9a7cfaefa74af9974ce7bacd545598031ce14cfc175f2c27705cac0f8cb860"
    )


def test_k1an_walsh_features_are_deterministic_bounded_and_sbox_sensitive() -> None:
    correct = build("correct_structure")
    wrong = build("wrong_sbox")
    fixture = torch.randint(0, 2, (7, 4, 2, 64), dtype=torch.float32)

    first = deterministic_sbox_transition_walsh_features(
        fixture,
        correct.runtime_structure,
    )
    replay = deterministic_sbox_transition_walsh_features(
        fixture,
        correct.runtime_structure,
    )
    changed = deterministic_sbox_transition_walsh_features(
        fixture,
        wrong.runtime_structure,
    )

    assert first.shape == (7, 2, 16, 64)
    assert torch.equal(first, replay)
    assert torch.isfinite(first).all()
    assert torch.all(first >= -1.0)
    assert torch.all(first <= 1.0)
    assert not torch.equal(first, changed)


def test_k1an_controls_share_geometry_initialization_and_fixed_basis() -> None:
    models: dict[str, torch.nn.Module] = {}
    for condition in CONTROL_CONDITIONS:
        torch.manual_seed(6)
        models[condition] = build(condition)

    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models.values()
    }
    counts = {
        model_metadata(model)["trainable_parameter_count"] for model in models.values()
    }
    initial_hashes = {
        tensor_mapping_sha256(model.state_dict()) for model in models.values()
    }
    correct = models["correct_structure"]
    wrong = models["wrong_sbox"]
    branch_off = models["transition_branch_off"]

    assert geometries and len(geometries) == 1
    assert counts == {EXPECTED_PARAMETER_COUNT}
    assert len(initial_hashes) == 1
    assert all(
        not any(
            token in name
            for token in ("transition_encoder", "transition_projection", "walsh")
        )
        for model in models.values()
        for name, _ in model.named_parameters()
    )
    assert correct.canonical_walsh_fingerprint == wrong.canonical_walsh_fingerprint
    assert (
        correct.sbox_transition_semantics_sha256
        != wrong.sbox_transition_semantics_sha256
    )
    assert (
        correct.sbox_transition_semantics_sha256
        == branch_off.sbox_transition_semantics_sha256
    )
    assert correct.transition_branch_enabled is True
    assert wrong.transition_branch_enabled is True
    assert branch_off.transition_branch_enabled is False


def test_k1an_control_preflight_passes_exact_six_task_matrix() -> None:
    checks = build_control_checks(synthetic_tasks())

    assert checks
    assert all(checks.values()), checks


def test_k1an_forward_uses_fixed_input_geometry() -> None:
    for condition in CONTROL_CONDITIONS:
        model = build(condition)
        output = model(torch.randint(0, 2, (5, 512), dtype=torch.float32))

        assert output.shape == (5, 1)
        assert torch.isfinite(output).all()
        assert model.canonical_walsh_features_per_stage == 64


def test_k1an_scratch_manifest_records_each_model_initialization() -> None:
    manifest = build_initialization_manifest()

    assert manifest["version"] == 1
    assert set(manifest["targets"]) == set(CONTROL_MODELS.values())
    assert all(
        entry == {"kind": "scratch", "target_mapping": "aligned"}
        for entry in manifest["targets"].values()
    )


def test_k1an_gate_passes_only_when_all_fresh_semantic_margins_hold() -> None:
    gate = synthetic_gate()

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("canonical_walsh_transition_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert gate["remote_scale"] == "no"


def test_k1an_gate_routes_wrong_sbox_substitution_to_shared_weight() -> None:
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
    assert gate["decision"].endswith("independent_wrong_sbox_substitute_unresolved")
    assert gate["research_checks"]["seed6_same_key_fresh_beats_wrong_sbox"] is False
    assert "shared-weight" in gate["next_action"]


def test_k1an_gate_rejects_unused_canonical_branch() -> None:
    rows = synthetic_evaluation_rows()
    for row in rows:
        if (
            row["condition"] == "transition_branch_off"
            and row["split"] == "same_key_fresh"
        ):
            row["auc"] = 0.649
    gate = synthetic_gate(evaluation_rows=rows)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("canonical_transition_signal_not_supported")


def test_k1an_plot_explains_anchor_loss_and_semantic_controls(tmp_path: Path) -> None:
    gate = synthetic_gate()
    output = tmp_path / "curves.svg"

    report = render_k1an_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["heatmaps_used_instead_of_overlapping_curves"] is True
    assert report["threshold_outcomes_visible"] is True
    assert "固定 Walsh 表示能否识别正确 S盒" in svg
    assert "正确模型 - 错误 S盒" in svg
    assert "正确模型 - 原 K1-AK" in svg


def synthetic_gate(
    *,
    evaluation_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows = synthetic_evaluation_rows() if evaluation_rows is None else evaluation_rows
    return adjudicate_k1an(
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
            "model_options": dict(OPTIONS),
        }
        for seed in EXPECTED_SEEDS
        for condition in CONTROL_CONDITIONS
    ]


def synthetic_training_rows() -> list[dict[str, object]]:
    aucs = {
        "correct_structure": 0.65,
        "wrong_sbox": 0.63,
        "transition_branch_off": 0.58,
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
            "initialization": {
                "kind": "scratch",
                "strict_state_dict_load": False,
                "initial_state_sha256": f"initial-{seed}",
            },
            "metrics": {"auc": aucs[condition]},
            "history": [{"epoch": epoch} for epoch in range(1, 11)],
            "training": {
                "batch_size": 64,
                "epochs": 10,
                "epochs_ran": 10,
                "optimizer_state_step_after": EXPECTED_OPTIMIZER_STEPS,
                "optimizer": "adam",
                "loss": "mse",
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
        "transition_branch_off": 0.58,
    }
    return [
        {
            "seed": seed,
            "split": split,
            "condition": condition,
            "auc": aucs[condition],
            "rows": 4096 if split == "train_seen" else 2048,
            "dataset_sha256": f"dataset-{seed}-{split}",
            "composition_sha256": (
                "wrong-composition"
                if condition == "wrong_sbox"
                else "correct-composition"
            ),
            "sbox_transition_semantics_sha256": (
                "wrong-semantic" if condition == "wrong_sbox" else "correct-semantic"
            ),
            "canonical_walsh_fingerprint": "canonical-basis",
            "transition_branch_enabled": condition != "transition_branch_off",
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
