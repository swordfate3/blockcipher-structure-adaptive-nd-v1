from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import torch

from blockcipher_nd.cli.gate_runtime_spn_dialga_d6 import audit_d3_cache_reuse
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE5GatedResidualSpnDistinguisher,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    adjudicate_runtime_spn_dialga_d3,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d6 import (
    D6_MODELS,
    EXPECTED_PARAMETER_COUNT,
    adjudicate_runtime_spn_dialga_d6,
)


PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_dialga128_runtime_e5_d6_r5_2048_seed0_seed1.csv"
)
COMMON_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/dialga128.json",
    "runtime_round_start": 3,
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "edge_gate",
    "cell_input_mode": "state_triplet",
    "round_window_mode": "recurrent_window",
    "runtime_structure_window_control": "full",
}


def _runtime_input(rows: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260725)
    return torch.randint(
        0,
        2,
        (rows, 4, 2, 128),
        generator=generator,
        dtype=torch.float32,
    )


def _build(role: str = "correct"):
    options = deepcopy(COMMON_OPTIONS)
    if role == "corrupted":
        options["topology_corruption_seed"] = 20260725
    model = build_model(
        D6_MODELS[role],
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options=options,
    )
    assert model is not None
    return model


def test_d6_zero_gate_is_exact_independent_base_and_nonconstant() -> None:
    torch.manual_seed(101)
    model = _build("correct").eval()
    assert isinstance(model.backbone, RuntimeE5GatedResidualSpnDistinguisher)
    inputs = _runtime_input()

    with torch.no_grad():
        logits = model.backbone(
            inputs,
            model.runtime_structure,
            relation_mode="true",
            topology_gate_override=0.0,
        )
        base_logits = model.backbone.base_logits(inputs, model.runtime_structure)

    assert torch.equal(logits, base_logits)
    assert torch.isfinite(base_logits).all()
    assert float(torch.std(base_logits)) > 0.0


def test_d6_independent_residual_is_zero_and_ignores_corrupted_topology() -> None:
    torch.manual_seed(103)
    model = _build("correct").eval()
    correct = dialga128_runtime_structure(2, round_start=3)
    corrupted = correct.corrupted(20260725)
    inputs = _runtime_input()

    with torch.no_grad():
        correct_base, correct_residual = model.backbone.encode_components(
            inputs, correct, relation_mode="independent"
        )
        corrupt_base, corrupt_residual = model.backbone.encode_components(
            inputs, corrupted, relation_mode="independent"
        )
        _, true_residual = model.backbone.encode_components(
            inputs, correct, relation_mode="true"
        )
        _, corrupted_true_residual = model.backbone.encode_components(
            inputs, corrupted, relation_mode="true"
        )
        correct_logits = model.backbone(
            inputs,
            correct,
            relation_mode="independent",
            topology_gate_override=0.9,
        )
        corrupt_logits = model.backbone(
            inputs,
            corrupted,
            relation_mode="independent",
            topology_gate_override=-0.9,
        )
        base_logits = model.backbone.base_logits(inputs, correct)

    assert torch.equal(correct_base, corrupt_base)
    assert torch.count_nonzero(correct_residual) == 0
    assert torch.count_nonzero(corrupt_residual) == 0
    assert torch.equal(correct_logits, base_logits)
    assert torch.equal(corrupt_logits, base_logits)
    assert float(torch.max(torch.abs(true_residual - corrupted_true_residual))) > 1e-7


def test_d6_cell_relabeling_invariance_and_finite_gradients() -> None:
    torch.manual_seed(107)
    model = _build("correct")
    correct = dialga128_runtime_structure(2, round_start=3)
    relabeled, bit_permutation = correct.relabel_cells(
        tuple(reversed(range(correct.cells)))
    )
    inputs = _runtime_input()
    relabeled_inputs = torch.empty_like(inputs)
    relabeled_inputs[..., bit_permutation] = inputs

    model.eval()
    with torch.no_grad():
        expected = model.backbone(
            inputs,
            correct,
            relation_mode="true",
            topology_gate_override=0.4,
        )
        actual = model.backbone(
            relabeled_inputs,
            relabeled,
            relation_mode="true",
            topology_gate_override=0.4,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    model.train()
    logits = model.backbone(inputs, correct, relation_mode="true")
    logits.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    assert any(
        gradient is not None and bool(torch.any(gradient != 0))
        for gradient in gradients
    )


def test_d6_registry_roles_have_identical_geometry_and_metadata() -> None:
    for role in D6_MODELS:
        model = _build(role)
        metadata = model_metadata(model)
        assert metadata["parameter_count"] == EXPECTED_PARAMETER_COUNT
        assert metadata["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
        assert metadata["topology_gate_initial"] == 0.0
        assert metadata["topology_gate_final_raw"] == 0.0
        assert metadata["topology_gate_final_bounded"] == 0.0
        assert metadata["topology_residual_mode"] == (
            "independent_base_plus_bounded_topology_logit_residual"
        )


def test_d6_no_topology_training_cannot_move_the_topology_gate() -> None:
    torch.manual_seed(109)
    model = _build("no_topology").train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.randint(0, 2, (4, 1024), dtype=torch.float32)

    loss = model(features).square().mean()
    loss.backward()
    optimizer.step()

    assert model.backbone.topology_gate.item() == 0.0


def test_real_d6_plan_parses_as_six_frozen_rows() -> None:
    tasks = tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )

    assert len(tasks) == 6
    assert {(task["seed"], task["model_key"]) for task in tasks} == {
        (seed, model) for seed in (0, 1) for model in D6_MODELS.values()
    }
    assert all(
        task["cipher_key"] == "dialga128"
        and task["rounds"] == 5
        and task["samples_per_class"] == 2048
        and task["pairs_per_sample"] == 4
        and task["target_epochs"] == 10
        and task["input_difference"] == 0x40
        and task["negative_mode"] == "encrypted_random_plaintexts"
        for task in tasks
    )


def _result_row(
    *,
    seed: int,
    role: str,
    model: str,
    auc: float,
    cache_root: Path,
    e5: bool,
) -> dict[str, object]:
    options = deepcopy(COMMON_OPTIONS)
    if role == "corrupted":
        options["topology_corruption_seed"] = 20260725
    final_gate = 0.0 if role == "no_topology" else 0.08 + 0.01 * seed
    row: dict[str, object] = {
        "cipher": "Dialga-128",
        "cipher_key": "dialga128",
        "model": model,
        "rounds": 5,
        "seed": seed,
        "samples_per_class": 2048,
        "dataset_label_mode": "balanced_per_class",
        "pairs_per_sample": 4,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "difference_profile": "",
        "difference_member": "",
        "input_difference": 0x40,
        "train_key": 0,
        "validation_key": int("11" * 32, 16),
        "parameter_count": EXPECTED_PARAMETER_COUNT if e5 else 442466,
        "trainable_parameter_count": EXPECTED_PARAMETER_COUNT if e5 else 442466,
        "runtime_structure_descriptor_name": (
            "Dialga-128 20-round heterogeneous runtime SPN structure"
        ),
        "runtime_structure_descriptor_path": (
            "/repo/configs/runtime/spn/dialga128.json"
        ),
        "runtime_structure_descriptor_sha256": "a" * 64,
        "runtime_structure_round_start": 3,
        "runtime_structure_available_rounds": 20,
        "runtime_structure_loaded_rounds": 2,
        "runtime_structure_unique_transition_count": 2,
        "runtime_structure_homogeneous": False,
        "runtime_structure_mode": {
            "correct": "true",
            "corrupted": "corrupted",
            "no_topology": "independent",
        }[role],
        "runtime_structure_window_control": "full",
        "runtime_structure_transition_sha256s": ["b" * 64, "c" * 64],
        "runtime_structure_window_sha256": (
            "d" * 64 if role == "corrupted" else "e" * 64
        ),
        "metrics": {"auc": auc},
        "history": [
            {"epoch": epoch + 1, "val_auc": auc - 0.009 + epoch * 0.001}
            for epoch in range(10)
        ],
        "training": {
            "epochs": 10,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "selected_checkpoint": "best",
            "train_rows": 4096,
            "validation_rows": 2048,
            "input_bits": 1024,
            "pair_bits": 256,
            "model_options": options,
            "train_dataset_storage": "disk",
            "validation_dataset_storage": "disk",
            "dataset_cache_root": str(cache_root),
        },
    }
    if e5:
        row.update(
            {
                "topology_residual_mode": (
                    "independent_base_plus_bounded_topology_logit_residual"
                ),
                "topology_gate_initial": 0.0,
                "topology_gate_final_raw": final_gate,
                "topology_gate_final_bounded": math.tanh(final_gate),
            }
        )
    return row


def _d3_source(cache_root: Path):
    models = {
        "correct": "runtime_spn_e4_equivariant_true",
        "corrupted": "runtime_spn_e4_equivariant_corrupted",
        "no_topology": "runtime_spn_e4_equivariant_independent",
    }
    aucs = {
        0: {"correct": 0.507, "corrupted": 0.504, "no_topology": 0.544},
        1: {"correct": 0.493, "corrupted": 0.528, "no_topology": 0.519},
    }
    rows = [
        _result_row(
            seed=seed,
            role=role,
            model=models[role],
            auc=aucs[seed][role],
            cache_root=cache_root,
            e5=False,
        )
        for seed in (0, 1)
        for role in models
    ]
    gate = adjudicate_runtime_spn_dialga_d3(
        run_id="i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725",
        rows=rows,
    )
    validation = {
        "run_id": gate["run_id"],
        "status": "pass",
        "checks": gate["protocol_checks"],
    }
    return rows, gate, validation


def _d6_rows(cache_root: Path, *, pass_gate: bool = True) -> list[dict[str, object]]:
    aucs = {
        0: {
            "correct": 0.56,
            "corrupted": 0.54,
            "no_topology": 0.53,
        },
        1: {
            "correct": 0.55 if pass_gate else 0.50,
            "corrupted": 0.53,
            "no_topology": 0.52,
        },
    }
    return [
        _result_row(
            seed=seed,
            role=role,
            model=model,
            auc=aucs[seed][role],
            cache_root=cache_root,
            e5=True,
        )
        for seed in (0, 1)
        for role, model in D6_MODELS.items()
    ]


def test_d6_gate_passes_only_with_two_seed_controls_and_d3_improvement(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    d3_rows, d3_gate, d3_validation = _d3_source(cache_root)
    gate = adjudicate_runtime_spn_dialga_d6(
        run_id="d6-pass",
        rows=_d6_rows(cache_root),
        d3_rows=d3_rows,
        persisted_d3_gate=d3_gate,
        replayed_d3_gate=deepcopy(d3_gate),
        d3_validation=d3_validation,
        expected_cache_root=cache_root,
        cache_audit={"status": "pass", "checks": {"reuse": True}},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_dialga_runtime_e5_d6_gated_residual_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_d6_gate_holds_failed_seed_and_fails_closed_on_cache_generation(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    d3_rows, d3_gate, d3_validation = _d3_source(cache_root)
    hold = adjudicate_runtime_spn_dialga_d6(
        run_id="d6-hold",
        rows=_d6_rows(cache_root, pass_gate=False),
        d3_rows=d3_rows,
        persisted_d3_gate=d3_gate,
        replayed_d3_gate=deepcopy(d3_gate),
        d3_validation=d3_validation,
        expected_cache_root=cache_root,
        cache_audit={"status": "pass", "checks": {"reuse": True}},
    )
    assert hold["status"] == "hold"

    invalid = adjudicate_runtime_spn_dialga_d6(
        run_id="d6-invalid",
        rows=_d6_rows(cache_root),
        d3_rows=d3_rows,
        persisted_d3_gate=d3_gate,
        replayed_d3_gate=deepcopy(d3_gate),
        d3_validation=d3_validation,
        expected_cache_root=cache_root,
        cache_audit={"status": "fail", "checks": {"reuse": False}},
    )
    assert invalid["status"] == "fail"
    assert invalid["protocol_checks"]["d3_cache_reused_without_generation"] is False


def test_d6_cache_audit_requires_all_six_rows_and_no_generation(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    leaves = {}
    for seed in (0, 1):
        for split in ("train", "validation"):
            leaf = cache_root / split / f"seed-{seed}"
            leaf.mkdir(parents=True)
            (leaf / "metadata.json").write_text("{}\n", encoding="utf-8")
            leaves[(seed, split)] = leaf
    progress = []
    for index in range(1, 7):
        seed = 0 if index <= 3 else 1
        for split in ("train", "validation"):
            progress.append(
                {
                    "stage": "dataset_cache",
                    "event": "cache_reuse",
                    "index": index,
                    "split": split,
                    "cache_path": str(leaves[(seed, split)]),
                }
            )

    audit = audit_d3_cache_reuse(
        progress_rows=progress,
        expected_cache_root=cache_root,
    )
    assert audit["status"] == "pass"
    invalid = audit_d3_cache_reuse(
        progress_rows=[*progress, {"stage": "dataset_cache", "event": "cache_start"}],
        expected_cache_root=cache_root,
    )
    assert invalid["status"] == "fail"
    assert invalid["checks"]["no_dataset_generation_events"] is False
