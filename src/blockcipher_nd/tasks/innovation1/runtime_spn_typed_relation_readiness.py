from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_readiness import (
    FIVE_CIPHER_PROTOCOLS,
    READINESS_SEED,
    _equal_task_contributions,
    _load_structures,
    _make_smoke_tasks,
    _official_runtime_equivalence,
    _smoke_complete,
    _training_finite,
    _training_result_payload,
)
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTrainingResult,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig


TYPED_ROLE_MODES = {
    "dense": "dense",
    "correct": "correct",
    "uniform": "agnostic",
    "shuffled": "shuffled",
}


def build_typed_relation_readiness(
    *,
    run_id: str,
    cache_root: Path,
    regression_tests_passed: bool,
    regression_test_command: list[str],
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    structures = _load_structures()
    equivalence = _official_runtime_equivalence(structures)
    relation_probe = _relation_probe(structures)
    geometry = _role_geometry()
    width_probe = _shared_width_probe(structures)
    control_probe = _control_probe(structures["uknit64"])
    gradient_probe = _gradient_probe(structures["uknit64"])
    relabel_probe = _cell_relabel_probe(structures)
    identity_probe = _forbidden_identity_probe()
    tasks = _make_smoke_tasks(cache_root, structures, progress_callback)
    smoke_results: dict[str, RuntimeSpnJointTrainingResult] = {}
    for role, mode in TYPED_ROLE_MODES.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(READINESS_SEED)
            model = RuntimeE4EquivariantSpnDistinguisher(_model_spec(mode))
        smoke_results[role] = train_runtime_spn_joint(
            model,
            tasks,
            _smoke_training_config(),
            progress_callback=progress_callback,
        )
    smoke = {
        role: _training_result_payload(result) for role, result in smoke_results.items()
    }
    coverage = _five_cipher_typed_coverage(smoke_results["correct"])
    checks = {
        "01_official_runtime_descriptors_exact_all_five": all(equivalence.values()),
        "02_relation_channels_reconstruct_exact_gf2": relation_probe["passed"],
        "03_controls_preserve_required_edge_support": control_probe[
            "support_preserved"
        ],
        "04_parameter_geometry_exactly_matched": geometry["passed"],
        "05_one_shared_state_loads_all_five_widths": width_probe["passed"],
        "06_control_logits_are_distinct_under_shared_weights": control_probe[
            "logits_distinct"
        ],
        "07_typed_parameters_have_finite_nonzero_gradients": (
            gradient_probe["passed"] and coverage
        ),
        "08_cell_relabeling_preserves_relation_channels_and_logits": all(
            relabel_probe.values()
        ),
        "09_no_cipher_identity_or_global_fingerprint_input": identity_probe["passed"],
        "10_existing_runtime_regressions_green": regression_tests_passed,
        "11_equal_task_weights_steps_and_finite_smoke": all(
            _training_finite(result)
            and _equal_task_contributions(result)
            and _smoke_complete(result)
            for result in smoke_results.values()
        ),
    }
    passed = all(checks.values())
    manifest = [
        {
            **asdict(protocol),
            "block_bits": structures[protocol.name].block_bits,
            "cells": structures[protocol.name].cells,
            "runtime_rounds": structures[protocol.name].rounds,
            "window_sha256": structures[protocol.name].window_sha256(),
            "official_runtime_equivalent": equivalence[protocol.name],
            "relation_round_sha256s": relation_probe["sha256s"][protocol.name],
            "nonempty_relation_types": relation_probe["nonempty_types"][
                protocol.name
            ],
            "cell_relabel_invariant": relabel_probe[protocol.name],
        }
        for protocol in FIVE_CIPHER_PROTOCOLS
    ]
    gate = {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_typed_relation_gnn_film_readiness",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_typed_relation_readiness_passed"
            if passed
            else "innovation1_runtime_spn_typed_relation_protocol_invalid"
        ),
        "checks": checks,
        "relation_probe": relation_probe,
        "geometry": geometry,
        "width_probe": width_probe,
        "control_probe": control_probe,
        "gradient_probe": gradient_probe,
        "forbidden_identity_probe": identity_probe,
        "regression_test_command": regression_test_command,
        "regression_tests_passed": regression_tests_passed,
        "training_performed": True,
        "training_scope": "32/class/cipher one-epoch CPU readiness smoke only",
        "claim_scope": (
            "engineering and protocol readiness only; no AUC, transfer, scale, "
            "attack, universality, or breakthrough claim"
        ),
        "next_action": (
            "run the preregistered 2048/class/cipher two-seed typed-relation matrix"
            if passed
            else "repair only failed readiness checks before any real matrix"
        ),
    }
    return manifest, gate, smoke


def _model_spec(mode: str) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=64,
        pair_embedding_dim=128,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
        typed_relation_mode=mode,
        typed_relation_scale=0.1,
    )


def _smoke_training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=1,
        batch_size=32,
        learning_rate=1e-4,
        seed=READINESS_SEED,
        device="cpu",
        optimizer="adam",
        weight_decay=1e-5,
        lr_scheduler="none",
        checkpoint_metric="val_macro_auc",
        restore_best_checkpoint=True,
        loss="mse",
    )


def _relation_probe(structures: dict[str, RuntimeSpnStructure]) -> dict[str, Any]:
    exact: dict[str, list[bool]] = {}
    sha256s: dict[str, list[str]] = {}
    nonempty_types: dict[str, list[int]] = {}
    edge_counts: dict[str, list[int]] = {}
    for name, structure in structures.items():
        exact[name] = []
        sha256s[name] = []
        nonempty_types[name] = []
        edge_counts[name] = []
        indices = _cell_role_indices(structure)
        for round_index in range(structure.rounds):
            relation = RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
                structure,
                round_index=round_index,
                mode="correct",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            reconstructed = relation.reshape(4, 4, structure.cells, structure.cells)
            reconstructed = reconstructed.permute(2, 0, 3, 1)
            expected = structure.inverse_linear_matrices[round_index][
                indices[:, :, None, None], indices[None, None, :, :]
            ].float()
            exact[name].append(bool(torch.equal(reconstructed, expected)))
            sha256s[name].append(
                hashlib.sha256(relation.numpy().tobytes()).hexdigest()
            )
            counts = relation.sum(dim=(1, 2))
            nonempty_types[name].append(int((counts > 0).sum()))
            edge_counts[name].append(int(counts.sum()))
    return {
        "passed": all(all(values) for values in exact.values()),
        "exact_reconstruction": exact,
        "sha256s": sha256s,
        "nonempty_types": nonempty_types,
        "edge_counts": edge_counts,
    }


def _role_geometry() -> dict[str, Any]:
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(_model_spec(mode))
        for role, mode in TYPED_ROLE_MODES.items()
    }
    geometry = {
        role: [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
        for role, model in models.items()
    }
    counts = {
        role: sum(parameter.numel() for parameter in model.parameters())
        for role, model in models.items()
    }
    return {
        "passed": bool(
            set(counts.values()) == {446_562}
            and all(value == geometry["correct"] for value in geometry.values())
        ),
        "parameter_counts": counts,
        "state_geometry_equal": all(
            value == geometry["correct"] for value in geometry.values()
        ),
        "typed_parameter_count": 4096,
    }


def _shared_width_probe(structures: dict[str, RuntimeSpnStructure]) -> dict[str, Any]:
    source = RuntimeE4EquivariantSpnDistinguisher(_model_spec("correct")).eval()
    state = source.state_dict()
    shapes: dict[str, list[int]] = {}
    finite: dict[str, bool] = {}
    for index, (name, structure) in enumerate(structures.items()):
        target = RuntimeE4EquivariantSpnDistinguisher(_model_spec("correct")).eval()
        target.load_state_dict(state, strict=True)
        generator = torch.Generator().manual_seed(READINESS_SEED + index)
        pairs = torch.randint(
            0,
            2,
            (2, 2, 2, structure.block_bits),
            generator=generator,
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = target(pairs, structure)
        shapes[name] = list(logits.shape)
        finite[name] = bool(torch.isfinite(logits).all())
    return {
        "passed": all(value == [2, 1] for value in shapes.values())
        and all(finite.values()),
        "strict_load": True,
        "output_shapes": shapes,
        "outputs_finite": finite,
    }


def _control_probe(structure: RuntimeSpnStructure) -> dict[str, Any]:
    relations = {
        mode: RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
            structure,
            round_index=0,
            mode=mode,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for mode in ("correct", "agnostic", "shuffled")
    }
    support_preserved = bool(
        torch.equal(relations["correct"].sum(0), relations["agnostic"].sum(0))
        and torch.equal(relations["correct"].sum(0), relations["shuffled"].sum(0))
    )
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(_model_spec(mode)).eval()
        for role, mode in TYPED_ROLE_MODES.items()
    }
    state = models["correct"].state_dict()
    for model in models.values():
        model.load_state_dict(state, strict=True)
    generator = torch.Generator().manual_seed(READINESS_SEED)
    pairs = torch.randint(
        0,
        2,
        (2, 2, 2, structure.block_bits),
        generator=generator,
        dtype=torch.float32,
    )
    with torch.no_grad():
        logits = {role: model(pairs, structure) for role, model in models.items()}
    distinct = {
        control: not torch.equal(logits["correct"], logits[control])
        for control in ("dense", "uniform", "shuffled")
    }
    return {
        "support_preserved": support_preserved,
        "logits_distinct": all(distinct.values()),
        "correct_differs_from": distinct,
        "logits": {role: value.flatten().tolist() for role, value in logits.items()},
    }


def _gradient_probe(structure: RuntimeSpnStructure) -> dict[str, Any]:
    model = RuntimeE4EquivariantSpnDistinguisher(_model_spec("correct"))
    generator = torch.Generator().manual_seed(READINESS_SEED)
    pairs = torch.randint(
        0,
        2,
        (4, 2, 2, structure.block_bits),
        generator=generator,
        dtype=torch.float32,
    )
    labels = torch.tensor((0.0, 1.0, 0.0, 1.0))
    logits = model(pairs, structure).squeeze(1)
    torch.nn.functional.mse_loss(torch.sigmoid(logits), labels).backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("typed_relation_message.")
    }
    finite = {
        name: value is not None and bool(torch.isfinite(value).all())
        for name, value in gradients.items()
    }
    nonzero = {
        name: value is not None and float(value.abs().sum()) > 0.0
        for name, value in gradients.items()
    }
    return {
        "passed": bool(
            gradients
            and all(finite.values())
            and all(nonzero.values())
            and model.last_primitive_adapter_traffic.get("typed_relation", 0.0) > 0.0
        ),
        "finite": finite,
        "nonzero": nonzero,
        "traffic": dict(model.last_primitive_adapter_traffic),
    }


def _cell_relabel_probe(
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, bool]:
    model = RuntimeE4EquivariantSpnDistinguisher(_model_spec("correct")).eval()
    result: dict[str, bool] = {}
    for index, (name, structure) in enumerate(structures.items()):
        permutation = tuple(reversed(range(structure.cells)))
        relabeled, bit_permutation = structure.relabel_cells(permutation)
        original_relation = RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
            structure,
            round_index=0,
            mode="correct",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        relabeled_relation = RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
            relabeled,
            round_index=0,
            mode="correct",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected_relation = torch.empty_like(original_relation)
        cell_permutation = torch.tensor(permutation)
        expected_relation[:, cell_permutation[:, None], cell_permutation[None, :]] = (
            original_relation
        )
        generator = torch.Generator().manual_seed(READINESS_SEED + index)
        pairs = torch.randint(
            0,
            2,
            (1, 2, 2, structure.block_bits),
            generator=generator,
            dtype=torch.float32,
        )
        relabeled_pairs = torch.empty_like(pairs)
        relabeled_pairs[..., bit_permutation] = pairs
        with torch.no_grad():
            original_logits = model(pairs, structure)
            relabeled_logits = model(relabeled_pairs, relabeled)
        result[name] = bool(
            torch.equal(relabeled_relation, expected_relation)
            and torch.allclose(original_logits, relabeled_logits, rtol=0.0, atol=1e-6)
        )
    return result


def _forbidden_identity_probe() -> dict[str, Any]:
    forbidden = {"cipher_id", "cipher_name", "block_width", "global_fingerprint"}
    spec_fields = set(RuntimeParameterizedSpnSpec.__dataclass_fields__)
    return {
        "passed": not (forbidden & spec_fields),
        "forbidden_spec_fields_present": sorted(forbidden & spec_fields),
        "condition_inputs": [
            "cell_membership",
            "bit_role",
            "inverse_gf2_edges",
            "existing_sbox_truth_path",
        ],
    }


def _five_cipher_typed_coverage(result: RuntimeSpnJointTrainingResult) -> bool:
    traffic = sum(
        task.get("typed_relation", 0.0) for task in result.router_traffic.values()
    )
    gradients = result.gradient_diagnostics["adapter_gradient_mean_abs_sum"]
    return bool(
        traffic > 0.0
        and gradients.get("typed_relation_message.gamma", 0.0) > 0.0
        and gradients.get("typed_relation_message.beta", 0.0) > 0.0
    )


def _cell_role_indices(structure: RuntimeSpnStructure) -> torch.Tensor:
    indices = torch.empty((structure.cells, 4), dtype=torch.long)
    bits = torch.arange(structure.block_bits)
    indices[structure.cell_membership, structure.bit_role] = bits
    return indices


__all__ = ["TYPED_ROLE_MODES", "build_typed_relation_readiness"]
