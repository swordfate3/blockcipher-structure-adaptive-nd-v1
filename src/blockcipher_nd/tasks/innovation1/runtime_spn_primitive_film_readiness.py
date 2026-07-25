from __future__ import annotations

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
    ROLE_MODES,
    _equal_task_contributions,
    _load_structures,
    _make_smoke_tasks,
    _official_runtime_equivalence,
    _smoke_complete,
    _smoke_training_config,
    _synthetic_mixed_fan_in_structure,
    _training_finite,
    _training_result_payload,
)
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTrainingResult,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import ProgressCallback


def build_primitive_true_film_readiness(
    *,
    run_id: str,
    cache_root: Path,
    regression_tests_passed: bool,
    regression_test_command: list[str],
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    structures = _load_structures()
    equivalence = _official_runtime_equivalence(structures)
    descriptors = _descriptor_probe(structures)
    geometry = _role_geometry()
    width_probe = _shared_width_probe(structures)
    control_probe = _control_probe()
    gradient_probe = _gradient_probe()
    relabel_probe = _cell_relabel_probe(structures)
    forbidden_probe = _forbidden_identity_probe(structures)
    tasks = _make_smoke_tasks(cache_root, structures, progress_callback)
    smoke_results: dict[str, RuntimeSpnJointTrainingResult] = {}
    for role, mode in ROLE_MODES.items():
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
    film_coverage = _five_cipher_film_coverage(smoke_results["correct"])
    checks = {
        "01_external_runtime_descriptors_exact_all_five": all(equivalence.values()),
        "02_descriptor_shape_finite_and_distinguishes_collisions": descriptors[
            "passed"
        ],
        "03_descriptor_responds_to_sbox_and_gf2_changes": descriptors[
            "mutation_sensitive"
        ],
        "04_parameter_geometry_exactly_matched": geometry["passed"],
        "05_one_shared_state_loads_all_five_widths": width_probe["passed"],
        "06_correct_uniform_shuffled_dense_outputs_distinct": control_probe["passed"],
        "07_every_film_parameter_has_finite_nonzero_gradient": (
            gradient_probe["passed"] and film_coverage
        ),
        "08_no_cipher_id_width_or_global_fingerprint_condition": forbidden_probe[
            "passed"
        ],
        "09_cell_relabeling_preserves_descriptors_and_logits": all(
            relabel_probe.values()
        ),
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
            "descriptor_shapes": descriptors["shapes"][protocol.name],
            "descriptor_round_sha256s": descriptors["sha256s"][protocol.name],
            "cell_relabel_invariant": relabel_probe[protocol.name],
        }
        for protocol in FIVE_CIPHER_PROTOCOLS
    ]
    gate = {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_primitive_true_film_five_cipher_readiness",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_primitive_true_film_readiness_passed"
            if passed
            else "innovation1_runtime_spn_primitive_true_film_protocol_invalid"
        ),
        "checks": checks,
        "descriptor_probe": descriptors,
        "geometry": geometry,
        "width_probe": width_probe,
        "control_probe": control_probe,
        "gradient_probe": gradient_probe,
        "forbidden_identity_probe": forbidden_probe,
        "regression_test_command": regression_test_command,
        "regression_tests_passed": regression_tests_passed,
        "training_performed": True,
        "training_scope": "32/class/cipher one-epoch CPU readiness smoke only",
        "claim_scope": (
            "engineering and protocol readiness only; no AUC, transfer, scale, "
            "attack, universality, or breakthrough claim"
        ),
        "next_action": (
            "run the preregistered 2048/class/cipher two-seed local True-FiLM matrix"
            if passed
            else "repair only the failed readiness checks before any real matrix"
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
        primitive_film_mode=mode,
        primitive_film_rank=10,
        primitive_film_scale=0.1,
    )


def _descriptor_probe(structures: dict[str, RuntimeSpnStructure]) -> dict[str, Any]:
    import hashlib

    shapes: dict[str, list[list[int]]] = {}
    sha256s: dict[str, list[str]] = {}
    finite: dict[str, bool] = {}
    signatures: dict[str, tuple[str, ...]] = {}
    for name, structure in structures.items():
        per_round = [
            RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
                structure,
                round_index=round_index,
                mode="correct",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for round_index in range(structure.rounds)
        ]
        shapes[name] = [list(value.shape) for value in per_round]
        finite[name] = all(bool(torch.isfinite(value).all()) for value in per_round)
        sha256s[name] = [
            hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in per_round
        ]
        signatures[name] = tuple(sha256s[name])

    synthetic = _synthetic_mixed_fan_in_structure()
    original = RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
        synthetic,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    changed_sbox = synthetic.sbox_truth_bits.clone()
    changed_sbox[:, 1] = torch.roll(changed_sbox[:, 1], shifts=4, dims=-1)
    from blockcipher_nd.models.structure.spn.runtime_structure import (
        runtime_spn_structure_from_truth_bits,
    )

    sbox_structure = runtime_spn_structure_from_truth_bits(
        synthetic.cell_membership,
        synthetic.bit_role,
        changed_sbox,
        synthetic.linear_matrices,
    )
    sbox_descriptor = RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
        sbox_structure,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    linear_descriptor = RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
        synthetic.corrupted(seed=20260726),
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    mutation_sensitive = bool(
        not torch.equal(original[:, :64], sbox_descriptor[:, :64])
        and not torch.equal(original[:, 64:], linear_descriptor[:, 64:])
    )
    passed = bool(
        all(finite.values())
        and all(
            shape == [structures[name].cells, 128]
            for name, round_shapes in shapes.items()
            for shape in round_shapes
        )
        and signatures["gift64"] != signatures["rectangle80"]
        and signatures["uknit64"] != signatures["dialga128"]
    )
    return {
        "passed": passed,
        "mutation_sensitive": mutation_sensitive,
        "shapes": shapes,
        "finite": finite,
        "sha256s": sha256s,
        "known_collision_pairs_distinguished": {
            "gift64_rectangle80": signatures["gift64"] != signatures["rectangle80"],
            "uknit64_dialga128": signatures["uknit64"] != signatures["dialga128"],
        },
    }


def _role_geometry() -> dict[str, Any]:
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(_model_spec(mode))
        for role, mode in ROLE_MODES.items()
    }
    additive = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=64,
            pair_embedding_dim=128,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="edge_gate",
            cell_input_mode="state_triplet",
            round_window_mode="recurrent_window",
            primitive_adapter_mode="correct",
            primitive_adapter_rank=8,
            primitive_adapter_scale=0.1,
        )
    )
    geometry = {
        role: [
            (name, tuple(parameter.shape))
            for name, parameter in model.named_parameters()
        ]
        for role, model in models.items()
    }
    counts = {
        role: sum(parameter.numel() for parameter in model.parameters())
        for role, model in models.items()
    }
    additive_count = sum(parameter.numel() for parameter in additive.parameters())
    return {
        "passed": bool(
            len(set(counts.values())) == 1
            and set(counts.values()) == {446_562}
            and additive_count == 446_562
            and all(value == geometry["correct"] for value in geometry.values())
        ),
        "parameter_counts": counts,
        "additive_source_parameter_count": additive_count,
        "state_geometry_equal": all(
            value == geometry["correct"] for value in geometry.values()
        ),
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


def _control_probe() -> dict[str, Any]:
    structure = _synthetic_mixed_fan_in_structure()
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(_model_spec(mode)).eval()
        for role, mode in ROLE_MODES.items()
    }
    state = models["correct"].state_dict()
    for model in models.values():
        model.load_state_dict(state, strict=True)
    generator = torch.Generator().manual_seed(READINESS_SEED)
    pairs = torch.randint(0, 2, (2, 2, 2, 8), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        logits = {role: model(pairs, structure) for role, model in models.items()}
    distinct = {
        control: not torch.equal(logits["correct"], logits[control])
        for control in ("dense", "uniform", "shuffled")
    }
    return {
        "passed": all(distinct.values()),
        "correct_differs_from": distinct,
        "logits": {role: value.flatten().tolist() for role, value in logits.items()},
    }


def _gradient_probe() -> dict[str, Any]:
    structure = _synthetic_mixed_fan_in_structure()
    model = RuntimeE4EquivariantSpnDistinguisher(_model_spec("correct"))
    generator = torch.Generator().manual_seed(READINESS_SEED)
    pairs = torch.randint(0, 2, (4, 2, 2, 8), generator=generator, dtype=torch.float32)
    labels = torch.tensor((0.0, 1.0, 0.0, 1.0))
    logits = model(pairs, structure).squeeze(1)
    torch.nn.functional.mse_loss(torch.sigmoid(logits), labels).backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("primitive_film_conditioner.")
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
            and model.last_primitive_adapter_traffic.get("film", 0.0) > 0.0
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
        original_descriptor = (
            RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
                structure,
                round_index=0,
                mode="correct",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
        relabeled_descriptor = (
            RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
                relabeled,
                round_index=0,
                mode="correct",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
        expected = torch.empty_like(original_descriptor)
        expected[torch.tensor(permutation)] = original_descriptor
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
            torch.equal(relabeled_descriptor, expected)
            and torch.allclose(original_logits, relabeled_logits, rtol=0.0, atol=1e-6)
        )
    return result


def _forbidden_identity_probe(
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, Any]:
    forbidden_fields = {"cipher_id", "cipher_name", "block_width", "global_fingerprint"}
    spec_fields = set(RuntimeParameterizedSpnSpec.__dataclass_fields__)
    descriptor_dims = {
        name: {
            RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
                structure,
                round_index=round_index,
                mode="correct",
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).shape[-1]
            for round_index in range(structure.rounds)
        }
        for name, structure in structures.items()
    }
    return {
        "passed": not (forbidden_fields & spec_fields)
        and all(values == {128} for values in descriptor_dims.values()),
        "forbidden_spec_fields_present": sorted(forbidden_fields & spec_fields),
        "descriptor_dims": {
            name: sorted(values) for name, values in descriptor_dims.items()
        },
        "condition_inputs": ["cell_sbox_truth_bits", "local_gf2_diffusion_statistics"],
    }


def _five_cipher_film_coverage(result: RuntimeSpnJointTrainingResult) -> bool:
    traffic = sum(task.get("film", 0.0) for task in result.router_traffic.values())
    gradients = result.gradient_diagnostics["adapter_gradient_mean_abs_sum"]
    return bool(
        traffic > 0.0
        and gradients.get("primitive_film_conditioner.down", 0.0) > 0.0
        and gradients.get("primitive_film_conditioner.affine", 0.0) > 0.0
    )


__all__ = ["build_primitive_true_film_readiness"]
