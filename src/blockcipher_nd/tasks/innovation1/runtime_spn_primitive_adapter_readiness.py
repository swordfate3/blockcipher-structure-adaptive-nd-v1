from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
    runtime_spn_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
    gift64_runtime_structure,
    rectangle80_runtime_structure,
    skinny64_runtime_structure,
    uknit64_runtime_structure,
)
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTask,
    RuntimeSpnJointTrainingResult,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig


READINESS_SEED = 20260725
ROLE_MODES = {
    "dense": "dense",
    "correct": "correct",
    "uniform": "uniform",
    "shuffled": "shuffled",
}


@dataclass(frozen=True)
class FiveCipherProtocol:
    name: str
    group: str
    cipher_key: str
    rounds: int
    input_difference: int
    train_key: int
    validation_key: int
    descriptor_path: str
    descriptor_round_start: int


FIVE_CIPHER_PROTOCOLS = (
    FiveCipherProtocol(
        name="gift64",
        group="core",
        cipher_key="gift64",
        rounds=6,
        input_difference=0x40,
        train_key=0,
        validation_key=int("11" * 16, 16),
        descriptor_path="configs/runtime/spn/gift64.json",
        descriptor_round_start=0,
    ),
    FiveCipherProtocol(
        name="skinny64",
        group="core",
        cipher_key="skinny64",
        rounds=7,
        input_difference=0x2000,
        train_key=0,
        validation_key=int("11" * 8, 16),
        descriptor_path="configs/runtime/spn/skinny64.json",
        descriptor_round_start=0,
    ),
    FiveCipherProtocol(
        name="rectangle80",
        group="core",
        cipher_key="rectangle80",
        rounds=6,
        input_difference=0x2100010020,
        train_key=0,
        validation_key=int("11" * 10, 16),
        descriptor_path="configs/runtime/spn/rectangle64.json",
        descriptor_round_start=0,
    ),
    FiveCipherProtocol(
        name="uknit64",
        group="stress",
        cipher_key="uknit64",
        rounds=5,
        input_difference=0x40,
        train_key=0,
        validation_key=int("11" * 16, 16),
        descriptor_path="configs/runtime/spn/uknit64.json",
        descriptor_round_start=3,
    ),
    FiveCipherProtocol(
        name="dialga128",
        group="stress",
        cipher_key="dialga128",
        rounds=4,
        input_difference=0x40,
        train_key=0,
        validation_key=int("11" * 32, 16),
        descriptor_path="configs/runtime/spn/dialga128.json",
        descriptor_round_start=2,
    ),
)


def build_primitive_adapter_readiness(
    *,
    run_id: str,
    cache_root: Path,
    regression_tests_passed: bool,
    regression_test_command: list[str],
    primitive_adapter_effect: str = "additive",
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if primitive_adapter_effect not in {"additive", "multiplicative_gate"}:
        raise ValueError("unsupported primitive adapter readiness effect")
    structures = _load_structures()
    equivalence = _official_runtime_equivalence(structures)
    geometry = _role_geometry(primitive_adapter_effect)
    width_probe = _shared_width_probe(structures, primitive_adapter_effect)
    routing_probe = _routing_probe(primitive_adapter_effect)
    synthetic_probe = _synthetic_gradient_probe(primitive_adapter_effect)
    relabel_probe = _cell_relabel_probe(structures, primitive_adapter_effect)
    tasks = _make_smoke_tasks(cache_root, structures, progress_callback)
    smoke_results: dict[str, RuntimeSpnJointTrainingResult] = {}
    for role, mode in ROLE_MODES.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(READINESS_SEED)
            model = RuntimeE4EquivariantSpnDistinguisher(
                _model_spec(mode, primitive_adapter_effect)
            )
        smoke_results[role] = train_runtime_spn_joint(
            model,
            tasks,
            _smoke_training_config(),
            progress_callback=progress_callback,
        )
    smoke = {
        role: _training_result_payload(result) for role, result in smoke_results.items()
    }
    checks = {
        "01_official_runtime_equivalence_all_five": all(equivalence.values()),
        "02_one_shared_state_loads_all_five_widths": width_probe["passed"],
        "03_all_roles_finite_joint_losses_and_gradients": all(
            _training_finite(result) for result in smoke_results.values()
        ),
        "04_equal_task_weights_and_step_counts": all(
            _equal_task_contributions(result) for result in smoke_results.values()
        ),
        "05_parameter_geometry_and_active_compute_matched": geometry["passed"],
        "06_correct_shuffled_distinct_uniform_assignment_invariant": routing_probe[
            "passed"
        ],
        "07_both_adapters_have_traffic_and_gradients": (
            synthetic_probe["passed"]
            and _five_cipher_adapter_coverage(smoke_results["correct"])
        ),
        "08_joint_cell_relabeling_preserves_logits": all(relabel_probe.values()),
        "09_existing_runtime_regressions_green": regression_tests_passed,
        "10_cpu_32_per_class_smoke_complete": all(
            _smoke_complete(result) for result in smoke_results.values()
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
            "cell_relabel_invariant": relabel_probe[protocol.name],
        }
        for protocol in FIVE_CIPHER_PROTOCOLS
    ]
    gate = {
        "run_id": run_id,
        "task": (
            "innovation1_runtime_spn_primitive_gated_modulation_five_cipher_readiness"
            if primitive_adapter_effect == "multiplicative_gate"
            else "innovation1_runtime_spn_primitive_adapter_five_cipher_readiness"
        ),
        "status": "pass" if passed else "fail",
        "decision": (
            (
                "innovation1_runtime_spn_primitive_gated_modulation_readiness_passed"
                if primitive_adapter_effect == "multiplicative_gate"
                else "innovation1_runtime_spn_primitive_adapter_readiness_passed"
            )
            if passed
            else "innovation1_runtime_spn_primitive_adapter_protocol_invalid"
        ),
        "primitive_adapter_effect": primitive_adapter_effect,
        "checks": checks,
        "geometry": geometry,
        "width_probe": width_probe,
        "routing_probe": routing_probe,
        "synthetic_probe": synthetic_probe,
        "regression_test_command": regression_test_command,
        "regression_tests_passed": regression_tests_passed,
        "training_performed": True,
        "training_scope": "32/class/cipher one-epoch CPU readiness smoke only",
        "claim_scope": (
            "engineering and protocol readiness only; no AUC, transfer, scale, "
            "attack, universality, or breakthrough claim"
        ),
        "next_action": (
            "freeze the readiness-passed implementation and prepare the preregistered "
            "2048/class/cipher two-seed local diagnostic matrix"
            if passed
            else "repair failed readiness checks before creating the real diagnostic matrix"
        ),
    }
    return manifest, gate, smoke


def _model_spec(
    mode: str,
    primitive_adapter_effect: str = "additive",
) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=64,
        pair_embedding_dim=128,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
        primitive_adapter_mode=mode,
        primitive_adapter_rank=8,
        primitive_adapter_scale=0.1,
        primitive_adapter_effect=primitive_adapter_effect,
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


def _load_structures() -> dict[str, RuntimeSpnStructure]:
    return {
        protocol.name: load_runtime_spn_descriptor(
            protocol.descriptor_path,
            rounds=2,
            round_start=protocol.descriptor_round_start,
        ).structure
        for protocol in FIVE_CIPHER_PROTOCOLS
    }


def _official_runtime_equivalence(
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, bool]:
    expected = {
        "gift64": gift64_runtime_structure(2),
        "skinny64": skinny64_runtime_structure(2),
        "rectangle80": rectangle80_runtime_structure(2),
        "uknit64": uknit64_runtime_structure(2, round_start=3),
        "dialga128": dialga128_runtime_structure(2, round_start=2),
    }
    fields = (
        "cell_membership",
        "bit_role",
        "sbox_truth_bits",
        "linear_matrices",
        "inverse_linear_matrices",
    )
    return {
        name: all(
            torch.equal(getattr(structure, field), getattr(expected[name], field))
            for field in fields
        )
        for name, structure in structures.items()
    }


def _role_geometry(primitive_adapter_effect: str = "additive") -> dict[str, Any]:
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(
            _model_spec(mode, primitive_adapter_effect)
        )
        for role, mode in ROLE_MODES.items()
    }
    routed_roles = ("correct", "uniform", "shuffled")
    routed_geometry = {
        role: [
            (name, tuple(parameter.shape))
            for name, parameter in models[role].named_parameters()
        ]
        for role in routed_roles
    }
    parameter_counts = {
        role: sum(parameter.numel() for parameter in model.parameters())
        for role, model in models.items()
    }
    active_compute = {
        role: model.primitive_adapter_summary()["active_adapter_evaluations"]
        for role, model in models.items()
    }
    routed_equal = all(
        routed_geometry[role] == routed_geometry["correct"] for role in routed_roles
    )
    dense_delta = abs(parameter_counts["dense"] - parameter_counts["correct"])
    dense_fraction = dense_delta / max(1, parameter_counts["correct"])
    return {
        "passed": bool(
            routed_equal
            and active_compute["correct"]
            == active_compute["uniform"]
            == active_compute["shuffled"]
            and dense_fraction <= 0.01
        ),
        "parameter_counts": parameter_counts,
        "dense_candidate_parameter_fraction": dense_fraction,
        "routed_state_geometry_equal": routed_equal,
        "active_adapter_evaluations": active_compute,
    }


def _shared_width_probe(
    structures: dict[str, RuntimeSpnStructure],
    primitive_adapter_effect: str = "additive",
) -> dict[str, Any]:
    source = RuntimeE4EquivariantSpnDistinguisher(
        _model_spec("correct", primitive_adapter_effect)
    ).eval()
    state = source.state_dict()
    outputs: dict[str, list[int]] = {}
    finite: dict[str, bool] = {}
    for index, (name, structure) in enumerate(structures.items()):
        target = RuntimeE4EquivariantSpnDistinguisher(
            _model_spec("correct", primitive_adapter_effect)
        ).eval()
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
        outputs[name] = list(logits.shape)
        finite[name] = bool(torch.isfinite(logits).all())
    return {
        "passed": all(shape == [2, 1] for shape in outputs.values())
        and all(finite.values()),
        "strict_load": True,
        "output_shapes": outputs,
        "outputs_finite": finite,
    }


def _routing_probe(primitive_adapter_effect: str = "additive") -> dict[str, Any]:
    structure = _synthetic_mixed_fan_in_structure()
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(
            _model_spec(mode, primitive_adapter_effect)
        ).eval()
        for role, mode in ROLE_MODES.items()
        if role != "dense"
    }
    source_state = models["correct"].state_dict()
    models["uniform"].load_state_dict(source_state, strict=True)
    models["shuffled"].load_state_dict(source_state, strict=True)
    generator = torch.Generator().manual_seed(READINESS_SEED)
    pairs = torch.randint(0, 2, (2, 2, 2, 8), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        logits = {role: model(pairs, structure) for role, model in models.items()}
    correct_weights = models["correct"].primitive_routing_weights(
        structure,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    shuffled_weights = models["correct"].primitive_routing_weights(
        structure,
        round_index=0,
        mode="shuffled",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    uniform_weights = models["correct"].primitive_routing_weights(
        structure,
        round_index=0,
        mode="uniform",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return {
        "passed": bool(
            torch.equal(shuffled_weights, correct_weights.flip(-1))
            and torch.equal(uniform_weights, torch.full_like(uniform_weights, 0.5))
            and not torch.equal(logits["correct"], logits["shuffled"])
            and not torch.equal(logits["correct"], logits["uniform"])
        ),
        "correct_weights": correct_weights.tolist(),
        "shuffled_weights": shuffled_weights.tolist(),
        "uniform_weights": uniform_weights.tolist(),
    }


def _synthetic_gradient_probe(
    primitive_adapter_effect: str = "additive",
) -> dict[str, Any]:
    structure = _synthetic_mixed_fan_in_structure()
    model = RuntimeE4EquivariantSpnDistinguisher(
        _model_spec("correct", primitive_adapter_effect)
    )
    generator = torch.Generator().manual_seed(READINESS_SEED)
    pairs = torch.randint(0, 2, (4, 2, 2, 8), generator=generator, dtype=torch.float32)
    labels = torch.tensor((0.0, 1.0, 0.0, 1.0))
    logits = model(pairs, structure).squeeze(1)
    torch.nn.functional.mse_loss(torch.sigmoid(logits), labels).backward()
    gradient_sums: dict[str, float] = {}
    gradients_finite = True
    assert model.primitive_adapters is not None
    for name, adapter in model.primitive_adapters.items():
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.grad is not None
        ]
        gradient_sums[name] = sum(float(value.abs().sum()) for value in gradients)
        gradients_finite = gradients_finite and bool(
            gradients and all(torch.isfinite(value).all() for value in gradients)
        )
    traffic = dict(model.last_primitive_adapter_traffic)
    return {
        "passed": bool(
            gradients_finite
            and all(value > 0.0 for value in gradient_sums.values())
            and all(value > 0.0 for value in traffic.values())
        ),
        "gradient_abs_sums": gradient_sums,
        "traffic": traffic,
        "gradients_finite": gradients_finite,
    }


def _cell_relabel_probe(
    structures: dict[str, RuntimeSpnStructure],
    primitive_adapter_effect: str = "additive",
) -> dict[str, bool]:
    model = RuntimeE4EquivariantSpnDistinguisher(
        _model_spec("correct", primitive_adapter_effect)
    ).eval()
    result: dict[str, bool] = {}
    for index, (name, structure) in enumerate(structures.items()):
        relabeled, bit_permutation = structure.relabel_cells(
            tuple(reversed(range(structure.cells)))
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
            original = model(pairs, structure)
            permuted = model(relabeled_pairs, relabeled)
        result[name] = bool(torch.allclose(original, permuted, rtol=0.0, atol=1e-6))
    return result


def _make_smoke_tasks(
    cache_root: Path,
    structures: dict[str, RuntimeSpnStructure],
    progress_callback: ProgressCallback | None,
) -> list[RuntimeSpnJointTask]:
    tasks: list[RuntimeSpnJointTask] = []
    for index, protocol in enumerate(FIVE_CIPHER_PROTOCOLS):
        datasets = {}
        for split, key, samples_per_class, seed_offset in (
            ("train", protocol.train_key, 32, 0),
            ("validation", protocol.validation_key, 16, 10_000),
        ):
            cipher = build_cipher(protocol.cipher_key, protocol.rounds, key=key)
            datasets[split] = make_chunked_differential_dataset(
                DifferentialDatasetConfig(
                    cipher=cipher,
                    input_difference=protocol.input_difference,
                    samples_per_class=samples_per_class,
                    seed=READINESS_SEED + seed_offset + index,
                    feature_encoding="ciphertext_pair_bits",
                    pairs_per_sample=4,
                    negative_mode="encrypted_random_plaintexts",
                    key_rotation_interval=0,
                    sample_structure="independent_pairs",
                ),
                cache_dir=cache_root / protocol.name / split,
                chunk_size=32,
                workers=1,
                progress_callback=progress_callback,
                progress_context={"cipher": protocol.name, "split": split},
            )
        tasks.append(
            RuntimeSpnJointTask(
                name=protocol.name,
                group=protocol.group,
                structure=structures[protocol.name],
                train_dataset=datasets["train"],
                validation_dataset=datasets["validation"],
            )
        )
    return tasks


def _synthetic_mixed_fan_in_structure() -> RuntimeSpnStructure:
    linear = torch.eye(8, dtype=torch.uint8)
    linear[4:, 4:] = torch.tensor(
        (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
        ),
        dtype=torch.uint8,
    )
    return runtime_spn_structure(
        cell_membership=(0, 0, 0, 0, 1, 1, 1, 1),
        bit_role=(3, 2, 1, 0, 3, 2, 1, 0),
        sbox_tables=PRESENT_SBOX,
        linear_matrices=linear.unsqueeze(0).repeat(2, 1, 1),
    )


def _training_finite(result: RuntimeSpnJointTrainingResult) -> bool:
    values = [
        value
        for row in result.history
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    return bool(
        values
        and all(np.isfinite(value) for value in values)
        and result.gradient_diagnostics["all_gradients_finite"]
    )


def _equal_task_contributions(result: RuntimeSpnJointTrainingResult) -> bool:
    metadata = result.metadata
    return bool(
        set(metadata["task_weights"].values()) == {0.2}
        and len(set(metadata["task_batch_counts"].values())) == 1
        and all(
            count == metadata["optimizer_steps"]
            for count in metadata["task_batch_counts"].values()
        )
    )


def _five_cipher_adapter_coverage(result: RuntimeSpnJointTrainingResult) -> bool:
    traffic = {
        name: sum(task.get(name, 0.0) for task in result.router_traffic.values())
        for name in ("fan_in_1", "multi_source")
    }
    gradients = result.gradient_diagnostics["adapter_gradient_mean_abs_sum"]
    return bool(
        all(value > 0.0 for value in traffic.values())
        and gradients.get("primitive_adapters.fan_in_1", 0.0) > 0.0
        and gradients.get("primitive_adapters.multi_source", 0.0) > 0.0
    )


def _smoke_complete(result: RuntimeSpnJointTrainingResult) -> bool:
    expected = {protocol.name for protocol in FIVE_CIPHER_PROTOCOLS}
    return bool(
        set(result.train_metrics) == expected
        and set(result.validation_metrics) == expected
        and len(result.history) == 1
        and result.metadata["shared_state_dict_count"] == 1
        and result.metadata["task_specific_trainable_state"] is False
    )


def _training_result_payload(result: RuntimeSpnJointTrainingResult) -> dict[str, Any]:
    groups = result.metadata["task_groups"]
    validation = result.validation_metrics
    core = [
        validation[name]["auc"] for name, group in groups.items() if group == "core"
    ]
    stress = [
        validation[name]["auc"] for name, group in groups.items() if group == "stress"
    ]
    return {
        "history": result.history,
        "train_metrics": result.train_metrics,
        "validation_metrics": result.validation_metrics,
        "aggregates": {
            "core_macro_auc": float(np.mean(core)),
            "stress_macro_auc": float(np.mean(stress)),
            "five_cipher_macro_auc": float(
                np.mean([metrics["auc"] for metrics in validation.values()])
            ),
        },
        "metadata": result.metadata,
        "router_traffic": result.router_traffic,
        "gradient_diagnostics": result.gradient_diagnostics,
    }


__all__ = [
    "FIVE_CIPHER_PROTOCOLS",
    "FiveCipherProtocol",
    "READINESS_SEED",
    "ROLE_MODES",
    "build_primitive_adapter_readiness",
]
