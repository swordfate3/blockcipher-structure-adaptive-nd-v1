from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch

from blockcipher_nd.ciphers.spn.gift import GIFT64_SBOX
from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeParameterizedSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    apply_gf2,
    permutation_matrix,
    runtime_spn_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    gift64_runtime_structure,
    present_runtime_structure,
    skinny64_runtime_structure,
    standard_four_bit_cells,
)


@dataclass(frozen=True)
class RuntimeSpnReadinessConfig:
    run_id: str
    seed: int = 0
    hidden_dim: int = 24
    pair_embedding_dim: int = 32
    processor_steps: int = 2

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if min(
            self.hidden_dim,
            self.pair_embedding_dim,
            self.processor_steps,
        ) <= 0:
            raise ValueError("model dimensions must be positive")


def run_runtime_spn_readiness(
    config: RuntimeSpnReadinessConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=config.hidden_dim,
        pair_embedding_dim=config.pair_embedding_dim,
        processor_steps=config.processor_steps,
        dropout=0.0,
    )
    model = RuntimeParameterizedSpnDistinguisher(spec).eval()
    parameter_geometry = _parameter_geometry(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    structures = _fixture_structures(config.processor_steps)

    rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    all_gradients_finite = True
    all_output_shapes_valid = True
    all_degree_controls_valid = True
    all_true_corrupted_distinct = True
    all_true_independent_distinct = True
    all_inverses_valid = True
    all_state_geometry_stable = True

    for fixture_index, (structure_name, structure) in enumerate(structures.items()):
        pairs = _binary_pairs(
            batch_size=2,
            pair_count=fixture_index + 2,
            block_bits=structure.block_bits,
            seed=config.seed + 100 + fixture_index,
        )
        corrupted = structure.corrupted()
        model.zero_grad(set_to_none=True)
        true_logits = model(pairs, structure)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            true_logits.squeeze(1), torch.tensor([0.0, 1.0])
        )
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        with torch.no_grad():
            corrupted_logits = model(pairs, corrupted)
            independent_logits = model(
                pairs,
                structure,
                relation_mode="independent",
            )
        true_corrupted_max_abs = float(
            torch.max(torch.abs(true_logits.detach() - corrupted_logits))
        )
        true_independent_max_abs = float(
            torch.max(torch.abs(true_logits.detach() - independent_logits))
        )
        row_degrees = structure.linear_matrices.sum(dim=2)
        corrupted_row_degrees = corrupted.linear_matrices.sum(dim=2)
        column_degrees = torch.sort(
            structure.linear_matrices.sum(dim=1), dim=1
        ).values
        corrupted_column_degrees = torch.sort(
            corrupted.linear_matrices.sum(dim=1), dim=1
        ).values
        degree_control_valid = bool(
            torch.equal(row_degrees, corrupted_row_degrees)
            and torch.equal(column_degrees, corrupted_column_degrees)
            and not torch.equal(
                structure.linear_matrices,
                corrupted.linear_matrices,
            )
        )
        inverse_valid = _inverse_is_exact(structure)
        output_shape_valid = tuple(true_logits.shape) == (2, 1)
        state_geometry_stable = parameter_geometry == _parameter_geometry(model)

        all_gradients_finite &= gradients_finite and bool(
            torch.isfinite(true_logits).all() and torch.isfinite(loss)
        )
        all_output_shapes_valid &= output_shape_valid
        all_degree_controls_valid &= degree_control_valid
        all_true_corrupted_distinct &= true_corrupted_max_abs > 1e-6
        all_true_independent_distinct &= true_independent_max_abs > 1e-6
        all_inverses_valid &= inverse_valid
        all_state_geometry_stable &= state_geometry_stable

        rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation1_runtime_parameterized_spn_r0_readiness",
                "structure": structure_name,
                "block_bits": structure.block_bits,
                "cells": structure.cells,
                "rounds": structure.rounds,
                "pair_count": fixture_index + 2,
                "linear_layer_kind": (
                    "permutation"
                    if bool(torch.all(row_degrees == 1))
                    else "general_gf2"
                ),
                "maximum_row_degree": int(row_degrees.max()),
                "linear_edges": int(structure.linear_matrices.sum()),
                "parameter_count": parameter_count,
                "output_shape_valid": output_shape_valid,
                "finite_forward_backward": gradients_finite,
                "exact_gf2_inverse_valid": inverse_valid,
                "degree_preserving_corruption_valid": degree_control_valid,
                "true_corrupted_max_abs_logit_delta": true_corrupted_max_abs,
                "true_independent_max_abs_logit_delta": true_independent_max_abs,
                "state_geometry_stable": state_geometry_stable,
                "training_performed": False,
                "metric_is_research_evidence": False,
            }
        )
        cell_rows.extend(_cell_rows(config.run_id, structure_name, structure))

    present = structures["PRESENT-64 permutation"]
    sbox_sensitive = _sbox_sensitivity(model, present, config.seed)
    permutation_exact = _permutation_gather_equivalence(present, config.seed)
    relabel_equivariant = _cell_relabel_equivariance(
        model,
        structures["SKINNY-64 sparse GF(2)"],
        config.seed,
    )
    invalid_inputs_rejected = _invalid_contract_inputs_rejected()
    state_names = tuple(parameter_geometry)
    runtime_absent_from_state = not any(
        token in name
        for name in state_names
        for token in (
            "cipher",
            "cell_membership",
            "topology",
            "linear_matrix",
            "sbox_truth",
        )
    )
    independent_same_capacity = parameter_geometry == _parameter_geometry(model)
    supports_permutation_and_general = (
        any(row["linear_layer_kind"] == "permutation" for row in rows)
        and any(row["linear_layer_kind"] == "general_gf2" for row in rows)
    )

    readiness_checks = {
        "four_runtime_structures_covered": len(rows) == 4,
        "shared_parameter_geometry_stable": all_state_geometry_stable,
        "runtime_structure_absent_from_state": runtime_absent_from_state,
        "variable_width_and_pair_shapes_valid": all_output_shapes_valid,
        "finite_forward_backward_all_structures": all_gradients_finite,
        "exact_gf2_inverses_valid": all_inverses_valid,
        "permutation_and_general_gf2_supported": supports_permutation_and_general,
        "permutation_gather_matches_gf2": permutation_exact,
        "degree_preserving_corruption_valid": all_degree_controls_valid,
        "true_and_corrupted_logits_distinct": all_true_corrupted_distinct,
        "true_and_independent_logits_distinct": all_true_independent_distinct,
        "independent_mode_preserves_capacity": independent_same_capacity,
        "sbox_descriptor_changes_context": sbox_sensitive,
        "cell_relabel_equivariance": relabel_equivariant,
        "invalid_contract_inputs_rejected": invalid_inputs_rejected,
    }
    status = "pass" if all(readiness_checks.values()) else "fail"
    if status == "pass":
        decision = "innovation1_runtime_spn_r0_readiness_passed"
        next_action = (
            "freeze an R1 local same-budget attribution matrix comparing the runtime "
            "true, degree-preserving corrupted, independent, and fixed E4 anchor roles"
        )
    else:
        decision = "innovation1_runtime_spn_r0_readiness_failed"
        next_action = (
            "repair the failed runtime contract checks before creating any R1 training config"
        )
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "checks_passed": sum(readiness_checks.values()),
        "checks_total": len(readiness_checks),
        "training_performed": False,
        "empirical_topology_superiority_tested": False,
        "claim_scope": (
            "implementation readiness only; random-initialization logit differences "
            "are sensitivity checks, not evidence that true topology is superior"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "claim topology superiority from R0 logits",
            "launch remote training",
            "call R0 an architecture-performance result",
        ],
    }
    contract = {
        "run_id": config.run_id,
        "model_spec": asdict(spec),
        "parameter_count": parameter_count,
        "parameter_geometry": parameter_geometry,
        "runtime_fields": {
            "cell_membership": "[bits] integer cell ids",
            "bit_role": "[bits] values 0..3 within each cell",
            "sbox_truth_bits": "[rounds, cells, 64] binary truth descriptors",
            "linear_matrices": "[rounds, bits, bits] target-by-source GF(2)",
            "inverse_linear_matrices": "validated exact GF(2) inverses",
        },
        "forward_input": "[batch, pairs, 2, bits] or flat equivalent",
        "forward_output": "[batch, 1] binary distinguisher logit",
        "relation_modes": ["true", "independent"],
        "corrupted_relation": (
            "separate deterministic structure with source-column permutation; "
            "row degrees and source-degree multiset preserved"
        ),
        "forbidden_structure_inputs": [
            "cipher_id",
            "secret_key",
            "DDT",
            "trail",
            "beam_score",
            "label_derived_feature",
        ],
    }
    summary = {
        "run_id": config.run_id,
        "task": "innovation1_runtime_parameterized_spn_r0_readiness",
        "seed": config.seed,
        "structures": len(rows),
        "block_widths": sorted({int(row["block_bits"]) for row in rows}),
        "linear_layer_kinds": sorted({str(row["linear_layer_kind"]) for row in rows}),
        "parameter_count": parameter_count,
        "gate": gate,
    }
    return {
        "rows": rows,
        "cell_rows": cell_rows,
        "contract": contract,
        "summary": summary,
        "gate": gate,
    }


def _fixture_structures(processor_steps: int) -> dict[str, RuntimeSpnStructure]:
    rounds = max(2, processor_steps)
    membership, roles = standard_four_bit_cells(128)
    synthetic_permutation = tuple((bit + 12) % 128 for bit in range(128))
    synthetic = runtime_spn_structure(
        cell_membership=membership,
        bit_role=roles,
        sbox_tables=PRESENT_SBOX,
        linear_matrices=permutation_matrix(synthetic_permutation)
        .unsqueeze(0)
        .repeat(rounds, 1, 1),
    )
    return {
        "PRESENT-64 permutation": present_runtime_structure(rounds),
        "GIFT-64 permutation": gift64_runtime_structure(rounds),
        "SKINNY-64 sparse GF(2)": skinny64_runtime_structure(rounds),
        "Synthetic-128 permutation": synthetic,
    }


def _binary_pairs(
    *, batch_size: int, pair_count: int, block_bits: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        2,
        (batch_size, pair_count, 2, block_bits),
        generator=generator,
        dtype=torch.float32,
    )


def _parameter_geometry(model: torch.nn.Module) -> dict[str, list[int]]:
    return {name: list(parameter.shape) for name, parameter in model.state_dict().items()}


def _inverse_is_exact(structure: RuntimeSpnStructure) -> bool:
    identity = torch.eye(structure.block_bits, dtype=torch.int64)
    return all(
        torch.equal(
            torch.remainder(linear.to(torch.int64) @ inverse.to(torch.int64), 2),
            identity,
        )
        for linear, inverse in zip(
            structure.linear_matrices,
            structure.inverse_linear_matrices,
            strict=True,
        )
    )


def _sbox_sensitivity(
    model: RuntimeParameterizedSpnDistinguisher,
    present: RuntimeSpnStructure,
    seed: int,
) -> bool:
    gift_sbox = runtime_spn_structure(
        cell_membership=present.cell_membership,
        bit_role=present.bit_role,
        sbox_tables=GIFT64_SBOX,
        linear_matrices=present.linear_matrices,
    )
    pairs = _binary_pairs(batch_size=2, pair_count=3, block_bits=64, seed=seed + 701)
    with torch.no_grad():
        delta = torch.max(torch.abs(model(pairs, present) - model(pairs, gift_sbox)))
    return float(delta) > 1e-6


def _permutation_gather_equivalence(
    structure: RuntimeSpnStructure,
    seed: int,
) -> bool:
    inverse = structure.inverse_linear_matrices[0]
    indices = torch.argmax(inverse, dim=1)
    values = _binary_pairs(
        batch_size=2,
        pair_count=4,
        block_bits=structure.block_bits,
        seed=seed + 702,
    )[:, :, 0]
    return bool(torch.equal(apply_gf2(inverse, values), values.index_select(2, indices)))


def _cell_relabel_equivariance(
    model: RuntimeParameterizedSpnDistinguisher,
    structure: RuntimeSpnStructure,
    seed: int,
) -> bool:
    cell_permutation = tuple(reversed(range(structure.cells)))
    relabeled, bit_permutation = structure.relabel_cells(cell_permutation)
    pairs = _binary_pairs(
        batch_size=2,
        pair_count=3,
        block_bits=structure.block_bits,
        seed=seed + 703,
    )
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    with torch.no_grad():
        original_logits = model(pairs, structure)
        relabeled_logits = model(relabeled_pairs, relabeled)
    return bool(torch.allclose(original_logits, relabeled_logits, atol=1e-6, rtol=0.0))


def _invalid_contract_inputs_rejected() -> bool:
    membership, roles = standard_four_bit_cells(64)
    identity = torch.eye(64, dtype=torch.uint8)
    bad_membership = list(membership)
    bad_membership[4] = 0
    invalid_factories: tuple[Callable[[], object], ...] = (
        lambda: runtime_spn_structure(
            cell_membership=bad_membership,
            bit_role=roles,
            sbox_tables=PRESENT_SBOX,
            linear_matrices=identity,
        ),
        lambda: runtime_spn_structure(
            cell_membership=membership,
            bit_role=roles,
            sbox_tables=tuple(range(15)) + (14,),
            linear_matrices=identity,
        ),
        lambda: runtime_spn_structure(
            cell_membership=membership,
            bit_role=roles,
            sbox_tables=PRESENT_SBOX,
            linear_matrices=torch.zeros(64, 64, dtype=torch.uint8),
        ),
    )
    return all(_raises_value_error(factory) for factory in invalid_factories)


def _raises_value_error(factory: Callable[[], object]) -> bool:
    try:
        factory()
    except ValueError:
        return True
    return False


def _cell_rows(
    run_id: str,
    structure_name: str,
    structure: RuntimeSpnStructure,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in range(structure.cells):
        bit_indices = torch.nonzero(
            structure.cell_membership == cell,
            as_tuple=False,
        ).flatten()
        truth = structure.sbox_truth_bits[0, cell].numpy().tobytes()
        rows.append(
            {
                "run_id": run_id,
                "structure": structure_name,
                "cell": cell,
                "bit_indices": " ".join(str(int(index)) for index in bit_indices),
                "bit_roles": " ".join(
                    str(int(structure.bit_role[index])) for index in bit_indices
                ),
                "sbox_truth_sha256": hashlib.sha256(truth).hexdigest(),
            }
        )
    return rows


def readiness_values_are_finite(result: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(row[key]))
        for row in result["rows"]
        for key in (
            "true_corrupted_max_abs_logit_delta",
            "true_independent_max_abs_logit_delta",
        )
    )


__all__ = [
    "RuntimeSpnReadinessConfig",
    "readiness_values_are_finite",
    "run_runtime_spn_readiness",
]
