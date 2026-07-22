from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputResidualCnn,
    SelectedOutputSpnResidualCnn,
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
)


RUN_ID = (
    "i2_output_prediction_opn1_present_r3_spn_rescnn_head_"
    "permutation_identifiability_audit_20260722"
)


@dataclass(frozen=True)
class SpnResCnnHeadIdentifiabilityConfig:
    run_id: str = RUN_ID
    channels: int = 252
    output_bits: int = 8
    positions: int = 64
    numerical_batch_rows: int = 3
    numerical_seed: int = 20260722
    maximum_equivalence_error: float = 1e-12

    def __post_init__(self) -> None:
        if (self.channels, self.output_bits, self.positions) != (252, 8, 64):
            raise ValueError("OPN1 is frozen to the formal OPC1 head dimensions")
        if self.numerical_batch_rows <= 0 or self.maximum_equivalence_error <= 0:
            raise ValueError("OPN1 numerical controls must be positive")


def _dense_head(model: nn.Module) -> nn.Linear:
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Sequential) or len(head) != 2:
        raise TypeError("expected Flatten plus Linear output head")
    flatten, linear = head
    if not isinstance(flatten, nn.Flatten) or not isinstance(linear, nn.Linear):
        raise TypeError("expected Flatten plus Linear output head")
    return linear


def absorb_final_position_permutation(
    weight: torch.Tensor,
    source_for_destination: torch.Tensor,
) -> torch.Tensor:
    if weight.ndim != 3:
        raise ValueError("weight must have [outputs, channels, positions] shape")
    positions = weight.shape[2]
    permutation = source_for_destination.to(dtype=torch.long, device=weight.device)
    if permutation.shape != (positions,) or sorted(permutation.tolist()) != list(
        range(positions)
    ):
        raise ValueError("source_for_destination must be a complete permutation")
    absorbed = torch.empty_like(weight)
    absorbed[:, :, permutation] = weight
    return absorbed


def _equivalence_row(
    *,
    mapping_name: str,
    mapping: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, Any]:
    routed = hidden.index_select(2, mapping)
    routed_output = torch.nn.functional.linear(
        routed.flatten(1), weight.flatten(1), bias
    )
    absorbed_weight = absorb_final_position_permutation(weight, mapping)
    absorbed_output = torch.nn.functional.linear(
        hidden.flatten(1), absorbed_weight.flatten(1), bias
    )
    return {
        "mapping": mapping_name,
        "positions": int(mapping.numel()),
        "is_complete_permutation": sorted(mapping.tolist()) == list(range(64)),
        "fixed_points": int(
            sum(index == int(source) for index, source in enumerate(mapping))
        ),
        "maximum_absolute_equivalence_error": float(
            (routed_output - absorbed_output).abs().max().item()
        ),
        "sample_classification": False,
    }


def audit_spn_rescnn_head_identifiability(
    config: SpnResCnnHeadIdentifiabilityConfig,
) -> dict[str, Any]:
    anchor = SelectedOutputResidualCnn(channels=config.channels, blocks=10)
    exact = SelectedOutputSpnResidualCnn(
        channels=config.channels,
        stage_blocks=(3, 3, 4),
        source_for_destination=_present_topology_mapping("exact"),
    )
    wrong = SelectedOutputSpnResidualCnn(
        channels=config.channels,
        stage_blocks=(3, 3, 4),
        source_for_destination=_present_topology_mapping("wrong"),
    )
    heads = [_dense_head(model) for model in (anchor, exact, wrong)]

    generator = torch.Generator().manual_seed(config.numerical_seed)
    hidden = torch.randn(
        config.numerical_batch_rows,
        config.channels,
        config.positions,
        dtype=torch.float64,
        generator=generator,
    )
    weight = torch.randn(
        config.output_bits,
        config.channels,
        config.positions,
        dtype=torch.float64,
        generator=generator,
    )
    bias = torch.randn(
        config.output_bits,
        dtype=torch.float64,
        generator=generator,
    )
    mappings = {
        "identity": torch.arange(config.positions),
        "exact_present_p": _present_topology_mapping("exact"),
        "fixed_wrong_p": _present_topology_mapping("wrong"),
    }
    rows = [
        {
            "run_id": config.run_id,
            **_equivalence_row(
                mapping_name=name,
                mapping=mapping,
                hidden=hidden,
                weight=weight,
                bias=bias,
            ),
        }
        for name, mapping in mappings.items()
    ]

    protocol_checks = {
        "formal_opc1_head_dimensions_match": all(
            head.in_features == config.channels * config.positions
            and head.out_features == config.output_bits
            for head in heads
        ),
        "anchor_and_hybrids_use_same_head_type": all(
            type(head) is type(heads[0]) for head in heads[1:]
        ),
        "every_output_connects_all_final_positions": all(
            head.in_features // config.channels == config.positions for head in heads
        ),
        "hybrid_has_three_routed_stages": len(exact.stages) == len(wrong.stages) == 3,
        "exact_and_wrong_mappings_differ_at_all_positions": bool(
            torch.all(mappings["exact_present_p"] != mappings["fixed_wrong_p"])
        ),
        "selected_outputs_match_op10_through_opc1": len(SELECTED_MSB_INDICES)
        == config.output_bits,
        "labels_are_output_values_not_sample_classes": True,
        "neural_training_is_not_used": True,
    }
    execution_checks = {
        "three_mapping_rows_complete": len(rows) == 3,
        "all_mappings_are_complete_permutations": all(
            row["is_complete_permutation"] for row in rows
        ),
        "all_equivalence_errors_are_finite": all(
            math.isfinite(float(row["maximum_absolute_equivalence_error"]))
            for row in rows
        ),
        "all_equivalence_errors_within_tolerance": all(
            float(row["maximum_absolute_equivalence_error"])
            <= config.maximum_equivalence_error
            for row in rows
        ),
    }
    valid = all(protocol_checks.values()) and all(execution_checks.values())
    if valid:
        status = "pass"
        decision = "innovation2_spn_rescnn_final_routing_absorbable_by_global_head"
        action = (
            "keep OPC1 unchanged; attribute any exact-versus-wrong gain only to the "
            "first two interstage routes, then follow the frozen OPC1 branch"
        )
    else:
        status = "fail"
        decision = "innovation2_spn_rescnn_head_identifiability_protocol_invalid"
        action = "repair only the actual head-shape, flatten-order, or permutation proof"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "metrics": {
            "head_in_features": heads[0].in_features,
            "head_out_features": heads[0].out_features,
            "head_connected_positions_per_output": config.positions,
            "maximum_absolute_equivalence_error": max(
                float(row["maximum_absolute_equivalence_error"]) for row in rows
            ),
            "absorbable_routed_stage_indices_zero_based": [2],
            "not_proven_absorbable_routed_stage_indices_zero_based": [0, 1],
        },
        "claim_scope": (
            "deterministic identifiability audit of the final OPC1 position routing "
            "under its global dense head; not an OPC1 performance result, proof about "
            "the first two routes, r4 evidence, sample classification, or SOTA"
        ),
        "next_action": {
            "action": action,
            "opc1_unchanged": True,
            "remote_training_authorized": False,
            "new_output_position_search_authorized": False,
        },
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_spn_rescnn_head_permutation_identifiability_audit",
        "cipher": "PRESENT-80",
        "config": asdict(config),
        "neural_training": False,
        "sample_classification": False,
        "claim_scope": gate["claim_scope"],
    }
    return {
        "rows": rows,
        "metadata": metadata,
        "summary": {
            "run_id": config.run_id,
            "metadata": metadata,
            "rows": rows,
            "gate": gate,
        },
        "gate": gate,
    }
