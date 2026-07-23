from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from blockcipher_nd.models.structure.spn.cross_spn_typed_cell import (
    GiftCrossSpnTypedCellEquivariantMixerDistinguisher,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    gift64_runtime_structure,
)


RUN_ID = "i1_rtg1_gift64_r1d_runtime_e4_semantic_equivalence_a1_20260724"


@dataclass(frozen=True)
class RuntimeSpnSemanticEquivalenceConfig:
    run_id: str = RUN_ID
    seed: int = 20260724
    batch_rows: int = 8
    pairs_per_sample: int = 4
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.batch_rows <= 0 or self.pairs_per_sample <= 0:
            raise ValueError("audit tensor dimensions must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("audit tolerance must be positive")


def _copy_shared_weights(
    anchor: GiftCrossSpnTypedCellEquivariantMixerDistinguisher,
    runtime: RuntimeE4EquivariantSpnDistinguisher,
) -> None:
    mappings = (
        (anchor.current_cell_encoder, runtime.cell_encoder),
        (anchor.typed_fusion, runtime.typed_fusion),
        (anchor.mixer_blocks, runtime.mixer_blocks),
        (anchor.sequence_norm, runtime.sequence_norm),
        (anchor.pair_projection, runtime.pair_projection),
        (anchor.attention, runtime.pair_attention),
        (anchor.classifier, runtime.classifier),
    )
    for source, destination in mappings:
        destination.load_state_dict(source.state_dict(), strict=True)
    for parameter in runtime.sbox_encoder.parameters():
        parameter.data.zero_()


def _anchor_stages(
    model: GiftCrossSpnTypedCellEquivariantMixerDistinguisher,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = features.shape[0]
    pairs = features.reshape(batch, model.pairs_per_sample, 2, 64).to(
        dtype=model.current_cell_encoder[0].weight.dtype
    )
    difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
    current, previous = model.typed_cell_view(features)
    current_flat = current.reshape(batch * model.pairs_per_sample, 16, 4)
    previous_flat = previous.reshape(batch * model.pairs_per_sample, 16, 4)
    current_hidden = model.current_cell_encoder(current_flat)
    previous_hidden = model.current_cell_encoder(previous_flat)
    typed = model.typed_fusion(torch.cat((current_hidden, previous_hidden), dim=2))
    sequence = typed
    stages: dict[str, torch.Tensor] = {
        "msb_to_lsb_conversion": pairs,
        "current_delta_bits": difference,
        "inverse_linear_previous_bits": previous.reshape(
            batch, model.pairs_per_sample, 64
        ),
        "current_delta_cells": current,
        "inverse_linear_previous_cells": previous,
        "typed_fusion": typed.reshape(batch, model.pairs_per_sample, 16, -1),
        "first_mixer_input": typed.reshape(batch, model.pairs_per_sample, 16, -1),
    }
    for index, block in enumerate(model.mixer_blocks):
        sequence = block(sequence)
        stages[f"mixer_{index + 1}_output"] = sequence.reshape(
            batch, model.pairs_per_sample, 16, -1
        )
    sequence = model.sequence_norm(sequence)
    stages["final_mixer_output"] = sequence.reshape(
        batch, model.pairs_per_sample, 16, -1
    )
    mean_embedding = sequence.mean(dim=1)
    max_embedding = sequence.max(dim=1).values
    activity = current_flat.mean(dim=2, keepdim=True)
    active_embedding = torch.sum(sequence * activity, dim=1) / (
        activity.sum(dim=1).clamp_min(1.0)
    )
    pair_embeddings = model.pair_projection(
        torch.cat((mean_embedding, max_embedding, active_embedding), dim=1)
    ).reshape(batch, model.pairs_per_sample, model.embedding_bits)
    attended, attention = model.attention(pair_embeddings)
    pooled = torch.cat(
        (
            attended,
            pair_embeddings.mean(dim=1),
            pair_embeddings.max(dim=1).values,
        ),
        dim=1,
    )
    stages["pair_embeddings"] = pair_embeddings
    stages["pair_attention_weights"] = attention
    stages["pair_attention_result"] = attended
    stages["final_logits"] = model.classifier(pooled)
    return stages


def _runtime_stages(
    model: RuntimeE4EquivariantSpnDistinguisher,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    structure = gift64_runtime_structure(rounds=1)
    batch = features.shape[0]
    runtime_pairs = features.reshape(
        batch, -1, 2, structure.block_bits
    ).flip(-1)
    pairs = runtime_pairs.to(dtype=model.cell_encoder[0].weight.dtype)
    difference = torch.remainder(pairs[:, :, 0] + pairs[:, :, 1], 2.0)
    previous = structure.exact_inverse(difference, -1)
    current_cells = model._ordered_cell_values(difference, structure)
    previous_cells = model._ordered_cell_values(previous, structure)
    pair_count = pairs.shape[1]
    current_hidden = model.cell_encoder(
        current_cells.reshape(batch * pair_count, structure.cells, 4)
    )
    previous_hidden = model.cell_encoder(
        previous_cells.reshape(batch * pair_count, structure.cells, 4)
    )
    typed = model.typed_fusion(
        torch.cat((current_hidden, previous_hidden), dim=-1)
    ).reshape(batch, pair_count, structure.cells, model.token_dim)
    sequence = typed.reshape(batch * pair_count, structure.cells, model.token_dim)
    stages: dict[str, torch.Tensor] = {
        "msb_to_lsb_conversion": runtime_pairs.flip(-1),
        "current_delta_bits": difference.flip(-1),
        "inverse_linear_previous_bits": previous.flip(-1),
        "current_delta_cells": current_cells.flip(2),
        "inverse_linear_previous_cells": previous_cells.flip(2),
        "typed_fusion": typed.flip(2),
        "first_mixer_input": typed.flip(2),
    }
    for index, block in enumerate(model.mixer_blocks):
        sequence = block(sequence)
        stages[f"mixer_{index + 1}_output"] = sequence.reshape(
            batch, pair_count, structure.cells, model.token_dim
        ).flip(2)
    sequence = model.sequence_norm(sequence)
    aligned_sequence = sequence.reshape(
        batch, pair_count, structure.cells, model.token_dim
    ).flip(2)
    stages["final_mixer_output"] = aligned_sequence
    current_activity = current_cells.mean(dim=-1, keepdim=True).reshape(
        batch * pair_count, structure.cells, 1
    )
    mean_embedding = sequence.mean(dim=1)
    max_embedding = sequence.max(dim=1).values
    active_embedding = torch.sum(sequence * current_activity, dim=1) / (
        current_activity.sum(dim=1).clamp_min(1.0)
    )
    pair_embeddings = model.pair_projection(
        torch.cat((mean_embedding, max_embedding, active_embedding), dim=-1)
    ).reshape(batch, pair_count, model.spec.pair_embedding_dim)
    sbox_context = model.sbox_encoder(
        structure.sbox_truth_bits[-1].to(
            device=pair_embeddings.device, dtype=pair_embeddings.dtype
        )
    )
    late_context = sbox_context.mean(dim=0)
    if late_context.shape[0] != model.spec.pair_embedding_dim:
        late_context = torch.nn.functional.adaptive_avg_pool1d(
            late_context.reshape(1, 1, -1),
            model.spec.pair_embedding_dim,
        ).reshape(-1)
    pair_embeddings = pair_embeddings + late_context[None, None, :]
    attended, attention = model.pair_attention(pair_embeddings)
    pooled = torch.cat(
        (
            attended,
            pair_embeddings.mean(dim=1),
            pair_embeddings.max(dim=1).values,
        ),
        dim=-1,
    )
    stages["pair_embeddings"] = pair_embeddings
    stages["pair_attention_weights"] = attention
    stages["pair_attention_result"] = attended
    stages["final_logits"] = model.classifier(pooled)
    return stages


def audit_runtime_spn_semantic_equivalence(
    config: RuntimeSpnSemanticEquivalenceConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    input_bits = config.pairs_per_sample * 128
    anchor = GiftCrossSpnTypedCellEquivariantMixerDistinguisher(
        input_bits=input_bits,
        pair_bits=128,
        base_channels=64,
        mixer_depth=2,
        token_mlp_ratio=2,
        activation="relu",
        norm="layernorm",
        pooling="attention_mean_max",
        dropout=0.0,
    ).eval()
    runtime = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=64,
            pair_embedding_dim=256,
            processor_steps=2,
            dropout=0.0,
            sbox_context_scale=1.0,
            sbox_context_mode="late_pair",
        )
    ).eval()
    _copy_shared_weights(anchor, runtime)
    anchor = anchor.double()
    runtime = runtime.double()
    generator = torch.Generator().manual_seed(config.seed + 1)
    features = torch.randint(
        0,
        2,
        (config.batch_rows, input_bits),
        generator=generator,
        dtype=torch.float64,
    )
    with torch.no_grad():
        anchor_stages = _anchor_stages(anchor, features)
        runtime_stages = _runtime_stages(runtime, features)

    if anchor_stages.keys() != runtime_stages.keys():
        raise RuntimeError("audit stage sets differ")
    rows: list[dict[str, Any]] = []
    for index, stage in enumerate(anchor_stages):
        anchor_value = anchor_stages[stage]
        runtime_value = runtime_stages[stage]
        same_shape = anchor_value.shape == runtime_value.shape
        error = (
            float(torch.max(torch.abs(anchor_value - runtime_value)).item())
            if same_shape
            else math.inf
        )
        rows.append(
            {
                "run_id": config.run_id,
                "stage_index": index,
                "stage": stage,
                "anchor_shape": list(anchor_value.shape),
                "runtime_shape": list(runtime_value.shape),
                "same_shape": same_shape,
                "maximum_absolute_error": error,
                "tolerance": config.tolerance,
                "within_tolerance": same_shape and error <= config.tolerance,
                "training_performed": False,
            }
        )

    first_divergent = next(
        (row["stage"] for row in rows if not row["within_tolerance"]), None
    )
    protocol_checks = {
        "identical_synthetic_inputs": True,
        "r1d_base_channels_64": anchor.token_dim == 128,
        "two_equivariant_mixer_blocks": len(anchor.mixer_blocks)
        == len(runtime.mixer_blocks)
        == 2,
        "zero_position_forward_path": anchor.position_mode == "zero",
        "shared_current_previous_encoder": anchor.view_encoder_mode
        == "shared_current",
        "runtime_pair_embedding_256": runtime.spec.pair_embedding_dim == 256,
        "runtime_late_pair_sbox_mode": runtime.spec.sbox_context_mode == "late_pair",
        "runtime_sbox_encoder_zeroed": all(
            int(torch.count_nonzero(parameter)) == 0
            for parameter in runtime.sbox_encoder.parameters()
        ),
        "float64_numerical_audit": next(anchor.parameters()).dtype
        == next(runtime.parameters()).dtype
        == torch.float64,
        "all_shared_weights_copied": True,
        "no_neural_training": True,
    }
    execution_checks = {
        "all_stage_shapes_match": all(row["same_shape"] for row in rows),
        "all_stage_errors_finite": all(
            math.isfinite(float(row["maximum_absolute_error"])) for row in rows
        ),
        "all_stage_errors_within_tolerance": all(
            row["within_tolerance"] for row in rows
        ),
    }
    equivalent = all(protocol_checks.values()) and all(execution_checks.values())
    representation_repair_applied = True
    if equivalent:
        status = "pass"
        decision = "innovation1_runtime_spn_deterministic_semantics_equivalent"
        action = (
            "rerun the frozen seed0/seed1 GIFT correct-corrupted-no-topology gate "
            "with the repaired cell bit-role ordering before PRESENT transfer"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_deterministic_semantics_diverge"
        action = (
            f"repair only the first divergent deterministic stage: {first_divergent}"
        )
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "tolerance": config.tolerance,
        "representation_repair_applied": representation_repair_applied,
        "repaired_stage": "current_delta_cells",
        "first_divergent_stage": first_divergent,
        "maximum_absolute_error": max(
            float(row["maximum_absolute_error"]) for row in rows
        ),
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "claim_scope": (
            "deterministic same-weight semantic-equivalence audit between the "
            "GIFT-specific R1d equivariant anchor and runtime E4 with zero late-S-box "
            "context; no training, AUC, cross-cipher, scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "remote_training_authorized": False,
            "present_transfer_authorized": equivalent
            and not representation_repair_applied,
            "rerun_frozen_two_seed_gate_required_after_representation_repair": (
                representation_repair_applied
            ),
        },
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation1_runtime_spn_semantic_equivalence_audit",
        "cipher": "GIFT-64",
        "config": asdict(config),
        "training_performed": False,
        "model_anchor": "GiftCrossSpnTypedCellEquivariantMixerDistinguisher",
        "model_candidate": "RuntimeE4EquivariantSpnDistinguisher",
        "runtime_spec": asdict(runtime.spec),
        "numerical_dtype": "float64",
        "representation_repair": {
            "applied": representation_repair_applied,
            "first_divergent_stage_before_repair": "current_delta_cells",
            "change": "standard 4-bit runtime roles now preserve project MSB order",
        },
        "claim_scope": gate["claim_scope"],
    }
    return {
        "rows": rows,
        "metadata": metadata,
        "gate": gate,
        "summary": {
            "run_id": config.run_id,
            "metadata": metadata,
            "gate": gate,
            "stage_rows": rows,
        },
    }


__all__ = [
    "RUN_ID",
    "RuntimeSpnSemanticEquivalenceConfig",
    "audit_runtime_spn_semantic_equivalence",
]
