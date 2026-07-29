from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    OPERATOR_TOKEN_DIM,
    PositionPreservingOperatorK1AzProbe,
    PositionPreservingOperatorSpec,
    relabel_runtime_pairs,
    trainable_parameter_geometry,
    transported_position_ids,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1ay import (
    build_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1az import (
    load_and_validate_config as load_k1az_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_linear_summary_collision_k1ba import (
    derive_collision_controls,
    load_and_validate_config as load_k1ba_config,
    load_authority as load_k1ba_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_position_preserving_operator_k1bb_readiness_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_position_preserving_operator_"
    "k1bb_readiness_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "2e2e02a19b39abed58a66b2095b5f9eca20544652078ca37897338dbde36424b"
)
EXPECTED_REPLICAS = (0, 1)
EXPECTED_ROWS = 12
PROBE_ROWS = 64
MINIMUM_OPERATOR_EMBEDDING_DELTA = 1e-4
MINIMUM_CROSS_EMBEDDING_DELTA = 1e-4
MINIMUM_MODULATION_DELTA = 1e-6
MINIMUM_LOGIT_DELTA = 1e-6
DISABLED_REPLAY_TOLERANCE = 0.0
RELABEL_EMBEDDING_TOLERANCE = 1e-6
RELABEL_MODULATION_TOLERANCE = 1e-5
RELABEL_LOGIT_TOLERANCE = 1e-5
CROSS_CIPHER_ORDER = {
    "uknit64": "midori64",
    "midori64": "dialga128",
    "dialga128": "uknit64",
}


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BB config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BB identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_position_preserving_operator_k1bb_readiness"
    ):
        raise ValueError("K1-BB experiment name drifted")
    if config.get("model") != {
        "hidden_dim": 32,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "modulation_scale": 0.05,
        "operator_token_dim": OPERATOR_TOKEN_DIM,
        "operator_input": (
            "round_position_source_cell_role_target_cell_role_"
            "nonzero_gf2_relation"
        ),
        "interaction_order": (
            "edge_token_then_sample_endpoint_then_target_aggregation"
        ),
        "shared_across_widths": True,
        "cipher_identity": False,
        "per_cipher_modules": False,
    }:
        raise ValueError("K1-BB model contract drifted")
    evaluation = config.get("evaluation", {})
    if evaluation != {
        "replicas": list(EXPECTED_REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "conditions": [
            "correct_operator",
            "same_summary_corrupted_operator",
            "cross_cipher_operator",
        ],
        "cross_cipher_order": CROSS_CIPHER_ORDER,
        "corruption_seed": 20260729,
        "encoder_initialization_seeds": [40, 41],
        "probe_rows_per_panel": PROBE_ROWS,
        "expected_rows": EXPECTED_ROWS,
        "training_performed": False,
        "optimizer_steps": 0,
        "device": "cpu",
        "execution": "local_readiness",
    }:
        raise ValueError("K1-BB evaluation contract drifted")
    if config.get("gates") != {
        "minimum_same_summary_operator_embedding_delta": (
            MINIMUM_OPERATOR_EMBEDDING_DELTA
        ),
        "minimum_cross_cipher_operator_embedding_delta": (
            MINIMUM_CROSS_EMBEDDING_DELTA
        ),
        "minimum_same_summary_edge_modulation_delta": MINIMUM_MODULATION_DELTA,
        "minimum_same_summary_logit_delta": MINIMUM_LOGIT_DELTA,
        "disabled_k1az_logit_replay_tolerance": DISABLED_REPLAY_TOLERANCE,
        "joint_relabel_embedding_tolerance": RELABEL_EMBEDDING_TOLERANCE,
        "joint_relabel_modulation_tolerance": RELABEL_MODULATION_TOLERANCE,
        "joint_relabel_logit_tolerance": RELABEL_LOGIT_TOLERANCE,
        "require_fixed_parameter_geometry": True,
        "require_zero_updates_and_immutable_states": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BB gate contract drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Mapping[tuple[str, int, str], Any],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor | None]],
    Mapping[int, Mapping[str, Any]],
    dict[str, Any],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    source_paths = {name: source_root / name for name in source["digests"]}
    source_gate = _read_json(source_paths["gate.json"])
    source_validation = _read_json(source_paths["validation.json"])
    source_collisions = _read_json(source_paths["structure_collisions.json"])
    source_config = load_k1ba_config(project_root / str(source["config"]))
    (
        readiness,
        dataset_rows,
        datasets,
        structures,
        controls,
        _summary_rows,
        checkpoints,
        _source_controls,
        inherited_checks,
    ) = load_k1ba_authority(
        source_config,
        project_root=project_root,
        device=device,
    )
    collision_summaries, collision_rows = derive_collision_controls(
        structures=structures,
        controls=controls,
        corruption_seed=int(config["evaluation"]["corruption_seed"]),
    )
    expected_source_collisions = source_collisions.get("collision_rows", [])
    checks = {
        "k1ba_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in source_paths.items()
        ),
        "k1ba_gate_authorizes_k1bb": (
            source_gate.get("run_id") == source["run_id"]
            and source_gate.get("status") == "pass"
            and source_gate.get("decision") == source["required_decision"]
            and not source_gate.get("failed_protocol_checks")
        ),
        "k1ba_validation_passes": (
            source_validation.get("status") == "pass"
            and not source_validation.get("errors")
            and int(source_validation.get("result_rows", -1)) == 36
        ),
        "k1ba_collision_controls_replay_exact": collision_rows
        == expected_source_collisions,
        "k1ba_three_collision_summaries_bound": set(collision_summaries)
        == set(EXPECTED_CIPHERS),
        **{f"inherited_{name}": bool(value) for name, value in inherited_checks.items()},
    }
    controls_payload = {
        "collision_summaries": collision_summaries,
        "collision_rows": collision_rows,
        "corrupted_structures": {
            cipher: structures[cipher].corrupted(
                seed=int(config["evaluation"]["corruption_seed"]) + index
            )
            for index, cipher in enumerate(EXPECTED_CIPHERS)
        },
    }
    return (
        readiness,
        dataset_rows,
        datasets,
        structures,
        controls,
        checkpoints,
        controls_payload,
        checks,
    )


def collect_readiness_rows(
    *,
    config: Mapping[str, Any],
    readiness: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, torch.Tensor | None]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    corrupted_structures: Mapping[str, Any],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    k1az_config = load_k1az_config(
        ROOT / str(load_k1ba_config(ROOT / str(config["source"]["config"]))["source"]["config"])
    )
    model_config = config["model"]
    spec = PositionPreservingOperatorSpec(
        hidden_dim=int(model_config["hidden_dim"]),
        pair_embedding_dim=int(model_config["pair_embedding_dim"]),
        dropout=float(model_config["dropout"]),
        modulation_scale=float(model_config["modulation_scale"]),
    )
    rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    for replica_config, initialization_seed in zip(
        k1az_config["replicas"],
        config["evaluation"]["encoder_initialization_seeds"],
        strict=True,
    ):
        replica = int(replica_config["replica"])
        with torch.random.fork_rng():
            torch.manual_seed(int(initialization_seed))
            anchor = build_candidate(
                cipher_configs[EXPECTED_CIPHERS[0]],
                readiness["model"],
                {"gate_hidden_dim": 12},
            ).to(device)
            incompatible = anchor.load_state_dict(
                checkpoints[replica]["state_dict"],
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError("K1-BB strict K1-AZ checkpoint load drifted")
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
            probe = PositionPreservingOperatorK1AzProbe(anchor, spec).to(device)
        probe.eval()
        encoder_geometry = trainable_parameter_geometry(probe.operator_encoder)
        state_before_replica = tensor_mapping_sha256(probe.state_dict())
        embeddings = {
            cipher: probe.operator_encoder.structure_embedding(structures[cipher])
            for cipher in EXPECTED_CIPHERS
        }
        for cipher in EXPECTED_CIPHERS:
            structure = structures[cipher]
            corrupted = corrupted_structures[cipher]
            cross_cipher = CROSS_CIPHER_ORDER[cipher]
            correct_embedding = embeddings[cipher]
            corrupted_embedding = probe.operator_encoder.structure_embedding(corrupted)
            cross_embedding = embeddings[cross_cipher]
            cell_permutation = torch.roll(
                torch.arange(structure.cells),
                shifts=-1,
            )
            relabeled_structure, bit_permutation = structure.relabel_cells(
                cell_permutation.tolist()
            )
            position_ids = transported_position_ids(cell_permutation)
            relabeled_embedding = probe.operator_encoder.structure_embedding(
                relabeled_structure,
                cell_position_ids=position_ids,
            )
            correct_tokens = probe.operator_encoder.operator_tokens(structure)
            corrupted_tokens = probe.operator_encoder.operator_tokens(corrupted)
            geometry_rows.append(
                {
                    "run_id": RUN_ID,
                    "replica": replica,
                    "cipher_key": cipher,
                    "block_bits": structure.block_bits,
                    "cells": structure.cells,
                    "rounds": structure.rounds,
                    "operator_token_dim": int(correct_tokens.values.shape[1]),
                    "correct_edge_count": int(correct_tokens.values.shape[0]),
                    "corrupted_edge_count": int(corrupted_tokens.values.shape[0]),
                    "trainable_parameter_count": sum(
                        int(parameter.numel())
                        for parameter in probe.operator_encoder.parameters()
                        if parameter.requires_grad
                    ),
                    "trainable_parameter_geometry": {
                        name: list(shape) for name, shape in encoder_geometry.items()
                    },
                    "uses_cipher_identity": probe.uses_cipher_identity,
                    "uses_per_cipher_parameters": probe.uses_per_cipher_parameters,
                    "uses_invariant_linear_summary": (
                        probe.uses_invariant_linear_summary
                    ),
                    "uses_actual_source_target_connectivity": (
                        probe.uses_actual_source_target_connectivity
                    ),
                    "operator_interaction_before_pooling": (
                        probe.operator_interaction_before_pooling
                    ),
                }
            )
            operator_rows.append(
                {
                    "run_id": RUN_ID,
                    "replica": replica,
                    "cipher_key": cipher,
                    "cross_cipher_key": cross_cipher,
                    "correct_vs_corrupted_embedding_max_abs_delta": _max_abs_delta(
                        correct_embedding,
                        corrupted_embedding,
                    ),
                    "correct_vs_cross_cipher_embedding_max_abs_delta": _max_abs_delta(
                        correct_embedding,
                        cross_embedding,
                    ),
                    "joint_relabel_embedding_max_abs_delta": _max_abs_delta(
                        correct_embedding,
                        relabeled_embedding,
                    ),
                    "correct_edge_count": int(correct_tokens.values.shape[0]),
                    "corrupted_edge_count": int(corrupted_tokens.values.shape[0]),
                }
            )
            correct_summary = summaries[cipher]["correct_descriptor"]
            if correct_summary is None:
                raise ValueError("K1-BB correct structure summary is missing")
            seed = int(replica_config["dataset_seeds"][cipher])
            for split in FRESH_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                features = torch.as_tensor(
                    np.array(dataset.features[:PROBE_ROWS], copy=True),
                    dtype=torch.float32,
                    device=device,
                )
                runtime_pairs = features.reshape(
                    features.shape[0],
                    -1,
                    2,
                    structure.block_bits,
                ).flip(-1)
                relabeled_pairs = relabel_runtime_pairs(
                    runtime_pairs,
                    bit_permutation,
                )
                relabeled_features = relabeled_pairs.flip(-1).reshape_as(features)
                with torch.inference_mode():
                    anchor_logits = probe.anchor.logits_with_runtime(
                        features,
                        structure,
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=correct_summary,
                        dual_path_enabled=True,
                        component_separation_enabled=True,
                    )
                    disabled_logits = probe.logits_with_operator(
                        features,
                        structure,
                        structure,
                        gate_summary=correct_summary,
                        enabled=False,
                    )
                    correct_modulation = probe.operator_encoder.sample_modulation(
                        runtime_pairs,
                        structure,
                        structure,
                    )
                    corrupted_modulation = probe.operator_encoder.sample_modulation(
                        runtime_pairs,
                        structure,
                        corrupted,
                    )
                    relabeled_modulation = probe.operator_encoder.sample_modulation(
                        relabeled_pairs,
                        relabeled_structure,
                        relabeled_structure,
                        cell_position_ids=position_ids,
                    )
                    correct_logits = probe.logits_with_operator(
                        features,
                        structure,
                        structure,
                        gate_summary=correct_summary,
                    )
                    corrupted_logits = probe.logits_with_operator(
                        features,
                        structure,
                        corrupted,
                        gate_summary=correct_summary,
                    )
                    relabeled_logits = probe.logits_with_operator(
                        relabeled_features,
                        relabeled_structure,
                        relabeled_structure,
                        gate_summary=correct_summary,
                        cell_position_ids=position_ids,
                    )
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "replica": replica,
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "probe_rows": PROBE_ROWS,
                        "correct_vs_corrupted_operator_embedding_delta": (
                            _max_abs_delta(correct_embedding, corrupted_embedding)
                        ),
                        "correct_vs_cross_cipher_operator_embedding_delta": (
                            _max_abs_delta(correct_embedding, cross_embedding)
                        ),
                        "correct_vs_corrupted_edge_modulation_delta": (
                            _max_abs_delta(correct_modulation, corrupted_modulation)
                        ),
                        "correct_vs_corrupted_logit_delta": _max_abs_delta(
                            correct_logits,
                            corrupted_logits,
                        ),
                        "disabled_k1az_logit_replay_delta": _max_abs_delta(
                            anchor_logits,
                            disabled_logits,
                        ),
                        "joint_relabel_embedding_delta": _max_abs_delta(
                            correct_embedding,
                            relabeled_embedding,
                        ),
                        "joint_relabel_modulation_delta": _max_abs_delta(
                            correct_modulation,
                            relabeled_modulation,
                        ),
                        "joint_relabel_logit_delta": _max_abs_delta(
                            correct_logits,
                            relabeled_logits,
                        ),
                        "runtime_structure_held_correct": True,
                        "operator_control_only": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
        state_after_replica = tensor_mapping_sha256(probe.state_dict())
        for row in rows:
            if int(row["replica"]) == replica:
                row["state_immutable"] = state_before_replica == state_after_replica
                row["checkpoint_sha256"] = checkpoints[replica]["sha256"]
                row["checkpoint_state_dict_sha256"] = checkpoints[replica][
                    "state_dict_sha256"
                ]
    return rows, geometry_rows, operator_rows


def adjudicate_readiness(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    operator_rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_panels = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    actual_panels = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        for row in rows
    }
    geometries = {
        json.dumps(row.get("trainable_parameter_geometry"), sort_keys=True)
        for row in geometry_rows
    }
    parameter_counts = {
        int(row.get("trainable_parameter_count", -1)) for row in geometry_rows
    }
    finite_fields = (
        "correct_vs_corrupted_operator_embedding_delta",
        "correct_vs_cross_cipher_operator_embedding_delta",
        "correct_vs_corrupted_edge_modulation_delta",
        "correct_vs_corrupted_logit_delta",
        "disabled_k1az_logit_replay_delta",
        "joint_relabel_embedding_delta",
        "joint_relabel_modulation_delta",
        "joint_relabel_logit_delta",
    )
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "twelve_complete_probe_panels": len(rows) == EXPECTED_ROWS
        and actual_panels == expected_panels,
        "six_operator_control_rows_complete": len(operator_rows) == 6,
        "fixed_parameter_geometry_across_ciphers_and_widths": (
            len(geometry_rows) == 6
            and len(geometries) == 1
            and len(parameter_counts) == 1
            and min(parameter_counts) > 0
            and all(int(row.get("operator_token_dim", -1)) == OPERATOR_TOKEN_DIM for row in geometry_rows)
        ),
        "actual_connectivity_before_pooling_without_cipher_modules": all(
            row.get("uses_actual_source_target_connectivity") is True
            and row.get("operator_interaction_before_pooling") is True
            and row.get("uses_cipher_identity") is False
            and row.get("uses_per_cipher_parameters") is False
            and row.get("uses_invariant_linear_summary") is False
            for row in geometry_rows
        ),
        "two_epoch9_checkpoints_bound": set(checkpoints) == set(EXPECTED_REPLICAS)
        and all(int(checkpoint.get("best_epoch", -1)) == 9 for checkpoint in checkpoints.values()),
        "zero_updates_and_immutable_states": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("state_immutable") is True
            and row.get("runtime_structure_held_correct") is True
            and row.get("operator_control_only") is True
            for row in rows
        ),
        "all_metrics_finite": all(
            all(math.isfinite(float(row.get(field, math.nan))) for field in finite_fields)
            for row in rows
        ),
    }
    panel_checks = {
        "same_summary_operator_embedding_separates": all(
            float(row["correct_vs_corrupted_operator_embedding_delta"])
            >= MINIMUM_OPERATOR_EMBEDDING_DELTA
            for row in rows
        ),
        "cross_cipher_operator_embedding_separates": all(
            float(row["correct_vs_cross_cipher_operator_embedding_delta"])
            >= MINIMUM_CROSS_EMBEDDING_DELTA
            for row in rows
        ),
        "same_summary_sample_modulation_responds": all(
            float(row["correct_vs_corrupted_edge_modulation_delta"])
            >= MINIMUM_MODULATION_DELTA
            for row in rows
        ),
        "same_summary_enabled_logits_respond": all(
            float(row["correct_vs_corrupted_logit_delta"])
            >= MINIMUM_LOGIT_DELTA
            for row in rows
        ),
        "disabled_path_exactly_replays_k1az": all(
            float(row["disabled_k1az_logit_replay_delta"])
            <= DISABLED_REPLAY_TOLERANCE
            for row in rows
        ),
        "transported_joint_relabel_is_equivariant": all(
            float(row["joint_relabel_embedding_delta"])
            <= RELABEL_EMBEDDING_TOLERANCE
            and float(row["joint_relabel_modulation_delta"])
            <= RELABEL_MODULATION_TOLERANCE
            and float(row["joint_relabel_logit_delta"])
            <= RELABEL_LOGIT_TOLERANCE
            for row in rows
        ),
    }
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_panels = [name for name, passed in panel_checks.items() if not passed]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1bb_protocol_invalid"
        next_action = (
            "Repair only the failed K1-BA source, checkpoint, geometry, zero-update "
            "or finite-metric binding and rerun K1-BB unchanged."
        )
    elif not failed_panels:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1bb_position_preserving_operator_"
            "readiness_authorized"
        )
        next_action = (
            "Preregister K1-BC and train one position-preserving operator candidate "
            "against K1-AZ plus same-summary corrupted and cross-cipher controls at "
            "the unchanged local 2048/class/cipher, four-pair, two-replica, ten-epoch "
            "budget. Do not increase pairs, data, width, seeds or use remote GPU."
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_k1bb_position_preserving_operator_"
            "response_incomplete"
        )
        next_action = (
            "Repair only the failed operator encoding, consumer response or transported "
            "relabel path; do not train, scale or change the frozen K1-AZ protocol."
        )
    minima = {
        "operator_embedding_delta": min(
            float(row["correct_vs_corrupted_operator_embedding_delta"])
            for row in rows
        ),
        "cross_cipher_embedding_delta": min(
            float(row["correct_vs_cross_cipher_operator_embedding_delta"])
            for row in rows
        ),
        "edge_modulation_delta": min(
            float(row["correct_vs_corrupted_edge_modulation_delta"])
            for row in rows
        ),
        "logit_delta": min(
            float(row["correct_vs_corrupted_logit_delta"]) for row in rows
        ),
    }
    maxima = {
        "disabled_replay_delta": max(
            float(row["disabled_k1az_logit_replay_delta"]) for row in rows
        ),
        "joint_relabel_embedding_delta": max(
            float(row["joint_relabel_embedding_delta"]) for row in rows
        ),
        "joint_relabel_modulation_delta": max(
            float(row["joint_relabel_modulation_delta"]) for row in rows
        ),
        "joint_relabel_logit_delta": max(
            float(row["joint_relabel_logit_delta"]) for row in rows
        ),
    }
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "panel_checks": panel_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_panel_checks": failed_panels,
        "minimum_response": minima,
        "maximum_compatibility_delta": maxima,
        "trainable_parameter_count": next(iter(parameter_counts), None),
        "remote_scale": "no",
        "claim_scope": (
            "Zero-training local representation readiness on frozen K1-AZ "
            "2048/class/cipher, four-pair evidence; not accuracy improvement, formal "
            "scale, an attack, unseen-cipher transfer or SOTA evidence."
        ),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
    }


def run_readiness(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    _require_fresh_output_root(output_root)
    output_root.mkdir(parents=True)
    _append_progress(output_root / "progress.jsonl", "run_start", run_id=RUN_ID)
    (
        readiness,
        dataset_rows,
        datasets,
        structures,
        summaries,
        checkpoints,
        control_payload,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-BB source preflight failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "evaluation": dict(config["evaluation"]),
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    rows, geometry_rows, operator_rows = collect_readiness_rows(
        config=config,
        readiness=readiness,
        datasets=datasets,
        structures=structures,
        summaries=summaries,
        checkpoints=checkpoints,
        corrupted_structures=control_payload["corrupted_structures"],
        device=device,
    )
    gate = adjudicate_readiness(
        config=config,
        source_checks=source_checks,
        rows=rows,
        geometry_rows=geometry_rows,
        operator_rows=operator_rows,
        checkpoints=checkpoints,
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {
                key: value
                for key, value in checkpoints[replica].items()
                if key != "state_dict"
            }
            for replica in EXPECTED_REPLICAS
        ],
    }
    serializable_controls = {
        "run_id": RUN_ID,
        "source_collision_rows": control_payload["collision_rows"],
        "operator_rows": operator_rows,
    }
    geometry = {
        "run_id": RUN_ID,
        "status": "pass",
        "rows": geometry_rows,
    }
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "result_rows": len(rows),
        "minimum_response": gate["minimum_response"],
        "maximum_compatibility_delta": gate["maximum_compatibility_delta"],
        "trainable_parameter_count": gate["trainable_parameter_count"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_json(output_root / "operator_controls.json", serializable_controls)
    _write_json(output_root / "geometry.json", geometry)
    _write_jsonl(output_root / "results.jsonl", rows)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(rows),
    )
    return {
        "preflight": preflight,
        "results": rows,
        "operator_controls": serializable_controls,
        "geometry": geometry,
        "checkpoint_manifest": checkpoint_manifest,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": event, "time": time.time(), **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(
        (path / name).exists()
        for name in ("preflight.json", "results.jsonl", "gate.json")
    ):
        raise ValueError("K1-BB output already exists")


__all__ = [
    "CONFIG_PATH",
    "CROSS_CIPHER_ORDER",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_ROWS",
    "ROOT",
    "RUN_ID",
    "adjudicate_readiness",
    "collect_readiness_rows",
    "load_and_validate_config",
    "load_authority",
    "run_readiness",
]
