from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.models.structure.spn.structure_conditioned_gate import (
    SBOX_SUMMARY_DIM,
    hybrid_structure_summary,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1ay import (
    build_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1az import (
    load_and_validate_config as load_k1az_config,
    load_sources as load_k1az_sources,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
    derive_structure_controls,
    load_and_validate_config as load_k1at_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1aw import (
    load_and_validate_config as load_k1aw_config,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_linear_summary_collision_k1ba_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_linear_summary_collision_k1ba_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "af7c9ad450372a116ed3dff5855c6e682a96a18caeec3bb8f67c748934bce7ef"
)
EXPECTED_REPLICAS = (0, 1)
CONDITIONS = (
    "correct_descriptor",
    "cross_cipher_linear_mismatch",
    "same_summary_corrupted_linear",
)
EXPECTED_ROWS = 36
EXPECTED_BATCH_SIZE = 64
COLLISION_SUMMARY_TOLERANCE = 0.0
MINIMUM_MATRIX_HAMMING_FRACTION = 0.001
MINIMUM_CROSS_EDGE_GATE_DELTA = 0.0005
MAXIMUM_CROSS_AUC_DELTA_ABS = 0.001
MINIMUM_CROSS_SPEARMAN = 0.999


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BA config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BA identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_linear_summary_collision_k1ba"
    ):
        raise ValueError("K1-BA experiment name drifted")
    if config.get("evaluation") != {
        "replicas": list(EXPECTED_REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "conditions": list(CONDITIONS),
        "corruption_seed": 20260729,
        "expected_rows": EXPECTED_ROWS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "training_performed": False,
        "optimizer_steps": 0,
        "execution": "local_audit",
    }:
        raise ValueError("K1-BA evaluation contract drifted")
    if config.get("gates") != {
        "collision_summary_max_abs_delta": COLLISION_SUMMARY_TOLERANCE,
        "minimum_corrupted_matrix_hamming_fraction": (
            MINIMUM_MATRIX_HAMMING_FRACTION
        ),
        "minimum_cross_cipher_edge_gate_delta": MINIMUM_CROSS_EDGE_GATE_DELTA,
        "maximum_cross_cipher_auc_delta_abs": MAXIMUM_CROSS_AUC_DELTA_ABS,
        "minimum_cross_cipher_probability_spearman": MINIMUM_CROSS_SPEARMAN,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BA gates drifted")
    if config.get("decision_order") != [
        "invariant_linear_summary_not_topology_identifying",
        "scalar_edge_modulation_rank_inertia",
        "mechanism_unresolved",
    ]:
        raise ValueError("K1-BA decision order drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int, str], DiskDifferentialDataset],
    dict[str, Any],
    dict[str, dict[str, torch.Tensor | None]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[tuple[int, str, str, str], dict[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    source_paths = {name: source_root / name for name in source["digests"]}
    source_gate = _read_json(source_paths["gate.json"])
    source_validation = _read_json(source_paths["validation.json"])
    source_results = _read_jsonl(source_paths["results.jsonl"])
    source_controls_rows = _read_jsonl(source_paths["controls.jsonl"])
    source_manifest = _read_json(source_paths["checkpoint_manifest.json"])
    source_summaries = _read_json(source_paths["structure_summaries.json"])
    source_dataset_rows = _read_jsonl(source_paths["dataset_manifest.jsonl"])
    source_config = load_k1az_config(project_root / str(source["config"]))
    (
        readiness,
        _k1as,
        _k1av,
        dataset_rows,
        datasets,
        _anchors,
        inherited_checks,
    ) = load_k1az_sources(source_config, project_root=project_root)
    k1aw_config = load_k1aw_config(
        project_root / str(source_config["same_budget_anchor"]["config"])
    )
    k1at_config = load_k1at_config(
        project_root / str(k1aw_config["same_budget_anchor"]["config"])
    )
    structures, controls, summary_rows, structure_checks = derive_structure_controls(
        readiness_config=readiness,
        config=k1at_config,
    )
    checkpoints, checkpoint_checks = _load_checkpoints(
        source_root=source_root,
        manifest=source_manifest,
        device=device,
    )
    source_controls = _index_source_controls(source_controls_rows)
    expected_source_keys = {
        (replica, cipher, split, condition)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
        for condition in ("correct_descriptor", "linear_only_mismatch")
    }
    checks = {
        "source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in source_paths.items()
        ),
        "source_gate_is_valid_hold": (
            source_gate.get("run_id") == source["run_id"]
            and source_gate.get("status") == "hold"
            and source_gate.get("decision") == source["required_decision"]
            and not source_gate.get("failed_protocol_checks")
        ),
        "source_validation_passes": (
            source_validation.get("status") == "pass"
            and not source_validation.get("errors")
            and int(source_validation.get("training_rows", -1)) == 2
            and int(source_validation.get("evaluation_rows", -1)) == 60
        ),
        "source_two_training_rows_and_sixty_controls": (
            len(source_results) == 2 and len(source_controls_rows) == 60
        ),
        "source_three_summaries_and_eighteen_datasets": (
            len(source_summaries.get("rows", [])) == 3
            and len(source_dataset_rows) == len(dataset_rows) == 18
        ),
        "source_required_control_rows_indexed": set(source_controls)
        == expected_source_keys,
        **{f"inherited_{name}": bool(value) for name, value in inherited_checks.items()},
        **{f"structure_{name}": bool(value) for name, value in structure_checks.items()},
        **checkpoint_checks,
    }
    return (
        readiness,
        dataset_rows,
        datasets,
        structures,
        controls,
        summary_rows,
        checkpoints,
        source_controls,
        checks,
    )


def derive_collision_controls(
    *,
    structures: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, torch.Tensor | None]],
    corruption_seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    collision_summaries: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for cipher_index, cipher in enumerate(EXPECTED_CIPHERS):
        structure = structures[cipher]
        correct = controls[cipher]["correct_descriptor"]
        cross = controls[cipher]["linear_only_mismatch"]
        if correct is None or cross is None:
            raise ValueError("K1-BA source summary cannot be absent")
        corrupted = structure.corrupted(seed=corruption_seed + cipher_index)
        collision = hybrid_structure_summary(
            sbox_structure=structure,
            linear_structure=corrupted,
        )
        if collision.shape != correct.shape:
            raise ValueError("K1-BA collision summary geometry drifted")
        collision_summaries[cipher] = collision
        correct_linear = torch.as_tensor(correct[SBOX_SUMMARY_DIM:])
        cross_linear = torch.as_tensor(cross[SBOX_SUMMARY_DIM:])
        collision_linear = torch.as_tensor(collision[SBOX_SUMMARY_DIM:])
        active = torch.nonzero(
            torch.abs(cross_linear - correct_linear) > 0.0,
            as_tuple=False,
        ).reshape(-1)
        matrix_difference = (
            structure.linear_matrices != corrupted.linear_matrices
        ).to(torch.float64)
        rows.append(
            {
                "run_id": RUN_ID,
                "cipher_key": cipher,
                "corruption_seed": corruption_seed + cipher_index,
                "correct_window_sha256": structure.window_sha256(),
                "corrupted_window_sha256": corrupted.window_sha256(),
                "correct_linear_matrices_sha256": _tensor_sha256(
                    structure.linear_matrices
                ),
                "corrupted_linear_matrices_sha256": _tensor_sha256(
                    corrupted.linear_matrices
                ),
                "matrix_hamming_fraction": float(matrix_difference.mean()),
                "correct_summary_sha256": _tensor_sha256(correct),
                "collision_summary_sha256": _tensor_sha256(collision),
                "collision_summary_max_abs_delta": float(
                    torch.max(torch.abs(collision - correct))
                ),
                "collision_linear_summary_max_abs_delta": float(
                    torch.max(torch.abs(collision_linear - correct_linear))
                ),
                "cross_cipher_active_linear_dimensions": [
                    int(index) for index in active.tolist()
                ],
                "cross_cipher_active_linear_dimension_count": int(active.numel()),
                "cross_cipher_linear_summary_l2": float(
                    torch.linalg.vector_norm(cross_linear - correct_linear)
                ),
                "operator_changed": not torch.equal(
                    structure.linear_matrices, corrupted.linear_matrices
                ),
                "summary_collision_exact": torch.equal(collision, correct),
            }
        )
    return collision_summaries, rows


def collect_panel_rows(
    *,
    model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    structure: Any,
    summaries: Mapping[str, torch.Tensor],
    source_controls: Mapping[tuple[int, str, str, str], Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    replica: int,
    cipher: str,
    seed: int,
    split: str,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    state_before = tensor_mapping_sha256(model.state_dict())
    probabilities: dict[str, np.ndarray] = {}
    gates: dict[str, tuple[float, float]] = {}
    for condition in CONDITIONS:
        summary = summaries[condition]
        gates[condition] = _path_gates(model, structure, summary)
        probabilities[condition] = _evaluate_probabilities(
            model=model,
            dataset=dataset,
            structure=structure,
            summary=summary,
            batch_size=batch_size,
            device=device,
        )
    state_after = tensor_mapping_sha256(model.state_dict())
    labels = np.asarray(dataset.labels, dtype=np.float32)
    correct = probabilities["correct_descriptor"]
    rows = []
    source_condition = {
        "correct_descriptor": "correct_descriptor",
        "cross_cipher_linear_mismatch": "linear_only_mismatch",
        "same_summary_corrupted_linear": "correct_descriptor",
    }
    for condition in CONDITIONS:
        current = probabilities[condition]
        auc = float(binary_auc(labels, current))
        reference = source_controls[
            (replica, cipher, split, source_condition[condition])
        ]
        edge_gate, transition_gate = gates[condition]
        rows.append(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "cipher_key": cipher,
                "seed": seed,
                "split": split,
                "condition": condition,
                "rows": int(len(labels)),
                "auc": auc,
                "correct_minus_condition_auc": float(
                    binary_auc(labels, correct) - auc
                ),
                "effective_edge_gate": edge_gate,
                "effective_transition_gate": transition_gate,
                "edge_gate_delta_from_correct": abs(
                    edge_gate - gates["correct_descriptor"][0]
                ),
                "transition_gate_delta_from_correct": abs(
                    transition_gate - gates["correct_descriptor"][1]
                ),
                "mean_abs_probability_delta_from_correct": float(
                    np.mean(np.abs(current - correct))
                ),
                "maximum_abs_probability_delta_from_correct": float(
                    np.max(np.abs(current - correct))
                ),
                "probability_spearman_from_correct": _spearman_correlation(
                    correct, current
                ),
                "probabilities_sha256": _array_sha256(current),
                "source_condition": source_condition[condition],
                "source_auc": float(reference["auc"]),
                "source_probabilities_sha256": reference[
                    "probabilities_sha256"
                ],
                "source_auc_replay_delta": auc - float(reference["auc"]),
                "source_probability_hash_replayed": _array_sha256(current)
                == reference["probabilities_sha256"],
                "descriptor_summary_sha256": _tensor_sha256(
                    summaries[condition]
                ),
                "dataset_sha256": differential_dataset_sha256(dataset),
                "checkpoint_sha256": checkpoint["sha256"],
                "state_dict_sha256": checkpoint["state_dict_sha256"],
                "state_immutable": state_before == state_after,
                "runtime_structure_cipher_key": cipher,
                "runtime_structure_held_correct": True,
                "training_performed": False,
                "optimizer_steps": 0,
            }
        )
    return rows


def adjudicate_audit(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    collision_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_panels = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    collision_by_cipher = {
        str(row["cipher_key"]): row for row in collision_rows
    }
    finite_fields = (
        "auc",
        "effective_edge_gate",
        "effective_transition_gate",
        "mean_abs_probability_delta_from_correct",
        "maximum_abs_probability_delta_from_correct",
        "probability_spearman_from_correct",
        "source_auc_replay_delta",
    )
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "three_distinct_operator_collision_rows_complete": (
            len(collision_rows) == 3
            and set(collision_by_cipher) == set(EXPECTED_CIPHERS)
            and all(row.get("operator_changed") is True for row in collision_rows)
        ),
        "thirty_six_rows_in_twelve_complete_panels": (
            len(rows) == EXPECTED_ROWS
            and set(grouped) == expected_panels
            and all(
                set(conditions) == set(CONDITIONS)
                for conditions in grouped.values()
            )
        ),
        "two_epoch9_checkpoints_exact": (
            set(checkpoints) == set(EXPECTED_REPLICAS)
            and all(
                Path(str(checkpoints[replica]["path"])).is_file()
                and file_sha256(Path(str(checkpoints[replica]["path"])))
                == checkpoints[replica]["sha256"]
                and int(checkpoints[replica]["best_epoch"]) == 9
                for replica in EXPECTED_REPLICAS
            )
        ),
        "source_probability_and_auc_replay_exact": all(
            row.get("source_probability_hash_replayed") is True
            and abs(float(row.get("source_auc_replay_delta", math.inf))) <= 1e-12
            for row in rows
        ),
        "zero_update_correct_runtime_immutable": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("runtime_structure_held_correct") is True
            and row.get("runtime_structure_cipher_key") == row.get("cipher_key")
            and row.get("state_immutable") is True
            for row in rows
        ),
        "all_metrics_finite": all(
            all(math.isfinite(float(row.get(field, math.nan))) for field in finite_fields)
            for row in rows
        ),
    }

    panel_results: dict[str, dict[str, Any]] = {}
    collision_panels = 0
    scalar_rank_inert_panels = 0
    for replica, cipher, split in sorted(expected_panels):
        conditions = grouped.get((replica, cipher, split), {})
        correct = conditions.get("correct_descriptor", {})
        cross = conditions.get("cross_cipher_linear_mismatch", {})
        collision = conditions.get("same_summary_corrupted_linear", {})
        structure_collision = collision_by_cipher.get(cipher, {})
        collision_pass = (
            float(
                structure_collision.get(
                    "matrix_hamming_fraction", -math.inf
                )
            )
            >= MINIMUM_MATRIX_HAMMING_FRACTION
            and float(
                structure_collision.get(
                    "collision_summary_max_abs_delta", math.inf
                )
            )
            <= COLLISION_SUMMARY_TOLERANCE
            and float(collision.get("edge_gate_delta_from_correct", math.inf))
            == 0.0
            and float(
                collision.get("transition_gate_delta_from_correct", math.inf)
            )
            == 0.0
            and collision.get("probabilities_sha256")
            == correct.get("probabilities_sha256")
            and float(collision.get("correct_minus_condition_auc", math.inf))
            == 0.0
        )
        scalar_rank_inert = (
            float(cross.get("edge_gate_delta_from_correct", -math.inf))
            >= MINIMUM_CROSS_EDGE_GATE_DELTA
            and abs(float(cross.get("correct_minus_condition_auc", math.inf)))
            <= MAXIMUM_CROSS_AUC_DELTA_ABS
            and float(cross.get("probability_spearman_from_correct", -math.inf))
            >= MINIMUM_CROSS_SPEARMAN
        )
        collision_panels += int(collision_pass)
        scalar_rank_inert_panels += int(scalar_rank_inert)
        panel_results[f"replica{replica}_{cipher}_{split}"] = {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "matrix_hamming_fraction": structure_collision.get(
                "matrix_hamming_fraction"
            ),
            "collision_summary_max_abs_delta": structure_collision.get(
                "collision_summary_max_abs_delta"
            ),
            "cross_cipher_active_linear_dimension_count": structure_collision.get(
                "cross_cipher_active_linear_dimension_count"
            ),
            "cross_cipher_edge_gate_delta": cross.get(
                "edge_gate_delta_from_correct"
            ),
            "cross_cipher_auc_delta": cross.get(
                "correct_minus_condition_auc"
            ),
            "cross_cipher_probability_spearman": cross.get(
                "probability_spearman_from_correct"
            ),
            "collision_edge_gate_delta": collision.get(
                "edge_gate_delta_from_correct"
            ),
            "collision_probability_hash_equal": collision.get(
                "probabilities_sha256"
            )
            == correct.get("probabilities_sha256"),
            "collision_pass": collision_pass,
            "scalar_rank_inert": scalar_rank_inert,
        }

    collision_results = {
        "passing_panels": collision_panels,
        "expected_panels": 12,
        "passing_ciphers": sum(
            bool(row.get("summary_collision_exact"))
            and float(row.get("matrix_hamming_fraction", -math.inf))
            >= MINIMUM_MATRIX_HAMMING_FRACTION
            for row in collision_rows
        ),
        "expected_ciphers": 3,
        "mechanism_supported": collision_panels == 12,
        "minimum_matrix_hamming_fraction": min(
            float(row["matrix_hamming_fraction"]) for row in collision_rows
        ),
        "maximum_summary_delta": max(
            float(row["collision_summary_max_abs_delta"])
            for row in collision_rows
        ),
    }
    scalar_rank_results = {
        "passing_panels": scalar_rank_inert_panels,
        "expected_panels": 12,
        "mechanism_supported": scalar_rank_inert_panels == 12,
        "minimum_edge_gate_delta": min(
            float(panel["cross_cipher_edge_gate_delta"])
            for panel in panel_results.values()
        ),
        "maximum_auc_delta_abs": max(
            abs(float(panel["cross_cipher_auc_delta"]))
            for panel in panel_results.values()
        ),
        "minimum_probability_spearman": min(
            float(panel["cross_cipher_probability_spearman"])
            for panel in panel_results.values()
        ),
    }
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1ba_protocol_invalid"
        next_action = (
            "Repair only the failed source, checkpoint, data, collision, replay or "
            "immutability binding and rerun K1-BA unchanged."
        )
    elif collision_results["mechanism_supported"]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1ba_invariant_linear_summary_"
            "not_topology_identifying_supported"
        )
        next_action = (
            "Open K1-BB zero-update readiness for one shared position-preserving "
            "linear-operator token encoder over actual source/target connectivity. "
            "Keep K1-AZ checkpoints, backbone, residual paths and data fixed; require "
            "correct-versus-corrupted operator separation before any training."
        )
    elif scalar_rank_results["mechanism_supported"]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1ba_scalar_edge_modulation_rank_inertia_supported"
        )
        next_action = (
            "Retain the descriptor and open zero-update readiness for bounded "
            "channelwise edge modulation; do not train before exact replay and "
            "wrong-descriptor response gates pass."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1ba_mechanism_unresolved"
        next_action = (
            "Inspect the failed collision or rank metric and hold architecture changes; "
            "do not add data, pairs, capacity or remote scale."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol,
        "panel_results": panel_results,
        "collision_results": collision_results,
        "scalar_rank_results": scalar_rank_results,
        "remote_scale": "no",
        "claim_scope": (
            "Zero-training local descriptor-identifiability audit of two frozen "
            "K1-AZ checkpoints on the existing 2048/class/cipher, four-pair data; "
            "not formal scale, an attack, unseen-cipher transfer or SOTA evidence."
        ),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
    }


def run_audit(
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
        controls,
        summary_rows,
        checkpoints,
        source_controls,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    collision_summaries, collision_rows = derive_collision_controls(
        structures=structures,
        controls=controls,
        corruption_seed=int(config["evaluation"]["corruption_seed"]),
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-BA source preflight failed: {source_checks}")
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
    _write_json(
        output_root / "structure_collisions.json",
        {
            "run_id": RUN_ID,
            "source_structure_summaries": summary_rows,
            "collision_rows": collision_rows,
        },
    )

    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    source_config = load_k1az_config(
        project_root / str(config["source"]["config"])
    )
    rows: list[dict[str, Any]] = []
    for replica_config in source_config["replicas"]:
        replica = int(replica_config["replica"])
        model = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            readiness["model"],
            {"gate_hidden_dim": 12},
        ).to(device)
        incompatible = model.load_state_dict(
            checkpoints[replica]["state_dict"], strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("K1-BA strict checkpoint load drifted")
        model.eval()
        for cipher in EXPECTED_CIPHERS:
            correct = controls[cipher]["correct_descriptor"]
            cross = controls[cipher]["linear_only_mismatch"]
            if correct is None or cross is None:
                raise ValueError("K1-BA descriptor summary cannot be absent")
            summaries = {
                "correct_descriptor": correct,
                "cross_cipher_linear_mismatch": cross,
                "same_summary_corrupted_linear": collision_summaries[cipher],
            }
            seed = int(replica_config["dataset_seeds"][cipher])
            for split in FRESH_SPLITS:
                panel_rows = collect_panel_rows(
                    model=model,
                    dataset=datasets[(cipher, seed, split)],
                    structure=structures[cipher],
                    summaries=summaries,
                    source_controls=source_controls,
                    checkpoint=checkpoints[replica],
                    replica=replica,
                    cipher=cipher,
                    seed=seed,
                    split=split,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )
                rows.extend(panel_rows)
                _append_progress(
                    output_root / "progress.jsonl",
                    "panel_done",
                    replica=replica,
                    cipher_key=cipher,
                    split=split,
                    rows=len(panel_rows),
                )

    gate = adjudicate_audit(
        config=config,
        source_checks=source_checks,
        collision_rows=collision_rows,
        rows=rows,
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
        "collision_results": gate["collision_results"],
        "scalar_rank_results": gate["scalar_rank_results"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", rows)
    _write_panel_csv(output_root / "panel_summary.csv", gate["panel_results"])
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
        "checkpoint_manifest": checkpoint_manifest,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _load_checkpoints(
    *,
    source_root: Path,
    manifest: Mapping[str, Any],
    device: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, bool]]:
    checkpoints: dict[int, dict[str, Any]] = {}
    entries = manifest.get("entries", [])
    for entry in entries:
        replica = int(entry["replica"])
        path = source_root / "checkpoints" / f"replica{replica}_best.pt"
        payload = torch.load(path, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"K1-BA checkpoint {replica} lacks state_dict")
        checkpoints[replica] = {
            "replica": replica,
            "path": str(path),
            "sha256": file_sha256(path),
            "state_dict_sha256": tensor_mapping_sha256(state_dict),
            "best_epoch": int(payload["best_epoch"]),
            "run_id": str(payload["run_id"]),
            "state_dict": state_dict,
        }
    checks = {
        "source_checkpoint_manifest_complete": set(checkpoints)
        == set(EXPECTED_REPLICAS),
        "source_checkpoint_file_digests_exact": len(entries) == 2
        and all(
            checkpoints[int(entry["replica"])]["sha256"] == entry["sha256"]
            for entry in entries
        ),
        "source_checkpoint_state_digests_exact": len(entries) == 2
        and all(
            checkpoints[int(entry["replica"])]["state_dict_sha256"]
            == entry["state_dict_sha256"]
            for entry in entries
        ),
        "source_checkpoint_identity_exact": all(
            checkpoint["run_id"]
            == "i1_uknit_family_component_separated_structure_gate_k1az_"
            "2048_replica0_replica1_20260729"
            and checkpoint["best_epoch"] == 9
            for checkpoint in checkpoints.values()
        ),
    }
    return checkpoints, checks


def _index_source_controls(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        condition = str(row["condition"])
        if condition not in {"correct_descriptor", "linear_only_mismatch"}:
            continue
        key = (
            int(row["replica"]),
            str(row["cipher_key"]),
            str(row["split"]),
            condition,
        )
        if key in indexed:
            raise ValueError(f"duplicate K1-BA source control: {key}")
        indexed[key] = dict(row)
    return indexed


def _evaluate_probabilities(
    *,
    model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    structure: Any,
    summary: torch.Tensor,
    batch_size: int,
    device: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset.labels), batch_size):
            features = torch.as_tensor(
                np.array(dataset.features[start : start + batch_size], copy=True),
                dtype=torch.float32,
                device=device,
            )
            logits = model.logits_with_runtime(
                features,
                structure,
                apply_sboxes=True,
                transition_branch_enabled=True,
                gate_summary=summary,
                dual_path_enabled=True,
                component_separation_enabled=True,
            )
            outputs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def _path_gates(
    model: torch.nn.Module,
    structure: Any,
    summary: torch.Tensor,
) -> tuple[float, float]:
    edge, transition = model.effective_path_gates(
        structure,
        summary=summary,
        dual_path_enabled=True,
        component_separation_enabled=True,
    )
    return float(edge.detach()), float(transition.detach())


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(np.asarray(left, dtype=np.float64))
    right_rank = _average_ranks(np.asarray(right, dtype=np.float64))
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _write_panel_csv(path: Path, panels: Mapping[str, Mapping[str, Any]]) -> None:
    if not panels:
        raise ValueError("K1-BA panel results are empty")
    fields = list(next(iter(panels.values())))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(panels.values())


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        raise ValueError("K1-BA output already exists")


__all__ = [
    "COLLISION_SUMMARY_TOLERANCE",
    "CONDITIONS",
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_ROWS",
    "MAXIMUM_CROSS_AUC_DELTA_ABS",
    "MINIMUM_CROSS_EDGE_GATE_DELTA",
    "MINIMUM_CROSS_SPEARMAN",
    "MINIMUM_MATRIX_HAMMING_FRACTION",
    "ROOT",
    "RUN_ID",
    "adjudicate_audit",
    "collect_panel_rows",
    "derive_collision_controls",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
