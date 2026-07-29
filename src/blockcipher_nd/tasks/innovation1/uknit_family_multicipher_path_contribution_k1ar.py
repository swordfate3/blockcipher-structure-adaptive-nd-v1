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
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_inverse_norm_k1aq import (
    load_and_validate_config as load_k1aq_config,
    load_authority as load_k1aq_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    build_runtime_model,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao_training import (
    EXPECTED_BATCH_SIZE,
    EXPECTED_STEPS_PER_REPLICA,
    FRESH_SPLITS,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_multicipher_path_contribution_k1ar_"
    "replica0_replica1_replay_fix_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_multicipher_path_contribution_k1ar_"
    "replica0_replica1_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "b34bdc65a41ceb068f4c5ab0495da661ad71001716d80891952d296bccfb0c36"
)
CHECKPOINT_FAMILIES = ("equal_loss_k1ao", "inverse_norm_k1aq")
REPLICAS = (0, 1)
EXPECTED_ROWS = 24
REPLAY_TOLERANCE = 1e-7
MIDORI_GAIN_DELTA_MIN = 0.010
MIDORI_PANELS_REQUIRED = 4
NON_MIDORI_GAIN_DELTA_MAX = 0.0
NON_MIDORI_PANELS_REQUIRED = 6


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AR config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AR identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_multicipher_path_contribution_k1ar"
    ):
        raise ValueError("K1-AR experiment name drifted")
    evaluation = config.get("evaluation")
    if evaluation != {
        "checkpoint_families": list(CHECKPOINT_FAMILIES),
        "replicas": list(REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "paths": ["pure_base", "edge_fused", "full_transition_fused"],
        "expected_rows": EXPECTED_ROWS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "training_performed": False,
        "optimizer_steps": 0,
        "full_and_branch_replay_tolerance": REPLAY_TOLERANCE,
    }:
        raise ValueError("K1-AR evaluation contract drifted")
    if config.get("gates") != {
        "midori_transition_gain_delta_min": MIDORI_GAIN_DELTA_MIN,
        "midori_panels_required": MIDORI_PANELS_REQUIRED,
        "non_midori_transition_gain_delta_max": NON_MIDORI_GAIN_DELTA_MAX,
        "non_midori_panels_required": NON_MIDORI_PANELS_REQUIRED,
        "stable_transition_demand_supports_next_readiness_only": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AR gates drifted")
    if tuple(config.get("sources", {})) != CHECKPOINT_FAMILIES:
        raise ValueError("K1-AR checkpoint families drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    dict[tuple[str, int, str], DiskDifferentialDataset],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, bool],
]:
    k1aq_config = load_k1aq_config()
    (
        readiness,
        dataset_rows,
        datasets,
        _anchors,
        _baseline_rows,
        inherited_checks,
    ) = load_k1aq_authority(k1aq_config, project_root=project_root)
    checks = {
        f"inherited_{name}": bool(value)
        for name, value in inherited_checks.items()
    }
    checkpoints: dict[str, dict[int, dict[str, Any]]] = {}
    controls: dict[str, list[dict[str, Any]]] = {}

    expected_decisions = {
        "equal_loss_k1ao": (
            "innovation1_uknit_family_k1ao_shared_training_"
            "retention_and_semantics_failed"
        ),
        "inverse_norm_k1aq": (
            "innovation1_uknit_family_k1aq_inverse_norm_scaling_not_supported"
        ),
    }
    for family in CHECKPOINT_FAMILIES:
        source = config["sources"][family]
        root = project_root / str(source["root"])
        paths = {name: root / name for name in source["digests"]}
        checks[f"{family}_digests_exact"] = all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        )
        gate = _read_json(paths["gate.json"])
        validation = _read_json(paths["validation.json"])
        family_controls = _read_jsonl(paths["controls.jsonl"])
        manifest = _read_json(paths["checkpoint_manifest.json"])
        checks[f"{family}_gate_is_valid_hold"] = (
            gate.get("run_id") == source["run_id"]
            and gate.get("status") == "hold"
            and gate.get("decision") == expected_decisions[family]
            and not gate.get("failed_protocol_checks")
        )
        checks[f"{family}_validation_passes"] = (
            validation.get("status") == "pass"
            and not validation.get("errors")
        )
        checks[f"{family}_controls_complete"] = _controls_complete(
            family_controls
        )
        family_checkpoints = _load_checkpoints(
            source=source,
            paths=paths,
            manifest=manifest,
            device=device,
        )
        checks[f"{family}_checkpoints_complete"] = (
            set(family_checkpoints) == set(REPLICAS)
        )
        checkpoints[family] = family_checkpoints
        controls[family] = family_controls

    return readiness, datasets, checkpoints, controls, dataset_rows, checks


def collect_path_metrics(
    *,
    model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    structure: Any,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    model.eval()
    state_before = tensor_mapping_sha256(model.state_dict())
    labels = np.asarray(dataset.labels, dtype=np.float32)
    base_logits_chunks: list[np.ndarray] = []
    edge_logits_chunks: list[np.ndarray] = []
    full_logits_chunks: list[np.ndarray] = []
    base_probability_chunks: list[np.ndarray] = []
    edge_probability_chunks: list[np.ndarray] = []
    full_probability_chunks: list[np.ndarray] = []
    base_embeddings: list[torch.Tensor] = []
    gated_edges: list[torch.Tensor] = []
    gated_transitions: list[torch.Tensor] = []
    edge_gate = float(torch.tanh(model.backbone.residual_gate.detach()))
    transition_gate = float(torch.tanh(model.backbone.transition_gate.detach()))

    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            features = torch.as_tensor(
                np.array(dataset.features[start : start + batch_size], copy=True),
                dtype=torch.float32,
                device=device,
            )
            runtime = features.reshape(
                features.shape[0], -1, 2, structure.block_bits
            ).flip(-1)
            base = model.backbone.base.encode(runtime, structure)
            edge_raw = model.backbone.edge_residual_embedding(
                runtime,
                structure,
                apply_sboxes=True,
            )
            transition_raw = model.backbone.transition_embedding(
                runtime,
                structure,
                apply_sboxes=True,
            ).repeat(1, 3)
            edge = edge_gate * torch.tanh(edge_raw)
            transition = transition_gate * torch.tanh(transition_raw)
            base_logits = model.backbone.base.classifier(base).squeeze(1)
            edge_logits = model.backbone.base.classifier(base + edge).squeeze(1)
            full_logits = model.backbone.base.classifier(
                base + edge + transition
            ).squeeze(1)
            base_logits_chunks.append(base_logits.cpu().numpy())
            edge_logits_chunks.append(edge_logits.cpu().numpy())
            full_logits_chunks.append(full_logits.cpu().numpy())
            base_probability_chunks.append(torch.sigmoid(base_logits).cpu().numpy())
            edge_probability_chunks.append(torch.sigmoid(edge_logits).cpu().numpy())
            full_probability_chunks.append(torch.sigmoid(full_logits).cpu().numpy())
            base_embeddings.append(base.cpu())
            gated_edges.append(edge.cpu())
            gated_transitions.append(transition.cpu())

    base_logits = np.concatenate(base_logits_chunks).astype(np.float64)
    edge_logits = np.concatenate(edge_logits_chunks).astype(np.float64)
    full_logits = np.concatenate(full_logits_chunks).astype(np.float64)
    base_probability = np.concatenate(base_probability_chunks).astype(np.float64)
    edge_probability = np.concatenate(edge_probability_chunks).astype(np.float64)
    full_probability = np.concatenate(full_probability_chunks).astype(np.float64)
    edge_delta = edge_logits - base_logits
    transition_delta = full_logits - edge_logits
    label_sign = 2.0 * labels.astype(np.float64) - 1.0
    transition_signed_probability = label_sign * (
        full_probability - edge_probability
    )
    base_embedding = torch.cat(base_embeddings)
    gated_edge = torch.cat(gated_edges)
    gated_transition = torch.cat(gated_transitions)
    base_rms = _tensor_rms(base_embedding)
    edge_rms = _tensor_rms(gated_edge)
    transition_rms = _tensor_rms(gated_transition)
    state_after = tensor_mapping_sha256(model.state_dict())
    recomputed_full = _recompute_runtime_logits(
        model=model,
        dataset=dataset,
        structure=structure,
        transition_branch_enabled=True,
        batch_size=batch_size,
        device=device,
    )
    recomputed_edge = _recompute_runtime_logits(
        model=model,
        dataset=dataset,
        structure=structure,
        transition_branch_enabled=False,
        batch_size=batch_size,
        device=device,
    )
    return {
        "rows": int(len(labels)),
        "pure_base_auc": float(binary_auc(labels, base_probability)),
        "edge_fused_auc": float(binary_auc(labels, edge_probability)),
        "full_auc": float(binary_auc(labels, full_probability)),
        "edge_gain_auc": float(
            binary_auc(labels, edge_probability)
            - binary_auc(labels, base_probability)
        ),
        "transition_gain_auc": float(
            binary_auc(labels, full_probability)
            - binary_auc(labels, edge_probability)
        ),
        "edge_delta_auc": float(binary_auc(labels, _sigmoid(edge_delta))),
        "transition_delta_auc": float(
            binary_auc(labels, _sigmoid(transition_delta))
        ),
        "mean_signed_transition_probability": float(
            transition_signed_probability.mean()
        ),
        "transition_helpful_fraction": float(
            np.mean(transition_signed_probability > 0.0)
        ),
        "mean_transition_mse_reduction": float(
            np.mean(np.square(edge_probability - labels))
            - np.mean(np.square(full_probability - labels))
        ),
        "base_embedding_rms": base_rms,
        "gated_edge_embedding_rms": edge_rms,
        "gated_transition_embedding_rms": transition_rms,
        "transition_to_base_rms_ratio": transition_rms / max(base_rms, 1e-12),
        "transition_to_edge_rms_ratio": transition_rms / max(edge_rms, 1e-12),
        "effective_edge_gate": edge_gate,
        "effective_transition_gate": transition_gate,
        "base_logits_sha256": _array_sha256(base_logits),
        "edge_logits_sha256": _array_sha256(edge_logits),
        "full_logits_sha256": _array_sha256(full_logits),
        "max_abs_full_forward_replay_delta": float(
            np.max(np.abs(full_logits - recomputed_full))
        ),
        "max_abs_edge_forward_replay_delta": float(
            np.max(np.abs(edge_logits - recomputed_edge))
        ),
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_immutable": state_before == state_after,
        "training_performed": False,
        "optimizer_steps": 0,
    }


def evaluate_paths(
    *,
    config: Mapping[str, Any],
    readiness: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    checkpoints: Mapping[str, Mapping[int, Mapping[str, Any]]],
    controls: Mapping[str, Sequence[Mapping[str, Any]]],
    device: str,
) -> list[dict[str, Any]]:
    k1aq_config = load_k1aq_config()
    replica_configs = {
        int(row["replica"]): row for row in k1aq_config["replicas"]
    }
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    structures = {
        cipher: build_runtime_model(
            cipher_configs[cipher], readiness["model"]
        ).runtime_structure
        for cipher in EXPECTED_CIPHERS
    }
    source_controls = {
        family: _group_controls(rows) for family, rows in controls.items()
    }
    results = []
    for family in CHECKPOINT_FAMILIES:
        for replica in REPLICAS:
            model = build_runtime_model(
                cipher_configs[EXPECTED_CIPHERS[0]], readiness["model"]
            ).to(device)
            checkpoint = checkpoints[family][replica]
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            loaded_state = tensor_mapping_sha256(model.state_dict())
            for cipher in EXPECTED_CIPHERS:
                seed = int(replica_configs[replica]["dataset_seeds"][cipher])
                for split in FRESH_SPLITS:
                    dataset = datasets[(cipher, seed, split)]
                    metrics = collect_path_metrics(
                        model=model,
                        dataset=dataset,
                        structure=structures[cipher],
                        batch_size=int(config["evaluation"]["batch_size"]),
                        device=device,
                    )
                    control_rows = source_controls[family][
                        (replica, cipher, split)
                    ]
                    full_source_auc = float(
                        control_rows["correct_runtime"]["auc"]
                    )
                    edge_source_auc = float(
                        control_rows[
                            "transition_branch_off_same_checkpoint"
                        ]["auc"]
                    )
                    results.append(
                        {
                            "run_id": RUN_ID,
                            "checkpoint_family": family,
                            "source_run_id": config["sources"][family]["run_id"],
                            "replica": replica,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "checkpoint_sha256": checkpoint["sha256"],
                            "state_dict_sha256": loaded_state,
                            "source_full_auc": full_source_auc,
                            "source_edge_fused_auc": edge_source_auc,
                            "full_auc_replay_delta": metrics["full_auc"]
                            - full_source_auc,
                            "edge_auc_replay_delta": metrics["edge_fused_auc"]
                            - edge_source_auc,
                            **metrics,
                        }
                    )
    return results


def adjudicate(
    *,
    source_checks: Mapping[str, bool],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = {
        (family, replica, cipher, split)
        for family in CHECKPOINT_FAMILIES
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    row_map = {
        (
            str(row["checkpoint_family"]),
            int(row["replica"]),
            str(row["cipher_key"]),
            str(row["split"]),
        ): row
        for row in rows
    }
    protocol_checks = {
        **{name: bool(value) for name, value in source_checks.items()},
        "result_rows_complete": len(rows) == EXPECTED_ROWS
        and set(row_map) == expected_keys,
        "all_rows_zero_step": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            for row in rows
        ),
        "all_states_immutable": all(row.get("state_immutable") is True for row in rows),
        "all_forward_replays_exact": all(
            abs(float(row.get("full_auc_replay_delta", math.inf)))
            <= REPLAY_TOLERANCE
            and abs(float(row.get("edge_auc_replay_delta", math.inf)))
            <= REPLAY_TOLERANCE
            and float(row.get("max_abs_full_forward_replay_delta", math.inf))
            <= REPLAY_TOLERANCE
            and float(row.get("max_abs_edge_forward_replay_delta", math.inf))
            <= REPLAY_TOLERANCE
            for row in rows
        ),
    }
    panel_results: dict[str, dict[str, Any]] = {}
    midori_pass_count = 0
    non_midori_pass_count = 0
    if protocol_checks["result_rows_complete"]:
        for replica in REPLICAS:
            for cipher in EXPECTED_CIPHERS:
                for split in FRESH_SPLITS:
                    baseline = row_map[("equal_loss_k1ao", replica, cipher, split)]
                    candidate = row_map[("inverse_norm_k1aq", replica, cipher, split)]
                    gain_delta = float(candidate["transition_gain_auc"]) - float(
                        baseline["transition_gain_auc"]
                    )
                    is_midori = cipher == "midori64"
                    passed = (
                        gain_delta >= MIDORI_GAIN_DELTA_MIN
                        if is_midori
                        else gain_delta <= NON_MIDORI_GAIN_DELTA_MAX
                    )
                    midori_pass_count += int(is_midori and passed)
                    non_midori_pass_count += int(not is_midori and passed)
                    key = f"replica{replica}_{cipher}_{split}"
                    panel_results[key] = {
                        "equal_loss_transition_gain_auc": float(
                            baseline["transition_gain_auc"]
                        ),
                        "inverse_norm_transition_gain_auc": float(
                            candidate["transition_gain_auc"]
                        ),
                        "transition_gain_delta": gain_delta,
                        "equal_loss_transition_delta_auc": float(
                            baseline["transition_delta_auc"]
                        ),
                        "inverse_norm_transition_delta_auc": float(
                            candidate["transition_delta_auc"]
                        ),
                        "equal_loss_transition_to_base_rms_ratio": float(
                            baseline["transition_to_base_rms_ratio"]
                        ),
                        "inverse_norm_transition_to_base_rms_ratio": float(
                            candidate["transition_to_base_rms_ratio"]
                        ),
                        "equal_loss_signed_probability_improvement": float(
                            baseline["mean_signed_transition_probability"]
                        ),
                        "inverse_norm_signed_probability_improvement": float(
                            candidate["mean_signed_transition_probability"]
                        ),
                        "direction_pass": passed,
                    }
    failed_protocol_checks = [
        name for name, value in protocol_checks.items() if not value
    ]
    heterogeneous_support = (
        not failed_protocol_checks
        and midori_pass_count == MIDORI_PANELS_REQUIRED
        and non_midori_pass_count >= NON_MIDORI_PANELS_REQUIRED
    )
    if failed_protocol_checks:
        status = "fail"
        decision = "innovation1_uknit_family_k1ar_protocol_invalid"
        next_action = "Repair only the failed source binding or replay check."
    elif heterogeneous_support:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1ar_heterogeneous_transition_demand_supported"
        )
        next_action = (
            "Design one local readiness test for a bounded structure-derived "
            "transition gate without cipher IDs, per-cipher heads, adapters, or experts."
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_k1ar_heterogeneous_transition_demand_not_supported"
        )
        next_action = (
            "Reject conditional gate design and audit transition projection geometry."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol_checks,
        "panel_results": panel_results,
        "midori_direction_pass_count": midori_pass_count,
        "midori_direction_required": MIDORI_PANELS_REQUIRED,
        "non_midori_direction_pass_count": non_midori_pass_count,
        "non_midori_direction_required": NON_MIDORI_PANELS_REQUIRED,
        "heterogeneous_transition_demand_supported": heterogeneous_support,
        "remote_scale": "no",
        "blocked_actions": [
            "loss-scale tuning or PCGrad",
            "16 pairs, larger data, epochs, width, or remote GPU",
            "cipher IDs, per-cipher heads, adapters, MoE, or experts",
        ],
        "next_action": next_action,
        "claim_scope": (
            "Zero-training replay of two local 2048/class/cipher, 4-pair shared "
            "checkpoint families; not formal scale, an attack, family-general proof, "
            "or SOTA evidence."
        ),
    }


def run_audit(
    *,
    config_path: Path = CONFIG_PATH,
    output_root: Path,
    device: str = "cpu",
    project_root: Path = ROOT,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"K1-AR output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    _append_progress(output_root / "progress.jsonl", "run_started")
    config = load_and_validate_config(config_path)
    (
        readiness,
        datasets,
        checkpoints,
        controls,
        dataset_rows,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    _write_json(
        output_root / "preflight.json",
        {
            "run_id": RUN_ID,
            "status": "pass" if all(source_checks.values()) else "fail",
            "config_sha256": file_sha256(config_path),
            "source_checks": source_checks,
            "training_performed": False,
            "optimizer_steps": 0,
        },
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "source_entries": [
            {
                key: value
                for key, value in checkpoints[family][replica].items()
                if key != "state_dict"
            }
            for family in CHECKPOINT_FAMILIES
            for replica in REPLICAS
        ],
    }
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    rows = evaluate_paths(
        config=config,
        readiness=readiness,
        datasets=datasets,
        checkpoints=checkpoints,
        controls=controls,
        device=device,
    )
    gate = adjudicate(source_checks=source_checks, rows=rows)
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "result_rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "optimizer_steps": 0,
        "errors": gate["failed_protocol_checks"],
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "midori_direction_pass_count": gate["midori_direction_pass_count"],
        "non_midori_direction_pass_count": gate[
            "non_midori_direction_pass_count"
        ],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", rows)
    _write_comparison_csv(output_root / "comparison.csv", gate["panel_results"])
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    return {
        "preflight": _read_json(output_root / "preflight.json"),
        "results": rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _load_checkpoints(
    *,
    source: Mapping[str, Any],
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    device: str,
) -> dict[int, dict[str, Any]]:
    entries = {int(row["replica"]): row for row in manifest.get("entries", [])}
    checkpoints = {}
    for replica in REPLICAS:
        name = f"checkpoints/replica{replica}_best.pt"
        path = paths[name]
        payload = torch.load(path, map_location=device, weights_only=False)
        state_dict = payload["state_dict"]
        state_sha = tensor_mapping_sha256(state_dict)
        entry = entries.get(replica, {})
        if (
            payload.get("run_id") != source["run_id"]
            or int(payload.get("replica", -1)) != replica
            or int(payload.get("optimizer_steps", -1)) != EXPECTED_STEPS_PER_REPLICA
            or entry.get("sha256") != file_sha256(path)
            or entry.get("state_dict_sha256") != state_sha
        ):
            raise ValueError(f"K1-AR checkpoint binding failed for {name}")
        checkpoints[replica] = {
            "checkpoint_family": next(
                family
                for family in CHECKPOINT_FAMILIES
                if source["run_id"].endswith(
                    "shared_weight_k1ao_2048_replica0_replica1_20260729"
                )
                == (family == "equal_loss_k1ao")
            ),
            "source_run_id": source["run_id"],
            "replica": replica,
            "path": str(path),
            "sha256": file_sha256(path),
            "state_dict_sha256": state_sha,
            "best_epoch": int(payload["best_epoch"]),
            "optimizer_steps": int(payload["optimizer_steps"]),
            "strict_state_dict_load": True,
            "state_dict": state_dict,
        }
    return checkpoints


def _controls_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    grouped = _group_controls(rows)
    expected = {
        (replica, cipher, split)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    required = {
        "correct_runtime",
        "wrong_sbox_same_checkpoint",
        "transition_branch_off_same_checkpoint",
    }
    return (
        len(rows) == 36
        and set(grouped) == expected
        and all(set(conditions) == required for conditions in grouped.values())
    )


def _group_controls(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], dict[str, Mapping[str, Any]]]:
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    return grouped


def _recompute_runtime_logits(
    *,
    model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    structure: Any,
    transition_branch_enabled: bool,
    batch_size: int,
    device: str,
) -> np.ndarray:
    chunks = []
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
                transition_branch_enabled=transition_branch_enabled,
            )
            chunks.append(logits.squeeze(1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _tensor_rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(values))).item())


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


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
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_comparison_csv(
    path: Path, panels: Mapping[str, Mapping[str, Any]]
) -> None:
    fieldnames = ["panel", *next(iter(panels.values())).keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel, values in sorted(panels.items()):
            writer.writerow({"panel": panel, **values})


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"run_id": RUN_ID, "event": event, "time": time.time(), **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


__all__ = [
    "CONFIG_PATH",
    "RUN_ID",
    "adjudicate",
    "collect_path_metrics",
    "evaluate_paths",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
