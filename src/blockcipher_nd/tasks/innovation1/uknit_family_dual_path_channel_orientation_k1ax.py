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
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
    derive_structure_controls,
    load_and_validate_config as load_k1at_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    build_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1aw import (
    load_and_validate_config as load_k1aw_config,
    load_sources as load_k1aw_sources,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_dual_path_channel_orientation_k1ax_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_dual_path_channel_orientation_k1ax_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "5480b4ed2ab1725ec3b0d0e0b00dfc6211c690fef23e9c50abbae8f3645a3dfa"
)
EXPECTED_REPLICAS = (0, 1)
CONDITIONS = (
    "correct_descriptor",
    "full_mismatch",
    "sbox_only_mismatch",
    "linear_only_mismatch",
)
PATH_STATES = (
    "pure_base",
    "edge_only",
    "transition_only",
    "full_dual_path",
)
EXPECTED_ROWS = 48
EXPECTED_BATCH_SIZE = 64
REPLAY_TOLERANCE = 1e-7
MINIMUM_COMPONENT_ROUTING_PANELS = 10
PATH_HARM_MECHANISM_PANELS = 3
CANCELLATION_FRACTION_THRESHOLD = 0.5
CANCELLATION_MECHANISM_PANELS = 3


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AX config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AX identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_dual_path_channel_orientation_k1ax"
    ):
        raise ValueError("K1-AX experiment name drifted")
    if config.get("evaluation") != {
        "replicas": list(EXPECTED_REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "conditions": list(CONDITIONS),
        "path_states": list(PATH_STATES),
        "expected_rows": EXPECTED_ROWS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "training_performed": False,
        "optimizer_steps": 0,
        "full_forward_replay_tolerance": REPLAY_TOLERANCE,
    }:
        raise ValueError("K1-AX evaluation contract drifted")
    if config.get("gates") != {
        "minimum_component_routing_panels": MINIMUM_COMPONENT_ROUTING_PANELS,
        "path_harm_mechanism_panels": PATH_HARM_MECHANISM_PANELS,
        "cancellation_fraction_threshold": CANCELLATION_FRACTION_THRESHOLD,
        "cancellation_mechanism_panels": CANCELLATION_MECHANISM_PANELS,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AX gates drifted")
    if config.get("decision_order") != [
        "component_routing_misalignment",
        "learned_path_harm",
        "path_cancellation",
        "mechanism_unresolved",
    ]:
        raise ValueError("K1-AX decision order drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int, str], DiskDifferentialDataset],
    dict[str, Any],
    dict[str, dict[str, torch.Tensor | None]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {name: source_root / name for name in source["digests"]}
    source_gate = _read_json(paths["gate.json"])
    source_validation = _read_json(paths["validation.json"])
    source_results = _read_jsonl(paths["results.jsonl"])
    source_controls = _read_jsonl(paths["controls.jsonl"])
    source_manifest = _read_json(paths["checkpoint_manifest.json"])
    source_summaries = _read_json(paths["structure_summaries.json"])
    k1aw_config = load_k1aw_config(project_root / str(source["config"]))
    (
        readiness,
        _k1as,
        k1av,
        dataset_rows,
        datasets,
        _anchors,
        inherited_checks,
    ) = load_k1aw_sources(k1aw_config, project_root=project_root)
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
    checks = {
        "source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
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
        ),
        "source_two_training_rows_complete": len(source_results) == 2,
        "source_sixty_controls_complete": len(source_controls) == 60,
        "source_three_structure_summaries_complete": len(
            source_summaries.get("rows", [])
        )
        == 3,
        **{f"inherited_{name}": bool(value) for name, value in inherited_checks.items()},
        **{f"structure_{name}": bool(value) for name, value in structure_checks.items()},
        **checkpoint_checks,
    }
    return (
        readiness,
        k1av,
        dataset_rows,
        datasets,
        structures,
        controls,
        summary_rows,
        checkpoints,
        checks,
    )


def collect_panel_metrics(
    *,
    model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    structure: Any,
    summaries: Mapping[str, torch.Tensor | None],
    replica: int,
    cipher: str,
    seed: int,
    split: str,
    checkpoint: Mapping[str, Any],
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    descriptor_summaries = {
        condition: summaries[condition] for condition in CONDITIONS
    }
    if any(summary is None for summary in descriptor_summaries.values()):
        raise ValueError("K1-AX descriptor summary cannot be absent")
    gates = {
        condition: _path_gates(model, structure, summary)
        for condition, summary in descriptor_summaries.items()
    }
    state_before = tensor_mapping_sha256(model.state_dict())
    labels = np.asarray(dataset.labels, dtype=np.float32)
    arrays = {
        condition: {state: [] for state in PATH_STATES}
        for condition in CONDITIONS
    }
    actual_full = {condition: [] for condition in CONDITIONS}

    model.eval()
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
            for condition, summary in descriptor_summaries.items():
                edge_gate, transition_gate = gates[condition]
                edge = edge_gate * torch.tanh(edge_raw)
                transition = transition_gate * torch.tanh(transition_raw)
                states = {
                    "pure_base": base,
                    "edge_only": base + edge,
                    "transition_only": base + transition,
                    "full_dual_path": base + edge + transition,
                }
                for state, embedding in states.items():
                    logits = model.backbone.base.classifier(embedding).squeeze(1)
                    arrays[condition][state].append(logits.cpu().numpy())
                replay = model.logits_with_runtime(
                    features,
                    structure,
                    apply_sboxes=True,
                    transition_branch_enabled=True,
                    gate_summary=summary,
                    dual_path_enabled=True,
                ).squeeze(1)
                actual_full[condition].append(replay.cpu().numpy())

    state_after = tensor_mapping_sha256(model.state_dict())
    rows = []
    for condition in CONDITIONS:
        logits = {
            state: np.concatenate(chunks).astype(np.float64, copy=False)
            for state, chunks in arrays[condition].items()
        }
        probabilities = {state: _sigmoid(values) for state, values in logits.items()}
        actual = np.concatenate(actual_full[condition]).astype(np.float64, copy=False)
        base_probability = probabilities["pure_base"]
        edge_probability = probabilities["edge_only"]
        transition_probability = probabilities["transition_only"]
        full_probability = probabilities["full_dual_path"]
        label_sign = 2.0 * labels.astype(np.float64) - 1.0
        edge_standalone = label_sign * (edge_probability - base_probability)
        transition_standalone = label_sign * (
            transition_probability - base_probability
        )
        edge_full_context = label_sign * (full_probability - transition_probability)
        transition_full_context = label_sign * (full_probability - edge_probability)
        full_change = label_sign * (full_probability - base_probability)
        opposition = edge_standalone * transition_standalone < 0.0
        cancellation = opposition & (
            np.abs(full_change)
            < np.maximum(np.abs(edge_standalone), np.abs(transition_standalone))
        )
        interaction = (
            logits["full_dual_path"]
            - logits["edge_only"]
            - logits["transition_only"]
            + logits["pure_base"]
        )
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
                "path_aucs": {
                    state: float(binary_auc(labels, probabilities[state]))
                    for state in PATH_STATES
                },
                "effective_edge_gate": float(edge_gate),
                "effective_transition_gate": float(transition_gate),
                "mean_signed_edge_standalone_probability": float(
                    edge_standalone.mean()
                ),
                "mean_signed_transition_standalone_probability": float(
                    transition_standalone.mean()
                ),
                "mean_signed_edge_full_context_probability": float(
                    edge_full_context.mean()
                ),
                "mean_signed_transition_full_context_probability": float(
                    transition_full_context.mean()
                ),
                "edge_standalone_helpful_fraction": float(
                    np.mean(edge_standalone > 0.0)
                ),
                "transition_standalone_helpful_fraction": float(
                    np.mean(transition_standalone > 0.0)
                ),
                "edge_full_context_helpful_fraction": float(
                    np.mean(edge_full_context > 0.0)
                ),
                "transition_full_context_helpful_fraction": float(
                    np.mean(transition_full_context > 0.0)
                ),
                "mean_abs_edge_standalone_logit_delta": float(
                    np.mean(np.abs(logits["edge_only"] - logits["pure_base"]))
                ),
                "mean_abs_transition_standalone_logit_delta": float(
                    np.mean(
                        np.abs(logits["transition_only"] - logits["pure_base"])
                    )
                ),
                "mean_abs_classifier_interaction_logit": float(
                    np.mean(np.abs(interaction))
                ),
                "path_opposition_fraction": float(np.mean(opposition)),
                "path_cancellation_fraction": float(np.mean(cancellation)),
                "maximum_full_forward_replay_delta": float(
                    np.max(np.abs(logits["full_dual_path"] - actual))
                ),
                "path_logit_sha256": {
                    state: _array_sha256(values) for state, values in logits.items()
                },
                "dataset_sha256": differential_dataset_sha256(dataset),
                "checkpoint_sha256": checkpoint["sha256"],
                "state_dict_sha256": checkpoint["state_dict_sha256"],
                "runtime_structure_cipher_key": cipher,
                "runtime_structure_held_correct": True,
                "descriptor_summary_sha256": _tensor_sha256(
                    descriptor_summaries[condition]
                ),
                "state_immutable": state_before == state_after,
                "training_performed": False,
                "optimizer_steps": 0,
            }
        )
    _add_correct_relative_metrics(rows)
    return rows


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
        k1av,
        dataset_rows,
        datasets,
        structures,
        controls,
        summary_rows,
        checkpoints,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-AX source preflight failed: {source_checks}")
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
        output_root / "structure_summaries.json",
        {"run_id": RUN_ID, "rows": summary_rows},
    )

    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    source_config = load_k1aw_config(
        project_root / str(config["source"]["config"])
    )
    rows: list[dict[str, Any]] = []
    for replica_config in source_config["replicas"]:
        replica = int(replica_config["replica"])
        model = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            readiness["model"],
            k1av["model"],
        ).to(device)
        model.load_state_dict(checkpoints[replica]["state_dict"], strict=True)
        for cipher in EXPECTED_CIPHERS:
            seed = int(replica_config["dataset_seeds"][cipher])
            for split in FRESH_SPLITS:
                panel_rows = collect_panel_metrics(
                    model=model,
                    dataset=datasets[(cipher, seed, split)],
                    structure=structures[cipher],
                    summaries=controls[cipher],
                    replica=replica,
                    cipher=cipher,
                    seed=seed,
                    split=split,
                    checkpoint=checkpoints[replica],
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
    gate = adjudicate_audit(
        config=config,
        source_checks=source_checks,
        rows=rows,
        checkpoints=checkpoints,
    )
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
        "routing_results": gate["routing_results"],
        "path_harm_results": gate["path_harm_results"],
        "cancellation_results": gate["cancellation_results"],
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


def adjudicate_audit(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    expected_panels = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    finite_fields = (
        "effective_edge_gate",
        "effective_transition_gate",
        "mean_signed_edge_standalone_probability",
        "mean_signed_transition_standalone_probability",
        "mean_signed_edge_full_context_probability",
        "mean_signed_transition_full_context_probability",
        "path_opposition_fraction",
        "path_cancellation_fraction",
        "maximum_full_forward_replay_delta",
    )
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "forty_eight_rows_in_twelve_complete_panels": len(rows) == EXPECTED_ROWS
        and set(grouped) == expected_panels
        and all(set(conditions) == set(CONDITIONS) for conditions in grouped.values()),
        "two_frozen_checkpoints_bound": set(checkpoints) == set(EXPECTED_REPLICAS)
        and all(
            Path(str(checkpoints[replica]["path"])).is_file()
            and file_sha256(Path(str(checkpoints[replica]["path"])))
            == checkpoints[replica]["sha256"]
            for replica in EXPECTED_REPLICAS
        ),
        "zero_update_correct_runtime_immutable": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("runtime_structure_held_correct") is True
            and row.get("runtime_structure_cipher_key") == row.get("cipher_key")
            and row.get("state_immutable") is True
            for row in rows
        ),
        "all_metrics_finite_and_auc_complete": all(
            all(math.isfinite(float(row.get(field, math.nan))) for field in finite_fields)
            and set(row.get("path_aucs", {})) == set(PATH_STATES)
            and all(
                math.isfinite(float(value))
                for value in row.get("path_aucs", {}).values()
            )
            for row in rows
        ),
        "full_forward_replay_exact": all(
            float(row.get("maximum_full_forward_replay_delta", math.inf))
            <= REPLAY_TOLERANCE
            for row in rows
        ),
    }

    panel_results: dict[str, dict[str, Any]] = {}
    sbox_aligned = 0
    linear_aligned = 0
    edge_harm = 0
    transition_harm = 0
    cancellation_heavy = 0
    for key in sorted(expected_panels):
        replica, cipher, split = key
        conditions = grouped.get(key, {})
        correct = conditions.get("correct_descriptor", {})
        sbox = conditions.get("sbox_only_mismatch", {})
        linear = conditions.get("linear_only_mismatch", {})
        sbox_edge_delta = abs(
            float(sbox.get("effective_edge_gate", math.nan))
            - float(correct.get("effective_edge_gate", math.nan))
        )
        sbox_transition_delta = abs(
            float(sbox.get("effective_transition_gate", math.nan))
            - float(correct.get("effective_transition_gate", math.nan))
        )
        linear_edge_delta = abs(
            float(linear.get("effective_edge_gate", math.nan))
            - float(correct.get("effective_edge_gate", math.nan))
        )
        linear_transition_delta = abs(
            float(linear.get("effective_transition_gate", math.nan))
            - float(correct.get("effective_transition_gate", math.nan))
        )
        sbox_pass = sbox_transition_delta > sbox_edge_delta
        linear_pass = linear_edge_delta > linear_transition_delta
        edge_harmful = (
            float(correct.get("mean_signed_edge_full_context_probability", math.nan))
            < 0.0
        )
        transition_harmful = (
            float(
                correct.get(
                    "mean_signed_transition_full_context_probability", math.nan
                )
            )
            < 0.0
        )
        cancellation_fraction = float(
            correct.get("path_cancellation_fraction", math.nan)
        )
        cancellation_pass = cancellation_fraction >= CANCELLATION_FRACTION_THRESHOLD
        sbox_aligned += int(sbox_pass)
        linear_aligned += int(linear_pass)
        edge_harm += int(edge_harmful)
        transition_harm += int(transition_harmful)
        cancellation_heavy += int(cancellation_pass)
        panel_results[f"replica{replica}_{cipher}_{split}"] = {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "sbox_edge_gate_delta": sbox_edge_delta,
            "sbox_transition_gate_delta": sbox_transition_delta,
            "sbox_routing_aligned": sbox_pass,
            "linear_edge_gate_delta": linear_edge_delta,
            "linear_transition_gate_delta": linear_transition_delta,
            "linear_routing_aligned": linear_pass,
            "correct_edge_full_context_signed_probability": correct.get(
                "mean_signed_edge_full_context_probability"
            ),
            "correct_transition_full_context_signed_probability": correct.get(
                "mean_signed_transition_full_context_probability"
            ),
            "edge_harmful": edge_harmful,
            "transition_harmful": transition_harmful,
            "correct_path_cancellation_fraction": cancellation_fraction,
            "cancellation_heavy": cancellation_pass,
        }

    routing_results = {
        "sbox_aligned_panels": sbox_aligned,
        "linear_aligned_panels": linear_aligned,
        "expected_panels": 12,
        "minimum_required": MINIMUM_COMPONENT_ROUTING_PANELS,
        "sbox_pass": sbox_aligned >= MINIMUM_COMPONENT_ROUTING_PANELS,
        "linear_pass": linear_aligned >= MINIMUM_COMPONENT_ROUTING_PANELS,
    }
    path_harm_results = {
        "edge_harmful_panels": edge_harm,
        "transition_harmful_panels": transition_harm,
        "mechanism_threshold": PATH_HARM_MECHANISM_PANELS,
        "mechanism_supported": max(edge_harm, transition_harm)
        >= PATH_HARM_MECHANISM_PANELS,
    }
    cancellation_results = {
        "cancellation_heavy_panels": cancellation_heavy,
        "expected_panels": 12,
        "fraction_threshold": CANCELLATION_FRACTION_THRESHOLD,
        "mechanism_threshold": CANCELLATION_MECHANISM_PANELS,
        "mechanism_supported": cancellation_heavy >= CANCELLATION_MECHANISM_PANELS,
    }
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1ax_protocol_invalid"
        next_action = (
            "Repair only the failed source, checkpoint, data, row, replay or "
            "immutability binding and rerun K1-AX unchanged."
        )
    elif not routing_results["sbox_pass"] or not routing_results["linear_pass"]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1ax_component_routing_misalignment_supported"
        )
        next_action = (
            "Open K1-AY zero-update readiness for component-separated gate encoders: "
            "the GF(2) edge gate may read only the 18 linear-summary features and "
            "the S-box transition gate only the 16 S-box-summary features. Preserve "
            "the K1-AW backbone and checkpoints; do not train before exact migration, "
            "wiring and mismatch-isolation gates pass."
        )
    elif path_harm_results["mechanism_supported"]:
        status = "pass"
        decision = "innovation1_uknit_family_k1ax_learned_path_harm_supported"
        next_action = (
            "Freeze descriptor routing and audit the harmful residual path sign and "
            "representation at the same checkpoints; do not change data or scale."
        )
    elif cancellation_results["mechanism_supported"]:
        status = "pass"
        decision = "innovation1_uknit_family_k1ax_path_cancellation_supported"
        next_action = (
            "Open one zero-update readiness for a bounded path-interaction or "
            "contribution-normalization candidate; keep both encoders and all data fixed."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1ax_orientation_mechanism_unresolved"
        next_action = (
            "Hold the dual-path gate route and return to structure representation "
            "design; do not scale an unexplained mechanism."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol,
        "panel_results": panel_results,
        "routing_results": routing_results,
        "path_harm_results": path_harm_results,
        "cancellation_results": cancellation_results,
        "remote_scale": "no",
        "claim_scope": (
            "Zero-update attribution of two frozen K1-AW checkpoints on the same "
            "local 2048/class/cipher, four-pair data authority; not formal scale, "
            "an attack, arbitrary-SPN generalization, unseen-cipher transfer or SOTA."
        ),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
    }


def _load_checkpoints(
    *,
    source_root: Path,
    manifest: Mapping[str, Any],
    device: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, bool]]:
    checkpoints = {}
    entries = manifest.get("entries", [])
    for entry in entries:
        replica = int(entry["replica"])
        path = source_root / "checkpoints" / f"replica{replica}_best.pt"
        payload = torch.load(path, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"K1-AX checkpoint {replica} lacks state_dict")
        checkpoints[replica] = {
            "replica": replica,
            "path": str(path),
            "sha256": file_sha256(path),
            "state_dict_sha256": tensor_mapping_sha256(state_dict),
            "best_epoch": int(payload["best_epoch"]),
            "optimizer_steps": int(payload["optimizer_steps"]),
            "strict_state_dict_load": True,
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
        "source_checkpoint_epoch_and_steps_exact": all(
            checkpoint["best_epoch"] == 10
            and checkpoint["optimizer_steps"] == 1920
            for checkpoint in checkpoints.values()
        ),
    }
    return checkpoints, checks


def _path_gates(
    model: torch.nn.Module,
    structure: Any,
    summary: torch.Tensor | None,
) -> tuple[float, float]:
    if summary is None:
        raise ValueError("K1-AX path gate summary cannot be absent")
    edge, transition = model.effective_path_gates(
        structure,
        summary=summary,
        dual_path_enabled=True,
    )
    return float(edge.detach()), float(transition.detach())


def _add_correct_relative_metrics(rows: list[dict[str, Any]]) -> None:
    correct = next(row for row in rows if row["condition"] == "correct_descriptor")
    for row in rows:
        row["correct_minus_condition_full_auc"] = float(
            correct["path_aucs"]["full_dual_path"]
            - row["path_aucs"]["full_dual_path"]
        )
        row["absolute_edge_gate_delta_from_correct"] = abs(
            float(row["effective_edge_gate"])
            - float(correct["effective_edge_gate"])
        )
        row["absolute_transition_gate_delta_from_correct"] = abs(
            float(row["effective_transition_gate"])
            - float(correct["effective_transition_gate"])
        )


def _write_panel_csv(path: Path, panels: Mapping[str, Mapping[str, Any]]) -> None:
    if not panels:
        raise ValueError("K1-AX panel results are empty")
    fields = list(next(iter(panels.values())))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(panels.values())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor | None) -> str:
    if value is None:
        raise ValueError("K1-AX tensor digest cannot hash None")
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
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
        raise ValueError("K1-AX output already exists")


__all__ = [
    "CANCELLATION_FRACTION_THRESHOLD",
    "CANCELLATION_MECHANISM_PANELS",
    "CONDITIONS",
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_ROWS",
    "MINIMUM_COMPONENT_ROUTING_PANELS",
    "PATH_HARM_MECHANISM_PANELS",
    "PATH_STATES",
    "ROOT",
    "RUN_ID",
    "adjudicate_audit",
    "collect_panel_metrics",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
