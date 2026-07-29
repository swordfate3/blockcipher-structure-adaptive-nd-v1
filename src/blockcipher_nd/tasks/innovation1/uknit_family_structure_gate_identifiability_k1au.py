from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1as import (
    build_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
    MISMATCH_CONDITIONS,
    derive_structure_controls,
    load_and_validate_config as load_k1at_config,
    load_sources as load_k1at_sources,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_structure_gate_identifiability_k1au_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_structure_gate_identifiability_k1au_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "275bc499134989c36000bd9be45d68e4af14be52e5136b963cdc9150677fd8da"
)
EXPECTED_REPLICAS = (0, 1)
REPLICA_DATASET_SEEDS = {
    0: {"uknit64": 3, "midori64": 6, "dialga128": 0},
    1: {"uknit64": 4, "midori64": 7, "dialga128": 1},
}
ROWS_PER_SPLIT = 32
EXPECTED_RESULT_ROWS = 6
EXPECTED_CONTROL_ROWS = 36
MINIMUM_RAW_DISTANCE = 1e-3
MINIMUM_HIDDEN_DISTANCE = 1e-4
REQUIRED_HIDDEN_RANK = 2
MINIMUM_COMPONENT_JACOBIAN = 1e-6
MINIMUM_PROJECTION_ALIGNMENT = 0.1
MINIMUM_ALIGNED_PANELS = 15
REQUIRED_GATE_RANK_CORRELATION = 1.0
MINIMUM_CROSS_REPLICA_JACOBIAN_COSINE = 0.5


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AU config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AU identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_structure_gate_identifiability_k1au"
    ):
        raise ValueError("K1-AU experiment name drifted")
    if config.get("audit") != {
        "replicas": list(EXPECTED_REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "mismatch_conditions": list(MISMATCH_CONDITIONS),
        "rows_per_split": ROWS_PER_SPLIT,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "expected_control_rows": EXPECTED_CONTROL_ROWS,
        "training_performed": False,
        "optimizer_steps": 0,
        "device": "cpu",
    }:
        raise ValueError("K1-AU audit protocol drifted")
    if config.get("gates") != {
        "minimum_raw_summary_l2_distance": MINIMUM_RAW_DISTANCE,
        "minimum_hidden_l2_distance": MINIMUM_HIDDEN_DISTANCE,
        "required_centered_correct_hidden_rank": REQUIRED_HIDDEN_RANK,
        "minimum_component_jacobian_l2": MINIMUM_COMPONENT_JACOBIAN,
        "minimum_projection_alignment_cosine": MINIMUM_PROJECTION_ALIGNMENT,
        "minimum_aligned_mismatch_panels": MINIMUM_ALIGNED_PANELS,
        "required_cross_replica_gate_rank_correlation": REQUIRED_GATE_RANK_CORRELATION,
        "minimum_cross_replica_jacobian_cosine": (
            MINIMUM_CROSS_REPLICA_JACOBIAN_COSINE
        ),
        "remote_scale": "no",
    }:
        raise ValueError("K1-AU gates drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[tuple[str, int, str], Any],
    dict[str, Any],
    dict[str, dict[str, torch.Tensor | None]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    source_paths = {name: source_root / name for name in source["digests"]}
    source_gate = _read_json(source_paths["gate.json"])
    source_validation = _read_json(source_paths["validation.json"])
    source_results = _read_jsonl(source_paths["results.jsonl"])
    source_controls = _read_jsonl(source_paths["controls.jsonl"])
    source_manifest = _read_json(source_paths["checkpoint_manifest.json"])
    source_summaries = _read_json(source_paths["structure_summaries.json"])
    k1at_config = load_k1at_config(project_root / str(source["config"]))
    (
        readiness_config,
        k1as_config,
        _dataset_rows,
        datasets,
        _anchors,
        inherited_source_checks,
    ) = load_k1at_sources(k1at_config, project_root=project_root)
    structures, structure_controls, replay_summaries, structure_checks = (
        derive_structure_controls(
            readiness_config=readiness_config,
            config=k1at_config,
        )
    )
    checkpoints, checkpoint_rows, checkpoint_checks = _load_checkpoints(
        manifest=source_manifest,
        source_root=source_root,
        device=device,
    )
    checks = {
        "source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in source_paths.items()
        ),
        "k1at_gate_is_valid_hold": (
            source_gate.get("run_id") == source["run_id"]
            and source_gate.get("status") == "hold"
            and source_gate.get("decision") == source["required_decision"]
            and not source_gate.get("failed_protocol_checks")
            and source_gate.get("descriptor_mismatch_gate_all") is False
            and source_gate.get("remote_scale") == "no"
        ),
        "k1at_validation_passes": (
            source_validation.get("run_id") == source["run_id"]
            and source_validation.get("status") == "pass"
            and not source_validation.get("errors")
        ),
        "k1at_two_training_rows_bound": (
            len(source_results) == 2
            and {int(row["replica"]) for row in source_results}
            == set(EXPECTED_REPLICAS)
        ),
        "k1at_sixty_controls_bound": len(source_controls) == 60,
        "k1at_three_summary_rows_bound": (
            source_summaries.get("run_id") == source["run_id"]
            and len(source_summaries.get("rows", [])) == 3
            and len(replay_summaries) == 3
        ),
        **{
            f"k1at_source_{name}": bool(value)
            for name, value in inherited_source_checks.items()
        },
        **{
            f"structure_{name}": bool(value) for name, value in structure_checks.items()
        },
        **checkpoint_checks,
    }
    return (
        readiness_config,
        k1as_config,
        datasets,
        structures,
        structure_controls,
        checkpoints,
        checkpoint_rows,
        checks,
    )


def audit_layers(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    k1as_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, Any],
    structure_controls: Mapping[str, Mapping[str, torch.Tensor | None]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    results: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    hidden_by_replica: dict[int, dict[str, torch.Tensor]] = {}
    jacobian_by_replica: dict[int, dict[str, torch.Tensor]] = {}
    gate_by_replica: dict[int, dict[str, float]] = {}
    hidden_rank_by_replica: dict[int, int] = {}

    for replica in EXPECTED_REPLICAS:
        model = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            readiness_config["model"],
            k1as_config["model"],
        ).to(device)
        model.load_state_dict(checkpoints[replica]["state_dict"], strict=True)
        model.eval()
        state_sha256 = tensor_mapping_sha256(model.state_dict())
        hidden_by_replica[replica] = {}
        jacobian_by_replica[replica] = {}
        gate_by_replica[replica] = {}
        result_indices: list[int] = []

        for cipher in EXPECTED_CIPHERS:
            correct_summary = structure_controls[cipher]["correct_descriptor"]
            if correct_summary is None:
                raise RuntimeError("K1-AU correct summary is unavailable")
            metrics = _descriptor_metrics(model, correct_summary, structures[cipher])
            hidden_by_replica[replica][cipher] = metrics["hidden_tensor"]
            jacobian_by_replica[replica][cipher] = metrics["jacobian_tensor"]
            gate_by_replica[replica][cipher] = metrics["effective_gate"]
            result_indices.append(len(results))
            results.append(
                {
                    "run_id": RUN_ID,
                    "replica": replica,
                    "cipher_key": cipher,
                    "checkpoint_sha256": checkpoints[replica]["sha256"],
                    "state_dict_sha256": state_sha256,
                    "correct_summary_sha256": _tensor_sha256(correct_summary),
                    "hidden_embedding": metrics["hidden_tensor"].tolist(),
                    "hidden_embedding_l2": metrics["hidden_l2"],
                    "projection_value": metrics["projection_value"],
                    "effective_gate": metrics["effective_gate"],
                    "sbox_jacobian_l2": metrics["sbox_jacobian_l2"],
                    "linear_jacobian_l2": metrics["linear_jacobian_l2"],
                    "jacobian": metrics["jacobian_tensor"].tolist(),
                    "uses_cipher_identity": bool(model.uses_cipher_identity),
                    "structure_gate_uses_cipher_identity": bool(
                        model.structure_gate_uses_cipher_identity
                    ),
                    "structure_gate_shared": bool(model.structure_gate_shared),
                    "state_immutable": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )

            for condition in MISMATCH_CONDITIONS:
                mismatch_summary = structure_controls[cipher][condition]
                if mismatch_summary is None:
                    raise RuntimeError("K1-AU mismatch summary is unavailable")
                mismatch_metrics = _descriptor_metrics(
                    model, mismatch_summary, structures[cipher]
                )
                hidden_delta = (
                    mismatch_metrics["hidden_tensor"] - metrics["hidden_tensor"]
                )
                final_weight = model.backbone.structure_gate.network[2].weight.squeeze(
                    0
                )
                alignment = abs(
                    float(
                        F.cosine_similarity(
                            hidden_delta,
                            final_weight,
                            dim=0,
                            eps=1e-12,
                        ).detach()
                    )
                )
                seed = int(REPLICA_DATASET_SEEDS[replica][cipher])
                for split in FRESH_SPLITS:
                    dataset = datasets[(cipher, seed, split)]
                    logit_metrics = _logit_sensitivity(
                        model=model,
                        structure=structures[cipher],
                        correct_summary=correct_summary,
                        mismatch_summary=mismatch_summary,
                        features=np.array(dataset.features[:ROWS_PER_SPLIT], copy=True),
                        device=device,
                    )
                    controls.append(
                        {
                            "run_id": RUN_ID,
                            "replica": replica,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "rows_inspected": ROWS_PER_SPLIT,
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "checkpoint_sha256": checkpoints[replica]["sha256"],
                            "correct_summary_sha256": _tensor_sha256(correct_summary),
                            "mismatch_summary_sha256": _tensor_sha256(mismatch_summary),
                            "raw_summary_l2_distance": float(
                                torch.linalg.vector_norm(
                                    mismatch_summary - correct_summary
                                )
                            ),
                            "hidden_l2_distance": float(
                                torch.linalg.vector_norm(hidden_delta)
                            ),
                            "projection_alignment_abs_cosine": alignment,
                            "projection_value_delta": abs(
                                mismatch_metrics["projection_value"]
                                - metrics["projection_value"]
                            ),
                            "effective_gate_delta": abs(
                                mismatch_metrics["effective_gate"]
                                - metrics["effective_gate"]
                            ),
                            **logit_metrics,
                            "runtime_structure_held_correct": True,
                            "state_immutable": tensor_mapping_sha256(model.state_dict())
                            == state_sha256,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )

        correct_hidden = torch.stack(
            [hidden_by_replica[replica][cipher] for cipher in EXPECTED_CIPHERS]
        )
        centered = correct_hidden - correct_hidden.mean(dim=0, keepdim=True)
        hidden_rank = int(torch.linalg.matrix_rank(centered, atol=1e-6, rtol=1e-5))
        hidden_rank_by_replica[replica] = hidden_rank
        for index in result_indices:
            results[index]["centered_correct_hidden_rank"] = hidden_rank
            results[index]["state_immutable"] = (
                tensor_mapping_sha256(model.state_dict()) == state_sha256
            )

    gate_rank_correlation = _rank_correlation(
        [gate_by_replica[0][cipher] for cipher in EXPECTED_CIPHERS],
        [gate_by_replica[1][cipher] for cipher in EXPECTED_CIPHERS],
    )
    jacobian_cosines = {
        cipher: float(
            F.cosine_similarity(
                jacobian_by_replica[0][cipher],
                jacobian_by_replica[1][cipher],
                dim=0,
                eps=1e-12,
            )
        )
        for cipher in EXPECTED_CIPHERS
    }
    cross_replica = {
        "gate_rank_correlation": gate_rank_correlation,
        "jacobian_cosine_by_cipher": jacobian_cosines,
        "gate_by_replica": gate_by_replica,
        "hidden_rank_by_replica": hidden_rank_by_replica,
    }
    for row in results:
        cipher = str(row["cipher_key"])
        row["cross_replica_gate_rank_correlation"] = gate_rank_correlation
        row["cross_replica_jacobian_cosine"] = jacobian_cosines[cipher]
    return results, controls, cross_replica


def adjudicate(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    results: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    cross_replica: Mapping[str, Any],
) -> dict[str, Any]:
    unique_controls = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["condition"])): row
        for row in controls
    }
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "six_result_rows_complete": len(results) == EXPECTED_RESULT_ROWS,
        "thirty_six_control_rows_complete": len(controls) == EXPECTED_CONTROL_ROWS,
        "eighteen_unique_mismatch_panels_complete": len(unique_controls) == 18,
        "two_checkpoints_strictly_bound": set(checkpoints) == set(EXPECTED_REPLICAS),
        "all_rows_zero_training_and_immutable": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("state_immutable") is True
            for row in [*results, *controls]
        ),
        "runtime_structure_held_correct": all(
            row.get("runtime_structure_held_correct") is True for row in controls
        ),
        "shared_gate_has_no_cipher_identity": all(
            row.get("uses_cipher_identity") is False
            and row.get("structure_gate_uses_cipher_identity") is False
            and row.get("structure_gate_shared") is True
            for row in results
        ),
        "all_metrics_finite": _all_numeric_metrics_finite(results, controls),
    }
    raw_preserved = all(
        float(row["raw_summary_l2_distance"]) >= MINIMUM_RAW_DISTANCE
        for row in unique_controls.values()
    )
    hidden_preserved = all(
        float(row["hidden_l2_distance"]) >= MINIMUM_HIDDEN_DISTANCE
        for row in unique_controls.values()
    )
    hidden_rank_preserved = all(
        int(row["centered_correct_hidden_rank"]) == REQUIRED_HIDDEN_RANK
        for row in results
    )
    component_jacobians_present = all(
        float(row["sbox_jacobian_l2"]) >= MINIMUM_COMPONENT_JACOBIAN
        and float(row["linear_jacobian_l2"]) >= MINIMUM_COMPONENT_JACOBIAN
        for row in results
    )
    aligned_count = sum(
        float(row["projection_alignment_abs_cosine"]) >= MINIMUM_PROJECTION_ALIGNMENT
        for row in unique_controls.values()
    )
    projection_alignment_pass = aligned_count >= MINIMUM_ALIGNED_PANELS
    gate_rank_correlation = float(cross_replica["gate_rank_correlation"])
    rank_stable = gate_rank_correlation >= REQUIRED_GATE_RANK_CORRELATION
    jacobian_cosines = cross_replica["jacobian_cosine_by_cipher"]
    jacobian_stable = all(
        float(jacobian_cosines[cipher]) >= MINIMUM_CROSS_REPLICA_JACOBIAN_COSINE
        for cipher in EXPECTED_CIPHERS
    )
    research_checks = {
        "raw_summary_distances_preserved": raw_preserved,
        "hidden_distances_preserved": hidden_preserved,
        "correct_hidden_rank_two_both_replicas": hidden_rank_preserved,
        "both_component_jacobians_present": component_jacobians_present,
        "final_projection_aligned_at_least_15_of_18": projection_alignment_pass,
        "cross_replica_gate_order_stable": rank_stable,
        "cross_replica_jacobian_directions_stable": jacobian_stable,
    }
    representation_preserved = all(
        (
            raw_preserved,
            hidden_preserved,
            hidden_rank_preserved,
            component_jacobians_present,
        )
    )
    scalar_mapping_stable = all(
        (projection_alignment_pass, rank_stable, jacobian_stable)
    )
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_research = [name for name, passed in research_checks.items() if not passed]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1au_protocol_invalid"
        next_action = (
            "Repair only the failed artifact, checkpoint, runtime, row, state or "
            "zero-update binding and replay K1-AU unchanged."
        )
    elif representation_preserved and not scalar_mapping_stable:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1au_final_scalar_projection_bottleneck_supported"
        )
        next_action = (
            "Open one readiness-only K1-AV bounded multi-channel modulation design "
            "that maps the same runtime summary to the existing GF(2) edge and "
            "S-box-transition paths; require exact disabled replay and descriptor "
            "controls before any training."
        )
    elif not representation_preserved:
        status = "hold"
        decision = (
            "innovation1_uknit_family_k1au_hidden_representation_not_identifiable"
        )
        next_action = (
            "Redesign the structure summary encoder before adding output channels; "
            "repeat a zero-update identifiability gate without scaling training."
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_k1au_modulation_target_mismatch_supported"
        next_action = (
            "Keep the stable descriptor encoder and audit which existing residual "
            "path should receive each structure component before implementing a new "
            "candidate; do not train or scale yet."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "representation_preserved_through_hidden": representation_preserved,
        "final_scalar_mapping_stable": scalar_mapping_stable,
        "aligned_mismatch_panels": aligned_count,
        "expected_mismatch_panels": 18,
        "gate_rank_correlation": gate_rank_correlation,
        "cross_replica_jacobian_cosine_by_cipher": jacobian_cosines,
        "minimum_raw_summary_l2_distance": min(
            float(row["raw_summary_l2_distance"]) for row in unique_controls.values()
        ),
        "minimum_hidden_l2_distance": min(
            float(row["hidden_l2_distance"]) for row in unique_controls.values()
        ),
        "minimum_projection_alignment_abs_cosine": min(
            float(row["projection_alignment_abs_cosine"])
            for row in unique_controls.values()
        ),
        "minimum_sbox_jacobian_l2": min(
            float(row["sbox_jacobian_l2"]) for row in results
        ),
        "minimum_linear_jacobian_l2": min(
            float(row["linear_jacobian_l2"]) for row in results
        ),
        "remote_scale": "no",
        "claim_scope": (
            "Zero-training local layerwise audit of two K1-AT checkpoints; not new "
            "AUC, formal scale, an attack, unseen-cipher transfer, or SOTA evidence."
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
        readiness_config,
        k1as_config,
        datasets,
        structures,
        structure_controls,
        checkpoints,
        checkpoint_rows,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-AU source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    results, controls, cross_replica = audit_layers(
        config=config,
        readiness_config=readiness_config,
        k1as_config=k1as_config,
        datasets=datasets,
        structures=structures,
        structure_controls=structure_controls,
        checkpoints=checkpoints,
        device=device,
    )
    gate = adjudicate(
        config=config,
        source_checks=source_checks,
        results=results,
        controls=controls,
        checkpoints=checkpoints,
        cross_replica=cross_replica,
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "source_run_id": config["source"]["run_id"],
        "entries": checkpoint_rows,
    }
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(results),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "control_rows": len(controls),
        "expected_control_rows": EXPECTED_CONTROL_ROWS,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "representation_preserved_through_hidden": gate[
            "representation_preserved_through_hidden"
        ],
        "final_scalar_mapping_stable": gate["final_scalar_mapping_stable"],
        "aligned_mismatch_panels": gate["aligned_mismatch_panels"],
        "gate_rank_correlation": gate["gate_rank_correlation"],
        "cross_replica_jacobian_cosine_by_cipher": gate[
            "cross_replica_jacobian_cosine_by_cipher"
        ],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", results)
    _write_jsonl(output_root / "controls.jsonl", controls)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(results),
        control_rows=len(controls),
    )
    return {
        "preflight": preflight,
        "results": results,
        "controls": controls,
        "checkpoint_manifest": checkpoint_manifest,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _load_checkpoints(
    *,
    manifest: Mapping[str, Any],
    source_root: Path,
    device: str,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    checkpoints: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    all_paths_scoped = True
    all_hashes_exact = True
    all_state_hashes_exact = True
    source_root_resolved = source_root.resolve()
    for entry in manifest.get("entries", []):
        replica = int(entry["replica"])
        path = Path(str(entry["path"]))
        all_paths_scoped &= path.resolve().is_relative_to(source_root_resolved)
        all_hashes_exact &= path.is_file() and file_sha256(path) == entry["sha256"]
        payload = torch.load(path, map_location=device, weights_only=False)
        state_dict = payload["state_dict"]
        state_sha256 = tensor_mapping_sha256(state_dict)
        all_state_hashes_exact &= state_sha256 == entry["state_dict_sha256"]
        checkpoints[replica] = {
            **dict(entry),
            "state_dict": state_dict,
        }
        rows.append(
            {
                **dict(entry),
                "loaded_state_dict_sha256": state_sha256,
                "strict_state_dict_load": True,
            }
        )
    return (
        checkpoints,
        rows,
        {
            "two_checkpoint_payloads_complete": set(checkpoints)
            == set(EXPECTED_REPLICAS),
            "checkpoint_paths_scoped_to_source_run": all_paths_scoped,
            "checkpoint_file_hashes_exact": all_hashes_exact,
            "checkpoint_state_hashes_exact": all_state_hashes_exact,
        },
    )


def _descriptor_metrics(
    model: torch.nn.Module,
    summary: torch.Tensor,
    structure: Any,
) -> dict[str, Any]:
    descriptor = summary.detach().clone().to(torch.float32).requires_grad_(True)
    network = model.backbone.structure_gate.network
    hidden = network[1](network[0](descriptor))
    projection = network[2](hidden).squeeze()
    effective_gate = model.effective_transition_gate(
        structure,
        summary=descriptor,
        enabled=True,
    )
    jacobian = torch.autograd.grad(effective_gate, descriptor)[0]
    return {
        "hidden_tensor": hidden.detach().cpu(),
        "hidden_l2": float(torch.linalg.vector_norm(hidden).detach()),
        "projection_value": float(projection.detach()),
        "effective_gate": float(effective_gate.detach()),
        "jacobian_tensor": jacobian.detach().cpu(),
        "sbox_jacobian_l2": float(torch.linalg.vector_norm(jacobian[:16]).detach()),
        "linear_jacobian_l2": float(torch.linalg.vector_norm(jacobian[16:]).detach()),
    }


def _logit_sensitivity(
    *,
    model: torch.nn.Module,
    structure: Any,
    correct_summary: torch.Tensor,
    mismatch_summary: torch.Tensor,
    features: np.ndarray,
    device: str,
) -> dict[str, float]:
    values = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.inference_mode():
        correct = model.logits_with_runtime(
            values,
            structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
            gate_summary=correct_summary,
            structure_gate_enabled=True,
        )
        mismatch = model.logits_with_runtime(
            values,
            structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
            gate_summary=mismatch_summary,
            structure_gate_enabled=True,
        )
    delta = torch.abs(correct - mismatch)
    return {
        "mean_abs_logit_delta": float(delta.mean()),
        "max_abs_logit_delta": float(delta.max()),
    }


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_ranks = np.argsort(np.argsort(np.asarray(left, dtype=np.float64)))
    right_ranks = np.argsort(np.argsort(np.asarray(right, dtype=np.float64)))
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _all_numeric_metrics_finite(
    results: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> bool:
    result_fields = (
        "hidden_embedding_l2",
        "projection_value",
        "effective_gate",
        "sbox_jacobian_l2",
        "linear_jacobian_l2",
        "cross_replica_gate_rank_correlation",
        "cross_replica_jacobian_cosine",
    )
    control_fields = (
        "raw_summary_l2_distance",
        "hidden_l2_distance",
        "projection_alignment_abs_cosine",
        "projection_value_delta",
        "effective_gate_delta",
        "mean_abs_logit_delta",
        "max_abs_logit_delta",
    )
    return all(
        math.isfinite(float(row[field])) for row in results for field in result_fields
    ) and all(
        math.isfinite(float(row[field])) for row in controls for field in control_fields
    )


def _tensor_sha256(value: torch.Tensor) -> str:
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
        raise ValueError("K1-AU output already exists")


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_CONTROL_ROWS",
    "EXPECTED_RESULT_ROWS",
    "ROOT",
    "RUN_ID",
    "adjudicate",
    "audit_layers",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
