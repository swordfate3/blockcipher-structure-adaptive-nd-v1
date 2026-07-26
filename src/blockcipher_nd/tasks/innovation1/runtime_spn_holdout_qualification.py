from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    RelationModeRuntimeE4,
    _load_structures,
    _plain_spec,
    load_and_validate_holdout_config,
)


CANDIDATES = ("rectangle80", "uknit64", "dialga128")
EXPECTED_SEEDS = (0, 1)
CONDITIONS = ("correct", "corrupted", "no_topology")
DISPLAY_NAMES = {
    "rectangle80": "RECTANGLE-80 r6",
    "uknit64": "uKNIT prefix-r5",
    "dialga128": "Dialga prefix-r4",
}


def load_and_validate_holdout_qualification_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("A7 config schema_version must be 1")
    if config.get("experiment") != "innovation1_runtime_spn_holdout_qualification_a7":
        raise ValueError("A7 experiment name drifted")

    protocol = config.get("protocol", {})
    expected_source_panels = {
        target: [candidate for candidate in ("gift64", "skinny64", *CANDIDATES) if candidate != target]
        for target in CANDIDATES
    }
    if tuple(protocol.get("candidate_order", ())) != CANDIDATES:
        raise ValueError("A7 candidate order drifted")
    if protocol.get("source_panels") != expected_source_panels:
        raise ValueError("A7 source panels drifted")
    if tuple(protocol.get("previous_whole_cipher_holdouts", ())) != (
        "rectangle80",
        "uknit64",
    ):
        raise ValueError("A7 previous holdout set drifted")
    if protocol.get("runtime_rounds") != 2:
        raise ValueError("A7 runtime window must contain two transitions")
    _verify_frozen_file(
        project_root,
        {"path": protocol["config_path"], "sha256": protocol["config_sha256"]},
    )
    load_and_validate_holdout_config(project_root / protocol["config_path"])

    evidence = config.get("evidence", {})
    if tuple(evidence) != (*CANDIDATES, "closed_mechanism"):
        raise ValueError("A7 evidence panel drifted")
    required_evidence = {
        "rectangle80": (
            "a3_zero_target_step_holdout",
            "innovation1_runtime_spn_h1_equalized_pcgrad_partial",
            ("config", "gate", "validation", "results"),
        ),
        "uknit64": (
            "u3_target_trained_attribution",
            "innovation1_runtime_spn_recurrent_window_not_supported",
            ("config", "gate", "validation", "results"),
        ),
        "dialga128": (
            "d1_target_trained_plus_d2_same_checkpoint",
            "innovation1_dialga_runtime_e4_d2_functional_topology_use_supported",
            (
                "d1_gate",
                "d1_validation",
                "d1_results",
                "gate",
                "validation",
                "results",
            ),
        ),
    }
    for candidate, (kind, decision, files) in required_evidence.items():
        row = evidence[candidate]
        if row.get("kind") != kind or row.get("required_decision") != decision:
            raise ValueError(f"A7 {candidate} evidence contract drifted")
        for key in files:
            _verify_frozen_file(project_root, row[key])

    closed = evidence["closed_mechanism"]
    if closed.get("required_decision") != (
        "innovation1_runtime_spn_uknit_heterogeneous_holdout_not_supported"
    ):
        raise ValueError("A7 closed-mechanism decision drifted")
    for key in ("config", "gate", "validation", "results"):
        _verify_frozen_file(project_root, closed[key])

    expected_audit = {
        "training_performed": False,
        "new_data_generated": False,
        "seeds": [0, 1],
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "pairs_per_sample": 4,
        "epochs": 10,
        "negative_mode": "encrypted_random_plaintexts",
        "atomic_gf2_type": "source_bit_role_to_target_bit_role",
        "cell_relabel_probe_seed": 26072607,
        "cell_relabel_probe_rows": 8,
    }
    if config.get("audit") != expected_audit:
        raise ValueError("A7 audit contract drifted")
    expected_gate = {
        "correct_auc_floor": 0.55,
        "topology_margin": 0.005,
        "atomic_gf2_coverage": 1.0,
        "cell_relabel_max_error": 0.000001,
        "require_both_seeds": True,
        "require_not_previously_used": True,
        "selection_metric": "minimum_correct_auc_then_candidate_order",
    }
    if config.get("gate") != expected_gate:
        raise ValueError("A7 gate contract drifted")
    return config


def atomic_gf2_relation_types(
    structure: RuntimeSpnStructure,
) -> set[tuple[int, int]]:
    relation_types: set[tuple[int, int]] = set()
    for matrix in structure.inverse_linear_matrices:
        for target_bit, source_bit in torch.nonzero(matrix, as_tuple=False).tolist():
            relation_types.add(
                (
                    int(structure.bit_role[source_bit]),
                    int(structure.bit_role[target_bit]),
                )
            )
    return relation_types


def sbox_truth_hashes(structure: RuntimeSpnStructure) -> set[str]:
    return {
        hashlib.sha256(truth.numpy().tobytes()).hexdigest()
        for truth in structure.sbox_truth_bits.reshape(-1, 64)
    }


def run_holdout_qualification_audit(
    *,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    base = load_and_validate_holdout_config(
        project_root / config["protocol"]["config_path"]
    )
    structures = _load_structures(base)
    evidence_metrics, evidence_checks = _load_frozen_evidence(config, project_root)
    closed_check = _closed_mechanism_valid(config, project_root)
    structure_profiles = _structure_profiles(config, base, structures)
    rows: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        profile = structure_profiles[candidate]
        for seed in EXPECTED_SEEDS:
            metrics = evidence_metrics[candidate][seed]
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "holdout_qualification",
                    "candidate": candidate,
                    "candidate_display_name": DISPLAY_NAMES[candidate],
                    "seed": seed,
                    **metrics,
                    "correct_minus_corrupted": (
                        metrics["correct_auc"] - metrics["corrupted_auc"]
                    ),
                    "correct_minus_no_topology": (
                        metrics["correct_auc"] - metrics["no_topology_auc"]
                    ),
                    "target_atomic_gf2_types": profile["target_atomic_gf2_types"],
                    "covered_atomic_gf2_types": profile[
                        "covered_atomic_gf2_types"
                    ],
                    "atomic_gf2_coverage": profile["atomic_gf2_coverage"],
                    "target_unique_sboxes": profile["target_unique_sboxes"],
                    "exact_source_sbox_overlap": profile[
                        "exact_source_sbox_overlap"
                    ],
                    "cell_relabel_max_error": profile["cell_relabel_max_error"],
                    "previous_whole_cipher_holdout": candidate
                    in config["protocol"]["previous_whole_cipher_holdouts"],
                    "evidence_valid": evidence_checks[candidate]["valid"],
                    "training_performed": False,
                    "new_data_generated": False,
                }
            )

    validation = _validate_audit(
        config=config,
        rows=rows,
        evidence_checks=evidence_checks,
        closed_check=closed_check,
        structures=structures,
    )
    gate = adjudicate_holdout_qualification(
        config=config,
        rows=rows,
        validation=validation,
    )
    summary = {
        "run_id": config["run_id"],
        "status": gate["status"],
        "decision": gate["decision"],
        "selected_holdout": gate["selected_holdout"],
        "training_performed": False,
        "new_data_generated": False,
        "next_action": gate["next_action"],
    }
    return {
        "rows": rows,
        "validation": validation,
        "gate": gate,
        "summary": summary,
        "structure_profiles": structure_profiles,
    }


def adjudicate_holdout_qualification(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    grouped = {
        candidate: {int(row["seed"]): row for row in rows if row["candidate"] == candidate}
        for candidate in CANDIDATES
    }
    per_candidate: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for candidate in CANDIDATES:
        seed_rows = grouped[candidate]
        seed_checks = {}
        for seed in EXPECTED_SEEDS:
            row = seed_rows.get(seed, {})
            seed_checks[str(seed)] = {
                "correct_auc_floor": _finite(row.get("correct_auc"))
                and row["correct_auc"] >= config["gate"]["correct_auc_floor"],
                "corrupted_margin": _finite(row.get("correct_minus_corrupted"))
                and row["correct_minus_corrupted"]
                >= config["gate"]["topology_margin"],
                "no_topology_margin": _finite(row.get("correct_minus_no_topology"))
                and row["correct_minus_no_topology"]
                >= config["gate"]["topology_margin"],
            }
        structural_checks = {
            "atomic_gf2_coverage": len(seed_rows) == 2
            and all(
                row.get("atomic_gf2_coverage", 0.0)
                >= config["gate"]["atomic_gf2_coverage"]
                for row in seed_rows.values()
            ),
            "cell_relabel_invariant": len(seed_rows) == 2
            and all(
                row.get("cell_relabel_max_error", math.inf)
                <= config["gate"]["cell_relabel_max_error"]
                for row in seed_rows.values()
            ),
            "evidence_valid": len(seed_rows) == 2
            and all(row.get("evidence_valid") is True for row in seed_rows.values()),
        }
        both_seeds_pass = len(seed_rows) == 2 and all(
            all(checks.values()) for checks in seed_checks.values()
        )
        technically_qualified = both_seeds_pass and all(structural_checks.values())
        previously_used = bool(
            seed_rows
            and next(iter(seed_rows.values())).get("previous_whole_cipher_holdout")
        )
        candidate_eligible = technically_qualified and not previously_used
        if candidate_eligible:
            eligible.append(candidate)
        per_candidate[candidate] = {
            "seed_checks": seed_checks,
            "structural_checks": structural_checks,
            "technically_qualified": technically_qualified,
            "previous_whole_cipher_holdout": previously_used,
            "eligible": candidate_eligible,
            "minimum_correct_auc": min(
                (row["correct_auc"] for row in seed_rows.values()),
                default=None,
            ),
            "target_atomic_gf2_types": next(
                (row["target_atomic_gf2_types"] for row in seed_rows.values()),
                None,
            ),
            "covered_atomic_gf2_types": next(
                (row["covered_atomic_gf2_types"] for row in seed_rows.values()),
                None,
            ),
            "target_unique_sboxes": next(
                (row["target_unique_sboxes"] for row in seed_rows.values()),
                None,
            ),
            "exact_source_sbox_overlap": next(
                (row["exact_source_sbox_overlap"] for row in seed_rows.values()),
                None,
            ),
        }

    order = {candidate: index for index, candidate in enumerate(CANDIDATES)}
    selected = max(
        eligible,
        key=lambda candidate: (
            per_candidate[candidate]["minimum_correct_auc"],
            -order[candidate],
        ),
        default=None,
    )
    protocol_valid = validation.get("status") == "pass"
    if not protocol_valid:
        status = "fail"
        decision = "innovation1_runtime_spn_holdout_qualification_protocol_invalid"
        selected = None
        next_action = "repair only the failed A7 evidence or protocol check"
    elif selected is None:
        status = "hold"
        decision = "innovation1_runtime_spn_holdout_qualification_none_selected"
        next_action = (
            "stop whole-cipher transfer training and redesign the atomic structure "
            "representation locally"
        )
    else:
        status = "pass"
        decision = f"innovation1_runtime_spn_holdout_qualification_{selected}_selected"
        next_action = (
            f"preregister A8 with {selected} as a zero-training-row whole-cipher "
            "holdout and the other four ciphers as sources"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "selected_holdout": selected,
        "eligible_candidates": eligible,
        "per_candidate": per_candidate,
        "training_performed": False,
        "new_data_generated": False,
        "claim_scope": (
            "zero-training holdout qualification audit only; no new model, "
            "cross-cipher success, formal scale, attack, SOTA or universality claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "retry uKNIT r5 or revive relation-mass pooling",
            "add MoE, Adapter, FiLM, typed residual or target-specific head",
            "increase samples, epochs or launch remote scale",
            "treat target-trained oracle AUC as cross-cipher transfer evidence",
        ],
    }


def _load_frozen_evidence(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, dict[str, Any]]]:
    metrics: dict[str, dict[int, dict[str, float]]] = {}
    checks: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        evidence = config["evidence"][candidate]
        gate = _read_json(project_root / evidence["gate"]["path"])
        validation = _read_json(project_root / evidence["validation"]["path"])
        rows = _read_jsonl(project_root / evidence["results"]["path"])
        if candidate == "rectangle80":
            candidate_metrics, budget_valid = _rectangle_metrics(rows)
            gate_matches = _rectangle_gate_matches(gate, candidate_metrics)
            protocol_valid = gate.get("protocol_valid") is True
        elif candidate == "uknit64":
            candidate_metrics, budget_valid = _uknit_metrics(rows)
            gate_matches = _uknit_gate_matches(gate, candidate_metrics)
            protocol_valid = all(gate.get("protocol_checks", {}).values())
        else:
            candidate_metrics, budget_valid = _dialga_metrics(rows)
            gate_matches = _dialga_gate_matches(gate, candidate_metrics)
            d1_gate = _read_json(project_root / evidence["d1_gate"]["path"])
            d1_validation = _read_json(
                project_root / evidence["d1_validation"]["path"]
            )
            d1_rows = _read_jsonl(project_root / evidence["d1_results"]["path"])
            budget_valid = budget_valid and _dialga_d1_budget_valid(d1_rows)
            protocol_valid = (
                all(gate.get("protocol_checks", {}).values())
                and d1_gate.get("status") == "pass"
                and all(d1_gate.get("protocol_checks", {}).values())
                and d1_validation.get("status") == "pass"
            )
        validation_valid = validation.get("status") == "pass" and all(
            validation.get("checks", {}).values()
        )
        decision_valid = gate.get("decision") == evidence["required_decision"]
        metrics[candidate] = candidate_metrics
        checks[candidate] = {
            "decision_valid": decision_valid,
            "validation_valid": validation_valid,
            "protocol_valid": protocol_valid,
            "budget_valid": budget_valid,
            "gate_matches_recomputed_metrics": gate_matches,
            "valid": all(
                (
                    decision_valid,
                    validation_valid,
                    protocol_valid,
                    budget_valid,
                    gate_matches,
                )
            ),
        }
    return metrics, checks


def _rectangle_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, float]], bool]:
    target = [
        row
        for row in rows
        if row.get("cipher") == "rectangle80"
        and row.get("row_kind") == "holdout_target"
    ]
    mapping = {
        "candidate_correct": "correct",
        "candidate_corrupted_target": "corrupted",
        "candidate_no_topology_target": "no_topology",
    }
    metrics = _group_condition_metrics(
        target,
        condition=lambda row: mapping.get(row.get("evaluation")),
        auc=lambda row: row.get("metrics", {}).get("validation", {}).get("auc"),
    )
    budget_valid = len(target) == 6 and all(
        row.get("optimizer_steps") == 0
        and row.get("training_samples_per_class") == 0
        and row.get("validation_samples_per_class") == 1024
        and row.get("pairs_per_sample") == 4
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("parameter_count") == 442466
        for row in target
    )
    return metrics, budget_valid


def _uknit_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, float]], bool]:
    model_map = {
        "runtime_spn_e4_equivariant_true": "correct",
        "runtime_spn_e4_equivariant_corrupted": "corrupted",
        "runtime_spn_e4_equivariant_independent": "no_topology",
    }
    target = [
        row
        for row in rows
        if row.get("runtime_round_window_mode") == "recurrent_window"
        and row.get("runtime_structure_window_control") == "full"
        and row.get("model") in model_map
    ]
    metrics = _group_condition_metrics(
        target,
        condition=lambda row: model_map.get(row.get("model")),
        auc=lambda row: row.get("metrics", {}).get("auc"),
    )
    budget_valid = len(target) == 6 and all(
        row.get("samples_per_class") == 2048
        and row.get("validation", {}).get("samples_per_class") == 1024
        and row.get("target_epochs") == 10
        and row.get("pairs_per_sample") == 4
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("parameter_count") == 442466
        for row in target
    )
    return metrics, budget_valid


def _dialga_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, float]], bool]:
    target = [row for row in rows if row.get("condition") in CONDITIONS]
    metrics = _group_condition_metrics(
        target,
        condition=lambda row: row.get("condition"),
        auc=lambda row: row.get("auc"),
    )
    budget_valid = len(target) == 6 and all(
        row.get("training_performed") is False
        and row.get("samples_total") == 2048
        and row.get("pairs_per_sample") == 4
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("parameter_count") == 442466
        for row in target
    )
    return metrics, budget_valid


def _dialga_d1_budget_valid(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == 6 and all(
        row.get("samples_per_class") == 2048
        and row.get("validation", {}).get("samples_per_class") == 1024
        and row.get("target_epochs") == 10
        and row.get("pairs_per_sample") == 4
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("parameter_count") == 442466
        for row in rows
    )


def _group_condition_metrics(
    rows: list[dict[str, Any]],
    *,
    condition: Any,
    auc: Any,
) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for row in rows:
        seed = int(row.get("seed", -1))
        name = condition(row)
        value = auc(row)
        if seed not in EXPECTED_SEEDS or name not in CONDITIONS or not _finite(value):
            continue
        if f"{name}_auc" in grouped[seed]:
            raise ValueError("duplicate A7 evidence condition")
        grouped[seed][f"{name}_auc"] = float(value)
    if any(tuple(grouped[seed]) != tuple(f"{name}_auc" for name in CONDITIONS) for seed in EXPECTED_SEEDS):
        raise ValueError("incomplete A7 evidence condition panel")
    return grouped


def _rectangle_gate_matches(
    gate: dict[str, Any], metrics: dict[int, dict[str, float]]
) -> bool:
    return all(
        _close(gate["per_seed"][str(seed)]["candidate_target_auc"], row["correct_auc"])
        and _close(
            gate["per_seed"][str(seed)]["target_margins"][
                "candidate_corrupted_target"
            ],
            row["correct_auc"] - row["corrupted_auc"],
        )
        and _close(
            gate["per_seed"][str(seed)]["target_margins"][
                "candidate_no_topology_target"
            ],
            row["correct_auc"] - row["no_topology_auc"],
        )
        for seed, row in metrics.items()
    )


def _uknit_gate_matches(
    gate: dict[str, Any], metrics: dict[int, dict[str, float]]
) -> bool:
    return all(
        _close(gate["seed_results"][str(seed)]["candidate_auc"], row["correct_auc"])
        and _close(
            gate["seed_results"][str(seed)]["candidate_minus_corrupted"],
            row["correct_auc"] - row["corrupted_auc"],
        )
        and _close(
            gate["seed_results"][str(seed)]["candidate_minus_no_topology"],
            row["correct_auc"] - row["no_topology_auc"],
        )
        for seed, row in metrics.items()
    )


def _dialga_gate_matches(
    gate: dict[str, Any], metrics: dict[int, dict[str, float]]
) -> bool:
    return all(
        _close(gate["seed_results"][str(seed)]["correct_auc"], row["correct_auc"])
        and _close(
            gate["seed_results"][str(seed)]["correct_minus_corrupted_auc"],
            row["correct_auc"] - row["corrupted_auc"],
        )
        and _close(
            gate["seed_results"][str(seed)]["correct_minus_no_topology_auc"],
            row["correct_auc"] - row["no_topology_auc"],
        )
        for seed, row in metrics.items()
    )


def _structure_profiles(
    config: dict[str, Any],
    base: dict[str, Any],
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, dict[str, Any]]:
    model = RelationModeRuntimeE4(_plain_spec(base["model"]), "true").eval()
    profiles: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        target = structures[candidate]
        sources = config["protocol"]["source_panels"][candidate]
        target_types = atomic_gf2_relation_types(target)
        source_types = set().union(
            *(atomic_gf2_relation_types(structures[source]) for source in sources)
        )
        covered_types = target_types & source_types
        target_sboxes = sbox_truth_hashes(target)
        source_sboxes = set().union(
            *(sbox_truth_hashes(structures[source]) for source in sources)
        )
        relabel_error = _cell_relabel_error(
            model=model,
            structure=target,
            seed=config["audit"]["cell_relabel_probe_seed"],
            rows=config["audit"]["cell_relabel_probe_rows"],
            pairs=config["audit"]["pairs_per_sample"],
        )
        profiles[candidate] = {
            "source_panel": list(sources),
            "target_atomic_gf2_types": len(target_types),
            "source_atomic_gf2_types": len(source_types),
            "covered_atomic_gf2_types": len(covered_types),
            "missing_atomic_gf2_types": [
                list(item) for item in sorted(target_types - source_types)
            ],
            "atomic_gf2_coverage": len(covered_types) / len(target_types),
            "target_unique_sboxes": len(target_sboxes),
            "source_unique_sboxes": len(source_sboxes),
            "exact_source_sbox_overlap": len(target_sboxes & source_sboxes),
            "cell_relabel_max_error": relabel_error,
            "runtime_window_sha256": target.window_sha256(),
        }
    return profiles


def _cell_relabel_error(
    *,
    model: RelationModeRuntimeE4,
    structure: RuntimeSpnStructure,
    seed: int,
    rows: int,
    pairs: int,
) -> float:
    generator = torch.Generator().manual_seed(seed + structure.block_bits)
    features = torch.randint(
        0,
        2,
        (rows, pairs, 2, structure.block_bits),
        generator=generator,
        dtype=torch.float32,
    )
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    relabeled_features = torch.empty_like(features)
    relabeled_features[..., bit_permutation] = features
    with torch.no_grad():
        original_logits = model(features, structure)
        relabeled_logits = model(relabeled_features, relabeled)
    return float((original_logits - relabeled_logits).abs().max())


def _closed_mechanism_valid(config: dict[str, Any], project_root: Path) -> bool:
    evidence = config["evidence"]["closed_mechanism"]
    gate = _read_json(project_root / evidence["gate"]["path"])
    validation = _read_json(project_root / evidence["validation"]["path"])
    return bool(
        gate.get("decision") == evidence["required_decision"]
        and gate.get("protocol_valid") is True
        and validation.get("status") == "pass"
        and all(validation.get("checks", {}).values())
    )


def _validate_audit(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    evidence_checks: dict[str, dict[str, Any]],
    closed_check: bool,
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, Any]:
    checks = {
        "six_rows_complete": len(rows) == 6
        and {
            (row.get("candidate"), row.get("seed")) for row in rows
        }
        == {(candidate, seed) for candidate in CANDIDATES for seed in EXPECTED_SEEDS},
        "all_frozen_evidence_valid": all(
            row["valid"] for row in evidence_checks.values()
        ),
        "a6_relation_mass_mechanism_closed": closed_check,
        "source_panels_exclude_target": all(
            candidate not in config["protocol"]["source_panels"][candidate]
            and len(config["protocol"]["source_panels"][candidate]) == 4
            for candidate in CANDIDATES
        ),
        "all_runtime_structures_loaded": set(structures)
        == {"gift64", "skinny64", "rectangle80", "uknit64", "dialga128"},
        "finite_metrics": all(
            _finite(row.get(field))
            for row in rows
            for field in (
                "correct_auc",
                "corrupted_auc",
                "no_topology_auc",
                "correct_minus_corrupted",
                "correct_minus_no_topology",
                "atomic_gf2_coverage",
                "cell_relabel_max_error",
            )
        ),
        "zero_training": config["audit"]["training_performed"] is False
        and all(row.get("training_performed") is False for row in rows),
        "zero_new_data": config["audit"]["new_data_generated"] is False
        and all(row.get("new_data_generated") is False for row in rows),
        "strict_negative_protocol": config["audit"]["negative_mode"]
        == "encrypted_random_plaintexts",
        "same_budget_contract": all(
            evidence_checks[candidate]["budget_valid"] for candidate in CANDIDATES
        ),
    }
    return {
        "run_id": config["run_id"],
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "evidence_checks": evidence_checks,
        "training_performed": False,
        "new_data_generated": False,
    }


def _verify_frozen_file(project_root: Path, evidence: dict[str, str]) -> None:
    path = project_root / evidence["path"]
    if not path.is_file():
        raise ValueError(f"A7 frozen evidence is missing: {evidence['path']}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != evidence.get("sha256"):
        raise ValueError(f"A7 frozen evidence hash drifted: {evidence['path']}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    return _finite(left) and _finite(right) and abs(float(left) - float(right)) <= tolerance
