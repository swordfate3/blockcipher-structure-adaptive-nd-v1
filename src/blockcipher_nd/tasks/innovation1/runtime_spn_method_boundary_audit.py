from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ID = "i1_runtime_spn_method_boundary_c2_20260726"
EVIDENCE_IDS = (
    "runtime_r0",
    "gift_r2g_seed0",
    "gift_r2g_seed1",
    "present_t1_seed0",
    "present_t1_seed1",
    "skinny_t2a",
    "skinny_rtg3a",
    "rectangle_h1",
    "uknit_a6",
    "dialga_a8",
    "sbox_s1",
    "sbox_s2",
    "topology_c1",
)
REQUIREMENTS = {
    "R1": "fixed parameter geometry",
    "R2": "runtime cell-membership support",
    "R3": "exact one-to-one and general-GF(2) operator support",
    "R4": "formal general-GF(2) topology attribution",
    "R5": "one-to-one P-layer topology attribution",
    "R6": "whole-cipher topology sensitivity",
    "R7": "stable whole-cipher anchor retention",
    "R8": "heterogeneous round-window support",
    "R9": "S-box descriptor responsiveness",
    "R10": "S-box semantic identifiability",
    "R11": "nonlinear S-box operator composability",
    "R12": "universal runtime-SPN adaptation",
}
ALLOWED_STATUSES = {"supported", "partial", "contradicted", "missing"}

ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_and_validate_audit_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("C2 config must be a JSON object")
    if config.get("run_id") != RUN_ID:
        raise ValueError(f"unexpected C2 run_id: {config.get('run_id')!r}")

    audit = _mapping(config, "audit")
    expected_audit = {
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"audit.{key} must equal {expected!r}")
    margin = audit.get("control_margin")
    if not isinstance(margin, (int, float)) or float(margin) <= 0:
        raise ValueError("audit.control_margin must be positive")

    evidence = _mapping(config, "evidence")
    if set(evidence) != set(EVIDENCE_IDS):
        missing = sorted(set(EVIDENCE_IDS) - set(evidence))
        extra = sorted(set(evidence) - set(EVIDENCE_IDS))
        raise ValueError(f"invalid evidence ids: missing={missing}, extra={extra}")
    root = project_root.resolve()
    for evidence_id in EVIDENCE_IDS:
        spec = _mapping(evidence, evidence_id)
        for field in ("path", "expected_run_id", "sha256", "provenance"):
            value = spec.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"evidence.{evidence_id}.{field} must be non-empty")
        relative_path = Path(spec["path"])
        if relative_path.is_absolute():
            raise ValueError(f"evidence path must be relative: {relative_path}")
        try:
            (root / relative_path).resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"evidence path escapes project root: {relative_path}"
            ) from exc
        digest = spec["sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA-256 for evidence {evidence_id}")
    return config


def run_method_boundary_audit(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    evidence, manifest = _load_evidence(
        config=config,
        project_root=project_root,
        progress_callback=progress_callback,
    )
    rows = _evaluate_requirements(
        evidence=evidence,
        manifest=manifest,
        control_margin=float(_mapping(config, "audit")["control_margin"]),
        progress_callback=progress_callback,
    )
    validation = _validate_audit(config=config, manifest=manifest, rows=rows)
    gate = _build_gate(rows=rows, validation=validation)
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "method_status": gate["method_status"],
        "universal_runtime_spn_supported": gate["universal_runtime_spn_supported"],
        "supported_method_boundary": gate["supported_method_boundary"],
        "unsupported_method_boundary": gate["unsupported_method_boundary"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    return {
        "results": rows,
        "validation": validation,
        "gate": gate,
        "summary": summary,
        "evidence_manifest": manifest,
    }


def write_method_boundary_artifacts(
    *,
    payload: Mapping[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    results = payload["results"]
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "gate.json", payload["gate"])
    _write_json(output_root / "summary.json", payload["summary"])


def _load_evidence(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    evidence_specs = _mapping(config, "evidence")
    evidence: dict[str, dict[str, Any]] = {}
    manifest: dict[str, dict[str, Any]] = {}
    root = project_root.resolve()
    for evidence_id in EVIDENCE_IDS:
        spec = _mapping(evidence_specs, evidence_id)
        path = (root / str(spec["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"evidence path escapes project root: {path}") from exc
        if not path.is_file():
            raise ValueError(f"missing evidence file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(
                f"evidence SHA-256 mismatch for {evidence_id}: "
                f"expected {spec['sha256']}, got {digest}"
            )
        gate = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(gate, dict):
            raise ValueError(f"evidence gate must be an object: {evidence_id}")
        if gate.get("run_id") != spec["expected_run_id"]:
            raise ValueError(
                f"evidence run_id mismatch for {evidence_id}: "
                f"expected {spec['expected_run_id']!r}, got {gate.get('run_id')!r}"
            )
        evidence[evidence_id] = gate
        manifest[evidence_id] = {
            "path": str(spec["path"]),
            "sha256": digest,
            "run_id": gate["run_id"],
            "provenance": spec["provenance"],
        }
        _emit(
            progress_callback,
            "evidence_verified",
            evidence_id=evidence_id,
            sha256=digest,
        )
    return evidence, manifest


def _evaluate_requirements(
    *,
    evidence: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    control_margin: float,
    progress_callback: ProgressCallback | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    r0_checks = _mapping(evidence["runtime_r0"], "readiness_checks")
    r1_checks = {
        "shared_parameter_geometry_stable": _truth(
            r0_checks, "shared_parameter_geometry_stable"
        ),
        "runtime_structure_absent_from_state": _truth(
            r0_checks, "runtime_structure_absent_from_state"
        ),
        "variable_width_and_pair_shapes_valid": _truth(
            r0_checks, "variable_width_and_pair_shapes_valid"
        ),
    }
    rows.append(
        _row(
            requirement_id="R1",
            status="supported" if all(r1_checks.values()) else "contradicted",
            evidence_ids=("runtime_r0",),
            exact_checks=r1_checks,
            claim_boundary=(
                "Implementation contract only; this does not prove shared weights "
                "retain accuracy on an unseen cipher."
            ),
            manifest=manifest,
        )
    )

    r2_checks = {
        "cell_relabel_equivariance": _truth(r0_checks, "cell_relabel_equivariance"),
        "four_runtime_structures_covered": _truth(
            r0_checks, "four_runtime_structures_covered"
        ),
        "variable_width_and_pair_shapes_valid": _truth(
            r0_checks, "variable_width_and_pair_shapes_valid"
        ),
    }
    rows.append(
        _row(
            requirement_id="R2",
            status="supported" if all(r2_checks.values()) else "contradicted",
            evidence_ids=("runtime_r0",),
            exact_checks=r2_checks,
            claim_boundary=(
                "The runtime cell contract and equivariance are implemented; "
                "semantic cross-cipher use is adjudicated separately."
            ),
            manifest=manifest,
        )
    )

    t2a = evidence["skinny_t2a"]
    t2a_general = _mapping(_mapping(t2a, "category_counts"), "general_gf2")
    r3_checks = {
        "exact_gf2_inverses_valid": _truth(r0_checks, "exact_gf2_inverses_valid"),
        "permutation_and_general_gf2_supported": _truth(
            r0_checks, "permutation_and_general_gf2_supported"
        ),
        "permutation_gather_matches_gf2": _truth(
            r0_checks, "permutation_gather_matches_gf2"
        ),
        "skinny_general_gf2_readiness_passed": (
            t2a.get("status") == "pass"
            and t2a_general.get("passed") == t2a_general.get("total") == 5
        ),
    }
    rows.append(
        _row(
            requirement_id="R3",
            status="supported" if all(r3_checks.values()) else "contradicted",
            evidence_ids=("runtime_r0", "skinny_t2a"),
            exact_checks=r3_checks,
            claim_boundary=(
                "Bit-exact operator readiness covers one-to-one and many-source "
                "GF(2) layers; superiority requires trained control evidence."
            ),
            manifest=manifest,
        )
    )

    formal = evidence["skinny_rtg3a"]
    formal_sources = formal.get("sources")
    if not isinstance(formal_sources, list):
        raise ValueError("skinny_rtg3a.sources must be a list")
    source_seeds = {source.get("seed") for source in formal_sources}
    per_source_pass = []
    formal_metrics: dict[str, Any] = {}
    for source in formal_sources:
        if not isinstance(source, dict):
            raise ValueError("skinny_rtg3a source rows must be objects")
        seed = source.get("seed")
        margins = _mapping(source, "margins")
        aucs = _mapping(source, "aucs")
        source_pass = (
            source.get("status") == "pass"
            and float(margins["true_minus_corrupted"]) >= control_margin
            and float(margins["true_minus_independent"]) >= control_margin
            and float(aucs["true"]) >= 0.55
        )
        per_source_pass.append(source_pass)
        formal_metrics[f"seed{seed}"] = {
            "status": source.get("status"),
            "true_auc": aucs["true"],
            "true_minus_corrupted": margins["true_minus_corrupted"],
            "true_minus_no_topology": margins["true_minus_independent"],
        }
    r4_checks = {
        "joint_gate_passed": formal.get("status") == "pass",
        "samples_per_class": formal.get("samples_per_class"),
        "formal_scale_exact": formal.get("samples_per_class") == 1_000_000,
        "two_distinct_seeds": source_seeds == {0, 1},
        "all_protocol_checks_passed": _all_true(_mapping(formal, "protocol_checks")),
        "per_seed_metrics": formal_metrics,
        "both_seed_control_gates_passed": len(per_source_pass) == 2
        and all(per_source_pass),
    }
    r4_pass = all(
        bool(r4_checks[key])
        for key in (
            "joint_gate_passed",
            "formal_scale_exact",
            "two_distinct_seeds",
            "all_protocol_checks_passed",
            "both_seed_control_gates_passed",
        )
    )
    rows.append(
        _row(
            requirement_id="R4",
            status="supported" if r4_pass else "contradicted",
            evidence_ids=("skinny_rtg3a",),
            exact_checks=r4_checks,
            claim_boundary=(
                "Fallback-retrieved project-formal SKINNY evidence only; not a "
                "paper reproduction, attack, SOTA result, breakthrough or "
                "universal-SPN proof."
            ),
            manifest=manifest,
        )
    )

    one_to_one_ids = (
        "gift_r2g_seed0",
        "gift_r2g_seed1",
        "present_t1_seed0",
        "present_t1_seed1",
    )
    one_to_one_metrics: dict[str, Any] = {}
    one_to_one_passes = []
    for evidence_id in one_to_one_ids:
        gate = evidence[evidence_id]
        margins = _mapping(gate, "margins")
        row_pass = (
            gate.get("status") == "pass"
            and _all_true(_mapping(gate, "protocol_checks"))
            and float(margins["true_minus_corrupted"]) >= control_margin
            and float(margins["true_minus_independent"]) >= control_margin
        )
        one_to_one_passes.append(row_pass)
        one_to_one_metrics[evidence_id] = {
            "status": gate.get("status"),
            "true_auc": _mapping(gate, "aucs")["true"],
            "true_minus_corrupted": margins["true_minus_corrupted"],
            "true_minus_no_topology": margins["true_minus_independent"],
            "passes": row_pass,
        }
    r5_checks = {
        "control_margin": control_margin,
        "per_seed_cipher_metrics": one_to_one_metrics,
        "four_local_gates_passed": len(one_to_one_passes) == 4
        and all(one_to_one_passes),
    }
    rows.append(
        _row(
            requirement_id="R5",
            status=(
                "supported" if r5_checks["four_local_gates_passed"] else "contradicted"
            ),
            evidence_ids=one_to_one_ids,
            exact_checks=r5_checks,
            claim_boundary=(
                "Two local seeds on GIFT r6 and PRESENT r7 support topology "
                "attribution with one shared parameter geometry; protocols differ "
                "and this is not formal-scale or zero-step weight transfer evidence."
            ),
            manifest=manifest,
        )
    )

    rectangle = evidence["rectangle_h1"]
    dialga = evidence["dialga_a8"]
    topology_c1 = evidence["topology_c1"]
    rectangle_seed_checks = {
        seed: _truth(
            _mapping(_mapping(rectangle, "per_seed"), seed),
            "checks",
            "target_controls",
        )
        for seed in ("0", "1")
    }
    dialga_seed_checks = {
        seed: _truth(
            _mapping(_mapping(dialga, "per_seed"), seed),
            "checks",
            "target_topology_margins",
        )
        for seed in ("0", "1")
    }
    c1_seed_checks = {
        seed: all(
            (
                _truth(
                    _mapping(_mapping(topology_c1, "per_seed"), seed),
                    "checks",
                    "corrupted_topology_margin",
                ),
                _truth(
                    _mapping(_mapping(topology_c1, "per_seed"), seed),
                    "checks",
                    "no_topology_margin",
                ),
            )
        )
        for seed in ("0", "1")
    }
    two_holdouts_all_seed = all(rectangle_seed_checks.values()) and all(
        dialga_seed_checks.values()
    )
    two_holdouts_any_seed = any(rectangle_seed_checks.values()) and any(
        dialga_seed_checks.values()
    )
    if two_holdouts_all_seed:
        r6_status = "supported"
    elif two_holdouts_any_seed:
        r6_status = "partial"
    else:
        r6_status = "contradicted"
    r6_checks = {
        "rectangle_target_control_by_seed": rectangle_seed_checks,
        "dialga_target_topology_by_seed": dialga_seed_checks,
        "dialga_topology_only_by_seed": c1_seed_checks,
        "two_distinct_holdouts_all_seed": two_holdouts_all_seed,
        "two_distinct_holdouts_any_seed": two_holdouts_any_seed,
    }
    rows.append(
        _row(
            requirement_id="R6",
            status=r6_status,
            evidence_ids=("rectangle_h1", "dialga_a8", "topology_c1"),
            exact_checks=r6_checks,
            claim_boundary=(
                "RECTANGLE is seed-unstable while Dialga has strong local topology "
                "sensitivity; this supports a mechanism signal, not stable transfer."
            ),
            manifest=manifest,
        )
    )

    whole_cipher_full_pass = {
        "rectangle_h1": rectangle.get("full_pass") is True,
        "uknit_a6": evidence["uknit_a6"].get("full_pass") is True,
        "dialga_a8": dialga.get("full_pass") is True,
        "topology_c1": topology_c1.get("full_pass") is True,
    }
    rows.append(
        _row(
            requirement_id="R7",
            status=(
                "supported" if all(whole_cipher_full_pass.values()) else "contradicted"
            ),
            evidence_ids=("rectangle_h1", "uknit_a6", "dialga_a8", "topology_c1"),
            exact_checks={"whole_cipher_full_pass": whole_cipher_full_pass},
            claim_boundary=(
                "The current shared checkpoint does not retain all source and target "
                "anchors across both seeds; this does not prove such transfer is "
                "impossible for every future architecture."
            ),
            manifest=manifest,
        )
    )

    uknit = evidence["uknit_a6"]
    uknit_seed_functional = {
        seed: _mapping(_mapping(uknit, "per_seed"), seed).get("functional_pass") is True
        for seed in ("0", "1")
    }
    r8_checks = {
        "protocol_valid": uknit.get("protocol_valid") is True,
        "target_training_rows_zero": uknit.get("target_training_rows") == 0,
        "target_optimizer_steps_zero": uknit.get("target_optimizer_steps") == 0,
        "functional_pass": uknit.get("functional_pass") is True,
        "per_seed_functional_pass": uknit_seed_functional,
    }
    r8_pass = all(
        bool(r8_checks[key])
        for key in (
            "protocol_valid",
            "target_training_rows_zero",
            "target_optimizer_steps_zero",
            "functional_pass",
        )
    ) and all(uknit_seed_functional.values())
    rows.append(
        _row(
            requirement_id="R8",
            status="supported" if r8_pass else "contradicted",
            evidence_ids=("uknit_a6",),
            exact_checks=r8_checks,
            claim_boundary=(
                "The completed heterogeneous uKNIT window fails under zero target "
                "training; the claim is limited to the current representation."
            ),
            manifest=manifest,
        )
    )

    s1 = evidence["sbox_s1"]
    responsiveness = _mapping(s1, "responsiveness")
    r9_checks = {
        "responsive_count": sum(value is True for value in responsiveness.values()),
        "expected_count": 10,
        "descriptor_responsive_every_seed_cipher": _truth(
            _mapping(s1, "research_checks"),
            "descriptor_responsive_every_seed_cipher",
        ),
    }
    r9_pass = (
        r9_checks["responsive_count"] == r9_checks["expected_count"]
        and r9_checks["descriptor_responsive_every_seed_cipher"] is True
    )
    rows.append(
        _row(
            requirement_id="R9",
            status="supported" if r9_pass else "contradicted",
            evidence_ids=("sbox_s1",),
            exact_checks=r9_checks,
            claim_boundary=(
                "Responsiveness means the descriptor changes logits; it does not "
                "mean the network identifies the correct Boolean function."
            ),
            manifest=manifest,
        )
    )

    s1_research = _mapping(s1, "research_checks")
    r10_checks = {
        "source_macro_sbox_identifiable": _truth(
            s1_research, "source_macro_sbox_identifiable"
        ),
        "dialga_holdout_sbox_identifiable": _truth(
            s1_research, "dialga_holdout_sbox_identifiable"
        ),
    }
    rows.append(
        _row(
            requirement_id="R10",
            status="supported" if all(r10_checks.values()) else "contradicted",
            evidence_ids=("sbox_s1",),
            exact_checks=r10_checks,
            claim_boundary=(
                "Correct S-box semantics are not identifiable from the completed "
                "counterfactuals; this closes the current truth-table conditioner."
            ),
            manifest=manifest,
        )
    )

    s2 = evidence["sbox_s2"]
    s2_seed_semantics: dict[str, bool] = {}
    for seed in ("0", "1"):
        checks = _mapping(_mapping(_mapping(s2, "per_seed"), seed), "checks")
        s2_seed_semantics[seed] = all(
            _truth(checks, key)
            for key in (
                "dialga_identity_margin",
                "dialga_input_permuted_margin",
                "source_identity_margin",
                "source_input_permuted_margin",
            )
        )
    r11_checks = {
        "protocol_valid": s2.get("protocol_valid") is True,
        "full_pass": s2.get("full_pass") is True,
        "per_seed_semantic_margins": s2_seed_semantics,
    }
    r11_pass = (
        r11_checks["protocol_valid"]
        and r11_checks["full_pass"]
        and all(s2_seed_semantics.values())
    )
    rows.append(
        _row(
            requirement_id="R11",
            status="supported" if r11_pass else "contradicted",
            evidence_ids=("sbox_s2",),
            exact_checks=r11_checks,
            claim_boundary=(
                "The exact ANF branch is responsive but does not dominate identity "
                "or input-permuted operators; nonlinear composability is unsupported."
            ),
            manifest=manifest,
        )
    )

    component_statuses = {row["requirement_id"]: row["status"] for row in rows}
    all_components_supported = all(
        component_statuses[requirement_id] == "supported"
        for requirement_id in REQUIREMENTS
        if requirement_id != "R12"
    )
    if all_components_supported:
        r12_status = "supported"
    elif "contradicted" in component_statuses.values():
        r12_status = "contradicted"
    elif "missing" in component_statuses.values():
        r12_status = "missing"
    else:
        r12_status = "partial"
    rows.append(
        _row(
            requirement_id="R12",
            status=r12_status,
            evidence_ids=EVIDENCE_IDS,
            exact_checks={
                "component_statuses": component_statuses,
                "all_components_supported": all_components_supported,
            },
            claim_boundary=(
                "The current implementation is not a universal composable SPN "
                "distinguisher. Contradiction applies to the completed architecture "
                "and evidence contract, not to the feasibility of future methods."
            ),
            manifest=manifest,
        )
    )

    for row in rows:
        _emit(
            progress_callback,
            "requirement_adjudicated",
            requirement_id=row["requirement_id"],
            status=row["status"],
        )
    return rows


def _validate_audit(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Mapping[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result_ids = [row.get("requirement_id") for row in rows]
    required_fields_complete = all(
        isinstance(row.get("evidence"), list)
        and bool(row["evidence"])
        and isinstance(row.get("exact_checks"), dict)
        and bool(row["exact_checks"])
        and isinstance(row.get("claim_boundary"), str)
        and bool(row["claim_boundary"])
        for row in rows
    )
    statuses_valid = all(row.get("status") in ALLOWED_STATUSES for row in rows)
    r12 = next((row for row in rows if row.get("requirement_id") == "R12"), None)
    other_statuses = [
        row["status"] for row in rows if row.get("requirement_id") != "R12"
    ]
    universal_consistent = bool(r12) and (
        (r12["status"] == "supported")
        == all(status == "supported" for status in other_statuses)
    )
    audit = _mapping(config, "audit")
    checks = {
        "evidence_ids_exact": set(manifest) == set(EVIDENCE_IDS),
        "evidence_count_exact": len(manifest) == len(EVIDENCE_IDS),
        "requirement_ids_exact": result_ids == list(REQUIREMENTS),
        "requirement_count_exact": len(rows) == len(REQUIREMENTS),
        "required_result_fields_complete": required_fields_complete,
        "statuses_valid": statuses_valid,
        "training_rows_zero": audit.get("training_rows") == 0,
        "optimizer_steps_zero": audit.get("optimizer_steps") == 0,
        "remote_disabled": audit.get("remote") is False,
        "universal_status_consistent": universal_consistent,
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "run_id": RUN_ID,
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
        "evidence_count": len(manifest),
        "requirement_count": len(rows),
        "training_rows": 0,
        "optimizer_steps": 0,
    }


def _build_gate(
    *,
    rows: list[dict[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    status_by_id = {row["requirement_id"]: row["status"] for row in rows}
    supported = [key for key, status in status_by_id.items() if status == "supported"]
    partial = [key for key, status in status_by_id.items() if status == "partial"]
    contradicted = [
        key for key, status in status_by_id.items() if status == "contradicted"
    ]
    missing = [key for key, status in status_by_id.items() if status == "missing"]
    universal_supported = status_by_id.get("R12") == "supported"
    audit_passed = validation.get("status") == "pass"
    return {
        "run_id": RUN_ID,
        "status": "pass" if audit_passed else "invalid",
        "decision": (
            "innovation1_runtime_spn_method_boundary_frozen"
            if audit_passed
            else "innovation1_runtime_spn_method_boundary_audit_invalid"
        ),
        "method_status": "supported" if universal_supported else "partial",
        "universal_runtime_spn_supported": universal_supported,
        "requirement_status": status_by_id,
        "supported_requirements": supported,
        "partial_requirements": partial,
        "contradicted_requirements": contradicted,
        "missing_requirements": missing,
        "supported_method_boundary": [
            "one shared parameter geometry across runtime SPN descriptors",
            "runtime cell partition and relabel-equivariant processing",
            "exact one-to-one and general GF(2) linear operators",
            "project-formal two-seed SKINNY general-GF(2) topology attribution",
            "local two-seed GIFT and PRESENT one-to-one topology attribution",
            "local topology sensitivity on RECTANGLE and Dialga holdouts",
        ],
        "unsupported_method_boundary": [
            "stable whole-cipher zero-training transfer across both seeds",
            "heterogeneous uKNIT round-window transfer",
            "S-box semantic identifiability",
            "composable nonlinear S-box operator use",
            "universal arbitrary-SPN adaptation",
        ],
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote_scale": "no",
        "claim_scope": (
            "zero-training evidence synthesis; supports a runtime GF(2)-topology-"
            "aware SPN neural distinguisher boundary, not universal composable SPN "
            "adaptation, a paper reproduction, attack, SOTA result or breakthrough"
        ),
        "next_action": (
            "freeze the current method as runtime GF(2)-topology-aware; prepare a "
            "separate C3 readiness plan for formal one-to-one P-layer attribution "
            "using the repaired PRESENT T1 protocol and exact correct/corrupted/no-"
            "topology controls, without reopening S-box, ANF, Adapter, FiLM, MoE, "
            "target-supervision or C1 rescue scaling"
        ),
        "blocked_actions": [
            "claim universal or composable arbitrary-SPN adaptation",
            "reopen truth-table, ANF, DDT, inverse-triplet or delta-U S-box routes",
            "rescue C1/S2 with more samples, epochs, pairs, experts or remote compute",
            "collapse local diagnostics and fallback-retrieved formal evidence into one scale claim",
        ],
    }


def _row(
    *,
    requirement_id: str,
    status: str,
    evidence_ids: tuple[str, ...],
    exact_checks: Mapping[str, Any],
    claim_boundary: str,
    manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "requirement_id": requirement_id,
        "requirement": REQUIREMENTS[requirement_id],
        "status": status,
        "evidence": [dict(manifest[evidence_id]) for evidence_id in evidence_ids],
        "exact_checks": dict(exact_checks),
        "claim_boundary": claim_boundary,
    }


def _mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _truth(source: Mapping[str, Any], *keys: str) -> bool:
    value: Any = source
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing required boolean field: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, bool):
        raise ValueError(f"required field is not boolean: {'.'.join(keys)}")
    return value


def _all_true(checks: Mapping[str, Any]) -> bool:
    return bool(checks) and all(value is True for value in checks.values())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)
