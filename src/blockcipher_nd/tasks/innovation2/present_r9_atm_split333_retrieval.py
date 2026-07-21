from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.atm_resumable_search_runner import (
    ArtifactIntegrityError,
    validate_completed_search_result,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_generation import (
    ATM_COMMIT,
    RUN_ID,
    SOURCE_HASHES,
    canonical_relations,
    search_config,
)
from blockcipher_nd.tasks.innovation2.present_sbox4_real_atm_runner_compatibility import (
    audit_relation_spaces,
)


VALIDATION_RUN_ID = "i2_present_r9_atm_split333_retrieval_validation_20260720"
EXPECTED_SOURCE_COMMIT = "85fc73200c56730894522034f5819bf72e0cb792"
EXPECTED_MODEL_SHA256 = "ccc91bfdb16e6104eeca6fde32ec71951f130c261e17b5b7202e6197304166d2"
GENERATION_DECISION = "innovation2_present_r9_split333_generation_passed"
CONFLICTING_SUCCESS_TERMINALS = (
    "pipeline_failed.marker",
    "probe_failed.marker",
    "resource_cap_hit.marker",
    "setup_failed.marker",
)


@dataclass(frozen=True)
class Split333RetrievalConfig:
    run_id: str = VALIDATION_RUN_ID
    source_run_id: str = RUN_ID
    expected_source_commit: str = EXPECTED_SOURCE_COMMIT
    expected_atm_commit: str = ATM_COMMIT
    expected_model_sha256: str = EXPECTED_MODEL_SHA256

    def __post_init__(self) -> None:
        if self.run_id != VALIDATION_RUN_ID or self.source_run_id != RUN_ID:
            raise ValueError("E104 retrieval run identifiers are frozen")
        if len(self.expected_source_commit) != 40:
            raise ValueError("expected source commit must be full-length")


def validate_split333_retrieval(
    config: Split333RetrievalConfig,
    *,
    raw_root: Path,
) -> dict[str, Any]:
    logs = raw_root / "logs"
    results = raw_root / "results"
    search_root = results / "search_state"
    required = {
        "pipeline_passed": logs / "pipeline_passed.marker",
        "source_revision": logs / "source_revision.txt",
        "atm_revision": logs / "atm_revision.txt",
        "source_status": logs / "source_status_after_sync.txt",
        "phase_a_gate": results / "phase_a_gate.json",
        "probe_gate": results / "probe_gate.json",
        "generation_gate": results / "gate.json",
        "generation_summary": results / "summary.json",
        "generation_relations": results / "relations.json",
        "generation_marker": results / "generation_passed.marker",
        "source_hashes": results / "source_hashes.json",
        "model_contract": results / "model_contract.json",
        "search_metadata": search_root / "metadata.json",
        "search_result": search_root / "result.json",
        "search_complete": search_root / "complete.marker",
        "search_progress": search_root / "progress.jsonl",
    }
    probe_paths = _probe_files(results)
    if len(probe_paths) >= 2:
        first_probe_path, second_probe_path = probe_paths[-2:]
        required["first_resume_probe"] = first_probe_path
        required["second_resume_probe"] = second_probe_path
        required["first_resume_probe_passed"] = first_probe_path.with_name(
            f"{first_probe_path.stem}_passed.marker"
        )
        required["second_resume_probe_passed"] = second_probe_path.with_name(
            f"{second_probe_path.stem}_passed.marker"
        )
    else:
        required["first_resume_probe"] = results / "missing_first_resume_probe.json"
        required["second_resume_probe"] = results / "missing_second_resume_probe.json"
    existence_checks = {
        f"required_{name}_exists": path.is_file() for name, path in required.items()
    }
    if not all(existence_checks.values()):
        return _invalid_result(
            config,
            existence_checks=existence_checks,
            checks={},
            metrics={},
            artifact_manifest=_artifact_manifest(raw_root, required, ()),
        )

    expected_parameter_hash = search_config().parameter_hash()
    source_revision = required["source_revision"].read_text(encoding="utf-8").strip()
    atm_revision = required["atm_revision"].read_text(encoding="utf-8").strip()
    source_status_lines = [
        line.strip()
        for line in required["source_status"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phase_a_gate = _load_json(required["phase_a_gate"])
    probe_gate = _load_json(required["probe_gate"])
    generation_gate = _load_json(required["generation_gate"])
    generation_summary = _load_json(required["generation_summary"])
    relations_payload = _load_json(required["generation_relations"])
    generation_marker = _load_json(required["generation_marker"])
    actual_source_hashes = _load_json(required["source_hashes"])
    model_contract = _load_json(required["model_contract"])
    first_probe = _load_json(required["first_resume_probe"])
    second_probe = _load_json(required["second_resume_probe"])
    search_metadata = _load_json(required["search_metadata"])

    completed_relations: tuple[tuple[tuple[int, int], ...], ...] = ()
    completed_integrity = True
    integrity_error = ""
    try:
        completed_relations = validate_completed_search_result(
            search_root,
            config=search_config(),
        )
    except (ArtifactIntegrityError, OSError, ValueError, TypeError) as error:
        completed_integrity = False
        integrity_error = str(error)
    serialized_relations = canonical_relations(relations_payload.get("relations", ()))
    completed_canonical = canonical_relations(completed_relations)
    relation_audit = audit_relation_spaces(completed_canonical, serialized_relations)
    candidate_audit = _audit_candidate_cache(
        search_root / "candidate_results",
        parameter_hash=expected_parameter_hash,
    )
    checks = {
        "no_conflicting_terminal_markers": not any(
            (logs / name).exists() for name in CONFLICTING_SUCCESS_TERMINALS
        ),
        "source_commit_matches": source_revision == config.expected_source_commit,
        "atm_commit_matches": atm_revision == config.expected_atm_commit,
        "source_tracked_worktree_clean": len(source_status_lines) == 1
        and source_status_lines[0].startswith("## "),
        "source_hashes_match": actual_source_hashes == SOURCE_HASHES,
        "model_hash_matches": model_contract.get("sha256")
        == config.expected_model_sha256,
        "phase_a_gate_pass": phase_a_gate.get("status") == "pass",
        "probe_gate_pass": probe_gate.get("status") == "pass",
        "first_resume_probe_completed_one_candidate": first_probe.get(
            "controlled_interruption"
        )
        is True
        and int(first_probe.get("new_durable_candidates", 0)) == 1
        and int(first_probe.get("candidate_reuse_events_total", 0)) > 0,
        "second_resume_probe_reused_candidates": second_probe.get(
            "controlled_interruption"
        )
        is True
        and int(second_probe.get("candidate_reuse_events_total", 0)) > 0
        and int(second_probe.get("new_durable_candidates", 0)) == 1,
        "generation_gate_pass": generation_gate.get("status") == "pass",
        "generation_decision_matches": generation_gate.get("decision")
        == GENERATION_DECISION,
        "search_parameter_hash_matches": search_metadata.get("parameter_hash")
        == expected_parameter_hash,
        "search_parameters_match": search_metadata.get("parameters")
        == search_config().parameter_payload(),
        "generation_marker_parameter_hash_matches": generation_marker.get(
            "parameter_hash"
        )
        == expected_parameter_hash,
        "completed_result_integrity": completed_integrity,
        "serialized_relations_match_completed_result": serialized_relations
        == completed_canonical,
        "relation_count_matches_summary": int(generation_summary.get("relations", -1))
        == len(completed_canonical),
        "relation_rank_matches_summary": int(
            generation_summary.get("relation_rank", -1)
        )
        == relation_audit["official_rank"],
        "relation_space_replays": relation_audit["rank_equal"]
        and relation_audit["anchor_span_in_runner"]
        and relation_audit["runner_span_in_anchor"],
        "candidate_cache_integrity": candidate_audit["valid"],
        "candidate_cache_nonempty": candidate_audit["files"] > 0,
    }
    metrics = {
        "relations": len(completed_canonical),
        "relation_rank": relation_audit["official_rank"],
        "support_coordinates": relation_audit["support_coordinates"],
        "candidate_files": candidate_audit["files"],
        "candidate_bytes": candidate_audit["bytes"],
        "candidate_cache_sha256": candidate_audit["aggregate_sha256"],
        "first_resume_probe": required["first_resume_probe"].stem,
        "second_resume_probe": required["second_resume_probe"].stem,
        "first_resume_probe_reuse_events": int(
            first_probe.get("candidate_reuse_events_total", 0)
        ),
        "second_resume_probe_reuse_events": int(
            second_probe.get("candidate_reuse_events_total", 0)
        ),
        "parameter_hash": expected_parameter_hash,
        "integrity_error": integrity_error,
    }
    artifact_manifest = _artifact_manifest(
        raw_root,
        required,
        tuple(sorted((search_root / "candidate_results").glob("*.json"))),
    )
    if all(existence_checks.values()) and all(checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_split333_retrieval_verified"
        action = "promote the verified copy and run E105 with frozen checkpoints"
    else:
        status = "fail"
        decision = "innovation2_present_r9_split333_retrieval_invalid"
        action = "keep raw fallback incomplete and repair only the failed evidence gate"
    return {
        "run_id": config.run_id,
        "source_run_id": config.source_run_id,
        "status": status,
        "decision": decision,
        "existence_checks": existence_checks,
        "checks": checks,
        "metrics": metrics,
        "relation_audit": relation_audit,
        "candidate_audit": candidate_audit,
        "artifact_manifest": artifact_manifest,
        "claim_scope": (
            "local integrity and plan-alignment validation of a remotely generated "
            "PRESENT r9 ATM split (3,3,3) relation set under independent round keys; "
            "not a published result, PRESENT-80, neural result, distinguisher, attack, or SOTA"
        ),
        "next_action": {"action": action, "e105_open": status == "pass"},
    }


def result_rows(validation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": validation["source_run_id"],
            "task": "innovation2_present_r9_atm_split333_retrieval_validation",
            "status": validation["status"],
            "decision": validation["decision"],
            "training_performed": False,
            **validation["metrics"],
        }
    ]


def serializable_config(config: Split333RetrievalConfig) -> dict[str, Any]:
    return asdict(config)


def _invalid_result(
    config: Split333RetrievalConfig,
    *,
    existence_checks: dict[str, bool],
    checks: dict[str, bool],
    metrics: dict[str, Any],
    artifact_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "source_run_id": config.source_run_id,
        "status": "fail",
        "decision": "innovation2_present_r9_split333_retrieval_invalid",
        "existence_checks": existence_checks,
        "checks": checks,
        "metrics": metrics,
        "artifact_manifest": artifact_manifest,
        "next_action": {
            "action": "keep raw fallback incomplete and wait for all terminal artifacts",
            "e105_open": False,
        },
    }


def _audit_candidate_cache(root: Path, *, parameter_hash: str) -> dict[str, Any]:
    paths = tuple(sorted(root.glob("*.json")))
    aggregate = hashlib.sha256()
    valid = True
    errors: list[str] = []
    identities: set[tuple[int, int, int]] = set()
    total_bytes = 0
    for path in paths:
        total_bytes += path.stat().st_size
        file_hash = _sha256(path)
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(file_hash.encode("ascii"))
        try:
            envelope = _load_json(path)
            payload = envelope["payload"]
            expected = envelope["payload_sha256"]
            actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
            identity = (int(payload["layer"]), int(payload["u"]), int(payload["v"]))
            if expected != actual:
                raise ValueError("payload checksum mismatch")
            if payload.get("parameter_hash") != parameter_hash:
                raise ValueError("parameter hash mismatch")
            if identity in identities:
                raise ValueError("duplicate candidate identity")
            identities.add(identity)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
            valid = False
            errors.append(f"{path.name}: {error}")
    return {
        "valid": valid,
        "files": len(paths),
        "bytes": total_bytes,
        "unique_identities": len(identities),
        "aggregate_sha256": aggregate.hexdigest(),
        "errors": errors,
    }


def _probe_files(results_root: Path) -> tuple[Path, ...]:
    numbered: list[tuple[int, Path]] = []
    for path in results_root.glob("probe_*.json"):
        match = re.fullmatch(r"probe_(\d+)", path.stem)
        if match:
            numbered.append((int(match.group(1)), path))
    return tuple(path for _, path in sorted(numbered))


def _artifact_manifest(
    root: Path,
    required: dict[str, Path],
    candidate_paths: tuple[Path, ...],
) -> dict[str, Any]:
    critical = {
        name: {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in required.items()
        if path.is_file()
    }
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in candidate_paths:
        digest = _sha256(path)
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
        total_bytes += path.stat().st_size
    return {
        "critical_files": critical,
        "candidate_cache": {
            "files": len(candidate_paths),
            "bytes": total_bytes,
            "aggregate_sha256": aggregate.hexdigest(),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
