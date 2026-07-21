from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ATM_COMMIT,
    Property,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_source_heldout_ranking import (
    _fold_training_overlap_audit,
    _source_novelty_audit,
    load_relations_json,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_neural_ranking import (
    PuNeuralRankingConfig,
    build_neural_folds,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    _canonical_coordinates,
    _relation_id,
    _rotation_candidates,
    audit_sources,
    load_relation_groups,
)


RUN_ID = "i2_present_r9_external_relation_source_readiness_20260721"
EXPECTED_E104_VALIDATION_SHA256 = (
    "53a46bf81db82cd980626dac16962ffe1e3dc7fe33f36e948001a24a6ef86418"
)
EXPECTED_E101_GATE_SHA256 = (
    "056366dcbe692306c830284348b0a28fbbd4dc685c9dc671237c7bf1a5519933"
)
EXPECTED_PAPER_MANIFEST_SHA256 = (
    "1f9edca7a34d0614be049af4ac1e43fad8c4931f99ded912cf1ebf0adbd2ac30"
)
EXPECTED_HWANG_TEXT_SHA256 = (
    "0be4709becf585bae2898f7275a70668d5e5aeedd9dc60ae7577f8a7c85f6e0d"
)
EXPECTED_SPLIT_TEXT_SHA256 = (
    "607f1d4db22fad31afe9600efa3dfbaf7d838d5a4c83ed0d92b3bb96e994f7f2"
)
EXPECTED_CLAASP_TEXT_SHA256 = (
    "255906a1bf3259b750e7aee099da41a3fa42f090f0f43dfb49d3a0ed822ecf2e"
)
EXPECTED_SPLIT_REPOSITORY_COMMIT = "aac4ab4d7430e4add3689214c9e69412a89d8fc1"
EXPECTED_SPLIT_README_SHA256 = (
    "ee4892cae6908b84fbf6205ffde9231900a8e0a2f0cbe5927fca1f39dc3d822b"
)


@dataclass(frozen=True)
class ExternalRelationSourceConfig:
    run_id: str = RUN_ID
    minimum_new_dimensions: int = 32

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E106 run_id is frozen")
        if self.minimum_new_dimensions != 32:
            raise ValueError("E106 novelty width is frozen")


def audit_external_relation_sources(
    config: ExternalRelationSourceConfig,
    *,
    public_results_root: Path,
    e104_root: Path,
    e101_gate: dict[str, Any],
    e101_gate_sha256: str,
    paper_manifest: Path,
    hwang_text: Path,
    split_text: Path,
    claasp_text: Path,
    split_repository: dict[str, Any],
) -> dict[str, Any]:
    public_audit = audit_sources(public_results_root, actual_commit=ATM_COMMIT)
    public_groups = load_relation_groups(public_results_root)
    public_relations = set().union(*public_groups.values())
    e104_validation_path = e104_root / "validation.json"
    e104_relations_path = e104_root / "results/relations.json"
    e104_validation = _load_json(e104_validation_path)
    e104_relations = load_relations_json(e104_relations_path)
    novelty = _source_novelty_audit(
        public_relations=public_relations,
        heldout_relations=e104_relations,
    )
    fold_overlap = _e104_fold_overlap(
        public_groups=public_groups,
        public_relations=public_relations,
        e104_relations=e104_relations,
    )
    relation_manifest = (
        e104_validation.get("artifact_manifest", {})
        .get("critical_files", {})
        .get("generation_relations", {})
    )
    hwang_content = hwang_text.read_text(encoding="utf-8")
    split_content = split_text.read_text(encoding="utf-8")
    claasp_content = claasp_text.read_text(encoding="utf-8")
    protocol_checks = {
        **{f"public_{name}": passed for name, passed in public_audit["checks"].items()},
        "public_rank_is_468": novelty["public_relation_rank"] == 468,
        "public_serialized_relations_are_470": novelty["public_relations"] == 470,
        "e104_validation_hash_matches": sha256(e104_validation_path)
        == EXPECTED_E104_VALIDATION_SHA256,
        "e104_validation_passes": e104_validation.get("status") == "pass"
        and e104_validation.get("decision")
        == "innovation2_present_r9_split333_retrieval_verified",
        "e104_relations_hash_matches_manifest": sha256(e104_relations_path)
        == relation_manifest.get("sha256"),
        "e101_gate_hash_matches": e101_gate_sha256 == EXPECTED_E101_GATE_SHA256,
        "e101_confirms_no_public_r10_results": e101_gate.get("metrics", {}).get(
            "public_r10_results"
        )
        == 0,
        "paper_manifest_hash_matches": sha256(paper_manifest)
        == EXPECTED_PAPER_MANIFEST_SHA256,
        "hwang_text_hash_matches": sha256(hwang_text) == EXPECTED_HWANG_TEXT_SHA256,
        "hwang_present_r9_four_masks_present": all(
            text in hwang_content
            for text in (
                "9-round has the same balance",
                "b4 ⊕ b12",
                "b16 ⊕ b48",
                "b20 ⊕ b28 ⊕ b52 ⊕ b60",
            )
        ),
        "split_text_hash_matches": sha256(split_text) == EXPECTED_SPLIT_TEXT_SHA256,
        "split_present_r9_rows_present": _split_present_rows_present(split_content),
        "claasp_text_hash_matches": sha256(claasp_text) == EXPECTED_CLAASP_TEXT_SHA256,
        "claasp_has_no_frozen_present_r9_result": "our objective is not to derive new"
        in claasp_content,
        "split_repository_commit_matches": split_repository.get("commit")
        == EXPECTED_SPLIT_REPOSITORY_COMMIT,
        "split_repository_readme_hash_matches": split_repository.get("readme_sha256")
        == EXPECTED_SPLIT_README_SHA256,
        "split_repository_is_documented_stub": split_repository.get("tracked_files")
        == ["README.md"]
        and split_repository.get("readme_text", "").strip()
        == "# splitandcancel\n\nWe will upload the codes soon. \nContact: `hossein.hadipour@rub.de`",
    }
    source_rows = _source_rows(
        config,
        novelty=novelty,
        fold_overlap=fold_overlap,
    )
    gate = adjudicate_external_relation_sources(
        config,
        protocol_checks=protocol_checks,
        source_rows=source_rows,
    )
    return {
        "protocol_checks": protocol_checks,
        "public_audit": public_audit,
        "e104_novelty": novelty,
        "e104_fold_overlap": fold_overlap,
        "source_rows": source_rows,
        "gate": gate,
    }


def adjudicate_external_relation_sources(
    config: ExternalRelationSourceConfig,
    *,
    protocol_checks: dict[str, bool],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [row for row in source_rows if row["direct_e99_eligible"]]
    candidates = [row for row in source_rows if row["role"] == "candidate"]
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_external_source_audit_protocol_invalid"
        action = "repair frozen source, paper, repository, or E104 evidence only"
    elif eligible:
        status = "pass"
        decision = "innovation2_present_r9_external_relation_source_ready"
        action = "preregister frozen-checkpoint zero-adaptation evaluation on the eligible source"
    else:
        status = "hold"
        decision = "innovation2_present_r9_external_relation_source_unavailable"
        action = (
            "stop E99 coordinate-identity transfer; audit deterministic reproduction of the "
            "four Hwang PRESENT r9 output masks as a new target representation"
        )
    known_novelty = [
        int(row["new_relation_space_dimensions"])
        for row in candidates
        if row["new_relation_space_dimensions"] is not None
    ]
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "metrics": {
            "candidate_sources": len(candidates),
            "eligible_external_sources": len(eligible),
            "machine_readable_candidate_sources": sum(
                bool(row["machine_readable_relations"]) for row in candidates
            ),
            "same_atm_semantics_candidate_sources": sum(
                bool(row["same_relation_semantics"]) for row in candidates
            ),
            "maximum_known_new_dimensions": max(known_novelty, default=0),
            "minimum_required_new_dimensions": config.minimum_new_dimensions,
        },
        "claim_scope": (
            "local zero-training source-eligibility audit for direct transfer of the frozen "
            "PRESENT r9 ATM relation-ranking model; paper output masks, weak-key observables, "
            "missing artifacts, and different-round results are not converted into ATM relations"
        ),
        "next_action": {
            "action": action,
            "e99_evaluation_open": bool(eligible) and status == "pass",
            "training": False,
            "remote_scale": False,
            "r10_generation": False,
            "target_representation_change_required": status == "hold",
        },
    }


def result_rows(
    config: ExternalRelationSourceConfig,
    *,
    source_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_external_relation_source_readiness",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
        "search_performed": False,
    }
    return [{**common, **row} for row in source_rows]


def serializable_config(config: ExternalRelationSourceConfig) -> dict[str, Any]:
    return asdict(config)


def _source_rows(
    config: ExternalRelationSourceConfig,
    *,
    novelty: dict[str, Any],
    fold_overlap: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _row(
            "atm_public_r9_eight",
            "ATM公开8个R9 split（训练锚点）",
            role="training_anchor",
            artifact_state="machine_readable",
            machine=True,
            same_rounds=True,
            same_key_model=True,
            same_semantics=True,
            disjoint=False,
            new_dimensions=0,
            reason="training source, not an external candidate",
        ),
        _row(
            "atm_e104_split333",
            "E104 ATM R9 (3,3,3)",
            role="candidate",
            artifact_state="verified_machine_readable",
            machine=True,
            same_rounds=True,
            same_key_model=True,
            same_semantics=True,
            disjoint=novelty["heldout_exact_public_overlap"] == 0
            and fold_overlap["maximum_fold_training_overlap"] == 0,
            new_dimensions=novelty["new_relation_space_dimensions"],
            reason="318 exact overlaps, 3 exact-new relations, zero new GF(2) dimensions",
        ),
        _row(
            "atm_r10_declared",
            "ATM声明的9个R10 split",
            role="candidate",
            artifact_state="declared_without_result",
            machine=False,
            same_rounds=False,
            same_key_model=True,
            same_semantics=True,
            disjoint=None,
            new_dimensions=None,
            reason="zero public result files and wrong round target for frozen E99",
        ),
        _row(
            "hwang_present_r9_masks",
            "Hwang 2026 PRESENT R9四个输出mask",
            role="candidate",
            artifact_state="paper_masks_only",
            machine=False,
            same_rounds=True,
            same_key_model=False,
            same_semantics=False,
            disjoint=None,
            new_dimensions=None,
            reason="PRESENT-80 output-mask kernel, not independent-round-key ATM relations",
            alternative_basis_dimension=4,
        ),
        _row(
            "splitandcancel_present",
            "Split-and-Cancel PRESENT输出组合",
            role="candidate",
            artifact_state="repository_stub_no_results",
            machine=False,
            same_rounds=True,
            same_key_model=False,
            same_semantics=False,
            disjoint=None,
            new_dimensions=None,
            reason="output-observable and weak-key semantics; public repository has README only",
            alternative_basis_dimension=3,
        ),
        _row(
            "claasp_mp_present",
            "CLAASP-MP通用monomial方法",
            role="candidate",
            artifact_state="method_code_without_present_r9_result",
            machine=False,
            same_rounds=False,
            same_key_model=False,
            same_semantics=False,
            disjoint=None,
            new_dimensions=None,
            reason="no frozen PRESENT r9 result; cube/monomial semantics differ",
        ),
    ]
    for row in rows:
        row["minimum_novelty_met"] = (
            row["new_relation_space_dimensions"] is not None
            and row["new_relation_space_dimensions"] >= config.minimum_new_dimensions
        )
        row["direct_e99_eligible"] = row["role"] == "candidate" and all(
            (
                row["machine_readable_relations"],
                row["same_rounds"],
                row["same_key_model"],
                row["same_relation_semantics"],
                row["training_identity_disjoint"],
                row["minimum_novelty_met"],
            )
        )
    return rows


def _row(
    source_id: str,
    source_name: str,
    *,
    role: str,
    artifact_state: str,
    machine: bool,
    same_rounds: bool,
    same_key_model: bool,
    same_semantics: bool,
    disjoint: bool | None,
    new_dimensions: int | None,
    reason: str,
    alternative_basis_dimension: int | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_name": source_name,
        "role": role,
        "artifact_state": artifact_state,
        "source_verified": True,
        "machine_readable_relations": machine,
        "same_cipher": True,
        "same_rounds": same_rounds,
        "same_key_model": same_key_model,
        "same_relation_semantics": same_semantics,
        "training_identity_disjoint": disjoint,
        "new_relation_space_dimensions": new_dimensions,
        "alternative_basis_dimension": alternative_basis_dimension,
        "reason": reason,
    }


def _e104_fold_overlap(
    *,
    public_groups: dict[str, set[Property]],
    public_relations: set[Property],
    e104_relations: set[Property],
) -> dict[str, Any]:
    all_known = public_relations | e104_relations
    pools = tuple(
        {
            "positive": relation,
            "positive_id": _relation_id(relation),
            "unlabeled_relations": _rotation_candidates(relation, all_known),
            "relations": (relation, *_rotation_candidates(relation, all_known)),
        }
        for relation in sorted(e104_relations, key=_canonical_coordinates)
    )
    return _fold_training_overlap_audit(
        fold_audit=build_neural_folds(public_groups, PuNeuralRankingConfig()),
        evaluation_pools=pools,
        heldout_relations=e104_relations,
    )


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _split_present_rows_present(text: str) -> bool:
    exact_row = re.search(
        r"^\s*9\s+260\s+3\s+1\s+16\.37\s+Subsection 5\.1\s*$",
        text,
        flags=re.MULTILINE,
    )
    weak_key_row = re.search(
        r"^\s*9\s+260\s+4\s+≥\s*2−2\s+16\.37\s+Subsection 5\.1\s*$",
        text,
        flags=re.MULTILINE,
    )
    return exact_row is not None and weak_key_row is not None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
