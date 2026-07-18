from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ATM_COMMIT,
    ATM_EXPECTED_RESULT_FILES,
    audit_atm_results,
    load_builtin_property_pickle,
)


AUDIT_ROUNDS = 9
AUDIT_MINIMUM_POSITIVES = 256
AUDIT_MINIMUM_NEGATIVES = 256


@dataclass(frozen=True)
class GeneralizedRelationContractConfig:
    run_id: str
    mode: str = "audit"
    rounds: int = AUDIT_ROUNDS
    minimum_positives: int = AUDIT_MINIMUM_POSITIVES
    minimum_negatives: int = AUDIT_MINIMUM_NEGATIVES
    expected_commit: str = ATM_COMMIT

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.rounds <= 0 or self.minimum_positives <= 0 or self.minimum_negatives <= 0:
            raise ValueError("round and label-width values must be positive")
        if len(self.expected_commit) != 40:
            raise ValueError("expected_commit must be a full hash")
        if self.mode == "audit" and (
            self.rounds != AUDIT_ROUNDS
            or self.minimum_positives != AUDIT_MINIMUM_POSITIVES
            or self.minimum_negatives != AUDIT_MINIMUM_NEGATIVES
            or self.expected_commit != ATM_COMMIT
        ):
            raise ValueError("E56 generalized-relation audit protocol is frozen")


def inspect_present_key_model(atm_root: Path) -> dict[str, Any]:
    notebook_path = atm_root / "Ciphers/PRESENT/PRESENT.ipynb"
    iterated_path = atm_root / "Construction/IteratedCipher.py"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    iterated_source = iterated_path.read_text(encoding="utf-8")
    splits = _literal_assignment(source, "splits")
    split_calls = [
        line.strip()
        for line in source.splitlines()
        if "construct_iterated_cipher" in line and "split[" in line
    ]
    key_mask_calls = [line for line in split_calls if "2**64-1" in line]
    checks = {
        "present_notebook_present": notebook_path.is_file(),
        "iterated_cipher_source_present": iterated_path.is_file(),
        "nine_round_split_loop_present": "for split in splits:" in source
        and isinstance(splits, (list, tuple))
        and bool(splits)
        and all(
            isinstance(split, (list, tuple))
            and len(split) == 3
            and sum(int(part) for part in split) == 9
            for split in splits
        ),
        "present_sbox_and_player_present": "PRESENT_sbox" in source
        and "PRESENT_bitpermutation" in source,
        "per_round_local_key_connections_present": "f.connect_to_key" in iterated_source,
        "all_round_key_masks_are_64_bit": len(key_mask_calls) >= 3,
    }
    semantics = {
        "independent_round_key_model": checks["per_round_local_key_connections_present"]
        and checks["all_round_key_masks_are_64_bit"],
        "present80_master_key_declared": "80" in source
        and "master" in source.lower()
        and "key" in source.lower(),
        "present80_key_schedule_implemented": any(
            token in source.lower()
            for token in ("key_schedule", "update_key", "rotate key", "key register")
        ),
        "actual_present80_key_model_match": False,
    }
    return {
        "notebook": str(notebook_path),
        "iterated_cipher": str(iterated_path),
        "checks": checks,
        "semantics": semantics,
        "split_constructor_calls": split_calls,
        "declared_splits": splits,
        "key_model": (
            "independent 64-bit local round-key variables on keyed boundaries; "
            "no PRESENT-80 master-key schedule"
        ),
    }


def audit_relation_overlap(results_root: Path) -> dict[str, Any]:
    file_properties = {
        name: load_builtin_property_pickle(results_root / name)
        for name in ATM_EXPECTED_RESULT_FILES
    }
    names = list(ATM_EXPECTED_RESULT_FILES)
    all_relations = set().union(*file_properties.values())
    common = set.intersection(*(set(file_properties[name]) for name in names))
    frequency = Counter(
        relation
        for properties in file_properties.values()
        for relation in properties
    )
    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = len(file_properties[left] & file_properties[right])
            union = len(file_properties[left] | file_properties[right])
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "intersection": overlap,
                    "jaccard": overlap / union if union else 0.0,
                }
            )
    relation_sizes = Counter(len(relation) for relation in all_relations)
    standard = [next(iter(relation)) for relation in all_relations if len(relation) == 1]
    linear = [coordinate for coordinate in standard if coordinate[1].bit_count() == 1]
    return {
        "files": [
            {
                "file": name,
                "relations": len(file_properties[name]),
                "common_to_all_files": len(file_properties[name] & common),
                "unique_to_file": sum(
                    frequency[relation] == 1 for relation in file_properties[name]
                ),
            }
            for name in names
        ],
        "pairwise": pairwise,
        "metrics": {
            "files": len(names),
            "deduplicated_relations": len(all_relations),
            "relations_common_to_all_files": len(common),
            "relations_unique_to_one_file": sum(value == 1 for value in frequency.values()),
            "maximum_file_frequency": max(frequency.values(), default=0),
            "minimum_pairwise_intersection": min(
                (row["intersection"] for row in pairwise), default=0
            ),
            "maximum_pairwise_intersection": max(
                (row["intersection"] for row in pairwise), default=0
            ),
            "relation_size_histogram": {
                str(size): count for size, count in sorted(relation_sizes.items())
            },
            "standard_basis_relations": len(standard),
            "linear_output_standard_relations": len(linear),
            "linear_input_weight_histogram": _histogram(
                coordinate[0].bit_count() for coordinate in linear
            ),
            "proven_key_dependent_negative_relations": 0,
            "negative_witnesses": 0,
        },
        "checks": {
            "all_expected_files_loaded": len(file_properties)
            == len(ATM_EXPECTED_RESULT_FILES),
            "deduplicated_positive_relations_exist": bool(all_relations),
            "file_split_has_shared_relations": bool(common),
            "file_split_is_relation_disjoint": not common,
            "proven_negative_relations_exist": False,
            "negative_witnesses_exist": False,
        },
    }


def evaluate_generalized_relation_contract(
    config: GeneralizedRelationContractConfig,
    *,
    actual_commit: str,
    atm_audit: dict[str, Any],
    key_model: dict[str, Any],
    overlap: dict[str, Any],
) -> dict[str, Any]:
    source_checks = {
        "commit_matches_frozen_version": actual_commit == config.expected_commit,
        **{f"atm_{key}": bool(value) for key, value in atm_audit["checks"].items()},
        **{f"source_{key}": bool(value) for key, value in key_model["checks"].items()},
        "all_expected_relation_files_loaded": overlap["checks"][
            "all_expected_files_loaded"
        ],
    }
    metrics = {
        **atm_audit["metrics"],
        **overlap["metrics"],
        "minimum_positive_relations": config.minimum_positives,
        "minimum_negative_relations": config.minimum_negatives,
    }
    generalized_checks = {
        "positive_membership_semantics_defined": True,
        "deduplicated_positive_width_at_least_256": metrics["deduplicated_relations"]
        >= config.minimum_positives,
        "proven_negative_width_at_least_256": metrics[
            "proven_key_dependent_negative_relations"
        ]
        >= config.minimum_negatives,
        "negative_witnesses_replayable": overlap["checks"]["negative_witnesses_exist"],
        "public_file_split_relation_disjoint": overlap["checks"][
            "file_split_is_relation_disjoint"
        ],
        "relation_size_degree_file_controls_definable": True,
    }
    original_target_checks = {
        "actual_present80_master_key_schedule": key_model["semantics"][
            "actual_present80_key_model_match"
        ],
        "constant_zero_or_one_known": atm_audit["semantic_checks"][
            "constant_value_zero_or_one_is_known"
        ],
        "linear_output_candidates_exist": atm_audit["semantic_checks"][
            "linear_output_candidates_exist"
        ],
        "eight_bit_affine_cube_and_offset_recoverable": False,
        "original_linear_mask_balance_mapping_complete": False,
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_generalized_relation_contract_protocol_invalid"
        action = "repair source commit, safe parsing, or notebook contract extraction"
    elif all(generalized_checks.values()) and all(original_target_checks.values()):
        status = "pass"
        decision = "innovation2_generalized_relation_original_target_ready"
        action = "build a PRESENT-80 linear-mask balance atlas before neural training"
    elif all(generalized_checks.values()):
        status = "pass"
        decision = "innovation2_generalized_relation_extension_ready"
        action = "preregister a separate generalized-relation prediction benchmark"
    else:
        status = "hold"
        decision = "innovation2_generalized_relation_label_contract_not_ready"
        action = (
            "keep neural training closed; require actual PRESENT-80 key scheduling and "
            "replayable key-dependent negatives before a generalized-relation benchmark"
        )
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "generalized_relation_checks": generalized_checks,
        "original_target_checks": original_target_checks,
        "metrics": metrics,
        "claim_scope": (
            "read-only PRESENT r9 generalized-relation label-contract audit at the "
            "frozen ATM commit; not PRESENT-80 key-schedule labels, zero-valued "
            "balance labels, neural training, an attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "neural_matrix_open": status == "pass",
            "closed_routes": (
                []
                if status == "pass"
                else [
                    "treating unknown relations as strict negatives",
                    "using file identity as a train-validation split",
                    "calling independent-round-key relations PRESENT-80 labels",
                    "mapping unknown constants to balanced zero",
                    "training generalized-relation neural models before negative witnesses",
                ]
            ),
        },
    }
    rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_generalized_integral_relation_contract",
            "contract": contract,
            "passed": bool(value),
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for group in (source_checks, generalized_checks, original_target_checks)
        for contract, value in group.items()
    ]
    return {"gate": gate, "metrics": metrics, "result_rows": rows}


def serializable_config(config: GeneralizedRelationContractConfig) -> dict[str, Any]:
    return asdict(config)


def _histogram(values: Any) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): count for key, count in sorted(counts.items())}


def _literal_assignment(source: str, name: str) -> Any:
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    return None
