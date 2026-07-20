from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ATM_COMMIT,
    ATM_EXPECTED_RESULT_FILES,
    Property,
    load_builtin_property_pickle,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    EXPECTED_FILE_HASHES,
)


RUN_ID = "i2_present_r9_atm_basis_merge_source_audit_20260720"
E98_GATE_SHA256 = "f4c560233616c720f8a9b7eea1bc93e29cda69978ce908b5cd9f8f01fc23bc5c"
EXPECTED_SOURCE_HASHES = {
    "Ciphers/PRESENT/PRESENT.ipynb": (
        "8bf13c770dfd66eb3fcaa5568282d16fe82d1043d2ddc2f5209a89c023b7ca92"
    ),
    "Ciphers/PRESENT/data_analysis.ipynb": (
        "c7e1ee11766e1445b348cb2ea89b03e6c5d6bed47ce649ab19ee082fb7ef4f6f"
    ),
    "Tools/BasisTools.py": (
        "a2e3adb0a7756534a2a09fd7eb18045333ad6e62ae487840570a8c7ff031a091"
    ),
}
EXPECTED_PAPER_HASH = "e9e85ff04275213234d9b991f1c126e0e48a011a9af019632b8de78526d94255"
EXPECTED_DECLARED_SPLITS = (
    (1, 7, 1),
    (1, 6, 2),
    (1, 5, 3),
    (2, 6, 1),
    (2, 5, 2),
    (2, 4, 3),
    (3, 5, 1),
    (3, 4, 2),
    (3, 3, 3),
)
EXPECTED_PUBLISHED_SPLITS = EXPECTED_DECLARED_SPLITS[:-1]
EXPECTED_FILE_DIMENSIONS = {
    (1, 7, 1): 455,
    (1, 6, 2): 420,
    (1, 5, 3): 338,
    (2, 6, 1): 425,
    (2, 5, 2): 401,
    (2, 4, 3): 331,
    (3, 5, 1): 338,
    (3, 4, 2): 331,
}


@dataclass(frozen=True)
class AtmBasisMergeAuditConfig:
    run_id: str = RUN_ID

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E98-A run_id is frozen")


def audit_source_contract(
    atm_root: Path,
    paper_text: Path,
    *,
    actual_commit: str,
) -> dict[str, Any]:
    results_root = atm_root / "Ciphers/PRESENT/Results"
    source_hashes = {
        relative: _sha256(atm_root / relative)
        for relative in EXPECTED_SOURCE_HASHES
    }
    result_hashes = {
        name: _sha256(results_root / name) for name in ATM_EXPECTED_RESULT_FILES
    }
    paper_hash = _sha256(paper_text)
    present_notebook = _load_notebook(atm_root / "Ciphers/PRESENT/PRESENT.ipynb")
    analysis_notebook = _load_notebook(
        atm_root / "Ciphers/PRESENT/data_analysis.ipynb"
    )
    declared_splits = tuple(_notebook_assignment(present_notebook, "splits", 7))
    analysis_splits = tuple(_notebook_assignment(analysis_notebook, "splits", 1))
    saved_dimensions = _saved_split_dimensions(present_notebook, 7)
    analysis_merge_count = _single_integer_output(analysis_notebook, 2)
    paper = paper_text.read_text(encoding="utf-8")
    paper_splits = _paper_table_splits(paper)
    paper_dimension = _paper_claimed_dimension(paper)
    merge_contract = inspect_merge_bases_source(atm_root / "Tools/BasisTools.py")
    checks = {
        "commit_matches_frozen_version": actual_commit == ATM_COMMIT,
        "source_hashes_match": source_hashes == EXPECTED_SOURCE_HASHES,
        "result_hashes_match_e98": result_hashes == EXPECTED_FILE_HASHES,
        "paper_hash_matches": paper_hash == EXPECTED_PAPER_HASH,
        "notebook_declares_frozen_nine_splits": declared_splits
        == EXPECTED_DECLARED_SPLITS,
        "data_analysis_loads_frozen_eight_splits": analysis_splits
        == EXPECTED_PUBLISHED_SPLITS,
        "notebook_saved_outputs_cover_eight_splits": set(saved_dimensions)
        == set(EXPECTED_PUBLISHED_SPLITS),
        "paper_table_covers_eight_splits": _same_split_coverage(
            paper_splits, EXPECTED_PUBLISHED_SPLITS
        ),
        "paper_claims_dimension_470": paper_dimension == 470,
        "data_analysis_saved_merge_count_is_470": analysis_merge_count == 470,
        "merge_function_found": merge_contract["function_found"],
    }
    return {
        "actual_commit": actual_commit,
        "checks": checks,
        "source_hashes": {
            **source_hashes,
            **{f"Ciphers/PRESENT/Results/{name}": value for name, value in result_hashes.items()},
            "paper_text": paper_hash,
        },
        "declared_splits": declared_splits,
        "analysis_splits": analysis_splits,
        "saved_dimensions": saved_dimensions,
        "paper_splits": paper_splits,
        "paper_dimension": paper_dimension,
        "analysis_merge_count": analysis_merge_count,
        "merge_contract": merge_contract,
    }


def inspect_merge_bases_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "merge_bases"
        ),
        None,
    )
    if function is None:
        return {
            "function_found": False,
            "row_reduce_calls": 0,
            "discarded_row_reduce_calls": 0,
            "captured_row_reduce_calls": 0,
            "row_reduce_return_value_applied": False,
        }
    parent = {
        child: node
        for node in ast.walk(function)
        for child in ast.iter_child_nodes(node)
    }
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "row_reduce"
    ]
    discarded = sum(isinstance(parent.get(call), ast.Expr) for call in calls)
    captured = len(calls) - discarded
    return {
        "function_found": True,
        "row_reduce_calls": len(calls),
        "discarded_row_reduce_calls": discarded,
        "captured_row_reduce_calls": captured,
        "row_reduce_return_value_applied": bool(captured),
    }


def audit_relation_bases(results_root: Path) -> dict[str, Any]:
    groups = {
        name: load_builtin_property_pickle(results_root / name)
        for name in ATM_EXPECTED_RESULT_FILES
    }
    file_rows: list[dict[str, Any]] = []
    for name in ATM_EXPECTED_RESULT_FILES:
        split = _split_from_name(name)
        serialized = len(groups[name])
        rank = _relation_rank(groups[name])
        file_rows.append(
            {
                "file": name,
                "split": _split_text(split),
                "serialized_basis_elements": serialized,
                "recomputed_rank": rank,
                "rank_deficiency": serialized - rank,
                "expected_dimension": EXPECTED_FILE_DIMENSIONS[split],
                "full_rank": serialized == rank == EXPECTED_FILE_DIMENSIONS[split],
            }
        )

    relations = tuple(
        sorted(set().union(*groups.values()), key=_canonical_coordinates)
    )
    coordinates = tuple(
        sorted({coordinate for relation in relations for coordinate in relation})
    )
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    bit_rows = tuple(
        sum(1 << coordinate_index[coordinate] for coordinate in relation)
        for relation in relations
    )
    rank, dependencies = _gf2_rank_and_dependencies(bit_rows)
    dependency_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for dependency_index, combination in enumerate(dependencies):
        dependency_id = f"dependency_{dependency_index:03d}"
        member_indices = tuple(
            index for index in range(len(relations)) if combination & (1 << index)
        )
        xor_coordinates: set[tuple[int, int]] = set()
        for member_order, relation_index in enumerate(member_indices):
            relation = relations[relation_index]
            xor_coordinates.symmetric_difference_update(relation)
            member_rows.append(
                {
                    "dependency_id": dependency_id,
                    "member_order": member_order,
                    "relation_index": relation_index,
                    "relation_id": _relation_id(relation),
                    "relation_terms": len(relation),
                    "coordinates": json.dumps(
                        _canonical_coordinates(relation), separators=(",", ":")
                    ),
                    "source_files": "|".join(
                        name for name, group in groups.items() if relation in group
                    ),
                }
            )
        dependency_rows.append(
            {
                "dependency_id": dependency_id,
                "members": len(member_indices),
                "member_relation_ids": "|".join(
                    _relation_id(relations[index]) for index in member_indices
                ),
                "coordinate_xor_is_zero": not xor_coordinates,
                "coefficient_mask_hex": hex(combination),
            }
        )
    return {
        "groups": groups,
        "file_rows": file_rows,
        "dependency_rows": dependency_rows,
        "member_rows": member_rows,
        "metrics": {
            "published_files": len(groups),
            "serialized_references": sum(len(group) for group in groups.values()),
            "deduplicated_relations": len(relations),
            "support_coordinates": len(coordinates),
            "recomputed_union_rank": rank,
            "union_nullity": len(relations) - rank,
            "recovered_dependencies": len(dependencies),
            "dependency_member_histogram": dict(
                sorted(Counter(row["members"] for row in dependency_rows).items())
            ),
            "all_files_individually_full_rank": all(
                row["full_rank"] for row in file_rows
            ),
            "all_dependencies_xor_to_zero": all(
                row["coordinate_xor_is_zero"] for row in dependency_rows
            ),
        },
    }


def build_split_coverage(
    source_audit: dict[str, Any],
    results_root: Path,
) -> list[dict[str, Any]]:
    saved = source_audit["saved_dimensions"]
    return [
        {
            "split": _split_text(split),
            "notebook_declared": split in source_audit["declared_splits"],
            "notebook_saved_output": split in saved,
            "notebook_saved_dimension": saved.get(split),
            "data_analysis_loaded": split in source_audit["analysis_splits"],
            "paper_table_reported": split in source_audit["paper_splits"],
            "result_pickle_present": (
                results_root / f"R9-complex-oracle-{_split_text(split)}.pkl"
            ).is_file(),
            "stats_file_present": (
                results_root / f"R9-complex-oracle-{_split_text(split)}-stats.txt"
            ).is_file(),
            "evidence_state": (
                "published_result"
                if split in source_audit["analysis_splits"]
                else "declared_without_public_result"
            ),
        }
        for split in EXPECTED_DECLARED_SPLITS
    ]


def adjudicate_basis_merge_audit(
    config: AtmBasisMergeAuditConfig,
    *,
    source_audit: dict[str, Any],
    relation_audit: dict[str, Any],
    split_rows: list[dict[str, Any]],
    e98_gate: dict[str, Any],
    e98_gate_hash: str,
) -> dict[str, Any]:
    metrics = {
        **relation_audit["metrics"],
        "paper_claimed_dimension": source_audit["paper_dimension"],
        "data_analysis_saved_merge_count": source_audit["analysis_merge_count"],
        "declared_splits": len(source_audit["declared_splits"]),
        "published_result_splits": sum(
            row["evidence_state"] == "published_result" for row in split_rows
        ),
        "missing_declared_splits": sum(
            row["evidence_state"] == "declared_without_public_result"
            for row in split_rows
        ),
        "e98_total_heldout_positives": e98_gate["metrics"][
            "total_heldout_positives"
        ],
        "e98_eligible_heldout_groups": e98_gate["metrics"][
            "eligible_heldout_groups"
        ],
    }
    merge = source_audit["merge_contract"]
    missing = [
        row for row in split_rows if row["evidence_state"] == "declared_without_public_result"
    ]
    protocol_checks = {
        **source_audit["checks"],
        "e98_gate_hash_matches": e98_gate_hash == E98_GATE_SHA256,
        "e98_gate_is_held": e98_gate["status"] == "hold",
        "e98_e99_is_closed": not e98_gate["next_action"]["e99_open"],
        "all_eight_files_individually_full_rank": metrics[
            "all_files_individually_full_rank"
        ],
        "all_dependencies_replay_to_zero": metrics[
            "all_dependencies_xor_to_zero"
        ],
    }
    explanation_checks = {
        "deduplicated_count_is_470": metrics["deduplicated_relations"] == 470,
        "corrected_union_rank_is_468": metrics["recomputed_union_rank"] == 468,
        "union_nullity_is_two": metrics["union_nullity"] == 2,
        "exactly_two_dependencies_recovered": metrics["recovered_dependencies"] == 2,
        "dependency_sizes_are_three_and_four": metrics[
            "dependency_member_histogram"
        ]
        == {3: 1, 4: 1},
        "merge_row_reduce_return_value_is_discarded": (
            merge["row_reduce_calls"] == 1
            and merge["discarded_row_reduce_calls"] == 1
            and not merge["row_reduce_return_value_applied"]
        ),
        "saved_470_equals_deduplicated_count_not_rank": (
            metrics["data_analysis_saved_merge_count"]
            == metrics["deduplicated_relations"]
            and metrics["data_analysis_saved_merge_count"]
            != metrics["recomputed_union_rank"]
        ),
        "exactly_one_declared_split_is_missing": len(missing) == 1,
        "missing_split_is_3_3_3": bool(missing)
        and missing[0]["split"] == "3-3-3",
        "missing_split_has_no_pickle_stats_or_saved_output": bool(missing)
        and not missing[0]["result_pickle_present"]
        and not missing[0]["stats_file_present"]
        and not missing[0]["notebook_saved_output"],
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_atm_basis_merge_audit_protocol_invalid"
        action = "repair frozen source, hash, safe parsing, E98 replay, or GF(2) dependency audit"
    elif all(explanation_checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_atm_public_merge_count_not_rank"
        action = (
            "treat the frozen public eight-file corpus as a 468-dimensional span; keep E99 "
            "closed and audit orbit-disjoint positive-family coverage before any neural run"
        )
    else:
        status = "hold"
        decision = "innovation2_present_r9_atm_470_468_mismatch_not_explained"
        action = "keep E99 closed and require an executable version-pinned author merge replay"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "explanation_checks": explanation_checks,
        "metrics": metrics,
        "merge_contract": merge,
        "claim_scope": (
            "frozen-source and GF(2) basis-merge audit of the public independent-round-key "
            "PRESENT r9 ATM corpus; it explains the executable 470-versus-468 count/rank "
            "mismatch but is not a correction of all historical environments, generation of "
            "the missing split, PRESENT-80 schedule validation, neural training, a distinguisher, "
            "an attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "e99_open": False,
            "remote_scale": False,
            "missing_split_generation": False,
            "next_adjudication": (
                "E98-B orbit-disjoint generalized-relation positive-family coverage audit"
                if status == "pass"
                else "version-pinned merge replay"
            ),
        },
    }


def result_rows(
    config: AtmBasisMergeAuditConfig,
    relation_audit: dict[str, Any],
    split_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_atm_basis_merge_source_audit",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
    }
    return [
        {**common, "result_kind": "file_rank", **row}
        for row in relation_audit["file_rows"]
    ] + [
        {**common, "result_kind": "dependency", **row}
        for row in relation_audit["dependency_rows"]
    ] + [
        {**common, "result_kind": "split", **row} for row in split_rows
    ]


def serializable_config(config: AtmBasisMergeAuditConfig) -> dict[str, Any]:
    return asdict(config)


def _gf2_rank_and_dependencies(rows: Iterable[int]) -> tuple[int, tuple[int, ...]]:
    pivots: dict[int, tuple[int, int]] = {}
    dependencies: list[int] = []
    for index, raw_value in enumerate(rows):
        row = int(raw_value)
        combination = 1 << index
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                pivot_row, pivot_combination = pivots[pivot]
                row ^= pivot_row
                combination ^= pivot_combination
            else:
                pivots[pivot] = (row, combination)
                break
        if not row:
            dependencies.append(combination)
    return len(pivots), tuple(dependencies)


def _relation_rank(relations: Iterable[Property]) -> int:
    relation_tuple = tuple(relations)
    coordinates = tuple(
        sorted({coordinate for relation in relation_tuple for coordinate in relation})
    )
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    rows = (
        sum(1 << coordinate_index[coordinate] for coordinate in relation)
        for relation in relation_tuple
    )
    return _gf2_rank_and_dependencies(rows)[0]


def _load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _notebook_assignment(
    notebook: dict[str, Any], name: str, cell_index: int
) -> list[tuple[int, int, int]]:
    source = "".join(notebook["cells"][cell_index].get("source", []))
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return [tuple(int(part) for part in split) for split in value]
    raise ValueError(f"missing notebook assignment: {name}")


def _saved_split_dimensions(
    notebook: dict[str, Any], cell_index: int
) -> dict[tuple[int, int, int], int]:
    text = "\n".join(
        "".join(output.get("text", []))
        for output in notebook["cells"][cell_index].get("outputs", [])
    )
    pattern = re.compile(
        r"\((\d+),\s*(\d+),\s*(\d+)\).*?dimension:\s*(\d+)", re.DOTALL
    )
    return {
        (int(left), int(middle), int(right)): int(dimension)
        for left, middle, right, dimension in pattern.findall(text)
    }


def _single_integer_output(notebook: dict[str, Any], cell_index: int) -> int:
    values = [
        value
        for output in notebook["cells"][cell_index].get("outputs", [])
        for value in output.get("data", {}).get("text/plain", [])
    ]
    if len(values) != 1:
        raise ValueError("expected exactly one notebook integer output")
    return int(values[0])


def _paper_table_splits(text: str) -> tuple[tuple[int, int, int], ...]:
    start = text.index("Table 1: Statistics on experiments on 9 rounds of Present")
    section = text[start : start + 1200]
    matches = re.findall(r"([123]),\s*([34567]),\s*([123])", section)
    return tuple((int(left), int(middle), int(right)) for left, middle, right in matches)


def _same_split_coverage(
    actual: tuple[tuple[int, int, int], ...],
    expected: tuple[tuple[int, int, int], ...],
) -> bool:
    return len(actual) == len(expected) and set(actual) == set(expected)


def _paper_claimed_dimension(text: str) -> int:
    match = re.search(
        r"9\s*rounds of Present, a subspace of constant generalized integral properties "
        r"of dimension\s+(\d+)",
        text,
    )
    if match is None:
        raise ValueError("paper dimension claim not found")
    return int(match.group(1))


def _split_from_name(name: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"R9-complex-oracle-(\d+)-(\d+)-(\d+)\.pkl", name)
    if match is None:
        raise ValueError(f"unexpected R9 filename: {name}")
    return tuple(int(part) for part in match.groups())


def _split_text(split: tuple[int, int, int]) -> str:
    return "-".join(str(part) for part in split)


def _canonical_coordinates(relation: Property) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(relation))


def _relation_id(relation: Property) -> str:
    payload = json.dumps(_canonical_coordinates(relation), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
