from __future__ import annotations

import importlib.util
import pickle
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


CLAASP_COMMIT = "f2239d639ae5c4a013947ce9121c6f4464584758"
ATM_COMMIT = "b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b"
ATM_PUBLISHED_DIMENSION = 470
ATM_EXPECTED_RESULT_FILES = (
    "R9-complex-oracle-1-5-3.pkl",
    "R9-complex-oracle-1-6-2.pkl",
    "R9-complex-oracle-1-7-1.pkl",
    "R9-complex-oracle-2-4-3.pkl",
    "R9-complex-oracle-2-5-2.pkl",
    "R9-complex-oracle-2-6-1.pkl",
    "R9-complex-oracle-3-4-2.pkl",
    "R9-complex-oracle-3-5-1.pkl",
)
Coordinate = tuple[int, int]
Property = frozenset[Coordinate]


@dataclass(frozen=True)
class ProviderAuditConfig:
    run_id: str
    claasp_commit: str = CLAASP_COMMIT
    atm_commit: str = ATM_COMMIT
    published_atm_dimension: int = ATM_PUBLISHED_DIMENSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if len(self.claasp_commit) != 40 or len(self.atm_commit) != 40:
            raise ValueError("provider commits must be full 40-character hashes")
        if self.published_atm_dimension <= 0:
            raise ValueError("published_atm_dimension must be positive")


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"global class loading is forbidden: {module}.{name}"
        )


def load_builtin_property_pickle(path: Path) -> set[Property]:
    if path.stat().st_size > 10_000_000:
        raise ValueError(f"property pickle is unexpectedly large: {path.name}")
    with path.open("rb") as handle:
        payload = _restricted_load(handle)
    if not isinstance(payload, set):
        raise ValueError(f"property pickle must contain a set: {path.name}")
    properties: set[Property] = set()
    for entry in payload:
        if not isinstance(entry, frozenset) or not entry:
            raise ValueError(f"property entries must be non-empty frozensets: {path.name}")
        coordinates: set[Coordinate] = set()
        for coordinate in entry:
            if (
                not isinstance(coordinate, tuple)
                or len(coordinate) != 2
                or type(coordinate[0]) is not int
                or type(coordinate[1]) is not int
                or coordinate[0] < 0
                or coordinate[1] < 0
            ):
                raise ValueError(f"invalid property coordinate in {path.name}")
            coordinates.add((coordinate[0], coordinate[1]))
        if len(coordinates) != len(entry):
            raise ValueError(f"duplicate property coordinate in {path.name}")
        properties.add(frozenset(coordinates))
    if len(properties) != len(payload):
        raise ValueError(f"duplicate property entry in {path.name}")
    return properties


def audit_atm_results(
    results_root: Path,
    *,
    expected_names: tuple[str, ...] = ATM_EXPECTED_RESULT_FILES,
    published_dimension: int = ATM_PUBLISHED_DIMENSION,
) -> dict[str, Any]:
    actual_names = tuple(sorted(path.name for path in results_root.glob("*.pkl")))
    file_names_match = actual_names == tuple(sorted(expected_names))
    rows: list[dict[str, Any]] = []
    all_properties: set[Property] = set()
    safe_shapes = True
    errors: list[str] = []
    for name in actual_names:
        try:
            properties = load_builtin_property_pickle(results_root / name)
        except (OSError, ValueError, pickle.UnpicklingError) as exc:
            safe_shapes = False
            errors.append(f"{name}: {exc}")
            continue
        singleton_count = sum(len(entry) == 1 for entry in properties)
        rows.append(
            {
                "file": name,
                "basis_elements": len(properties),
                "singleton_elements": singleton_count,
                "multi_term_elements": len(properties) - singleton_count,
            }
        )
        all_properties.update(properties)

    coordinates = sorted(
        {coordinate for entry in all_properties for coordinate in entry}
    )
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    bit_rows = tuple(
        sum(1 << coordinate_index[coordinate] for coordinate in entry)
        for entry in all_properties
    )
    pivots = _gf2_pivots(bit_rows)
    standard_members = tuple(
        coordinate
        for index, coordinate in enumerate(coordinates)
        if _is_in_span(1 << index, pivots)
    )
    linear_members = tuple(
        coordinate for coordinate in standard_members if coordinate[1].bit_count() == 1
    )
    return {
        "files": rows,
        "errors": errors,
        "checks": {
            "expected_eight_result_files_present": file_names_match,
            "safe_builtin_pickle_shapes": safe_shapes and bool(rows),
            "all_basis_entries_nonempty": bool(all_properties)
            and all(all(entry) for entry in all_properties),
            "all_coordinates_are_nonnegative_integer_pairs": bool(coordinates),
        },
        "metrics": {
            "result_files": len(actual_names),
            "unique_serialized_basis_elements": len(all_properties),
            "union_gf2_rank": len(pivots),
            "published_dimension": published_dimension,
            "support_coordinates": len(coordinates),
            "standard_basis_members": len(standard_members),
            "linear_output_standard_basis_members": len(linear_members),
            "multi_term_serialized_basis_elements": sum(
                len(entry) > 1 for entry in all_properties
            ),
            "input_exponent_weight_histogram": _histogram(
                coordinate[0].bit_count() for coordinate in standard_members
            ),
            "output_exponent_weight_histogram": _histogram(
                coordinate[1].bit_count() for coordinate in standard_members
            ),
        },
        "semantic_checks": {
            "linear_output_candidates_exist": bool(linear_members),
            "constant_value_zero_or_one_is_known": False,
            "absence_from_found_subspace_is_complete_negative": False,
            "multi_term_relations_are_single_mask_labels": False,
            "published_dimension_matches_recomputed_union_rank": len(pivots)
            == published_dimension,
        },
    }


def inspect_claasp_contract(root: Path) -> dict[str, Any]:
    present_path = root / "claasp/ciphers/block_ciphers/present_block_cipher.py"
    module_path = (
        root
        / "claasp/cipher_modules/models/milp/milp_models/Gurobi/monomial_prediction.py"
    )
    test_path = (
        root
        / "tests/unit/cipher_modules/models/milp/milp_models/Gurobi/monomial_prediction_test.py"
    )
    module_text = module_path.read_text(encoding="utf-8") if module_path.exists() else ""
    test_text = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
    gurobi_installed = importlib.util.find_spec("gurobipy") is not None
    return {
        "checks": {
            "present_model_present": present_path.is_file(),
            "monomial_prediction_module_present": module_path.is_file(),
            "module_declares_gurobi_license_requirement": (
                "Gurobi license" in module_text and "from gurobipy import" in module_text
            ),
            "tests_are_skipped_for_gurobi_license": (
                "Requires Gurobi license" in test_text
                and "pytest.mark.skip" in test_text
            ),
        },
        "runtime": {
            "sage_executable_available": shutil.which("sage") is not None,
            "gurobipy_available": gurobi_installed,
            "current_runtime_available": gurobi_installed,
        },
        "semantic_checks": {
            "selected_output_bit_api_present": (
                "find_superpoly_of_specific_output_bit" in module_text
            ),
            "current_runtime_can_verify_label_value": gurobi_installed,
            "current_runtime_can_generate_complete_negatives": False,
        },
    }


def evaluate_provider_contract(
    config: ProviderAuditConfig,
    *,
    claasp: dict[str, Any],
    atm: dict[str, Any],
    actual_claasp_commit: str,
    actual_atm_commit: str,
) -> dict[str, Any]:
    readiness = {
        "claasp_commit_matches_frozen_version": actual_claasp_commit
        == config.claasp_commit,
        "atm_commit_matches_frozen_version": actual_atm_commit == config.atm_commit,
        **{f"claasp_{key}": bool(value) for key, value in claasp["checks"].items()},
        **{f"atm_{key}": bool(value) for key, value in atm["checks"].items()},
    }
    claasp_ready = (
        bool(claasp["runtime"]["current_runtime_available"])
        and bool(claasp["semantic_checks"]["current_runtime_can_verify_label_value"])
        and bool(claasp["semantic_checks"]["current_runtime_can_generate_complete_negatives"])
    )
    atm_ready = all(
        bool(atm["semantic_checks"][key])
        for key in (
            "linear_output_candidates_exist",
            "constant_value_zero_or_one_is_known",
            "absence_from_found_subspace_is_complete_negative",
            "published_dimension_matches_recomputed_union_rank",
        )
    )
    contract_checks = {
        "claasp_provider_ready": claasp_ready,
        "atm_provider_ready": atm_ready,
        "at_least_one_provider_ready": claasp_ready or atm_ready,
    }
    if not readiness or not all(readiness.values()):
        status = "fail"
        decision = "innovation2_deterministic_provider_protocol_invalid"
        action = "repair source-version, safe-pickle, file-set, or provider contract validation"
    elif contract_checks["at_least_one_provider_ready"]:
        status = "pass"
        decision = "innovation2_deterministic_provider_ready"
        action = "build E32 high-round structure-by-linear-mask label atlas and shortcut gate"
    elif (
        not atm["semantic_checks"]["constant_value_zero_or_one_is_known"]
        or not atm["semantic_checks"]["absence_from_found_subspace_is_complete_negative"]
    ):
        status = "hold"
        decision = "innovation2_deterministic_provider_semantics_mismatch"
        action = (
            "do not train on unknown constants or incomplete negatives; prepare an exact "
            "small-state SPN label-width audit while retaining these tools as controls"
        )
    else:
        status = "hold"
        decision = "innovation2_deterministic_provider_runtime_unavailable"
        action = "evaluate an open executable provider or exact small-state SPN labels"

    rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_deterministic_label_provider_contract",
            "provider": "CLAASP-MP",
            "source_commit": actual_claasp_commit,
            "current_runtime_available": claasp["runtime"]["current_runtime_available"],
            "linear_output_candidates_exist": claasp["semantic_checks"][
                "selected_output_bit_api_present"
            ],
            "label_value_known": claasp["semantic_checks"][
                "current_runtime_can_verify_label_value"
            ],
            "complete_negative_semantics": claasp["semantic_checks"][
                "current_runtime_can_generate_complete_negatives"
            ],
            "training_performed": False,
        },
        {
            "run_id": config.run_id,
            "task": "innovation2_deterministic_label_provider_contract",
            "provider": "AlgebraicTransitionMatrices",
            "source_commit": actual_atm_commit,
            "current_runtime_available": False,
            "linear_output_candidates_exist": atm["semantic_checks"][
                "linear_output_candidates_exist"
            ],
            "label_value_known": atm["semantic_checks"][
                "constant_value_zero_or_one_is_known"
            ],
            "complete_negative_semantics": atm["semantic_checks"][
                "absence_from_found_subspace_is_complete_negative"
            ],
            "published_dimension_matches": atm["semantic_checks"][
                "published_dimension_matches_recomputed_union_rank"
            ],
            "training_performed": False,
        },
    ]
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "contract_checks": contract_checks,
        "metrics": atm["metrics"],
        "provider_states": {
            "claasp": {**claasp["runtime"], **claasp["semantic_checks"]},
            "atm": atm["semantic_checks"],
        },
        "claim_scope": (
            "read-only source and precomputed-result contract audit at frozen public commits; "
            "not a reproduction of CLAASP-MP, the complete 470-dimensional ATM space, neural "
            "training, or evidence that an absent candidate is not an integral property"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
        },
    }
    return {
        "rows": rows,
        "gate": gate,
        "summary": {
            "run_id": config.run_id,
            "claasp": claasp,
            "atm": atm,
            "gate": gate,
        },
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_deterministic_label_provider_contract",
            "claasp_commit": actual_claasp_commit,
            "atm_commit": actual_atm_commit,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def _restricted_load(handle: BinaryIO) -> Any:
    return RestrictedUnpickler(handle).load()


def _gf2_pivots(rows: Iterable[int]) -> dict[int, int]:
    pivots: dict[int, int] = {}
    for value in rows:
        row = int(value)
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return pivots


def _is_in_span(row: int, pivots: dict[int, int]) -> bool:
    value = row
    while value:
        pivot = value.bit_length() - 1
        if pivot not in pivots:
            return False
        value ^= pivots[pivot]
    return True


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}
