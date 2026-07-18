from __future__ import annotations

import importlib.util
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ACTIVE_DIMENSION,
    EXPECTED_OUTPUT_MASKS,
    ActiveStructure,
    LinearOutputMask,
    build_checkerboard_benchmark,
    build_raw_atlas,
    make_output_masks,
    make_structures,
    reconstruct_present_sbox_from_anf,
    validate_scalar_negative_witnesses,
)


AUDIT_ROUNDS = 5
AUDIT_STRUCTURES = 96
AUDIT_WITNESS_KEYS = 16
AUDIT_OFFSETS_PER_STRUCTURE = 8
AUDIT_STRUCTURE_SEED = 20260718
AUDIT_KEY_SEED = 407
AUDIT_OFFSET_SEED = 1701
FROZEN_CLAASP_COMMIT = "f2239d639ae5c4a013947ce9121c6f4464584758"
FULL_ACTIVE_SUPPORT = 1 << ACTIVE_DIMENSION


@dataclass(frozen=True)
class StrictLabelCoverageConfig:
    run_id: str
    mode: str = "audit"
    rounds: int = AUDIT_ROUNDS
    structure_count: int = AUDIT_STRUCTURES
    witness_keys: int = AUDIT_WITNESS_KEYS
    offsets_per_structure: int = AUDIT_OFFSETS_PER_STRUCTURE
    structure_seed: int = AUDIT_STRUCTURE_SEED
    key_seed: int = AUDIT_KEY_SEED
    offset_seed: int = AUDIT_OFFSET_SEED
    match_attempts: int = 64

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.rounds <= 0 or self.structure_count <= 1:
            raise ValueError("rounds and structure_count must be positive")
        if self.witness_keys <= 0 or self.offsets_per_structure <= 0:
            raise ValueError("witness bank dimensions must be positive")
        if self.mode == "audit" and (
            self.rounds != AUDIT_ROUNDS
            or self.structure_count != AUDIT_STRUCTURES
            or self.witness_keys != AUDIT_WITNESS_KEYS
            or self.offsets_per_structure != AUDIT_OFFSETS_PER_STRUCTURE
            or self.structure_seed != AUDIT_STRUCTURE_SEED
            or self.key_seed != AUDIT_KEY_SEED
            or self.offset_seed != AUDIT_OFFSET_SEED
            or self.match_attempts != 64
        ):
            raise ValueError("E52 audit protocol is frozen")


def inspect_claasp_p1(
    root: Path,
    *,
    actual_commit: str | None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    present_path = root / "claasp/ciphers/block_ciphers/present_block_cipher.py"
    module_path = (
        root
        / "claasp/cipher_modules/models/milp/milp_models/Gurobi"
        / "monomial_prediction.py"
    )
    test_path = (
        root
        / "tests/unit/cipher_modules/models/milp/milp_models/Gurobi"
        / "monomial_prediction_test.py"
    )
    present_text = _read_if_present(present_path)
    module_text = _read_if_present(module_path)
    test_text = _read_if_present(test_path)
    superpoly_body = _function_body(
        module_text,
        "def find_superpoly_of_specific_output_bit",
        "def find_exact_degree_of_superpoly_of_specific_output_bit",
    )
    keycoeff_body = _function_body(
        module_text,
        "def find_keycoeff_of_cube_monomial_of_specific_output_bit",
        "def _parse_cube_positions",
    )
    alternative_backends = sorted(
        str(path.relative_to(root))
        for path in (root / "claasp/cipher_modules/models").rglob(
            "*monomial_prediction*.py"
        )
        if "/Gurobi/" not in path.as_posix()
    ) if root.is_dir() else []
    runtime = runtime or {
        "sage_executable": shutil.which("sage"),
        "sage_version": None,
        "sage_modules": {},
        "claasp_present_importable": False,
        "claasp_present_import_error": "runtime probe not supplied",
        "gurobi_license_status": "not_checked",
        "relevant_docker_image_found": False,
    }
    gurobipy_available = bool(
        runtime.get("sage_modules", {}).get("gurobipy", False)
        or importlib.util.find_spec("gurobipy") is not None
    )
    source_checks = {
        "frozen_commit_matches": actual_commit == FROZEN_CLAASP_COMMIT,
        "present_model_present": present_path.is_file(),
        "present80_round_parameter_present": (
            "class PresentBlockCipher" in present_text
            and "number_of_rounds" in present_text
            and "INPUT_KEY" in present_text
            and "cipher_inputs=[INPUT_PLAINTEXT, INPUT_KEY]" in present_text
        ),
        "monomial_prediction_module_present": module_path.is_file(),
        "full_superpoly_api_present": (
            "def find_superpoly_of_specific_output_bit" in module_text
        ),
        "full_superpoly_keeps_non_cube_public_variables_symbolic": (
            "poly = self.get_solutions()" in superpoly_body
            and "Fix all other non-key input bits to 0" not in superpoly_body
        ),
        "key_coefficient_api_present": (
            "def find_keycoeff_of_cube_monomial_of_specific_output_bit"
            in module_text
        ),
        "key_coefficient_is_zero_offset_only": (
            "Fix all other non-key input bits to 0" in keycoeff_body
        ),
        "independent_cube_sum_verifier_present": (
            "def check_correctness_of_keycoeff_of_cube_monomial_or_superpoly"
            in module_text
        ),
        "gurobi_dependency_explicit": "from gurobipy import Model, GRB, Env" in module_text,
        "gurobi_license_requirement_explicit": (
            "This module can only be used if the user possesses a Gurobi license."
            in module_text
        ),
        "gurobi_tests_skipped_for_license": (
            "Requires Gurobi license" in test_text and "pytest.mark.skip" in test_text
        ),
        "non_gurobi_monomial_backend_absent": not alternative_backends,
    }
    license_verified = runtime.get("gurobi_license_status") == "verified"
    execution_available = bool(
        source_checks["frozen_commit_matches"]
        and source_checks["present_model_present"]
        and source_checks["full_superpoly_api_present"]
        and runtime.get("sage_executable")
        and runtime.get("claasp_present_importable")
        and gurobipy_available
        and license_verified
    )
    if not all(source_checks.values()):
        status = "protocol_mismatch"
    elif execution_available:
        status = "ready_for_fixed_subset_execution"
    else:
        status = "execution_unavailable"
    return {
        "provider_id": "P1_claasp_mp_full_superpoly",
        "status": status,
        "source": {
            "repository": "https://github.com/Crypto-TII/claasp",
            "license": "GPL-3.0",
            "frozen_commit": FROZEN_CLAASP_COMMIT,
            "actual_commit": actual_commit,
            "alternative_monomial_backends": alternative_backends,
        },
        "source_checks": source_checks,
        "runtime": {
            **runtime,
            "gurobipy_available": gurobipy_available,
            "execution_available": execution_available,
        },
        "target_mapping": {
            "required_api": "find_superpoly_of_specific_output_bit",
            "required_certificate": (
                "XOR of the selected output-bit full superpolies is the zero "
                "polynomial over all key and inactive plaintext variables"
            ),
            "key_coefficient_api_is_insufficient": (
                "it fixes every non-cube public plaintext bit to zero and therefore "
                "does not prove the property for every inactive offset"
            ),
            "output_bit_order_fixture_required_before_execution": True,
            "multi_bit_mask_requires_polynomial_xor": True,
        },
        "readiness": {
            "semantically_selected": all(source_checks.values()),
            "fixed_subset_executable": execution_available,
            "full_pool_executable": False,
        },
    }


def evaluate_coverage(
    config: StrictLabelCoverageConfig,
    structures: tuple[ActiveStructure, ...],
    masks: tuple[LinearOutputMask, ...],
    raw: dict[str, Any],
    p1: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray(raw["labels"], dtype=np.int8)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    unknown = int(np.sum(labels < 0))
    mixed_structures = int(
        sum(np.any(row == 1) and np.any(row == 0) for row in labels)
    )
    positive_structures = int(sum(np.any(row == 1) for row in labels))
    negative_structures = int(sum(np.any(row == 0) for row in labels))
    support_sizes = np.asarray(raw["support_sizes"], dtype=np.int64)
    scalar = validate_scalar_negative_witnesses(
        raw["rows"], structures, masks, config.rounds
    )
    matched = build_checkerboard_benchmark(
        labels=labels,
        structures=structures,
        masks=masks,
        attempts=config.match_attempts,
    )
    train = matched["split_metrics"]["train"]
    validation = matched["split_metrics"]["validation"]
    protocol_frozen = config.mode != "audit" or (
        config.rounds == AUDIT_ROUNDS
        and config.structure_count == AUDIT_STRUCTURES
        and config.witness_keys == AUDIT_WITNESS_KEYS
        and config.offsets_per_structure == AUDIT_OFFSETS_PER_STRUCTURE
        and config.structure_seed == AUDIT_STRUCTURE_SEED
        and config.key_seed == AUDIT_KEY_SEED
        and config.offset_seed == AUDIT_OFFSET_SEED
        and config.match_attempts == 64
    )
    correctness_checks = {
        "audit_protocol_frozen": protocol_frozen,
        "official_present_vector_matches": Present80(rounds=31, key=0).encrypt(0)
        == 0x5579C1387B228445,
        "present_sbox_anf_reconstructs": all(
            reconstruct_present_sbox_from_anf(value) == PRESENT_SBOX[value]
            for value in range(16)
        ),
        "candidate_pool_shape_matches": labels.shape
        == (len(structures), len(masks)),
        "candidate_pool_is_full_width": config.mode != "audit"
        or labels.shape == (AUDIT_STRUCTURES, EXPECTED_OUTPUT_MASKS),
        "all_positive_rows_have_sound_certificate": all(
            row["certificate"]
            == "full_cube_monomial_absent_from_support_overapprox"
            for row in raw["rows"]
            if row["status"] == "positive"
        ),
        "all_negative_rows_have_concrete_witness": all(
            row["witness_key_hex"] is not None
            and row["witness_offset_hex"] is not None
            and row["witness_parity_word_hex"] is not None
            for row in raw["rows"]
            if row["status"] == "negative"
        ),
        "sampled_negative_witnesses_scalar_validate": scalar["all_pass"],
        "unknown_is_separate_from_provider_error": True,
    }
    coverage_checks = {
        "mixed_positive_negative_structures_at_least_48": mixed_structures >= 48,
        "matched_train_each_class_at_least_400": train["positive"] >= 400
        and train["negative"] >= 400,
        "matched_validation_each_class_at_least_118": validation["positive"]
        >= 118
        and validation["negative"] >= 118,
        "train_validation_structures_disjoint": not set(
            matched["split_indices"]["train"]
        ).intersection(matched["split_indices"]["validation"]),
        "strongest_unary_auc_at_most_0p65": matched["marginal_baselines"][
            "strongest_auc"
        ]
        <= 0.65,
        "duplicate_structure_mask_pairs_zero": matched["balance"][
            "duplicate_edges"
        ]
        == 0,
    }
    p0_complete = all(correctness_checks.values())
    p0_sufficient = p0_complete and all(coverage_checks.values())
    p1_executable = bool(p1["readiness"]["fixed_subset_executable"])
    if not p0_complete:
        status = "fail"
        decision = "innovation2_present_r5_strict_label_provider_protocol_invalid"
        action = "repair PRESENT, certificate, witness, or artifact protocol and rerun E52"
    elif p0_sufficient:
        status = "pass"
        decision = "innovation2_present_r5_strict_label_bank_ready"
        action = "run deterministic shortcut attribution before freezing the first r5 neural matrix"
    elif p1_executable:
        status = "hold"
        decision = "innovation2_present_r5_strict_label_p1_subset_required"
        action = "run CLAASP-MP full-superpoly coverage on the frozen 16x64 subset"
    else:
        status = "hold"
        decision = "innovation2_present_r5_strict_label_bank_not_ready"
        action = (
            "do not train r5 networks; obtain an approved Gurobi runtime for the "
            "CLAASP-MP full-superpoly subset or implement a verifiable open 3SDP provider"
        )
    metrics = {
        "candidate_pairs": int(labels.size),
        "positive": positive,
        "negative": negative,
        "unknown": unknown,
        "positive_rate": positive / labels.size,
        "negative_rate": negative / labels.size,
        "unknown_rate": unknown / labels.size,
        "positive_structures": positive_structures,
        "negative_structures": negative_structures,
        "mixed_positive_negative_structures": mixed_structures,
        "support_size_minimum": int(support_sizes.min()),
        "support_size_maximum": int(support_sizes.max()),
        "fully_saturated_output_supports": int(
            np.sum(support_sizes == FULL_ACTIVE_SUPPORT)
        ),
        "output_supports": int(support_sizes.size),
        "support_saturation_fraction": float(
            np.mean(support_sizes == FULL_ACTIVE_SUPPORT)
        ),
        "matched_train_positive": train["positive"],
        "matched_train_negative": train["negative"],
        "matched_validation_positive": validation["positive"],
        "matched_validation_negative": validation["negative"],
        "strongest_unary_auc": matched["marginal_baselines"]["strongest_auc"],
        "scalar_negative_witness_validation": scalar,
    }
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "correctness_checks": correctness_checks,
        "coverage_checks": coverage_checks,
        "provider_checks": {
            "p0_completed": p0_complete,
            "p0_coverage_sufficient": p0_sufficient,
            "p1_semantically_selected": p1["readiness"][
                "semantically_selected"
            ],
            "p1_fixed_subset_executable": p1_executable,
        },
        "metrics": metrics,
        "claim_scope": (
            "PRESENT-80 r5 strict all-key/all-inactive-offset label-provider "
            "coverage audit; not neural training, a five-round distinguisher, an "
            "attack, or a SOTA result"
        ),
        "next_action": {
            "action": action,
            "training": status == "pass",
            "remote_scale": False,
            "closed_routes": [
                "finite-key empirical balance labels",
                "zero-offset-only key-coefficient labels",
                "PRESENT r4 architecture tuning",
                "seed1 or longer epochs",
                "remote GPU scale before the r5 label gate",
            ],
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_r5_strict_label_provider_coverage",
            "provider": "P0_active_variable_support_overapprox",
            "provider_status": "completed_insufficient"
            if not p0_sufficient
            else "completed_sufficient",
            "rounds": config.rounds,
            "structures": len(structures),
            "masks": len(masks),
            "candidate_pairs": int(labels.size),
            "positive": positive,
            "negative": negative,
            "unknown": unknown,
            "training_performed": False,
            "status": status,
            "decision": decision,
        },
        {
            "run_id": config.run_id,
            "task": "innovation2_present_r5_strict_label_provider_coverage",
            "provider": p1["provider_id"],
            "provider_status": p1["status"],
            "rounds": config.rounds,
            "structures": 0,
            "masks": 0,
            "candidate_pairs": 0,
            "positive": 0,
            "negative": 0,
            "unknown": 0,
            "training_performed": False,
            "status": status,
            "decision": decision,
        },
    ]
    return {
        "metrics": metrics,
        "gate": gate,
        "result_rows": result_rows,
        "matched_rows": matched["rows"],
    }


def serializable_config(config: StrictLabelCoverageConfig) -> dict[str, Any]:
    return asdict(config)


def _read_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _function_body(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    body = text.split(start, 1)[1]
    return body.split(end, 1)[0] if end in body else body
