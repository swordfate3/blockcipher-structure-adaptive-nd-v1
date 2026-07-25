from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from blockcipher_nd.ciphers.spn.dialga import Dialga128
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.training.metrics import binary_auc


BLOCK_BITS = 128
EXPECTED_BIT_INDICES = tuple(range(BLOCK_BITS))
EXPECTED_KEY_ROLES = ("train_key", "validation_key")
REFERENCE_BIT_INDEX = 6
REFERENCE_DIFFERENCE = 0x40
CALIBRATION_ROWS_PER_CLASS = 512
EVALUATION_ROWS_PER_CLASS = 512
PAIRS_PER_ROW = 4
AUC_FLOOR = 0.520
ANCHOR_MARGIN = 0.010
LAPLACE_ALPHA = 1.0
TRAIN_KEY = 0
VALIDATION_KEY = int("11" * 32, 16)
PANEL_SPECS = {
    "train_key": (TRAIN_KEY, 41000),
    "validation_key": (VALIDATION_KEY, 42000),
}


@dataclass(frozen=True)
class DifferenceScreenPanel:
    key_role: str
    key: int
    seed: int
    calibration_rows_per_class: int
    evaluation_rows_per_class: int
    pairs_per_row: int
    calibration_positive_plaintexts: tuple[int, ...]
    evaluation_positive_plaintexts: tuple[int, ...]
    calibration_negative_plaintexts: tuple[tuple[int, int], ...]
    evaluation_negative_plaintexts: tuple[tuple[int, int], ...]
    calibration_negative_bits: np.ndarray
    evaluation_negative_bits: np.ndarray
    panel_sha256: str
    calibration_negative_sha256: str
    evaluation_negative_sha256: str


def prepare_difference_screen_panel(
    *,
    key_role: str,
    key: int,
    seed: int,
    calibration_rows_per_class: int = CALIBRATION_ROWS_PER_CLASS,
    evaluation_rows_per_class: int = EVALUATION_ROWS_PER_CLASS,
    pairs_per_row: int = PAIRS_PER_ROW,
) -> DifferenceScreenPanel:
    if key_role not in EXPECTED_KEY_ROLES:
        raise ValueError(f"unexpected D5 key role: {key_role}")
    if calibration_rows_per_class <= 0 or evaluation_rows_per_class <= 0:
        raise ValueError("D5 screen row counts must be positive")
    if pairs_per_row <= 0:
        raise ValueError("D5 screen pairs_per_row must be positive")
    rng = np.random.default_rng(seed)
    calibration_pairs = calibration_rows_per_class * pairs_per_row
    evaluation_pairs = evaluation_rows_per_class * pairs_per_row
    calibration_positive = _random_blocks(rng, calibration_pairs)
    evaluation_positive = _random_blocks(rng, evaluation_pairs)
    calibration_negative = tuple(
        zip(
            _random_blocks(rng, calibration_pairs),
            _random_blocks(rng, calibration_pairs),
            strict=True,
        )
    )
    evaluation_negative = tuple(
        zip(
            _random_blocks(rng, evaluation_pairs),
            _random_blocks(rng, evaluation_pairs),
            strict=True,
        )
    )
    cipher = Dialga128(rounds=5, key=key, tweak=0)
    calibration_negative_bits = _encrypted_xor_bits(cipher, calibration_negative)
    evaluation_negative_bits = _encrypted_xor_bits(cipher, evaluation_negative)
    panel_sha256 = _panel_sha256(
        key_role=key_role,
        key=key,
        seed=seed,
        calibration_positive=calibration_positive,
        evaluation_positive=evaluation_positive,
        calibration_negative=calibration_negative,
        evaluation_negative=evaluation_negative,
    )
    return DifferenceScreenPanel(
        key_role=key_role,
        key=key,
        seed=seed,
        calibration_rows_per_class=calibration_rows_per_class,
        evaluation_rows_per_class=evaluation_rows_per_class,
        pairs_per_row=pairs_per_row,
        calibration_positive_plaintexts=calibration_positive,
        evaluation_positive_plaintexts=evaluation_positive,
        calibration_negative_plaintexts=calibration_negative,
        evaluation_negative_plaintexts=evaluation_negative,
        calibration_negative_bits=calibration_negative_bits,
        evaluation_negative_bits=evaluation_negative_bits,
        panel_sha256=panel_sha256,
        calibration_negative_sha256=_array_sha256(calibration_negative_bits),
        evaluation_negative_sha256=_array_sha256(evaluation_negative_bits),
    )


def evaluate_difference_candidate(
    *,
    bit_index: int,
    panel: DifferenceScreenPanel,
    runtime_structure: RuntimeSpnStructure,
) -> dict[str, Any]:
    if bit_index not in EXPECTED_BIT_INDICES:
        raise ValueError("D5 bit index must be in [0, 127]")
    if runtime_structure.block_bits != BLOCK_BITS:
        raise ValueError("D5 runtime structure must describe 128 input bits")
    difference = 1 << bit_index
    cipher = Dialga128(rounds=5, key=panel.key, tweak=0)
    calibration_positive_bits = _encrypted_xor_bits(
        cipher,
        tuple(
            (plaintext, plaintext ^ difference)
            for plaintext in panel.calibration_positive_plaintexts
        ),
    )
    evaluation_positive_bits = _encrypted_xor_bits(
        cipher,
        tuple(
            (plaintext, plaintext ^ difference)
            for plaintext in panel.evaluation_positive_plaintexts
        ),
    )
    positive_probability = _bernoulli_probability(calibration_positive_bits)
    negative_probability = _bernoulli_probability(panel.calibration_negative_bits)
    positive_scores = _row_log_likelihood_ratio(
        evaluation_positive_bits,
        positive_probability=positive_probability,
        negative_probability=negative_probability,
        rows=panel.evaluation_rows_per_class,
        pairs_per_row=panel.pairs_per_row,
    )
    negative_scores = _row_log_likelihood_ratio(
        panel.evaluation_negative_bits,
        positive_probability=positive_probability,
        negative_probability=negative_probability,
        rows=panel.evaluation_rows_per_class,
        pairs_per_row=panel.pairs_per_row,
    )
    labels = np.concatenate(
        (
            np.ones(panel.evaluation_rows_per_class, dtype=np.uint8),
            np.zeros(panel.evaluation_rows_per_class, dtype=np.uint8),
        )
    )
    scores = np.concatenate((positive_scores, negative_scores)).astype(np.float64)
    bit_bias = positive_probability - negative_probability
    ranked_bits = np.argsort(-np.abs(bit_bias), kind="stable")[:8]
    cell_id = int(runtime_structure.cell_membership[bit_index])
    bit_role = int(runtime_structure.bit_role[bit_index])
    return {
        "key_role": panel.key_role,
        "key": panel.key,
        "panel_seed": panel.seed,
        "bit_index_lsb": bit_index,
        "input_difference": difference,
        "input_hamming_weight": difference.bit_count(),
        "input_cell_id": cell_id,
        "input_cell_bit_role": bit_role,
        "cipher": "Dialga-128",
        "rounds": 5,
        "tweak": 0,
        "calibration_rows_per_class": panel.calibration_rows_per_class,
        "evaluation_rows_per_class": panel.evaluation_rows_per_class,
        "pairs_per_row": panel.pairs_per_row,
        "negative_mode": "encrypted_random_plaintexts",
        "screen_model": "bit_marginal_bernoulli_naive_bayes",
        "screen_auc": binary_auc(labels, scores),
        "max_abs_output_bit_bias": float(np.max(np.abs(bit_bias))),
        "mean_abs_output_bit_bias": float(np.mean(np.abs(bit_bias))),
        "top_biased_output_bits_lsb": [int(index) for index in ranked_bits],
        "panel_sha256": panel.panel_sha256,
        "calibration_negative_sha256": panel.calibration_negative_sha256,
        "evaluation_negative_sha256": panel.evaluation_negative_sha256,
        "calibration_positive_xor_sha256": _array_sha256(calibration_positive_bits),
        "evaluation_positive_xor_sha256": _array_sha256(evaluation_positive_bits),
        "evaluation_score_sha256": _array_sha256(scores),
        "runtime_structure_window_sha256": runtime_structure.window_sha256(),
        "runtime_structure_cell_membership_sha256": _array_sha256(
            runtime_structure.cell_membership.numpy()
        ),
        "runtime_structure_bit_role_sha256": _array_sha256(
            runtime_structure.bit_role.numpy()
        ),
        "training_performed": False,
        "data_generation_performed": True,
    }


def adjudicate_difference_screen(
    *, run_id: str, rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    rows = list(rows)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row.get("bit_index_lsb", -1))][str(row.get("key_role"))].append(row)
    complete = all(
        len(grouped[bit].get(key_role, ())) == 1
        for bit in EXPECTED_BIT_INDICES
        for key_role in EXPECTED_KEY_ROLES
    )
    flat_rows = [
        grouped[bit][key_role][0]
        for bit in EXPECTED_BIT_INDICES
        for key_role in EXPECTED_KEY_ROLES
        if len(grouped[bit].get(key_role, ())) == 1
    ]
    protocol_checks = {
        "two_hundred_fifty_six_rows_complete": len(rows) == 256,
        "full_single_bit_by_key_panel": complete
        and set(grouped) == set(EXPECTED_BIT_INDICES),
        "exact_candidate_space": complete and _candidate_contract(grouped),
        "exact_key_panels": complete and _key_panel_contract(grouped),
        "shared_plaintext_panels_within_key": complete
        and _same_key_role_fields(
            grouped,
            (
                "panel_sha256",
                "calibration_negative_sha256",
                "evaluation_negative_sha256",
            ),
        ),
        "distinct_key_panels": complete
        and grouped[0]["train_key"][0].get("panel_sha256")
        != grouped[0]["validation_key"][0].get("panel_sha256"),
        "runtime_cell_mapping_consistent": complete
        and _runtime_mapping_contract(grouped),
        "frozen_screen_protocol": len(flat_rows) == 256
        and all(_row_has_frozen_protocol(row) for row in flat_rows),
        "finite_metrics": len(flat_rows) == 256
        and all(
            _finite(row.get(field))
            for row in flat_rows
            for field in (
                "screen_auc",
                "max_abs_output_bit_bias",
                "mean_abs_output_bit_bias",
            )
        ),
        "artifact_hashes_present": len(flat_rows) == 256
        and all(
            _is_sha256(row.get(field))
            for row in flat_rows
            for field in (
                "panel_sha256",
                "calibration_negative_sha256",
                "evaluation_negative_sha256",
                "calibration_positive_xor_sha256",
                "evaluation_positive_xor_sha256",
                "evaluation_score_sha256",
                "runtime_structure_window_sha256",
                "runtime_structure_cell_membership_sha256",
                "runtime_structure_bit_role_sha256",
            )
        ),
        "no_neural_training": len(flat_rows) == 256
        and all(
            row.get("training_performed") is False
            and row.get("data_generation_performed") is True
            for row in flat_rows
        ),
    }
    aggregates = _candidate_aggregates(grouped) if complete else []
    anchor = next(
        (
            candidate
            for candidate in aggregates
            if candidate["bit_index_lsb"] == REFERENCE_BIT_INDEX
        ),
        None,
    )
    eligible: list[dict[str, Any]] = []
    if anchor is not None:
        eligible = [
            candidate
            for candidate in aggregates
            if candidate["bit_index_lsb"] != REFERENCE_BIT_INDEX
            and candidate["train_key_auc"] >= AUC_FLOOR
            and candidate["validation_key_auc"] >= AUC_FLOOR
            and candidate["worst_key_auc"] >= anchor["worst_key_auc"] + ANCHOR_MARGIN
        ]
    eligible.sort(
        key=lambda candidate: (
            -candidate["worst_key_auc"],
            -candidate["mean_auc"],
            candidate["bit_index_lsb"],
        )
    )
    shortlist = eligible[:2]
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_dialga_runtime_e4_d5_screen_protocol_invalid"
        next_action = (
            "repair only the D5 shortlist implementation while keeping candidate "
            "space, keys, seeds, row counts, metric, and thresholds frozen"
        )
    elif shortlist:
        status = "pass"
        decision = "innovation1_dialga_runtime_e4_d5_difference_candidate_supported"
        next_action = (
            "train only the top shortlisted difference under the exact D3 two-seed "
            "correct/corrupted/no-topology budget; reuse D3 0x40 as the anchor"
        )
    else:
        status = "hold"
        decision = "innovation1_dialga_runtime_e4_d5_no_difference_candidate"
        next_action = (
            "stop mechanical Dialga input-difference search and implement the "
            "residual/gated topology processor as a separate same-data hypothesis"
        )
    return {
        "run_id": run_id,
        "task": "innovation1_dialga128_runtime_e4_d5_difference_screen",
        "cipher": "Dialga-128",
        "status": status,
        "decision": decision,
        "thresholds": {
            "per_key_auc": AUC_FLOOR,
            "worst_key_auc_margin_over_0x40": ANCHOR_MARGIN,
        },
        "reference": anchor,
        "shortlist": shortlist,
        "top_candidates": aggregates[:12],
        "protocol_checks": protocol_checks,
        "claim_scope": (
            "Dialga-128 prefix-r5 two-key 128-single-bit local cipher-statistical "
            "shortlist; no neural training, formal scale, attack, trail reproduction, "
            "SOTA, or universal-SPN claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "train more than the top candidate before its same-budget gate",
            "add multi-bit, DDT, trail, partial-decryption, or guessed-key candidates",
            "change Runtime-E4 in the same input-difference experiment",
            "launch remote GPU or increase samples",
        ],
    }


def _candidate_aggregates(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    aggregates = []
    for bit in EXPECTED_BIT_INDICES:
        train_auc = float(grouped[bit]["train_key"][0]["screen_auc"])
        validation_auc = float(grouped[bit]["validation_key"][0]["screen_auc"])
        row = grouped[bit]["train_key"][0]
        aggregates.append(
            {
                "bit_index_lsb": bit,
                "input_difference": 1 << bit,
                "input_difference_hex": f"0x{1 << bit:032x}",
                "input_cell_id": int(row["input_cell_id"]),
                "input_cell_bit_role": int(row["input_cell_bit_role"]),
                "train_key_auc": train_auc,
                "validation_key_auc": validation_auc,
                "worst_key_auc": min(train_auc, validation_auc),
                "mean_auc": (train_auc + validation_auc) / 2.0,
            }
        )
    aggregates.sort(
        key=lambda candidate: (
            -candidate["worst_key_auc"],
            -candidate["mean_auc"],
            candidate["bit_index_lsb"],
        )
    )
    return aggregates


def _candidate_contract(grouped: dict[int, dict[str, list[dict[str, Any]]]]) -> bool:
    return (
        all(
            row.get("bit_index_lsb") == bit
            and row.get("input_difference") == 1 << bit
            and row.get("input_hamming_weight") == 1
            for bit in EXPECTED_BIT_INDICES
            for key_role in EXPECTED_KEY_ROLES
            for row in (grouped[bit][key_role][0],)
        )
        and grouped[REFERENCE_BIT_INDEX]["train_key"][0].get("input_difference")
        == REFERENCE_DIFFERENCE
    )


def _key_panel_contract(grouped: dict[int, dict[str, list[dict[str, Any]]]]) -> bool:
    return all(
        row.get("key_role") == key_role
        and row.get("key") == PANEL_SPECS[key_role][0]
        and row.get("panel_seed") == PANEL_SPECS[key_role][1]
        for bit in EXPECTED_BIT_INDICES
        for key_role in EXPECTED_KEY_ROLES
        for row in (grouped[bit][key_role][0],)
    )


def _runtime_mapping_contract(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
) -> bool:
    fingerprints = set()
    for bit in EXPECTED_BIT_INDICES:
        left = grouped[bit]["train_key"][0]
        right = grouped[bit]["validation_key"][0]
        if any(
            left.get(field) != right.get(field)
            for field in (
                "input_cell_id",
                "input_cell_bit_role",
                "runtime_structure_window_sha256",
                "runtime_structure_cell_membership_sha256",
                "runtime_structure_bit_role_sha256",
            )
        ):
            return False
        fingerprints.add(left.get("runtime_structure_window_sha256"))
    return len(fingerprints) == 1


def _same_key_role_fields(
    grouped: dict[int, dict[str, list[dict[str, Any]]]], fields: Sequence[str]
) -> bool:
    return all(
        len({grouped[bit][key_role][0].get(field) for bit in EXPECTED_BIT_INDICES}) == 1
        for key_role in EXPECTED_KEY_ROLES
        for field in fields
    )


def _row_has_frozen_protocol(row: dict[str, Any]) -> bool:
    key_role = str(row.get("key_role"))
    return bool(
        key_role in PANEL_SPECS
        and row.get("cipher") == "Dialga-128"
        and row.get("rounds") == 5
        and row.get("tweak") == 0
        and row.get("key") == PANEL_SPECS[key_role][0]
        and row.get("panel_seed") == PANEL_SPECS[key_role][1]
        and row.get("calibration_rows_per_class") == CALIBRATION_ROWS_PER_CLASS
        and row.get("evaluation_rows_per_class") == EVALUATION_ROWS_PER_CLASS
        and row.get("pairs_per_row") == PAIRS_PER_ROW
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("screen_model") == "bit_marginal_bernoulli_naive_bayes"
    )


def _bernoulli_probability(bits: np.ndarray) -> np.ndarray:
    count = bits.shape[0]
    return (bits.sum(axis=0, dtype=np.float64) + LAPLACE_ALPHA) / (
        count + 2.0 * LAPLACE_ALPHA
    )


def _row_log_likelihood_ratio(
    bits: np.ndarray,
    *,
    positive_probability: np.ndarray,
    negative_probability: np.ndarray,
    rows: int,
    pairs_per_row: int,
) -> np.ndarray:
    log_one = np.log(positive_probability / negative_probability)
    log_zero = np.log((1.0 - positive_probability) / (1.0 - negative_probability))
    pair_scores = bits @ (log_one - log_zero) + log_zero.sum()
    return pair_scores.reshape(rows, pairs_per_row).sum(axis=1)


def _encrypted_xor_bits(
    cipher: Dialga128, plaintext_pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    encrypt = cipher.encrypt
    differences = [encrypt(left) ^ encrypt(right) for left, right in plaintext_pairs]
    payload = b"".join(value.to_bytes(16, byteorder="big") for value in differences)
    msb_first = np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).reshape(
        len(differences), BLOCK_BITS
    )
    return msb_first[:, ::-1].copy()


def _random_blocks(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    payload = rng.bytes(16 * count)
    return tuple(
        int.from_bytes(payload[offset : offset + 16], byteorder="big")
        for offset in range(0, len(payload), 16)
    )


def _panel_sha256(
    *,
    key_role: str,
    key: int,
    seed: int,
    calibration_positive: Sequence[int],
    evaluation_positive: Sequence[int],
    calibration_negative: Sequence[tuple[int, int]],
    evaluation_negative: Sequence[tuple[int, int]],
) -> str:
    digest = hashlib.sha256()
    digest.update(key_role.encode())
    digest.update(key.to_bytes(32, byteorder="big"))
    digest.update(seed.to_bytes(8, byteorder="big"))
    for values in (
        calibration_positive,
        evaluation_positive,
        (value for pair in calibration_negative for value in pair),
        (value for pair in evaluation_negative for value in pair),
    ):
        for value in values:
            digest.update(int(value).to_bytes(16, byteorder="big"))
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ANCHOR_MARGIN",
    "AUC_FLOOR",
    "CALIBRATION_ROWS_PER_CLASS",
    "EVALUATION_ROWS_PER_CLASS",
    "EXPECTED_BIT_INDICES",
    "EXPECTED_KEY_ROLES",
    "PAIRS_PER_ROW",
    "PANEL_SPECS",
    "REFERENCE_BIT_INDEX",
    "REFERENCE_DIFFERENCE",
    "DifferenceScreenPanel",
    "adjudicate_difference_screen",
    "evaluate_difference_candidate",
    "prepare_difference_screen_panel",
]
