from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from blockcipher_nd.ciphers.spn.rectangle import (
    RECTANGLE_ROUND_CONSTANTS,
    RECTANGLE_SBOX,
    Rectangle80,
    rectangle_player,
    update_rectangle80_key,
)
from blockcipher_nd.tasks.innovation2.gift64_unit_balance_profile_readiness import (
    build_gift_checkerboard,
)
from blockcipher_nd.tasks.innovation2.present_active_dimension_zero_shot_transfer import (
    compatible_prefix_features,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ActiveStructure,
    _cube_assignments,
)


AUDIT_ROUNDS = 4
AUDIT_STRUCTURES = 96
AUDIT_WITNESS_KEYS = 16
AUDIT_OFFSETS = 8
AUDIT_MATCH_ATTEMPTS = 64
EXPANSION_STRUCTURES = 192
ACTIVE_DIMENSION = 8
OUTPUT_BITS = 64
OFFICIAL_ZERO_VECTOR = 0x0874E8B1E3542D96


def _anf_coefficients(truth: tuple[int, ...]) -> tuple[int, ...]:
    coefficients = list(truth)
    for bit in range(4):
        for mask in range(16):
            if mask & (1 << bit):
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    return tuple(coefficients)


RECTANGLE_SBOX_ANF = tuple(
    tuple(
        mask
        for mask, coefficient in enumerate(
            _anf_coefficients(
                tuple((value >> output_bit) & 1 for value in RECTANGLE_SBOX)
            )
        )
        if coefficient
    )
    for output_bit in range(4)
)


@dataclass(frozen=True)
class Rectangle80UnitProfileConfig:
    run_id: str
    rounds: int = AUDIT_ROUNDS
    structure_count: int = AUDIT_STRUCTURES
    witness_keys: int = AUDIT_WITNESS_KEYS
    offsets_per_structure: int = AUDIT_OFFSETS
    match_attempts: int = AUDIT_MATCH_ATTEMPTS
    structure_seed: int = 20260718
    key_seed: int = 8701
    offset_seed: int = 18701

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.rounds != AUDIT_ROUNDS
            or self.structure_count != AUDIT_STRUCTURES
            or self.witness_keys != AUDIT_WITNESS_KEYS
            or self.offsets_per_structure != AUDIT_OFFSETS
            or self.match_attempts != AUDIT_MATCH_ATTEMPTS
        ):
            raise ValueError("E87 protocol is frozen")


@dataclass(frozen=True)
class Rectangle80UnitProfileExpansionConfig:
    run_id: str
    rounds: int = AUDIT_ROUNDS
    structure_count: int = EXPANSION_STRUCTURES
    witness_keys: int = AUDIT_WITNESS_KEYS
    offsets_per_structure: int = AUDIT_OFFSETS
    match_attempts: int = AUDIT_MATCH_ATTEMPTS
    structure_seed: int = 20260718
    key_seed: int = 8701
    offset_seed: int = 18701

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.rounds != AUDIT_ROUNDS
            or self.structure_count != EXPANSION_STRUCTURES
            or self.witness_keys != AUDIT_WITNESS_KEYS
            or self.offsets_per_structure != AUDIT_OFFSETS
            or self.match_attempts != AUDIT_MATCH_ATTEMPTS
        ):
            raise ValueError("E88 protocol is frozen")


Rectangle80ProfileConfig = (
    Rectangle80UnitProfileConfig | Rectangle80UnitProfileExpansionConfig
)


def reconstruct_rectangle_sbox_from_anf(value: int) -> int:
    if value not in range(16):
        raise ValueError("value must be a nibble")
    output = 0
    for output_bit, terms in enumerate(RECTANGLE_SBOX_ANF):
        coordinate = 0
        for term in terms:
            coordinate ^= int((value & term) == term)
        output |= coordinate << output_bit
    return output


def make_rectangle80_keys(count: int, seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.getrandbits(80) for _ in range(count))


def rectangle80_round_keys(keys: tuple[int, ...], rounds: int) -> np.ndarray:
    if rounds < 1 or rounds > 25:
        raise ValueError("RECTANGLE-80 supports 1..25 rounds")
    round_keys = np.zeros((len(keys), rounds + 1), dtype=np.uint64)
    for key_index, seed_key in enumerate(keys):
        key = seed_key
        for round_index in range(rounds):
            round_keys[key_index, round_index] = np.uint64(key & ((1 << 64) - 1))
            key = update_rectangle80_key(key, round_index)
        round_keys[key_index, rounds] = np.uint64(key & ((1 << 64) - 1))
    return round_keys


def encrypt_rectangle80_words(
    plaintexts: np.ndarray,
    round_keys: np.ndarray,
) -> np.ndarray:
    words = np.asarray(plaintexts, dtype=np.uint64)
    keys = np.asarray(round_keys, dtype=np.uint64)
    if keys.ndim != 2 or keys.shape[1] < 2:
        raise ValueError("round_keys must have shape keys x (rounds + 1)")
    state = np.broadcast_to(words[None, :], (len(keys), len(words))).copy()
    sbox = np.asarray(RECTANGLE_SBOX, dtype=np.uint64)
    for round_index in range(keys.shape[1] - 1):
        state ^= keys[:, round_index, None]
        after_sbox = np.zeros_like(state)
        for column in range(16):
            values = np.zeros(state.shape, dtype=np.uint64)
            for row in range(4):
                source = 16 * row + column
                values |= ((state >> np.uint64(source)) & np.uint64(1)) << np.uint64(row)
            substituted = sbox[values.astype(np.intp)]
            for row in range(4):
                target = 16 * row + column
                after_sbox |= ((substituted >> np.uint64(row)) & np.uint64(1)) << np.uint64(target)
        shifted = np.zeros_like(state)
        for source, target in enumerate(rectangle_player()):
            shifted |= ((after_sbox >> np.uint64(source)) & np.uint64(1)) << np.uint64(target)
        state = shifted
    return state ^ keys[:, -1, None]


def rectangle80_variable_supports(
    active_bits: tuple[int, ...], rounds: int
) -> tuple[frozenset[int], ...]:
    variable_by_bit = {bit: variable for variable, bit in enumerate(active_bits)}
    state: list[set[int]] = [
        {0, 1 << variable_by_bit[bit]} if bit in variable_by_bit else {0}
        for bit in range(64)
    ]
    player = rectangle_player()
    for _ in range(rounds):
        after_sbox: list[set[int]] = [set() for _ in range(64)]
        for column in range(16):
            inputs = [state[16 * row + column] for row in range(4)]
            for output_row in range(4):
                after_sbox[16 * output_row + column] = _sbox_support(
                    inputs, output_row
                )
        after_player: list[set[int]] = [set() for _ in range(64)]
        for source, support in enumerate(after_sbox):
            after_player[player[source]] = support | {0}
        state = after_player
    return tuple(frozenset(support) for support in state)


def _sbox_support(inputs: list[set[int]], output_bit: int) -> set[int]:
    output: set[int] = set()
    for local_term in RECTANGLE_SBOX_ANF[output_bit]:
        product = {0}
        for input_bit in range(4):
            if local_term & (1 << input_bit):
                product = {
                    left | right
                    for left in product
                    for right in inputs[input_bit]
                }
        output.update(product)
    return output


def build_rectangle80_unit_atlas(
    config: Rectangle80UnitProfileConfig,
    structures: tuple[ActiveStructure, ...],
) -> dict[str, Any]:
    keys = make_rectangle80_keys(config.witness_keys, config.key_seed)
    round_keys = rectangle80_round_keys(keys, config.rounds)
    labels = np.full((len(structures), OUTPUT_BITS), -1, dtype=np.int8)
    prefix = np.empty((len(structures), OUTPUT_BITS, 39), dtype=np.float64)
    witness_key_indices = np.full(labels.shape, -1, dtype=np.int16)
    witness_offsets = np.zeros(labels.shape, dtype=np.uint64)
    rows: list[dict[str, Any]] = []
    support_sizes: list[int] = []
    for structure in structures:
        supports = {
            round_count: rectangle80_variable_supports(
                structure.active_bits, round_count
            )
            for round_count in (1, 2, 3, 4)
        }
        prefix[structure.index] = compatible_prefix_features(
            {round_count: supports[round_count] for round_count in (1, 2, 3)},
            ACTIVE_DIMENSION,
        )
        support_sizes.extend(len(support) for support in supports[4])
        full_cube = (1 << ACTIVE_DIMENSION) - 1
        assignments = _cube_assignments(structure.active_bits)
        rng = random.Random(
            config.offset_seed
            + sum(
                (index + 1) * bit
                for index, bit in enumerate(structure.active_bits)
            )
        )
        negative_bits: dict[int, tuple[int, int, int]] = {}
        for offset_index in range(config.offsets_per_structure):
            offset = rng.getrandbits(64) & ~structure.active_mask
            ciphertexts = encrypt_rectangle80_words(
                assignments ^ np.uint64(offset), round_keys
            )
            parity_words = np.bitwise_xor.reduce(ciphertexts, axis=1)
            for key_index, parity_word in enumerate(parity_words):
                word = int(parity_word)
                for output_bit in range(OUTPUT_BITS):
                    if output_bit not in negative_bits and word & (1 << output_bit):
                        negative_bits[output_bit] = (
                            key_index,
                            offset_index,
                            offset,
                        )
        for output_bit in range(OUTPUT_BITS):
            witness = negative_bits.get(output_bit)
            if full_cube not in supports[4][output_bit]:
                labels[structure.index, output_bit] = 1
                status = "positive"
                certificate = "full_cube_monomial_absent_from_support_overapprox"
            elif witness is not None:
                labels[structure.index, output_bit] = 0
                status = "negative"
                witness_key_indices[structure.index, output_bit] = witness[0]
                witness_offsets[structure.index, output_bit] = np.uint64(witness[2])
                certificate = "concrete_key_offset_unit_xor_one"
            else:
                status = "unknown"
                certificate = "unresolved"
            rows.append(
                {
                    "run_id": config.run_id,
                    "structure_index": structure.index,
                    "structure_id": structure.structure_id,
                    "structure_role": structure.role,
                    "active_bits": list(structure.active_bits),
                    "active_mask_hex": f"0x{structure.active_mask:016X}",
                    "output_bit": output_bit,
                    "status": status,
                    "label": None
                    if labels[structure.index, output_bit] < 0
                    else int(labels[structure.index, output_bit]),
                    "certificate": certificate,
                    "witness_key_index": None if witness is None else witness[0],
                    "witness_key_hex": None
                    if witness is None
                    else f"0x{keys[witness[0]]:020X}",
                    "witness_offset_index": None if witness is None else witness[1],
                    "witness_offset_hex": None
                    if witness is None
                    else f"0x{witness[2]:016X}",
                }
            )
    return {
        "rows": rows,
        "labels": labels,
        "prefix_features": prefix,
        "keys": keys,
        "witness_key_indices": witness_key_indices,
        "witness_offsets": witness_offsets,
        "support_sizes": support_sizes,
    }


def build_rectangle80_checkerboard(
    labels: np.ndarray,
    structures: tuple[ActiveStructure, ...],
    attempts: int,
) -> dict[str, Any]:
    return build_gift_checkerboard(labels, structures, attempts)


def evaluate_rectangle80_unit_profile(
    config: Rectangle80ProfileConfig,
    structures: tuple[ActiveStructure, ...],
    raw: dict[str, Any],
    matched: dict[str, Any],
    anchor_checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    labels = raw["labels"]
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    unknown = int(np.sum(labels < 0))
    resolved = positive + negative
    mixed = sum(bool(np.any(row == 1) and np.any(row == 0)) for row in labels)
    signatures = {
        hashlib.sha256(np.asarray(row, dtype=np.int8).tobytes()).hexdigest()
        for row in labels
    }
    scalar = validate_rectangle80_negative_witnesses(config, structures, raw)
    vector_fixture = validate_rectangle80_vectorized_fixture(config)
    train = matched["split_metrics"]["train"]
    validation = matched["split_metrics"]["validation"]
    metrics = {
        "raw_rows": int(labels.size),
        "raw_positive": positive,
        "raw_negative": negative,
        "raw_unknown": unknown,
        "resolved_positive_prevalence": positive / resolved if resolved else 0.0,
        "mixed_structures": mixed,
        "distinct_ternary_signatures": len(signatures),
        "support_size_minimum": min(raw["support_sizes"]),
        "support_size_maximum": max(raw["support_sizes"]),
        "matched_split_metrics": matched["split_metrics"],
        "matched_total_structures": len(
            {row["structure_index"] for row in matched["rows"]}
        ),
        "matched_marginal_baselines": matched["marginal_baselines"],
        "matched_balance": matched["balance"],
        "scalar_witness_validation": scalar,
        "vectorized_fixture": vector_fixture,
    }
    protocol_checks = {
        "official_zero_vector_matches": Rectangle80().encrypt(0)
        == OFFICIAL_ZERO_VECTOR,
        "rectangle_sbox_anf_reconstructs": all(
            reconstruct_rectangle_sbox_from_anf(value) == RECTANGLE_SBOX[value]
            for value in range(16)
        ),
        "rectangle_round_constants_match_final_spec": RECTANGLE_ROUND_CONSTANTS
        == (
            0x01,
            0x02,
            0x04,
            0x09,
            0x12,
            0x05,
            0x0B,
            0x16,
            0x0C,
            0x19,
            0x13,
            0x07,
            0x0F,
            0x1F,
            0x1E,
            0x1C,
            0x18,
            0x11,
            0x03,
            0x06,
            0x0D,
            0x1B,
            0x17,
            0x0E,
            0x1D,
        ),
        "rectangle_player_is_bijective": sorted(rectangle_player())
        == list(range(64)),
        "vectorized_scalar_fixture_matches": vector_fixture["all_pass"],
        "structure_count_matches": len(structures) == config.structure_count,
        "raw_shape_matches": labels.shape
        == (config.structure_count, OUTPUT_BITS),
        "all_positive_rows_have_certificate": all(
            row["certificate"]
            == "full_cube_monomial_absent_from_support_overapprox"
            for row in raw["rows"]
            if row["status"] == "positive"
        ),
        "all_negative_rows_have_witness": all(
            row["witness_key_hex"] is not None
            and row["witness_offset_hex"] is not None
            for row in raw["rows"]
            if row["status"] == "negative"
        ),
        "positive_negative_conflicts_zero": all(
            row["witness_key_hex"] is None
            for row in raw["rows"]
            if row["status"] == "positive"
        ),
        "sampled_negative_witnesses_scalar_validate": scalar["all_pass"],
        "train_validation_structures_disjoint": not set(
            matched["split_indices"]["train"]
        ).intersection(matched["split_indices"]["validation"]),
        **(anchor_checks or {}),
    }
    raw_width_checks = {
        "raw_each_class_at_least_256": positive >= 256 and negative >= 256,
        "resolved_prevalence_in_0p10_0p90": 0.10
        <= metrics["resolved_positive_prevalence"]
        <= 0.90,
        "mixed_structures_at_least_32": mixed >= 32,
        "distinct_signatures_at_least_4": len(signatures) >= 4,
    }
    matching_width_checks = {
        "matched_train_each_class_at_least_150": train["positive"] >= 150
        and train["negative"] >= 150,
        "matched_validation_each_class_at_least_50": validation["positive"] >= 50
        and validation["negative"] >= 50,
        "matched_total_structures_at_least_32": metrics[
            "matched_total_structures"
        ]
        >= 32,
        "matched_validation_structures_at_least_8": validation["structures"]
        >= 8,
        "matched_validation_output_bits_at_least_16": validation["output_bits"]
        >= 16,
    }
    shortcut_checks = {
        "strongest_unary_auc_at_most_0p65": matched["marginal_baselines"][
            "strongest_auc"
        ]
        <= 0.65,
        "duplicate_edges_zero": matched["balance"]["duplicate_edges"] == 0,
        "structure_class_delta_zero": matched["balance"][
            "maximum_structure_class_delta"
        ]
        == 0,
        "output_class_delta_zero": matched["balance"][
            "maximum_mask_class_delta"
        ]
        == 0,
    }
    status, decision, action = adjudicate_rectangle80_profile_checks(
        protocol_checks,
        raw_width_checks,
        matching_width_checks,
        shortcut_checks,
        experiment=_profile_experiment(config),
    )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "raw_width_checks": raw_width_checks,
        "matching_width_checks": matching_width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": metrics,
        "claim_scope": (
            "strict RECTANGLE-80 r4 8-bit-cube unit-output profile label "
            "readiness; no neural gain, seven-round reproduction, high-round "
            "distinguisher, cross-cipher transfer, attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "neural_readiness": status == "pass"
            and _profile_experiment(config) == "e88",
            "remote_scale": False,
        },
    }


def adjudicate_rectangle80_profile_checks(
    protocol_checks: dict[str, bool],
    raw_width_checks: dict[str, bool],
    matching_width_checks: dict[str, bool],
    shortcut_checks: dict[str, bool],
    *,
    experiment: str = "e87",
) -> tuple[str, str, str]:
    if experiment not in {"e87", "e88"}:
        raise ValueError("experiment must be e87 or e88")
    if experiment == "e88":
        if not all(protocol_checks.values()):
            return (
                "fail",
                "innovation2_rectangle80_unit_profile_expansion_protocol_invalid",
                "repair E87 replay or RECTANGLE label protocol",
            )
        if (
            not all(raw_width_checks.values())
            or not all(matching_width_checks.values())
            or not all(shortcut_checks.values())
        ):
            return (
                "hold",
                "innovation2_rectangle80_unit_profile_expansion_not_ready",
                "close the current RECTANGLE r4 unit-profile neural route",
            )
        return (
            "pass",
            "innovation2_rectangle80_unit_profile_expansion_ready",
            "run a local RECTANGLE r3-only three-row neural readiness matrix",
        )
    if not all(protocol_checks.values()):
        return (
            "fail",
            "innovation2_rectangle80_unit_profile_protocol_invalid",
            "repair final specification, row order, vector path, support, or witnesses",
        )
    if not all(raw_width_checks.values()):
        return (
            "hold",
            "innovation2_rectangle80_unit_profile_raw_labels_not_ready",
            "run E88 changing only rounds from four to five at 96 structures",
        )
    if not all(matching_width_checks.values()) or not all(shortcut_checks.values()):
        return (
            "hold",
            "innovation2_rectangle80_unit_profile_matching_not_ready",
            "run E88 changing only structures from 96 to 192",
        )
    return (
        "pass",
        "innovation2_rectangle80_unit_profile_ready",
        "expand to 192 structures before any neural readiness",
    )


def validate_rectangle80_vectorized_fixture(
    config: Rectangle80ProfileConfig,
) -> dict[str, Any]:
    keys = (0, 1, (1 << 80) - 1, 0x0123456789ABCDEFFEDC)
    words = np.asarray(
        (0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210),
        dtype=np.uint64,
    )
    vector = encrypt_rectangle80_words(
        words, rectangle80_round_keys(keys, config.rounds)
    )
    expected = np.asarray(
        [
            [
                Rectangle80(rounds=config.rounds, key=key).encrypt(int(word))
                for word in words
            ]
            for key in keys
        ],
        dtype=np.uint64,
    )
    return {
        "checked": int(vector.size),
        "maximum_xor_difference": int(np.max(np.bitwise_xor(vector, expected))),
        "all_pass": bool(np.array_equal(vector, expected)),
    }


def validate_rectangle80_negative_witnesses(
    config: Rectangle80ProfileConfig,
    structures: tuple[ActiveStructure, ...],
    raw: dict[str, Any],
    sample_count: int = 24,
) -> dict[str, Any]:
    negatives = np.argwhere(raw["labels"] == 0)
    if not len(negatives):
        return {"checked": 0, "passed": 0, "all_pass": False}
    selected = np.linspace(
        0,
        len(negatives) - 1,
        num=min(sample_count, len(negatives)),
        dtype=np.int64,
    )
    passed = 0
    for index in selected:
        structure_index, output_bit = negatives[int(index)]
        key_index = int(raw["witness_key_indices"][structure_index, output_bit])
        offset = int(raw["witness_offsets"][structure_index, output_bit])
        cipher = Rectangle80(
            rounds=config.rounds, key=raw["keys"][key_index]
        )
        parity = 0
        for assignment in _cube_assignments(
            structures[int(structure_index)].active_bits
        ):
            parity ^= cipher.encrypt(int(assignment) ^ offset)
        passed += int(bool(parity & (1 << int(output_bit))))
    return {
        "checked": int(len(selected)),
        "passed": passed,
        "all_pass": passed == len(selected),
    }


def result_rows_for_rectangle80_profile(
    config: Rectangle80ProfileConfig,
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = gate["metrics"]
    common = {
        "run_id": config.run_id,
        "task": _profile_task(config),
        "cipher": "RECTANGLE-80",
        "rounds": config.rounds,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
    }
    return [
        {
            **common,
            "split": "raw_atlas",
            "raw_positive": metrics["raw_positive"],
            "raw_negative": metrics["raw_negative"],
            "raw_unknown": metrics["raw_unknown"],
        },
        *[
            {
                **common,
                "split": split,
                **metrics["matched_split_metrics"][split],
                "strongest_unary_auc": metrics["matched_marginal_baselines"][
                    "strongest_auc"
                ],
            }
            for split in ("train", "validation")
        ],
    ]


def _profile_experiment(config: Rectangle80ProfileConfig) -> str:
    return (
        "e88"
        if isinstance(config, Rectangle80UnitProfileExpansionConfig)
        else "e87"
    )


def _profile_task(config: Rectangle80ProfileConfig) -> str:
    if _profile_experiment(config) == "e88":
        return "innovation2_rectangle80_unit_balance_profile_expansion"
    return "innovation2_rectangle80_unit_balance_profile_readiness"


def serializable_config(config: Rectangle80ProfileConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "OFFICIAL_ZERO_VECTOR",
    "RECTANGLE_SBOX_ANF",
    "Rectangle80ProfileConfig",
    "Rectangle80UnitProfileConfig",
    "Rectangle80UnitProfileExpansionConfig",
    "adjudicate_rectangle80_profile_checks",
    "build_rectangle80_checkerboard",
    "build_rectangle80_unit_atlas",
    "encrypt_rectangle80_words",
    "evaluate_rectangle80_unit_profile",
    "make_rectangle80_keys",
    "rectangle80_round_keys",
    "rectangle80_variable_supports",
    "reconstruct_rectangle_sbox_from_anf",
    "result_rows_for_rectangle80_profile",
    "serializable_config",
    "validate_rectangle80_vectorized_fixture",
]
