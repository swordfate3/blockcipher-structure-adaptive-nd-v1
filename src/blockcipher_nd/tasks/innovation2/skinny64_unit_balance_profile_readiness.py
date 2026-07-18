from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from blockcipher_nd.ciphers.spn.skinny import (
    SKINNY64_SBOX,
    SKINNY64_TK_PERMUTATION,
    Skinny64,
    generate_round_constants,
    int_to_cells,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ActiveStructure,
    _cube_assignments,
    _select_checkerboards,
    checkerboard_balance,
)
from blockcipher_nd.training.metrics import binary_auc


AUDIT_ROUNDS = 4
TRANSITION_ROUNDS = 5
AUDIT_STRUCTURES = 96
AUDIT_WITNESS_KEYS = 16
AUDIT_OFFSETS = 8
AUDIT_MATCH_ATTEMPTS = 64
ACTIVE_DIMENSION = 8
OUTPUT_BITS = 64
SHIFT_ROWS_SOURCES = (0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12)


def _anf_coefficients(truth: tuple[int, ...]) -> tuple[int, ...]:
    coefficients = list(truth)
    variables = (len(truth) - 1).bit_length()
    for bit in range(variables):
        for mask in range(len(coefficients)):
            if mask & (1 << bit):
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    return tuple(coefficients)


SKINNY64_SBOX_ANF = tuple(
    tuple(
        mask
        for mask, coefficient in enumerate(
            _anf_coefficients(
                tuple((value >> output_bit) & 1 for value in SKINNY64_SBOX)
            )
        )
        if coefficient
    )
    for output_bit in range(4)
)


@dataclass(frozen=True)
class Skinny64UnitProfileConfig:
    run_id: str
    rounds: int = AUDIT_ROUNDS
    structure_count: int = AUDIT_STRUCTURES
    witness_keys: int = AUDIT_WITNESS_KEYS
    offsets_per_structure: int = AUDIT_OFFSETS
    match_attempts: int = AUDIT_MATCH_ATTEMPTS
    structure_seed: int = 20260718
    key_seed: int = 7401
    offset_seed: int = 17401

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.rounds != AUDIT_ROUNDS
            or self.structure_count != AUDIT_STRUCTURES
            or self.witness_keys != AUDIT_WITNESS_KEYS
            or self.offsets_per_structure != AUDIT_OFFSETS
            or self.match_attempts != AUDIT_MATCH_ATTEMPTS
            or self.structure_seed != 20260718
            or self.key_seed != 7401
            or self.offset_seed != 17401
        ):
            raise ValueError("E81 protocol is frozen")


@dataclass(frozen=True)
class Skinny64UnitProfileTransitionConfig:
    run_id: str
    rounds: int = TRANSITION_ROUNDS
    structure_count: int = AUDIT_STRUCTURES
    witness_keys: int = AUDIT_WITNESS_KEYS
    offsets_per_structure: int = AUDIT_OFFSETS
    match_attempts: int = AUDIT_MATCH_ATTEMPTS
    structure_seed: int = 20260718
    key_seed: int = 7401
    offset_seed: int = 17401

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.rounds != TRANSITION_ROUNDS
            or self.structure_count != AUDIT_STRUCTURES
            or self.witness_keys != AUDIT_WITNESS_KEYS
            or self.offsets_per_structure != AUDIT_OFFSETS
            or self.match_attempts != AUDIT_MATCH_ATTEMPTS
            or self.structure_seed != 20260718
            or self.key_seed != 7401
            or self.offset_seed != 17401
        ):
            raise ValueError("E82 protocol is frozen")


Skinny64ProfileConfig = (
    Skinny64UnitProfileConfig | Skinny64UnitProfileTransitionConfig
)


def reconstruct_skinny_sbox_from_anf(value: int) -> int:
    if value not in range(16):
        raise ValueError("value must be a nibble")
    output = 0
    for output_bit, terms in enumerate(SKINNY64_SBOX_ANF):
        coordinate = 0
        for term in terms:
            coordinate ^= int((value & term) == term)
        output |= coordinate << output_bit
    return output


def make_skinny_keys(count: int, seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.getrandbits(64) for _ in range(count))


def skinny_round_tweakeys(keys: tuple[int, ...], rounds: int) -> np.ndarray:
    tweakeys = np.empty((len(keys), rounds, 8), dtype=np.uint8)
    for key_index, key in enumerate(keys):
        cells = int_to_cells(key)
        for round_index in range(rounds):
            tweakeys[key_index, round_index] = cells[:8]
            cells = tuple(cells[index] for index in SKINNY64_TK_PERMUTATION)
    return tweakeys


def encrypt_skinny_words(
    plaintexts: np.ndarray,
    round_tweakeys: np.ndarray,
    round_constants: tuple[int, ...],
) -> np.ndarray:
    words = np.asarray(plaintexts, dtype=np.uint64)
    key_count = round_tweakeys.shape[0]
    state = np.empty((key_count, len(words), 16), dtype=np.uint8)
    for cell in range(16):
        shift = 4 * (15 - cell)
        state[:, :, cell] = (
            (words[None, :] >> np.uint64(shift)) & np.uint64(0xF)
        ).astype(np.uint8)
    sbox = np.asarray(SKINNY64_SBOX, dtype=np.uint8)
    for round_index, constant in enumerate(round_constants):
        state = sbox[state]
        state[:, :, 0] ^= np.uint8(constant & 0xF)
        state[:, :, 4] ^= np.uint8((constant >> 4) & 0x3)
        state[:, :, 8] ^= np.uint8(0x2)
        state[:, :, :8] ^= round_tweakeys[:, round_index, None, :]
        shifted = state[:, :, SHIFT_ROWS_SOURCES]
        mixed = np.empty_like(shifted)
        for column in range(4):
            s0 = shifted[:, :, column]
            s1 = shifted[:, :, 4 + column]
            s2 = shifted[:, :, 8 + column]
            s3 = shifted[:, :, 12 + column]
            mixed[:, :, column] = s0 ^ s2 ^ s3
            mixed[:, :, 4 + column] = s0
            mixed[:, :, 8 + column] = s1 ^ s2
            mixed[:, :, 12 + column] = s0 ^ s2
        state = mixed
    ciphertexts = np.zeros((key_count, len(words)), dtype=np.uint64)
    for cell in range(16):
        ciphertexts |= state[:, :, cell].astype(np.uint64) << np.uint64(
            4 * (15 - cell)
        )
    return ciphertexts


def skinny_variable_supports(
    active_bits: tuple[int, ...], rounds: int
) -> tuple[frozenset[int], ...]:
    variable_by_bit = {bit: variable for variable, bit in enumerate(active_bits)}
    cells: list[list[set[int]]] = []
    for cell in range(16):
        lanes: list[set[int]] = []
        for lane in range(4):
            bit = _cell_lane_to_bit(cell, lane)
            lanes.append(
                {0, 1 << variable_by_bit[bit]} if bit in variable_by_bit else {0}
            )
        cells.append(lanes)
    for _ in range(rounds):
        after_sbox: list[list[set[int]]] = []
        for cell in range(16):
            after_sbox.append(
                [_sbox_support(cells[cell], output_bit) | {0} for output_bit in range(4)]
            )
        shifted = [after_sbox[source] for source in SHIFT_ROWS_SOURCES]
        mixed: list[list[set[int]]] = [[set() for _ in range(4)] for _ in range(16)]
        for column in range(4):
            for lane in range(4):
                s0 = shifted[column][lane]
                s1 = shifted[4 + column][lane]
                s2 = shifted[8 + column][lane]
                s3 = shifted[12 + column][lane]
                mixed[column][lane] = s0 | s2 | s3
                mixed[4 + column][lane] = set(s0)
                mixed[8 + column][lane] = s1 | s2
                mixed[12 + column][lane] = s0 | s2
        cells = mixed
    output: list[frozenset[int] | None] = [None] * 64
    for cell in range(16):
        for lane in range(4):
            output[_cell_lane_to_bit(cell, lane)] = frozenset(cells[cell][lane])
    if any(support is None for support in output):
        raise AssertionError("all output coordinates must be assigned")
    return tuple(support for support in output if support is not None)


def _sbox_support(inputs: list[set[int]], output_bit: int) -> set[int]:
    output: set[int] = set()
    for local_term in SKINNY64_SBOX_ANF[output_bit]:
        product = {0}
        for input_bit in range(4):
            if local_term & (1 << input_bit):
                product = {
                    left | right for left in product for right in inputs[input_bit]
                }
        output.update(product)
    return output


def _cell_lane_to_bit(cell: int, lane: int) -> int:
    return 4 * (15 - cell) + lane


def build_skinny_unit_atlas(
    config: Skinny64ProfileConfig,
    structures: tuple[ActiveStructure, ...],
) -> dict[str, Any]:
    keys = make_skinny_keys(config.witness_keys, config.key_seed)
    round_tweakeys = skinny_round_tweakeys(keys, config.rounds)
    round_constants = generate_round_constants(config.rounds)
    labels = np.full((len(structures), OUTPUT_BITS), -1, dtype=np.int8)
    prefix_rounds = tuple(range(1, config.rounds))
    prefix = np.empty(
        (len(structures), OUTPUT_BITS, 13 * len(prefix_rounds)), dtype=np.float64
    )
    witness_key_indices = np.full(labels.shape, -1, dtype=np.int16)
    witness_offsets = np.zeros(labels.shape, dtype=np.uint64)
    rows: list[dict[str, Any]] = []
    support_sizes: list[int] = []
    for structure in structures:
        supports = {
            rounds: skinny_variable_supports(structure.active_bits, rounds)
            for rounds in range(1, config.rounds + 1)
        }
        prefix[structure.index] = profile_prefix_features(
            supports,
            ACTIVE_DIMENSION,
            prefix_rounds,
        )
        support_sizes.extend(len(support) for support in supports[config.rounds])
        full_cube = (1 << ACTIVE_DIMENSION) - 1
        assignments = _cube_assignments(structure.active_bits)
        rng = random.Random(
            config.offset_seed
            + sum(
                (index + 1) * bit for index, bit in enumerate(structure.active_bits)
            )
        )
        negative_bits: dict[int, tuple[int, int, int]] = {}
        for offset_index in range(config.offsets_per_structure):
            offset = rng.getrandbits(64) & ~structure.active_mask
            ciphertexts = encrypt_skinny_words(
                assignments ^ np.uint64(offset), round_tweakeys, round_constants
            )
            parity_words = np.bitwise_xor.reduce(ciphertexts, axis=1)
            for key_index, parity_word in enumerate(parity_words):
                word = int(parity_word)
                for output_bit in range(OUTPUT_BITS):
                    if output_bit not in negative_bits and word & (1 << output_bit):
                        negative_bits[output_bit] = (key_index, offset_index, offset)
        for output_bit in range(OUTPUT_BITS):
            witness = negative_bits.get(output_bit)
            if full_cube not in supports[config.rounds][output_bit]:
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
                    else f"0x{keys[witness[0]]:016X}",
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


def profile_prefix_features(
    supports_by_round: dict[int, tuple[frozenset[int], ...]],
    active_dimension: int,
    prefix_rounds: tuple[int, ...],
) -> np.ndarray:
    features = np.empty((64, 13 * len(prefix_rounds)), dtype=np.float64)
    universe = float(1 << active_dimension)
    for output_bit in range(64):
        values: list[float] = []
        for rounds in prefix_rounds:
            support = supports_by_round[rounds][output_bit]
            size = len(support)
            values.extend((size / universe,) * 4)
            degree_counts = np.zeros(9, dtype=np.float64)
            for monomial in support:
                degree_counts[min(monomial.bit_count(), 8)] += 1.0
            degree_counts /= max(1.0, float(degree_counts.sum()))
            values.extend(degree_counts.tolist())
        features[output_bit] = np.asarray(values, dtype=np.float64)
    return features


def build_skinny_checkerboard(
    labels: np.ndarray,
    structures: tuple[ActiveStructure, ...],
    attempts: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, int]] = {}
    split_indices = {
        "train": tuple(
            structure.index for structure in structures if structure.index % 4
        ),
        "validation": tuple(
            structure.index for structure in structures if not structure.index % 4
        ),
    }
    for split_index, split in enumerate(("train", "validation")):
        edges, rectangles = _select_checkerboards(
            labels=labels,
            structure_indices=split_indices[split],
            attempts=attempts,
            seed=17400 + 1000 * split_index,
        )
        for structure_index, output_bit in sorted(edges):
            rows.append(
                {
                    "split": split,
                    "structure_index": structure_index,
                    "structure_id": structures[structure_index].structure_id,
                    "output_bit": output_bit,
                    "label": int(labels[structure_index, output_bit]),
                }
            )
        selected = [row for row in rows if row["split"] == split]
        split_metrics[split] = {
            "rows": len(selected),
            "positive": sum(row["label"] == 1 for row in selected),
            "negative": sum(row["label"] == 0 for row in selected),
            "structures": len({row["structure_index"] for row in selected}),
            "output_bits": len({row["output_bit"] for row in selected}),
            "rectangles": len(rectangles),
        }
    return {
        "rows": rows,
        "split_indices": split_indices,
        "split_metrics": split_metrics,
        "balance": checkerboard_balance(
            [{**row, "mask_index": row["output_bit"]} for row in rows]
        ),
        "marginal_baselines": _marginal_baselines(rows, structures),
    }


def _marginal_baselines(
    rows: list[dict[str, Any]], structures: tuple[ActiveStructure, ...]
) -> dict[str, float]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        return {
            "global": 0.5,
            "output_bit": 0.5,
            "active_bit": 0.5,
            "strongest_auc": 0.5,
        }
    global_rate = float(np.mean([row["label"] for row in train]))
    output_rates = {
        bit: _rate(
            [row["label"] for row in train if row["output_bit"] == bit],
            global_rate,
        )
        for bit in range(64)
    }
    active_rates = {
        bit: _rate(
            [
                row["label"]
                for row in train
                if bit in structures[row["structure_index"]].active_bits
            ],
            global_rate,
        )
        for bit in range(64)
    }
    labels = np.asarray([row["label"] for row in validation], dtype=np.float32)
    predictors = {
        "global": np.full(len(validation), global_rate),
        "output_bit": np.asarray(
            [output_rates[row["output_bit"]] for row in validation]
        ),
        "active_bit": np.asarray(
            [
                np.mean(
                    [
                        active_rates[bit]
                        for bit in structures[row["structure_index"]].active_bits
                    ]
                )
                for row in validation
            ]
        ),
    }
    aucs = {name: _safe_auc(labels, score) for name, score in predictors.items()}
    return {**aucs, "strongest_auc": max(aucs.values())}


def _rate(values: list[int], default: float) -> float:
    return float(np.mean(values)) if values else default


def evaluate_skinny_unit_profile(
    config: Skinny64ProfileConfig,
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
    scalar = validate_skinny_negative_witnesses(config, structures, raw)
    vector_fixture = validate_skinny_vectorized_fixture(config)
    support_fixture = validate_skinny_support_fixture()
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
        "support_fixture": support_fixture,
    }
    protocol_checks = {
        "official_skinny_vector_matches": Skinny64(
            rounds=32, key=0xF5269826FC681238
        ).encrypt(0x06034F957724D19D)
        == 0xBB39DFB2429B8AC7,
        "skinny_sbox_anf_reconstructs": all(
            reconstruct_skinny_sbox_from_anf(value) == SKINNY64_SBOX[value]
            for value in range(16)
        ),
        "vectorized_scalar_fixture_matches": vector_fixture["all_pass"],
        "support_fixture_exact_terms_are_covered": support_fixture["all_pass"],
        "structure_count_matches": len(structures) == config.structure_count,
        "raw_shape_matches": labels.shape == (config.structure_count, OUTPUT_BITS),
        "prefix_shape_matches": raw["prefix_features"].shape
        == (config.structure_count, OUTPUT_BITS, 13 * (config.rounds - 1)),
        "all_positive_rows_have_certificate": all(
            row["certificate"] == "full_cube_monomial_absent_from_support_overapprox"
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
    width_checks = {
        "raw_each_class_at_least_256": positive >= 256 and negative >= 256,
        "resolved_prevalence_in_0p10_0p90": 0.10
        <= metrics["resolved_positive_prevalence"]
        <= 0.90,
        "mixed_structures_at_least_32": mixed >= 32,
        "distinct_signatures_at_least_4": len(signatures) >= 4,
        "matched_train_each_class_at_least_150": train["positive"] >= 150
        and train["negative"] >= 150,
        "matched_validation_each_class_at_least_50": validation["positive"] >= 50
        and validation["negative"] >= 50,
        "matched_total_structures_at_least_32": metrics["matched_total_structures"]
        >= 32,
        "matched_validation_structures_at_least_8": validation["structures"] >= 8,
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
        "output_class_delta_zero": matched["balance"]["maximum_mask_class_delta"]
        == 0,
    }
    status, decision, action = adjudicate_skinny_profile_checks(
        protocol_checks,
        width_checks,
        shortcut_checks,
        experiment=_profile_experiment(config),
    )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": metrics,
        "claim_scope": (
            f"strict SKINNY-64/64 r{config.rounds} 8-bit-cube unit-output "
            "profile label readiness; "
            "no neural gain, high-round, cross-cipher generalization, attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "neural_readiness": status == "pass",
            "remote_scale": False,
        },
    }


def adjudicate_skinny_profile_checks(
    protocol_checks: dict[str, bool],
    width_checks: dict[str, bool],
    shortcut_checks: dict[str, bool],
    *,
    experiment: str = "e81",
) -> tuple[str, str, str]:
    if experiment not in {"e81", "e82"}:
        raise ValueError("experiment must be e81 or e82")
    stem = (
        "innovation2_skinny64_r5_unit_balance_profile_transition"
        if experiment == "e82"
        else "innovation2_skinny64_unit_balance_profile"
    )
    if not all(protocol_checks.values()):
        return (
            "fail",
            f"{stem}_protocol_invalid",
            "repair SKINNY coordinates, vectorization, support, witness, or split",
        )
    raw_width_keys = (
        "raw_each_class_at_least_256",
        "resolved_prevalence_in_0p10_0p90",
        "mixed_structures_at_least_32",
        "distinct_signatures_at_least_4",
    )
    raw_width_ready = all(width_checks.get(key, False) for key in raw_width_keys)
    if not all(shortcut_checks.values()):
        return (
            "hold",
            f"{stem}_not_ready",
            "stop the shortcut-prone label route and rank a new sound representation",
        )
    if not all(width_checks.values()):
        action = (
            "audit 192-structure packing capacity before any expansion; do not train"
            if raw_width_ready
            else (
                "stop the SKINNY 8-bit unit-profile round scan and rank a new sound label"
                if experiment == "e82"
                else "stop r4 expansion and audit an r5 label-distribution transition"
            )
        )
        return (
            "hold",
            f"{stem}_not_ready",
            action,
        )
    return (
        "pass",
        f"{stem}_ready",
        (
            "run E83 local r4-only SKINNY profile operator readiness"
            if experiment == "e82"
            else "run a local r3-only SKINNY profile operator readiness matrix"
        ),
    )


def validate_skinny_vectorized_fixture(
    config: Skinny64ProfileConfig,
) -> dict[str, Any]:
    keys = make_skinny_keys(4, config.key_seed + 99)
    words = np.asarray(
        [0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0xFFFFFFFFFFFFFFFF],
        dtype=np.uint64,
    )
    vector = encrypt_skinny_words(
        words,
        skinny_round_tweakeys(keys, config.rounds),
        generate_round_constants(config.rounds),
    )
    expected = np.asarray(
        [
            [Skinny64(rounds=config.rounds, key=key).encrypt(int(word)) for word in words]
            for key in keys
        ],
        dtype=np.uint64,
    )
    return {
        "checked": int(vector.size),
        "maximum_xor_difference": int(np.max(np.bitwise_xor(vector, expected))),
        "all_pass": bool(np.array_equal(vector, expected)),
    }


def validate_skinny_support_fixture() -> dict[str, Any]:
    active_bits = (0, 5, 58, 63)
    assignments = _cube_assignments(active_bits)
    ciphertexts = [Skinny64(rounds=1, key=0).encrypt(int(word)) for word in assignments]
    overapprox = skinny_variable_supports(active_bits, 1)
    missing: list[tuple[int, int]] = []
    exact_terms = 0
    for output_bit in range(64):
        truth = tuple((word >> output_bit) & 1 for word in ciphertexts)
        terms = {
            mask
            for mask, coefficient in enumerate(_anf_coefficients(truth))
            if coefficient
        }
        exact_terms += len(terms)
        missing.extend(
            (output_bit, term) for term in sorted(terms - overapprox[output_bit])
        )
    return {
        "active_bits": list(active_bits),
        "exact_terms": exact_terms,
        "missing_terms": len(missing),
        "all_pass": not missing,
    }


def validate_skinny_negative_witnesses(
    config: Skinny64ProfileConfig,
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
        cipher = Skinny64(rounds=config.rounds, key=raw["keys"][key_index])
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


def result_rows_for_skinny_profile(
    config: Skinny64ProfileConfig, gate: dict[str, Any]
) -> list[dict[str, Any]]:
    metrics = gate["metrics"]
    common = {
        "run_id": config.run_id,
        "task": (
            "innovation2_skinny64_r5_unit_balance_profile_transition"
            if _profile_experiment(config) == "e82"
            else "innovation2_skinny64_unit_balance_profile_readiness"
        ),
        "cipher": "SKINNY-64/64",
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


def serializable_config(config: Skinny64ProfileConfig) -> dict[str, Any]:
    return asdict(config)


def _profile_experiment(config: Skinny64ProfileConfig) -> str:
    return "e82" if isinstance(config, Skinny64UnitProfileTransitionConfig) else "e81"


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels, scores))


__all__ = [
    "SKINNY64_SBOX_ANF",
    "Skinny64ProfileConfig",
    "Skinny64UnitProfileConfig",
    "Skinny64UnitProfileTransitionConfig",
    "adjudicate_skinny_profile_checks",
    "build_skinny_checkerboard",
    "build_skinny_unit_atlas",
    "encrypt_skinny_words",
    "evaluate_skinny_unit_profile",
    "make_skinny_keys",
    "profile_prefix_features",
    "reconstruct_skinny_sbox_from_anf",
    "result_rows_for_skinny_profile",
    "serializable_config",
    "skinny_round_tweakeys",
    "skinny_variable_supports",
    "validate_skinny_support_fixture",
    "validate_skinny_vectorized_fixture",
]
