from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from blockcipher_nd.ciphers.spn.gift import (
    Gift64,
    _GIFT64_PERM,
    _ROUND_CONSTANTS,
    _SBOX,
    _key_nibbles_from_int,
    _update_key_nibbles,
)
from blockcipher_nd.tasks.innovation2.present_active_dimension_zero_shot_transfer import (
    compatible_prefix_features,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ActiveStructure,
    _cube_assignments,
    _select_checkerboards,
    checkerboard_balance,
    make_structures,
)
from blockcipher_nd.training.metrics import binary_auc


AUDIT_ROUNDS = 4
AUDIT_STRUCTURES = 96
AUDIT_WITNESS_KEYS = 16
AUDIT_OFFSETS = 8
AUDIT_MATCH_ATTEMPTS = 64
EXPANSION_STRUCTURES = 192
ACTIVE_DIMENSION = 8
OUTPUT_BITS = 64


def _anf_coefficients(truth: tuple[int, ...]) -> tuple[int, ...]:
    coefficients = list(truth)
    for bit in range(4):
        for mask in range(16):
            if mask & (1 << bit):
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    return tuple(coefficients)


GIFT64_SBOX_ANF = tuple(
    tuple(
        mask
        for mask, coefficient in enumerate(
            _anf_coefficients(tuple((value >> output_bit) & 1 for value in _SBOX))
        )
        if coefficient
    )
    for output_bit in range(4)
)


@dataclass(frozen=True)
class Gift64UnitProfileConfig:
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
        ):
            raise ValueError("E74 protocol is frozen")


@dataclass(frozen=True)
class Gift64UnitProfileExpansionConfig:
    run_id: str
    rounds: int = AUDIT_ROUNDS
    structure_count: int = EXPANSION_STRUCTURES
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
            or self.structure_count != EXPANSION_STRUCTURES
            or self.witness_keys != AUDIT_WITNESS_KEYS
            or self.offsets_per_structure != AUDIT_OFFSETS
            or self.match_attempts != AUDIT_MATCH_ATTEMPTS
        ):
            raise ValueError("E75 protocol is frozen")


Gift64ProfileConfig = Gift64UnitProfileConfig | Gift64UnitProfileExpansionConfig


def reconstruct_gift_sbox_from_anf(value: int) -> int:
    if value not in range(16):
        raise ValueError("value must be a nibble")
    output = 0
    for output_bit, terms in enumerate(GIFT64_SBOX_ANF):
        coordinate = 0
        for term in terms:
            coordinate ^= int((value & term) == term)
        output |= coordinate << output_bit
    return output


def make_gift_keys(count: int, seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.getrandbits(128) for _ in range(count))


def gift_round_injections(keys: tuple[int, ...], rounds: int) -> np.ndarray:
    injections = np.zeros((len(keys), rounds), dtype=np.uint64)
    for key_index, key_value in enumerate(keys):
        key = _key_nibbles_from_int(key_value)
        for round_index in range(rounds):
            key_bits = [((key[index] >> bit) & 1) for index in range(32) for bit in range(4)]
            injection = 1 << 63
            for index in range(16):
                injection ^= key_bits[index] << (4 * index)
                injection ^= key_bits[index + 16] << (4 * index + 1)
            constant = _ROUND_CONSTANTS[round_index]
            for bit, target in enumerate((3, 7, 11, 15, 19, 23)):
                injection ^= ((constant >> bit) & 1) << target
            injections[key_index, round_index] = np.uint64(injection)
            key = _update_key_nibbles(key)
    return injections


def encrypt_gift_words(
    plaintexts: np.ndarray,
    injections: np.ndarray,
) -> np.ndarray:
    words = np.asarray(plaintexts, dtype=np.uint64)
    state = np.broadcast_to(words[None, :], (len(injections), len(words))).copy()
    sbox = np.asarray(_SBOX, dtype=np.uint64)
    for round_index in range(injections.shape[1]):
        after_sbox = np.zeros_like(state)
        for nibble in range(16):
            shift = 4 * nibble
            values = ((state >> np.uint64(shift)) & np.uint64(0xF)).astype(np.intp)
            after_sbox |= sbox[values] << np.uint64(shift)
        after_perm = np.zeros_like(state)
        for source, target in enumerate(_GIFT64_PERM):
            after_perm |= ((after_sbox >> np.uint64(source)) & np.uint64(1)) << np.uint64(target)
        state = after_perm ^ injections[:, round_index, None]
    return state


def gift_variable_supports(
    active_bits: tuple[int, ...], rounds: int
) -> tuple[frozenset[int], ...]:
    variable_by_bit = {bit: variable for variable, bit in enumerate(active_bits)}
    state: list[set[int]] = [
        {0, 1 << variable_by_bit[bit]} if bit in variable_by_bit else {0}
        for bit in range(64)
    ]
    for _ in range(rounds):
        after_sbox: list[set[int]] = [set() for _ in range(64)]
        for nibble in range(16):
            inputs = state[4 * nibble : 4 * nibble + 4]
            for output_bit in range(4):
                after_sbox[4 * nibble + output_bit] = _sbox_support(
                    inputs, output_bit
                )
        after_perm: list[set[int]] = [set() for _ in range(64)]
        for source, support in enumerate(after_sbox):
            after_perm[_GIFT64_PERM[source]] = support | {0}
        state = after_perm
    return tuple(frozenset(support) for support in state)


def _sbox_support(inputs: list[set[int]], output_bit: int) -> set[int]:
    output: set[int] = set()
    for local_term in GIFT64_SBOX_ANF[output_bit]:
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


def build_gift_unit_atlas(
    config: Gift64ProfileConfig,
    structures: tuple[ActiveStructure, ...],
) -> dict[str, Any]:
    keys = make_gift_keys(config.witness_keys, config.key_seed)
    injections = gift_round_injections(keys, config.rounds)
    labels = np.full((len(structures), OUTPUT_BITS), -1, dtype=np.int8)
    prefix = np.empty((len(structures), OUTPUT_BITS, 39), dtype=np.float64)
    witness_key_indices = np.full(labels.shape, -1, dtype=np.int16)
    witness_offsets = np.zeros(labels.shape, dtype=np.uint64)
    rows: list[dict[str, Any]] = []
    support_sizes: list[int] = []
    for structure in structures:
        supports = {
            rounds: gift_variable_supports(structure.active_bits, rounds)
            for rounds in (1, 2, 3, 4)
        }
        prefix[structure.index] = compatible_prefix_features(
            {rounds: supports[rounds] for rounds in (1, 2, 3)},
            ACTIVE_DIMENSION,
        )
        support_sizes.extend(len(support) for support in supports[4])
        full_cube = (1 << ACTIVE_DIMENSION) - 1
        assignments = _cube_assignments(structure.active_bits)
        rng = random.Random(
            config.offset_seed
            + sum((index + 1) * bit for index, bit in enumerate(structure.active_bits))
        )
        negative_bits: dict[int, tuple[int, int, int]] = {}
        for offset_index in range(config.offsets_per_structure):
            offset = rng.getrandbits(64) & ~structure.active_mask
            ciphertexts = encrypt_gift_words(
                assignments ^ np.uint64(offset), injections
            )
            parity_words = np.bitwise_xor.reduce(ciphertexts, axis=1)
            for key_index, parity_word in enumerate(parity_words):
                word = int(parity_word)
                for output_bit in range(OUTPUT_BITS):
                    if output_bit not in negative_bits and word & (1 << output_bit):
                        negative_bits[output_bit] = (key_index, offset_index, offset)
        for output_bit in range(OUTPUT_BITS):
            certificate = None
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
                    else f"0x{keys[witness[0]]:032X}",
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


def build_gift_checkerboard(
    labels: np.ndarray,
    structures: tuple[ActiveStructure, ...],
    attempts: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, int]] = {}
    split_indices = {
        "train": tuple(structure.index for structure in structures if structure.index % 4),
        "validation": tuple(structure.index for structure in structures if not structure.index % 4),
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
            [
                {
                    **row,
                    "mask_index": row["output_bit"],
                }
                for row in rows
            ]
        ),
        "marginal_baselines": _marginal_baselines(rows, structures),
    }


def _marginal_baselines(
    rows: list[dict[str, Any]], structures: tuple[ActiveStructure, ...]
) -> dict[str, float]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        return {"global": 0.5, "output_bit": 0.5, "active_bit": 0.5, "strongest_auc": 0.5}
    global_rate = float(np.mean([row["label"] for row in train]))
    output_rates = {
        bit: _rate([row["label"] for row in train if row["output_bit"] == bit], global_rate)
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
        "output_bit": np.asarray([output_rates[row["output_bit"]] for row in validation]),
        "active_bit": np.asarray(
            [
                np.mean(
                    [active_rates[bit] for bit in structures[row["structure_index"]].active_bits]
                )
                for row in validation
            ]
        ),
    }
    aucs = {name: _safe_auc(labels, score) for name, score in predictors.items()}
    return {**aucs, "strongest_auc": max(aucs.values())}


def _rate(values: list[int], default: float) -> float:
    return float(np.mean(values)) if values else default


def evaluate_gift_unit_profile(
    config: Gift64ProfileConfig,
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
    scalar = validate_gift_negative_witnesses(config, structures, raw)
    vector_fixture = validate_gift_vectorized_fixture(config)
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
        "matched_total_structures": len({row["structure_index"] for row in matched["rows"]}),
        "matched_marginal_baselines": matched["marginal_baselines"],
        "matched_balance": matched["balance"],
        "scalar_witness_validation": scalar,
        "vectorized_fixture": vector_fixture,
    }
    protocol_checks = {
        "official_gift_vector_matches": Gift64(rounds=28, key=0).encrypt(0)
        == 0xF62BC3EF34F775AC,
        "gift_sbox_anf_reconstructs": all(
            reconstruct_gift_sbox_from_anf(value) == _SBOX[value] for value in range(16)
        ),
        "gift_permutation_is_bijective": sorted(_GIFT64_PERM) == list(range(64)),
        "vectorized_scalar_fixture_matches": vector_fixture["all_pass"],
        "structure_count_matches": len(structures) == config.structure_count,
        "raw_shape_matches": labels.shape == (config.structure_count, OUTPUT_BITS),
        "all_positive_rows_have_certificate": all(
            row["certificate"] == "full_cube_monomial_absent_from_support_overapprox"
            for row in raw["rows"]
            if row["status"] == "positive"
        ),
        "all_negative_rows_have_witness": all(
            row["witness_key_hex"] is not None and row["witness_offset_hex"] is not None
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
        "matched_total_structures_at_least_32": metrics["matched_total_structures"] >= 32,
        "matched_validation_structures_at_least_8": validation["structures"] >= 8,
        "matched_validation_output_bits_at_least_16": validation["output_bits"] >= 16,
    }
    shortcut_checks = {
        "strongest_unary_auc_at_most_0p65": matched["marginal_baselines"]["strongest_auc"] <= 0.65,
        "duplicate_edges_zero": matched["balance"]["duplicate_edges"] == 0,
        "structure_class_delta_zero": matched["balance"]["maximum_structure_class_delta"] == 0,
        "output_class_delta_zero": matched["balance"]["maximum_mask_class_delta"] == 0,
    }
    status, decision, action = adjudicate_gift_profile_checks(
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
            "strict GIFT-64 r4 8-bit-cube unit-output profile label readiness; "
            "no neural gain, high-round, cross-cipher generalization, attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "neural_readiness": status == "pass",
            "remote_scale": False,
        },
    }


def adjudicate_gift_profile_checks(
    protocol_checks: dict[str, bool],
    width_checks: dict[str, bool],
    shortcut_checks: dict[str, bool],
    *,
    experiment: str = "e74",
) -> tuple[str, str, str]:
    if experiment not in {"e74", "e75"}:
        raise ValueError("experiment must be e74 or e75")
    decision_stem = (
        "innovation2_gift64_unit_balance_profile_expansion"
        if experiment == "e75"
        else "innovation2_gift64_unit_balance_profile"
    )
    if not all(protocol_checks.values()):
        status = "fail"
        decision = f"{decision_stem}_protocol_invalid"
        action = "repair GIFT semantics, vectorization, witness, or certificate protocol"
    elif not all(width_checks.values()) or not all(shortcut_checks.values()):
        status = "hold"
        decision = f"{decision_stem}_not_ready"
        action = "redesign strict GIFT labels or matching before neural training"
    else:
        status = "pass"
        decision = f"{decision_stem}_ready"
        action = "run a local r3-only GIFT profile operator readiness matrix"
    return status, decision, action


def validate_gift_vectorized_fixture(config: Gift64ProfileConfig) -> dict[str, Any]:
    keys = make_gift_keys(4, config.key_seed + 99)
    words = np.asarray(
        [0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0xFFFFFFFFFFFFFFFF],
        dtype=np.uint64,
    )
    vector = encrypt_gift_words(words, gift_round_injections(keys, config.rounds))
    expected = np.asarray(
        [
            [Gift64(rounds=config.rounds, key=key).encrypt(int(word)) for word in words]
            for key in keys
        ],
        dtype=np.uint64,
    )
    return {
        "checked": int(vector.size),
        "maximum_xor_difference": int(np.max(np.bitwise_xor(vector, expected))),
        "all_pass": bool(np.array_equal(vector, expected)),
    }


def validate_gift_negative_witnesses(
    config: Gift64ProfileConfig,
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
        cipher = Gift64(rounds=config.rounds, key=raw["keys"][key_index])
        parity = 0
        for assignment in _cube_assignments(structures[int(structure_index)].active_bits):
            parity ^= cipher.encrypt(int(assignment) ^ offset)
        passed += int(bool(parity & (1 << int(output_bit))))
    return {"checked": int(len(selected)), "passed": passed, "all_pass": passed == len(selected)}


def result_rows_for_gift_profile(
    config: Gift64ProfileConfig, gate: dict[str, Any]
) -> list[dict[str, Any]]:
    metrics = gate["metrics"]
    common = {
        "run_id": config.run_id,
        "task": _profile_task(config),
        "cipher": "GIFT-64",
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


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def _profile_experiment(config: Gift64ProfileConfig) -> str:
    return "e75" if isinstance(config, Gift64UnitProfileExpansionConfig) else "e74"


def _profile_task(config: Gift64ProfileConfig) -> str:
    if _profile_experiment(config) == "e75":
        return "innovation2_gift64_unit_balance_profile_expansion"
    return "innovation2_gift64_unit_balance_profile_readiness"


def serializable_config(config: Gift64ProfileConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "GIFT64_SBOX_ANF",
    "Gift64ProfileConfig",
    "Gift64UnitProfileConfig",
    "Gift64UnitProfileExpansionConfig",
    "adjudicate_gift_profile_checks",
    "build_gift_checkerboard",
    "build_gift_unit_atlas",
    "encrypt_gift_words",
    "evaluate_gift_unit_profile",
    "gift_round_injections",
    "gift_variable_supports",
    "make_gift_keys",
    "reconstruct_gift_sbox_from_anf",
    "result_rows_for_gift_profile",
    "serializable_config",
    "validate_gift_vectorized_fixture",
]
