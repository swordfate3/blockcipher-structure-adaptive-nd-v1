from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_linear_subspace_diversity import (
    _encrypt_present_words,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys
from blockcipher_nd.training.metrics import binary_auc


AUDIT_ROUNDS = 4
AUDIT_STRUCTURES = 96
AUDIT_WITNESS_KEYS = 16
AUDIT_OFFSETS_PER_STRUCTURE = 8
AUDIT_MATCH_ATTEMPTS = 64
ACTIVE_DIMENSION = 8
EXPECTED_OUTPUT_MASKS = 300
MAXIMUM_UNARY_AUC = 0.65


@dataclass(frozen=True)
class UniversalBalanceAtlasConfig:
    run_id: str
    mode: str = "audit"
    rounds: int = AUDIT_ROUNDS
    structure_count: int = AUDIT_STRUCTURES
    witness_keys: int = AUDIT_WITNESS_KEYS
    offsets_per_structure: int = AUDIT_OFFSETS_PER_STRUCTURE
    match_attempts: int = AUDIT_MATCH_ATTEMPTS
    structure_seed: int = 20260718
    key_seed: int = 407
    offset_seed: int = 1701

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.rounds <= 0 or self.structure_count <= 1:
            raise ValueError("rounds and structure_count must be positive")
        if self.witness_keys <= 0 or self.offsets_per_structure <= 0:
            raise ValueError("witness bank dimensions must be positive")
        if self.match_attempts <= 0:
            raise ValueError("match_attempts must be positive")
        if self.mode == "audit" and (
            self.rounds != AUDIT_ROUNDS
            or self.structure_count != AUDIT_STRUCTURES
            or self.witness_keys != AUDIT_WITNESS_KEYS
            or self.offsets_per_structure != AUDIT_OFFSETS_PER_STRUCTURE
            or self.match_attempts != AUDIT_MATCH_ATTEMPTS
        ):
            raise ValueError("E43 audit protocol is frozen")


@dataclass(frozen=True)
class ActiveStructure:
    index: int
    structure_id: str
    role: str
    active_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.role not in {"coordinate_nibble_pair", "random_coordinate_cube"}:
            raise ValueError("unsupported structure role")
        if len(self.active_bits) != ACTIVE_DIMENSION:
            raise ValueError("E43 structures must contain eight active bits")
        if tuple(sorted(set(self.active_bits))) != self.active_bits:
            raise ValueError("active bits must be sorted and unique")
        if self.active_bits[0] < 0 or self.active_bits[-1] >= 64:
            raise ValueError("active bits must be in [0, 63]")

    @property
    def active_mask(self) -> int:
        return sum(1 << bit for bit in self.active_bits)


@dataclass(frozen=True)
class LinearOutputMask:
    index: int
    mask_id: str
    family: str
    value: int

    @property
    def bits(self) -> tuple[int, ...]:
        return tuple(bit for bit in range(64) if self.value & (1 << bit))


@dataclass(frozen=True)
class WitnessRecord:
    key_index: int
    key: int
    offset_index: int
    offset: int
    parity_word: int


def coordinate_anf_terms(output_bit: int) -> tuple[int, ...]:
    if output_bit not in range(4):
        raise ValueError("output_bit must be in [0, 3]")
    coefficients = [(PRESENT_SBOX[value] >> output_bit) & 1 for value in range(16)]
    for bit in range(4):
        for mask in range(16):
            if mask & (1 << bit):
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    return tuple(mask for mask, coefficient in enumerate(coefficients) if coefficient)


PRESENT_SBOX_ANF = tuple(coordinate_anf_terms(bit) for bit in range(4))


def reconstruct_present_sbox_from_anf(value: int) -> int:
    if value not in range(16):
        raise ValueError("value must be a nibble")
    output = 0
    for output_bit, terms in enumerate(PRESENT_SBOX_ANF):
        coordinate = 0
        for term in terms:
            coordinate ^= int((value & term) == term)
        output |= coordinate << output_bit
    return output


def make_structures(config: UniversalBalanceAtlasConfig) -> tuple[ActiveStructure, ...]:
    active_sets: list[tuple[str, tuple[int, ...]]] = []
    for left, right in combinations(range(16), 2):
        bits = tuple(
            bit
            for nibble in (left, right)
            for bit in range(4 * nibble, 4 * nibble + 4)
        )
        active_sets.append(("coordinate_nibble_pair", bits))
        if len(active_sets) >= min(24, config.structure_count):
            break
    rng = random.Random(config.structure_seed)
    seen = {bits for _, bits in active_sets}
    while len(active_sets) < config.structure_count:
        bits = tuple(sorted(rng.sample(range(64), ACTIVE_DIMENSION)))
        if bits in seen:
            continue
        seen.add(bits)
        active_sets.append(("random_coordinate_cube", bits))
    return tuple(
        ActiveStructure(
            index=index,
            structure_id=f"cube_{index:03d}",
            role=role,
            active_bits=bits,
        )
        for index, (role, bits) in enumerate(active_sets)
    )


def make_output_masks() -> tuple[LinearOutputMask, ...]:
    values: list[tuple[str, int]] = []
    seen: set[int] = set()

    def add(family: str, value: int) -> None:
        if value and value not in seen:
            seen.add(value)
            values.append((family, value))

    for bit in range(64):
        add("unit", 1 << bit)
    for nibble in range(16):
        add("nibble", 0xF << (4 * nibble))
    for bit in range(64):
        add("player_pair", (1 << bit) | (1 << _player_target(bit)))
    for nibble in range(16):
        for left, right in combinations(range(4), 2):
            add(
                "same_nibble_pair",
                (1 << (4 * nibble + left)) | (1 << (4 * nibble + right)),
            )
    for nibble in range(16):
        neighbor = (nibble + 1) % 16
        for within in range(4):
            add(
                "adjacent_nibble_pair",
                (1 << (4 * nibble + within))
                | (1 << (4 * neighbor + within)),
            )
    if len(values) != EXPECTED_OUTPUT_MASKS:
        raise AssertionError(
            f"expected {EXPECTED_OUTPUT_MASKS} output masks, got {len(values)}"
        )
    return tuple(
        LinearOutputMask(
            index=index,
            mask_id=f"mask_{index:03d}",
            family=family,
            value=value,
        )
        for index, (family, value) in enumerate(values)
    )


def possible_active_monomials(
    active_bits: tuple[int, ...], rounds: int
) -> tuple[frozenset[int], ...]:
    if len(active_bits) != ACTIVE_DIMENSION:
        raise ValueError("active_bits must contain eight coordinates")
    variable_by_bit = {bit: variable for variable, bit in enumerate(active_bits)}
    state: list[set[int]] = [
        {0, 1 << variable_by_bit[bit]} if bit in variable_by_bit else {0}
        for bit in range(64)
    ]
    for _ in range(rounds):
        keyed_state = [support | {0} for support in state]
        after_sbox: list[set[int]] = [set() for _ in range(64)]
        for nibble in range(16):
            inputs = keyed_state[4 * nibble : 4 * nibble + 4]
            for output_bit in range(4):
                after_sbox[4 * nibble + output_bit] = _sbox_output_support(
                    inputs, output_bit
                )
        after_permutation: list[set[int]] = [set() for _ in range(64)]
        for bit, support in enumerate(after_sbox):
            after_permutation[_player_target(bit)] = support
        state = after_permutation
    return tuple(frozenset(support) for support in state)


def build_witness_bank(
    config: UniversalBalanceAtlasConfig,
    structure: ActiveStructure,
    keys: tuple[int, ...],
) -> tuple[WitnessRecord, ...]:
    assignments = _cube_assignments(structure.active_bits)
    round_keys = present_round_key_matrix(keys, rounds=config.rounds)
    rng = random.Random(
        config.offset_seed
        + sum((index + 1) * bit for index, bit in enumerate(structure.active_bits))
    )
    records: list[WitnessRecord] = []
    for offset_index in range(config.offsets_per_structure):
        offset = rng.getrandbits(64) & ~structure.active_mask
        ciphertexts = _encrypt_present_words(
            assignments ^ np.uint64(offset), round_keys
        )
        parity_words = np.bitwise_xor.reduce(ciphertexts, axis=1)
        records.extend(
            WitnessRecord(
                key_index=key_index,
                key=key,
                offset_index=offset_index,
                offset=offset,
                parity_word=int(parity_words[key_index]),
            )
            for key_index, key in enumerate(keys)
        )
    return tuple(records)


def build_raw_atlas(
    config: UniversalBalanceAtlasConfig,
    structures: tuple[ActiveStructure, ...],
    masks: tuple[LinearOutputMask, ...],
) -> dict[str, Any]:
    keys = make_keys(count=config.witness_keys, seed=config.key_seed)
    labels = np.full((len(structures), len(masks)), -1, dtype=np.int8)
    rows: list[dict[str, Any]] = []
    support_sizes: list[int] = []
    for structure in structures:
        supports = possible_active_monomials(
            structure.active_bits, rounds=config.rounds
        )
        support_sizes.extend(len(support) for support in supports)
        full_cube = (1 << len(structure.active_bits)) - 1
        witnesses = build_witness_bank(config, structure, keys)
        for mask in masks:
            selected_supports = [supports[bit] for bit in mask.bits]
            certified = all(full_cube not in support for support in selected_supports)
            witness = None
            if not certified:
                witness = next(
                    (
                        record
                        for record in witnesses
                        if (record.parity_word & mask.value).bit_count() & 1
                    ),
                    None,
                )
            if certified:
                label = 1
                status = "positive"
                certificate = "full_cube_monomial_absent_from_support_overapprox"
            elif witness is not None:
                label = 0
                status = "negative"
                certificate = "concrete_key_offset_masked_xor_one"
            else:
                label = -1
                status = "unknown"
                certificate = "unresolved"
            labels[structure.index, mask.index] = label
            row = {
                "run_id": config.run_id,
                "structure_index": structure.index,
                "structure_id": structure.structure_id,
                "structure_role": structure.role,
                "active_bits": list(structure.active_bits),
                "active_mask_hex": f"0x{structure.active_mask:016X}",
                "mask_index": mask.index,
                "mask_id": mask.mask_id,
                "mask_family": mask.family,
                "mask_hex": f"0x{mask.value:016X}",
                "mask_weight": mask.value.bit_count(),
                "status": status,
                "label": None if label < 0 else label,
                "certificate": certificate,
                "witness_key_index": None,
                "witness_key_hex": None,
                "witness_offset_index": None,
                "witness_offset_hex": None,
                "witness_parity_word_hex": None,
            }
            if witness is not None:
                row.update(
                    {
                        "witness_key_index": witness.key_index,
                        "witness_key_hex": f"0x{witness.key:020X}",
                        "witness_offset_index": witness.offset_index,
                        "witness_offset_hex": f"0x{witness.offset:016X}",
                        "witness_parity_word_hex": f"0x{witness.parity_word:016X}",
                    }
                )
            rows.append(row)
    return {
        "rows": rows,
        "labels": labels,
        "keys": keys,
        "support_sizes": support_sizes,
    }


def build_checkerboard_benchmark(
    *,
    labels: np.ndarray,
    structures: tuple[ActiveStructure, ...],
    masks: tuple[LinearOutputMask, ...],
    attempts: int,
) -> dict[str, Any]:
    split_indices = {
        "train": tuple(structure.index for structure in structures if structure.index % 4),
        "validation": tuple(
            structure.index for structure in structures if not structure.index % 4
        ),
    }
    rows: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(("train", "validation")):
        selected, rectangles = _select_checkerboards(
            labels=labels,
            structure_indices=split_indices[split],
            attempts=attempts,
            seed=9000 + split_index * 1000,
        )
        for rectangle_index, rectangle in enumerate(rectangles):
            for structure_index, mask_index in rectangle:
                structure = structures[structure_index]
                mask = masks[mask_index]
                rows.append(
                    {
                        "split": split,
                        "rectangle_index": rectangle_index,
                        "structure_index": structure_index,
                        "structure_id": structure.structure_id,
                        "structure_role": structure.role,
                        "active_mask_hex": f"0x{structure.active_mask:016X}",
                        "mask_index": mask_index,
                        "mask_id": mask.mask_id,
                        "mask_family": mask.family,
                        "mask_hex": f"0x{mask.value:016X}",
                        "label": int(labels[structure_index, mask_index]),
                    }
                )
        split_rows = [row for row in rows if row["split"] == split]
        positives = sum(row["label"] == 1 for row in split_rows)
        structure_ids = {row["structure_index"] for row in split_rows}
        mask_ids = {row["mask_index"] for row in split_rows}
        split_metrics[split] = {
            "rows": len(split_rows),
            "positive": positives,
            "negative": len(split_rows) - positives,
            "structures": len(structure_ids),
            "masks": len(mask_ids),
            "rectangles": len(rectangles),
            "selected_edges": len(selected),
        }
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    baselines = marginal_baselines(train_rows, validation_rows, structures, masks)
    balance = checkerboard_balance(rows)
    return {
        "rows": rows,
        "split_indices": split_indices,
        "split_metrics": split_metrics,
        "marginal_baselines": baselines,
        "balance": balance,
    }


def marginal_baselines(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    structures: tuple[ActiveStructure, ...],
    masks: tuple[LinearOutputMask, ...],
) -> dict[str, float]:
    if not train_rows or not validation_rows:
        return {
            "global": 0.5,
            "mask_only": 0.5,
            "mask_family": 0.5,
            "active_bit": 0.5,
            "strongest_auc": 0.5,
        }
    global_rate = float(np.mean([row["label"] for row in train_rows]))
    mask_rates = {
        mask.index: _rate_or_default(
            [row["label"] for row in train_rows if row["mask_index"] == mask.index],
            global_rate,
        )
        for mask in masks
    }
    families = {mask.family for mask in masks}
    family_rates = {
        family: _rate_or_default(
            [row["label"] for row in train_rows if row["mask_family"] == family],
            global_rate,
        )
        for family in families
    }
    active_rates = {
        bit: _rate_or_default(
            [
                row["label"]
                for row in train_rows
                if bit in structures[row["structure_index"]].active_bits
            ],
            global_rate,
        )
        for bit in range(64)
    }
    labels = np.asarray([row["label"] for row in validation_rows], dtype=np.float32)
    predictors = {
        "global": np.full(len(validation_rows), global_rate, dtype=np.float32),
        "mask_only": np.asarray(
            [mask_rates[row["mask_index"]] for row in validation_rows],
            dtype=np.float32,
        ),
        "mask_family": np.asarray(
            [family_rates[row["mask_family"]] for row in validation_rows],
            dtype=np.float32,
        ),
        "active_bit": np.asarray(
            [
                np.mean(
                    [
                        active_rates[bit]
                        for bit in structures[row["structure_index"]].active_bits
                    ]
                )
                for row in validation_rows
            ],
            dtype=np.float32,
        ),
    }
    aucs = {name: _safe_auc(labels, score) for name, score in predictors.items()}
    return {**aucs, "strongest_auc": max(aucs.values())}


def checkerboard_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_edges = len(rows) - len(
        {(row["split"], row["structure_index"], row["mask_index"]) for row in rows}
    )
    structure_deltas: list[int] = []
    mask_deltas: list[int] = []
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        for structure_index in {row["structure_index"] for row in split_rows}:
            values = [
                row["label"]
                for row in split_rows
                if row["structure_index"] == structure_index
            ]
            structure_deltas.append(abs(sum(values) - (len(values) - sum(values))))
        for mask_index in {row["mask_index"] for row in split_rows}:
            values = [
                row["label"]
                for row in split_rows
                if row["mask_index"] == mask_index
            ]
            mask_deltas.append(abs(sum(values) - (len(values) - sum(values))))
    return {
        "duplicate_edges": duplicate_edges,
        "maximum_structure_class_delta": max(structure_deltas, default=0),
        "maximum_mask_class_delta": max(mask_deltas, default=0),
    }


def evaluate_atlas(
    config: UniversalBalanceAtlasConfig,
    structures: tuple[ActiveStructure, ...],
    masks: tuple[LinearOutputMask, ...],
    raw: dict[str, Any],
    matched: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray(raw["labels"], dtype=np.int8)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    unknown = int(np.sum(labels < 0))
    resolved = positive + negative
    mixed_structures = sum(
        bool(np.any(row == 1) and np.any(row == 0)) for row in labels
    )
    signatures = {
        hashlib.sha256(np.asarray(row, dtype=np.int8).tobytes()).hexdigest()
        for row in labels
    }
    scalar_witness = validate_scalar_negative_witnesses(
        raw["rows"], structures, masks, config.rounds
    )
    raw_train = [
        _raw_baseline_row(row)
        for row in raw["rows"]
        if row["label"] is not None and row["structure_index"] % 4
    ]
    raw_validation = [
        _raw_baseline_row(row)
        for row in raw["rows"]
        if row["label"] is not None and not row["structure_index"] % 4
    ]
    raw_baselines = marginal_baselines(raw_train, raw_validation, structures, masks)
    metrics = {
        "raw_rows": int(labels.size),
        "raw_positive": positive,
        "raw_negative": negative,
        "raw_unknown": unknown,
        "raw_resolved_positive_prevalence": positive / resolved if resolved else 0.0,
        "mixed_structures": mixed_structures,
        "distinct_ternary_signatures": len(signatures),
        "support_size_minimum": min(raw["support_sizes"]),
        "support_size_maximum": max(raw["support_sizes"]),
        "raw_marginal_baselines": raw_baselines,
        "matched_split_metrics": matched["split_metrics"],
        "matched_marginal_baselines": matched["marginal_baselines"],
        "matched_balance": matched["balance"],
        "matched_total_structures": len(
            {row["structure_index"] for row in matched["rows"]}
        ),
        "scalar_witness_validation": scalar_witness,
    }
    readiness_checks = {
        "audit_protocol_frozen": config.mode != "audit"
        or (
            config.rounds == AUDIT_ROUNDS
            and config.structure_count == AUDIT_STRUCTURES
            and config.witness_keys == AUDIT_WITNESS_KEYS
            and config.offsets_per_structure == AUDIT_OFFSETS_PER_STRUCTURE
            and config.match_attempts == AUDIT_MATCH_ATTEMPTS
        ),
        "official_present_vector_matches": Present80(rounds=31, key=0).encrypt(0)
        == 0x5579C1387B228445,
        "present_sbox_anf_reconstructs": all(
            reconstruct_present_sbox_from_anf(value) == PRESENT_SBOX[value]
            for value in range(16)
        ),
        "structure_count_matches": len(structures) == config.structure_count,
        "output_mask_count_is_300": len(masks) == EXPECTED_OUTPUT_MASKS,
        "raw_shape_matches": labels.shape == (len(structures), len(masks)),
        "all_positive_rows_have_absence_certificate": all(
            row["certificate"] == "full_cube_monomial_absent_from_support_overapprox"
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
        "sampled_negative_witnesses_scalar_validate": config.mode == "smoke"
        or scalar_witness["all_pass"],
        "train_validation_structures_disjoint": not set(
            matched["split_indices"]["train"]
        ).intersection(matched["split_indices"]["validation"]),
    }
    train = matched["split_metrics"]["train"]
    validation = matched["split_metrics"]["validation"]
    width_checks = {
        "raw_positive_at_least_1000": positive >= 1000,
        "raw_negative_at_least_1000": negative >= 1000,
        "raw_resolved_positive_prevalence_in_0p10_0p90": 0.10
        <= metrics["raw_resolved_positive_prevalence"]
        <= 0.90,
        "mixed_structures_at_least_32": mixed_structures >= 32,
        "distinct_ternary_signatures_at_least_4": len(signatures) >= 4,
        "matched_train_each_class_at_least_250": train["positive"] >= 250
        and train["negative"] >= 250,
        "matched_validation_each_class_at_least_100": validation["positive"]
        >= 100
        and validation["negative"] >= 100,
        "matched_total_structures_at_least_32": metrics[
            "matched_total_structures"
        ]
        >= 32,
        "matched_validation_structures_at_least_8": validation["structures"] >= 8,
    }
    shortcut_checks = {
        "matched_strongest_unary_auc_at_most_0p65": matched[
            "marginal_baselines"
        ]["strongest_auc"]
        <= MAXIMUM_UNARY_AUC,
        "matched_edges_unique": matched["balance"]["duplicate_edges"] == 0,
        "matched_each_structure_balanced": matched["balance"][
            "maximum_structure_class_delta"
        ]
        == 0,
        "matched_each_mask_balanced": matched["balance"][
            "maximum_mask_class_delta"
        ]
        == 0,
    }
    if not all(readiness_checks.values()):
        status = "fail"
        decision = "innovation2_present_universal_balance_atlas_protocol_invalid"
        action = "repair ANF, witness, split, mask, or certificate protocol"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_present_universal_balance_atlas_too_narrow"
        action = "widen coordinate-cube structures or tighten the sound certificate"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_present_universal_balance_atlas_shortcut_dominated"
        action = "redesign the matched benchmark before neural training"
    else:
        status = "pass"
        decision = "innovation2_present_universal_balance_atlas_ready"
        action = (
            "prepare E44 local seed0 unary baseline vs pair-local vs triangle with "
            "fair-corrupted P-layer control"
        )
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": metrics,
        "claim_scope": (
            "real PRESENT-80 r4 universal-balance certificate/counterexample atlas "
            "readiness; not a high-round distinguisher, neural result, complete "
            "classification, or SOTA attack"
        ),
        "next_action": {
            "action": action,
            "training": status == "pass",
            "training_scope": "local seed0 readiness only" if status == "pass" else None,
            "remote_scale": False,
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_universal_balance_atlas",
            "split": "raw_atlas",
            "positive": positive,
            "negative": negative,
            "unknown": unknown,
            "structures": len(structures),
            "masks": len(masks),
            "strongest_marginal_auc": raw_baselines["strongest_auc"],
            "status": status,
            "decision": decision,
            "training_performed": False,
        },
        *[
            {
                "run_id": config.run_id,
                "task": "innovation2_present_universal_balance_atlas",
                "split": split,
                **matched["split_metrics"][split],
                "unknown": 0,
                "strongest_marginal_auc": matched["marginal_baselines"][
                    "strongest_auc"
                ],
                "status": status,
                "decision": decision,
                "training_performed": False,
            }
            for split in ("train", "validation")
        ],
    ]
    return {"gate": gate, "result_rows": result_rows, "metrics": metrics}


def validate_scalar_negative_witnesses(
    raw_rows: list[dict[str, Any]],
    structures: tuple[ActiveStructure, ...],
    masks: tuple[LinearOutputMask, ...],
    rounds: int,
    sample_count: int = 32,
) -> dict[str, Any]:
    negatives = [row for row in raw_rows if row["status"] == "negative"]
    if not negatives:
        return {"checked": 0, "passed": 0, "all_pass": False}
    indices = np.linspace(
        0, len(negatives) - 1, num=min(sample_count, len(negatives)), dtype=np.int64
    )
    passed = 0
    for index in indices:
        row = negatives[int(index)]
        structure = structures[int(row["structure_index"])]
        mask = masks[int(row["mask_index"])]
        key = int(str(row["witness_key_hex"]), 16)
        offset = int(str(row["witness_offset_hex"]), 16)
        parity_word = _scalar_parity_word(structure.active_bits, rounds, key, offset)
        passed += int((parity_word & mask.value).bit_count() & 1 == 1)
    return {
        "checked": int(len(indices)),
        "passed": passed,
        "all_pass": passed == len(indices),
    }


def serializable_config(config: UniversalBalanceAtlasConfig) -> dict[str, Any]:
    return asdict(config)


def _select_checkerboards(
    *,
    labels: np.ndarray,
    structure_indices: tuple[int, ...],
    attempts: int,
    seed: int,
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], ...]]]:
    mask_count = labels.shape[1]
    pair_patterns: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    for left, right in combinations(structure_indices, 2):
        forward = [
            mask
            for mask in range(mask_count)
            if labels[left, mask] == 1 and labels[right, mask] == 0
        ]
        reverse = [
            mask
            for mask in range(mask_count)
            if labels[left, mask] == 0 and labels[right, mask] == 1
        ]
        if forward and reverse:
            pair_patterns[(left, right)] = (forward, reverse)
    best_edges: set[tuple[int, int]] = set()
    best_rectangles: list[tuple[tuple[int, int], ...]] = []
    best_score = (-1, -1)
    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        pairs = list(pair_patterns)
        rng.shuffle(pairs)
        edges: set[tuple[int, int]] = set()
        rectangles: list[tuple[tuple[int, int], ...]] = []
        for left, right in pairs:
            forward = list(pair_patterns[(left, right)][0])
            reverse = list(pair_patterns[(left, right)][1])
            rng.shuffle(forward)
            rng.shuffle(reverse)
            for first, second in zip(forward, reverse, strict=False):
                rectangle = (
                    (left, first),
                    (left, second),
                    (right, first),
                    (right, second),
                )
                if any(edge in edges for edge in rectangle):
                    continue
                edges.update(rectangle)
                rectangles.append(rectangle)
        score = (len({structure for structure, _ in edges}), len(edges))
        if score > best_score:
            best_score = score
            best_edges = edges
            best_rectangles = rectangles
    return best_edges, best_rectangles


def _sbox_output_support(inputs: list[set[int]], output_bit: int) -> set[int]:
    output: set[int] = set()
    for local_term in PRESENT_SBOX_ANF[output_bit]:
        product = {0}
        for input_bit in range(4):
            if local_term & (1 << input_bit):
                product = {left | right for left in product for right in inputs[input_bit]}
        output.update(product)
    return output


def _player_target(bit: int) -> int:
    return (16 * bit) % 63 if bit < 63 else 63


def _cube_assignments(active_bits: tuple[int, ...]) -> np.ndarray:
    values = np.zeros(1 << len(active_bits), dtype=np.uint64)
    for assignment in range(len(values)):
        value = 0
        for variable, bit in enumerate(active_bits):
            value |= ((assignment >> variable) & 1) << bit
        values[assignment] = np.uint64(value)
    return values


def _scalar_parity_word(
    active_bits: tuple[int, ...], rounds: int, key: int, offset: int
) -> int:
    cipher = Present80(rounds=rounds, key=key)
    parity = 0
    for assignment in _cube_assignments(active_bits):
        parity ^= cipher.encrypt(int(assignment) ^ offset)
    return parity


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    return float(binary_auc(labels, scores))


def _rate_or_default(values: list[int], default: float) -> float:
    return float(np.mean(values)) if values else default


def _raw_baseline_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "structure_index": int(row["structure_index"]),
        "mask_index": int(row["mask_index"]),
        "mask_family": str(row["mask_family"]),
        "label": int(row["label"]),
    }
