from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.ciphers.spn.rectangle import Rectangle80
from blockcipher_nd.tasks.innovation2.gift64_unit_balance_profile_readiness import (
    build_gift_checkerboard,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    _cube_assignments,
)
from blockcipher_nd.tasks.innovation2.rectangle80_unit_balance_profile_readiness import (
    OFFICIAL_ZERO_VECTOR,
    encrypt_rectangle80_words,
    make_rectangle80_keys,
    rectangle80_round_keys,
    rectangle80_variable_supports,
    validate_rectangle80_vectorized_fixture,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719"
E88_RUN_ID = "i2_rectangle80_r4_unit_balance_profile_192_structures_20260719"
E88_DECISION = "innovation2_rectangle80_unit_profile_expansion_ready"
E88_GATE_SHA256 = "1ba3206a603bc0a3ef9acf6be8bc013038acfd1ca8ce0e46e2fbae00b98d07d1"
E88_ATLAS_SHA256 = "296c40a5d5ffc7f616a485bdd72222b9fa1287e11163dcf5f4839dd8c5fa38aa"
E88_STRUCTURES_SHA256 = "3622974c24b885288ba44bbba21e9aba401e0a49c657abd3c762a79281f8c74b"
DIMENSIONS = (7, 8, 9)
OUTPUT_BITS = 64


@dataclass(frozen=True)
class Rectangle80NestedCubeConfig:
    run_id: str = RUN_ID
    rounds: int = 4
    chain_count: int = 192
    witness_keys: int = 16
    offsets_per_node: int = 8
    match_attempts: int = 64
    key_seed: int = 8701
    offset_seed: int = 18701

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.rounds != 4
            or self.chain_count != 192
            or self.witness_keys != 16
            or self.offsets_per_node != 8
            or self.match_attempts != 64
            or self.key_seed != 8701
            or self.offset_seed != 18701
        ):
            raise ValueError("E94 protocol is frozen")


@dataclass(frozen=True)
class NestedCubeChain:
    index: int
    chain_id: str
    active_bits_7: tuple[int, ...]
    active_bits_8: tuple[int, ...]
    active_bits_9: tuple[int, ...]
    removed_bit: int
    added_bit: int
    split: str

    def active_bits(self, dimension: int) -> tuple[int, ...]:
        return {
            7: self.active_bits_7,
            8: self.active_bits_8,
            9: self.active_bits_9,
        }[dimension]


@dataclass(frozen=True)
class _NodeStructure:
    index: int
    structure_id: str
    active_bits: tuple[int, ...]


def load_e88_anchor(anchor_root: Path) -> dict[str, Any]:
    files = {
        "gate": anchor_root / "gate.json",
        "atlas": anchor_root / "atlas.jsonl",
        "structures": anchor_root / "structures.json",
        "metadata": anchor_root / "metadata.json",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E88 anchor is missing: {', '.join(missing)}")

    gate = json.loads(files["gate"].read_text(encoding="utf-8"))
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    structure_rows = json.loads(
        files["structures"].read_text(encoding="utf-8")
    )["structures"]
    atlas_rows = [
        json.loads(line)
        for line in files["atlas"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    labels = np.full((192, OUTPUT_BITS), -1, dtype=np.int8)
    atlas_by_edge: dict[tuple[int, int], dict[str, Any]] = {}
    for row in atlas_rows:
        edge = (int(row["structure_index"]), int(row["output_bit"]))
        if edge in atlas_by_edge:
            raise ValueError("duplicate E88 atlas edge")
        atlas_by_edge[edge] = row
        if row["label"] is not None:
            labels[edge] = int(row["label"])

    hashes = {
        name: hashlib.sha256(files[name].read_bytes()).hexdigest()
        for name in ("gate", "atlas", "structures")
    }
    config = metadata.get("config", {})
    checks = {
        "e88_status_is_pass": gate.get("status") == "pass",
        "e88_decision_matches": gate.get("decision") == E88_DECISION,
        "e88_run_id_matches": gate.get("run_id") == E88_RUN_ID,
        "e88_gate_hash_matches": hashes["gate"] == E88_GATE_SHA256,
        "e88_atlas_hash_matches": hashes["atlas"] == E88_ATLAS_SHA256,
        "e88_structures_hash_matches": hashes["structures"]
        == E88_STRUCTURES_SHA256,
        "e88_structure_count_matches": len(structure_rows) == 192,
        "e88_atlas_shape_matches": len(atlas_by_edge) == 192 * OUTPUT_BITS,
        "e88_protocol_dimensions_match": config.get("rounds") == 4
        and config.get("witness_keys") == 16
        and config.get("offsets_per_structure") == 8,
    }
    return {
        "gate": gate,
        "metadata": metadata,
        "structures": structure_rows,
        "atlas_rows": atlas_rows,
        "atlas_by_edge": atlas_by_edge,
        "labels": labels,
        "hashes": hashes,
        "checks": checks,
    }


def make_nested_chains(
    structure_rows: list[dict[str, Any]],
) -> tuple[NestedCubeChain, ...]:
    chains: list[NestedCubeChain] = []
    for index, row in enumerate(structure_rows):
        active_8 = tuple(int(bit) for bit in row["active_bits"])
        if len(active_8) != 8 or tuple(sorted(set(active_8))) != active_8:
            raise ValueError("E88 structure must contain eight sorted active bits")
        removed = active_8[index % 8]
        active_7 = tuple(bit for bit in active_8 if bit != removed)
        inactive = tuple(bit for bit in range(64) if bit not in active_8)
        added = inactive[index % len(inactive)]
        active_9 = tuple(sorted((*active_8, added)))
        chains.append(
            NestedCubeChain(
                index=index,
                chain_id=f"nested_cube_{index:03d}",
                active_bits_7=active_7,
                active_bits_8=active_8,
                active_bits_9=active_9,
                removed_bit=removed,
                added_bit=added,
                split="validation" if not index % 4 else "train",
            )
        )
    return tuple(chains)


def build_nested_atlas(
    config: Rectangle80NestedCubeConfig,
    chains: tuple[NestedCubeChain, ...],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    keys = make_rectangle80_keys(config.witness_keys, config.key_seed)
    round_keys = rectangle80_round_keys(keys, config.rounds)
    direct_labels = np.full(
        (config.chain_count, len(DIMENSIONS), OUTPUT_BITS), -1, dtype=np.int8
    )
    witness_key_indices = np.full(direct_labels.shape, -1, dtype=np.int16)
    witness_offsets = np.zeros(direct_labels.shape, dtype=np.uint64)
    rows_by_edge: dict[tuple[int, int, int], dict[str, Any]] = {}
    support_sizes: dict[str, list[int]] = {f"d{dimension}": [] for dimension in DIMENSIONS}

    for chain in chains:
        for dimension_index, dimension in enumerate(DIMENSIONS):
            if dimension == 8:
                for output_bit in range(OUTPUT_BITS):
                    source = anchor["atlas_by_edge"][(chain.index, output_bit)]
                    label = -1 if source["label"] is None else int(source["label"])
                    direct_labels[chain.index, dimension_index, output_bit] = label
                    if source.get("witness_key_index") is not None:
                        witness_key_indices[
                            chain.index, dimension_index, output_bit
                        ] = int(source["witness_key_index"])
                    if source.get("witness_offset_hex") is not None:
                        witness_offsets[
                            chain.index, dimension_index, output_bit
                        ] = np.uint64(int(source["witness_offset_hex"], 16))
                    rows_by_edge[(chain.index, dimension, output_bit)] = _nested_row(
                        config,
                        chain,
                        dimension,
                        output_bit,
                        source["status"],
                        label,
                        str(source["certificate"]),
                        source.get("witness_key_index"),
                        source.get("witness_key_hex"),
                        source.get("witness_offset_index"),
                        source.get("witness_offset_hex"),
                    )
                continue

            active_bits = chain.active_bits(dimension)
            supports = rectangle80_variable_supports(active_bits, config.rounds)
            support_sizes[f"d{dimension}"].extend(len(item) for item in supports)
            full_cube = (1 << dimension) - 1
            assignments = _cube_assignments(active_bits)
            rng = random.Random(
                config.offset_seed
                + sum((position + 1) * bit for position, bit in enumerate(active_bits))
            )
            negative_bits: dict[int, tuple[int, int, int]] = {}
            active_mask = sum(1 << bit for bit in active_bits)
            for offset_index in range(config.offsets_per_node):
                offset = rng.getrandbits(64) & ~active_mask
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
                if full_cube not in supports[output_bit]:
                    label = 1
                    status = "positive"
                    certificate = "full_cube_monomial_absent_from_support_overapprox"
                elif witness is not None:
                    label = 0
                    status = "negative"
                    certificate = "concrete_key_offset_unit_xor_one"
                    witness_key_indices[
                        chain.index, dimension_index, output_bit
                    ] = witness[0]
                    witness_offsets[
                        chain.index, dimension_index, output_bit
                    ] = np.uint64(witness[2])
                else:
                    label = -1
                    status = "unknown"
                    certificate = "unresolved"
                direct_labels[chain.index, dimension_index, output_bit] = label
                rows_by_edge[(chain.index, dimension, output_bit)] = _nested_row(
                    config,
                    chain,
                    dimension,
                    output_bit,
                    status,
                    label,
                    certificate,
                    None if witness is None else witness[0],
                    None if witness is None else f"0x{keys[witness[0]]:020X}",
                    None if witness is None else witness[1],
                    None if witness is None else f"0x{witness[2]:016X}",
                )

    closed_labels, monotonicity = apply_positive_monotonic_closure(direct_labels)
    for chain_index, dimension_index, output_bit in np.argwhere(
        (direct_labels < 0) & (closed_labels == 1)
    ):
        dimension = DIMENSIONS[int(dimension_index)]
        row = rows_by_edge[(int(chain_index), dimension, int(output_bit))]
        source_dimension = 7 if direct_labels[chain_index, 0, output_bit] == 1 else 8
        row.update(
            {
                "status": "positive",
                "label": 1,
                "certificate": "positive_inherited_by_cube_superset_monotonicity",
                "certificate_source": f"d{source_dimension}",
            }
        )

    return {
        "rows": [rows_by_edge[key] for key in sorted(rows_by_edge)],
        "direct_labels": direct_labels,
        "labels": closed_labels,
        "keys": keys,
        "witness_key_indices": witness_key_indices,
        "witness_offsets": witness_offsets,
        "support_sizes": support_sizes,
        "monotonicity": monotonicity,
    }


def apply_positive_monotonic_closure(
    direct_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.asarray(direct_labels, dtype=np.int8)
    if labels.ndim != 3 or labels.shape[1] != 3:
        raise ValueError("nested labels must have shape chains x 3 x outputs")
    closed = labels.copy()
    violations = {
        "d7_positive_to_d8_negative": int(
            np.sum((labels[:, 0] == 1) & (labels[:, 1] == 0))
        ),
        "d8_positive_to_d9_negative": int(
            np.sum((labels[:, 1] == 1) & (labels[:, 2] == 0))
        ),
        "d7_positive_to_d9_negative": int(
            np.sum((labels[:, 0] == 1) & (labels[:, 2] == 0))
        ),
    }
    inherited_d8 = (closed[:, 0] == 1) & (closed[:, 1] < 0)
    closed[:, 1][inherited_d8] = 1
    inherited_d9 = ((closed[:, 0] == 1) | (closed[:, 1] == 1)) & (
        closed[:, 2] < 0
    )
    closed[:, 2][inherited_d9] = 1
    return closed, {
        **violations,
        "inherited_positive_d8": int(np.sum(inherited_d8)),
        "inherited_positive_d9": int(np.sum(inherited_d9)),
    }


def build_nested_checkerboard(
    labels: np.ndarray,
    chains: tuple[NestedCubeChain, ...],
    attempts: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dimension_metrics: dict[str, Any] = {}
    dimension_balance: dict[str, Any] = {}
    for dimension_index, dimension in enumerate(DIMENSIONS):
        structures = tuple(
            _NodeStructure(
                index=chain.index,
                structure_id=f"{chain.chain_id}_d{dimension}",
                active_bits=chain.active_bits(dimension),
            )
            for chain in chains
        )
        matched = build_gift_checkerboard(
            labels[:, dimension_index], structures, attempts
        )
        dimension_metrics[f"d{dimension}"] = matched["split_metrics"]
        dimension_balance[f"d{dimension}"] = matched["balance"]
        for row in matched["rows"]:
            chain_index = int(row["structure_index"])
            chain = chains[chain_index]
            rows.append(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "structure_index"
                    },
                    "chain_index": chain_index,
                    "chain_id": chain.chain_id,
                    "dimension": dimension,
                    "removed_bit": chain.removed_bit,
                    "added_bit": chain.added_bit,
                    "active_bits": list(chain.active_bits(dimension)),
                }
            )
    duplicate_rows = len(rows) - len(
        {(row["chain_index"], row["dimension"], row["output_bit"]) for row in rows}
    )
    balance = {
        "duplicate_rows": duplicate_rows,
        "maximum_chain_class_delta": max(
            item["maximum_structure_class_delta"]
            for item in dimension_balance.values()
        ),
        "maximum_output_class_delta": max(
            item["maximum_mask_class_delta"] for item in dimension_balance.values()
        ),
    }
    return {
        "rows": rows,
        "dimension_metrics": dimension_metrics,
        "dimension_balance": dimension_balance,
        "balance": balance,
        "unary_baselines": nested_unary_baselines(rows),
    }


def nested_unary_baselines(rows: list[dict[str, Any]]) -> dict[str, float]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        return {
            name: 0.5
            for name in (
                "global",
                "dimension",
                "output_bit",
                "removed_bit",
                "added_bit",
                "active_position",
                "strongest_auc",
            )
        }
    global_rate = float(np.mean([row["label"] for row in train]))

    def rates(field: str, values: range | tuple[int, ...]) -> dict[int, float]:
        return {
            value: _rate(
                [row["label"] for row in train if int(row[field]) == value],
                global_rate,
            )
            for value in values
        }

    dimension_rates = rates("dimension", DIMENSIONS)
    output_rates = rates("output_bit", range(OUTPUT_BITS))
    removed_rates = rates("removed_bit", range(64))
    added_rates = rates("added_bit", range(64))
    active_rates = {
        bit: _rate(
            [row["label"] for row in train if bit in row["active_bits"]],
            global_rate,
        )
        for bit in range(64)
    }
    labels = np.asarray([row["label"] for row in validation], dtype=np.float32)
    predictors = {
        "global": np.full(len(validation), global_rate),
        "dimension": np.asarray(
            [dimension_rates[int(row["dimension"])] for row in validation]
        ),
        "output_bit": np.asarray(
            [output_rates[int(row["output_bit"])] for row in validation]
        ),
        "removed_bit": np.asarray(
            [removed_rates[int(row["removed_bit"])] for row in validation]
        ),
        "added_bit": np.asarray(
            [added_rates[int(row["added_bit"])] for row in validation]
        ),
        "active_position": np.asarray(
            [np.mean([active_rates[bit] for bit in row["active_bits"]]) for row in validation]
        ),
    }
    aucs = {
        name: _safe_auc(labels, np.asarray(scores, dtype=np.float64))
        for name, scores in predictors.items()
    }
    return {**aucs, "strongest_auc": max(aucs.values())}


def evaluate_nested_cube_readiness(
    config: Rectangle80NestedCubeConfig,
    chains: tuple[NestedCubeChain, ...],
    anchor: dict[str, Any],
    raw: dict[str, Any],
    matched: dict[str, Any],
) -> dict[str, Any]:
    direct = raw["direct_labels"]
    labels = raw["labels"]
    direct_counts = _dimension_counts(direct)
    closed_counts = _dimension_counts(labels)
    prevalence = {
        name: counts["positive"] / (counts["positive"] + counts["negative"])
        for name, counts in closed_counts.items()
    }
    mixed = {
        f"d{dimension}": sum(
            bool(np.any(labels[index, dimension_index] == 1))
            and bool(np.any(labels[index, dimension_index] == 0))
            for index in range(len(chains))
        )
        for dimension_index, dimension in enumerate(DIMENSIONS)
    }
    transition_chains = sum(
        any(
            set(int(value) for value in labels[chain_index, :, output_bit] if value >= 0)
            == {0, 1}
            for output_bit in range(OUTPUT_BITS)
        )
        for chain_index in range(len(chains))
    )
    scalar_witnesses = validate_nested_negative_witnesses(config, chains, raw)
    vector_fixture = validate_rectangle80_vectorized_fixture(config)  # type: ignore[arg-type]
    chain_checks = [
        set(chain.active_bits_7) < set(chain.active_bits_8) < set(chain.active_bits_9)
        and chain.removed_bit in chain.active_bits_8
        and chain.removed_bit not in chain.active_bits_7
        and chain.added_bit not in chain.active_bits_8
        and chain.added_bit in chain.active_bits_9
        for chain in chains
    ]
    protocol_checks = {
        **anchor["checks"],
        "official_zero_vector_matches": Rectangle80().encrypt(0)
        == OFFICIAL_ZERO_VECTOR,
        "vectorized_scalar_fixture_matches": vector_fixture["all_pass"],
        "all_192_chains_present": len(chains) == config.chain_count,
        "all_chains_are_strictly_nested": all(chain_checks),
        "chain_splits_are_disjoint": not {
            chain.index for chain in chains if chain.split == "train"
        }.intersection(
            chain.index for chain in chains if chain.split == "validation"
        ),
        "nested_label_shape_matches": labels.shape
        == (config.chain_count, 3, OUTPUT_BITS),
        "matched_rows_preserve_chain_split": all(
            row["split"] == chains[int(row["chain_index"])].split
            for row in matched["rows"]
        ),
        "all_positive_rows_have_sound_certificate": all(
            row["certificate"]
            in {
                "full_cube_monomial_absent_from_support_overapprox",
                "positive_inherited_by_cube_superset_monotonicity",
            }
            for row in raw["rows"]
            if row["status"] == "positive"
        ),
        "all_negative_rows_have_witness": all(
            row["witness_key_hex"] is not None
            and row["witness_offset_hex"] is not None
            for row in raw["rows"]
            if row["status"] == "negative"
        ),
        "sampled_negative_witnesses_scalar_validate": scalar_witnesses["all_pass"],
    }
    monotonicity = raw["monotonicity"]
    monotonic_checks = {
        "d7_positive_to_d8_negative_zero": monotonicity[
            "d7_positive_to_d8_negative"
        ]
        == 0,
        "d8_positive_to_d9_negative_zero": monotonicity[
            "d8_positive_to_d9_negative"
        ]
        == 0,
        "d7_positive_to_d9_negative_zero": monotonicity[
            "d7_positive_to_d9_negative"
        ]
        == 0,
        "positive_prevalence_nondecreasing": prevalence["d7"]
        <= prevalence["d8"]
        <= prevalence["d9"],
    }
    width_checks: dict[str, bool] = {}
    for dimension in DIMENSIONS:
        name = f"d{dimension}"
        width_checks[f"{name}_raw_each_class_at_least_128"] = (
            direct_counts[name]["positive"] >= 128
            and direct_counts[name]["negative"] >= 128
        )
        width_checks[f"{name}_mixed_chains_at_least_24"] = mixed[name] >= 24
        for split, minimum in (("train", 96), ("validation", 32)):
            metrics = matched["dimension_metrics"][name][split]
            width_checks[f"{name}_{split}_each_class_at_least_{minimum}"] = (
                metrics["positive"] >= minimum and metrics["negative"] >= minimum
            )
        validation = matched["dimension_metrics"][name]["validation"]
        width_checks[f"{name}_validation_chains_at_least_8"] = (
            validation["structures"] >= 8
        )
        width_checks[f"{name}_validation_outputs_at_least_16"] = (
            validation["output_bits"] >= 16
        )
    width_checks["transition_chains_at_least_32"] = transition_chains >= 32
    shortcut_checks = {
        "duplicate_rows_zero": matched["balance"]["duplicate_rows"] == 0,
        "chain_class_delta_zero": matched["balance"][
            "maximum_chain_class_delta"
        ]
        == 0,
        "output_class_delta_zero": matched["balance"][
            "maximum_output_class_delta"
        ]
        == 0,
        "strongest_unary_auc_at_most_0p65": matched["unary_baselines"][
            "strongest_auc"
        ]
        <= 0.65,
    }
    status, decision, action = adjudicate_nested_cube_checks(
        protocol_checks, monotonic_checks, width_checks, shortcut_checks
    )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "monotonic_checks": monotonic_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": {
            "direct_counts": direct_counts,
            "closed_counts": closed_counts,
            "closed_positive_prevalence": prevalence,
            "mixed_chains": mixed,
            "transition_chains": transition_chains,
            "monotonicity": monotonicity,
            "matched_dimension_metrics": matched["dimension_metrics"],
            "matched_balance": matched["balance"],
            "matched_unary_baselines": matched["unary_baselines"],
            "scalar_witness_validation": scalar_witnesses,
            "vectorized_fixture": vector_fixture,
            "support_size_ranges": {
                name: {
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in raw["support_sizes"].items()
                if values
            },
            "anchor_hashes": anchor["hashes"],
        },
        "claim_scope": (
            "strict RECTANGLE-80 r4 nested 7/8/9-bit cube unit-output label "
            "and monotonic-mechanism readiness; no neural gain, high-round "
            "distinguisher, attack, cross-cipher transfer, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "neural_readiness": False,
            "deterministic_mechanism_readiness": status == "pass",
        },
    }


def adjudicate_nested_cube_checks(
    protocol_checks: dict[str, bool],
    monotonic_checks: dict[str, bool],
    width_checks: dict[str, bool],
    shortcut_checks: dict[str, bool],
) -> tuple[str, str, str]:
    if not all(protocol_checks.values()) or not all(monotonic_checks.values()):
        return (
            "fail",
            "innovation2_rectangle80_nested_cube_monotonic_protocol_invalid",
            "repair E88 replay, nesting, certificate closure, or witness semantics",
        )
    if not all(width_checks.values()) or not all(shortcut_checks.values()):
        return (
            "hold",
            "innovation2_rectangle80_nested_cube_monotonic_labels_not_ready",
            "close the nested-cube neural route at the label gate",
        )
    return (
        "pass",
        "innovation2_rectangle80_nested_cube_monotonic_labels_ready",
        "run a no-training true/shuffled/wrong-superset monotonic mechanism gate",
    )


def validate_nested_negative_witnesses(
    config: Rectangle80NestedCubeConfig,
    chains: tuple[NestedCubeChain, ...],
    raw: dict[str, Any],
    samples_per_dimension: int = 12,
) -> dict[str, Any]:
    checked = 0
    passed = 0
    by_dimension: dict[str, dict[str, int | bool]] = {}
    direct = raw["direct_labels"]
    for dimension_index, dimension in enumerate(DIMENSIONS):
        negatives = np.argwhere(direct[:, dimension_index] == 0)
        selected = np.linspace(
            0,
            len(negatives) - 1,
            num=min(samples_per_dimension, len(negatives)),
            dtype=np.int64,
        ) if len(negatives) else np.asarray([], dtype=np.int64)
        dimension_passed = 0
        for selected_index in selected:
            chain_index, output_bit = negatives[int(selected_index)]
            key_index = int(
                raw["witness_key_indices"][chain_index, dimension_index, output_bit]
            )
            offset = int(
                raw["witness_offsets"][chain_index, dimension_index, output_bit]
            )
            if key_index < 0:
                continue
            cipher = Rectangle80(
                rounds=config.rounds, key=raw["keys"][key_index]
            )
            parity = 0
            for assignment in _cube_assignments(
                chains[int(chain_index)].active_bits(dimension)
            ):
                parity ^= cipher.encrypt(int(assignment) ^ offset)
            dimension_passed += int(bool(parity & (1 << int(output_bit))))
        count = int(len(selected))
        by_dimension[f"d{dimension}"] = {
            "checked": count,
            "passed": dimension_passed,
            "all_pass": count > 0 and dimension_passed == count,
        }
        checked += count
        passed += dimension_passed
    return {
        "checked": checked,
        "passed": passed,
        "all_pass": checked > 0
        and passed == checked
        and all(bool(item["all_pass"]) for item in by_dimension.values()),
        "by_dimension": by_dimension,
    }


def result_rows_for_nested_cube(
    config: Rectangle80NestedCubeConfig, gate: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = gate["metrics"]
    common = {
        "run_id": config.run_id,
        "task": "innovation2_rectangle80_nested_cube_monotonic_readiness",
        "cipher": "RECTANGLE-80",
        "rounds": config.rounds,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
    }
    for dimension in DIMENSIONS:
        name = f"d{dimension}"
        rows.append(
            {
                **common,
                "split": "raw_atlas",
                "dimension": dimension,
                **{f"raw_{key}": value for key, value in metrics["direct_counts"][name].items()},
                "closed_positive_prevalence": metrics[
                    "closed_positive_prevalence"
                ][name],
            }
        )
        for split in ("train", "validation"):
            rows.append(
                {
                    **common,
                    "split": split,
                    "dimension": dimension,
                    **metrics["matched_dimension_metrics"][name][split],
                    "strongest_unary_auc": metrics["matched_unary_baselines"][
                        "strongest_auc"
                    ],
                }
            )
    return rows


def serializable_config(config: Rectangle80NestedCubeConfig) -> dict[str, Any]:
    return asdict(config)


def _nested_row(
    config: Rectangle80NestedCubeConfig,
    chain: NestedCubeChain,
    dimension: int,
    output_bit: int,
    status: str,
    label: int,
    certificate: str,
    witness_key_index: int | None,
    witness_key_hex: str | None,
    witness_offset_index: int | None,
    witness_offset_hex: str | None,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "chain_index": chain.index,
        "chain_id": chain.chain_id,
        "split": chain.split,
        "dimension": dimension,
        "active_bits": list(chain.active_bits(dimension)),
        "active_mask_hex": f"0x{sum(1 << bit for bit in chain.active_bits(dimension)):016X}",
        "removed_bit": chain.removed_bit,
        "added_bit": chain.added_bit,
        "output_bit": output_bit,
        "raw_status": status,
        "status": status,
        "label": None if label < 0 else label,
        "certificate": certificate,
        "certificate_source": "direct" if label >= 0 else None,
        "witness_key_index": witness_key_index,
        "witness_key_hex": witness_key_hex,
        "witness_offset_index": witness_offset_index,
        "witness_offset_hex": witness_offset_hex,
    }


def _dimension_counts(labels: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        f"d{dimension}": {
            "positive": int(np.sum(labels[:, dimension_index] == 1)),
            "negative": int(np.sum(labels[:, dimension_index] == 0)),
            "unknown": int(np.sum(labels[:, dimension_index] < 0)),
        }
        for dimension_index, dimension in enumerate(DIMENSIONS)
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def _rate(values: list[int], default: float) -> float:
    return float(np.mean(values)) if values else default


__all__ = [
    "DIMENSIONS",
    "E88_ATLAS_SHA256",
    "E88_DECISION",
    "E88_GATE_SHA256",
    "E88_RUN_ID",
    "E88_STRUCTURES_SHA256",
    "NestedCubeChain",
    "RUN_ID",
    "Rectangle80NestedCubeConfig",
    "adjudicate_nested_cube_checks",
    "apply_positive_monotonic_closure",
    "build_nested_atlas",
    "build_nested_checkerboard",
    "evaluate_nested_cube_readiness",
    "load_e88_anchor",
    "make_nested_chains",
    "nested_unary_baselines",
    "result_rows_for_nested_cube",
    "serializable_config",
    "validate_nested_negative_witnesses",
]
