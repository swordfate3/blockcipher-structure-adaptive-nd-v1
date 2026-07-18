from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX_ANF
from blockcipher_nd.models.structure.spn.present_balance_profile_operator import (
    PresentBalanceProfileOperator,
    PresentBalanceProfileOperatorSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_linear_subspace_diversity import (
    _encrypt_present_words,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    _cube_assignments,
    _scalar_parity_word,
    _select_checkerboards,
    checkerboard_balance,
)
from blockcipher_nd.training.metrics import binary_auc


ACTIVE_DIMENSIONS = (4, 12)
STRUCTURES_PER_DIMENSION = 16
WITNESS_KEYS = 8
OFFSETS_PER_STRUCTURE = 4
ROUNDS = 4
CHECKERBOARD_ATTEMPTS = 64
E65_RUN_ID = "i2_present_r4_unit_balance_profile_readiness_20260718"
E68_RUN_ID = "i2_present_r4_prefix_guided_profile_operator_seed1_20260718"
SUPPORT_COMBINATION_CAP = 2_000_000


class SupportGrowthCapExceeded(RuntimeError):
    def __init__(
        self,
        *,
        round_index: int,
        nibble: int,
        output_bit: int,
        local_term: int,
        input_bit: int,
        candidate_count: int,
        cap: int,
    ) -> None:
        super().__init__(
            f"support combination {candidate_count} exceeds frozen cap {cap}"
        )
        self.details = {
            "round": round_index,
            "nibble": nibble,
            "output_bit": output_bit,
            "local_term": local_term,
            "input_bit": input_bit,
            "candidate_count": candidate_count,
            "cap": cap,
        }


@dataclass(frozen=True)
class ActiveDimensionTransferConfig:
    run_id: str
    rounds: int = ROUNDS
    structures_per_dimension: int = STRUCTURES_PER_DIMENSION
    witness_keys: int = WITNESS_KEYS
    offsets_per_structure: int = OFFSETS_PER_STRUCTURE
    checkerboard_attempts: int = CHECKERBOARD_ATTEMPTS
    key_seed: int = 701
    offset_seed: int = 17001

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.rounds != ROUNDS
            or self.structures_per_dimension != STRUCTURES_PER_DIMENSION
            or self.witness_keys != WITNESS_KEYS
            or self.offsets_per_structure != OFFSETS_PER_STRUCTURE
            or self.checkerboard_attempts != CHECKERBOARD_ATTEMPTS
        ):
            raise ValueError("E70 audit protocol is frozen")


@dataclass(frozen=True)
class TransferStructure:
    index: int
    dimension: int
    structure_id: str
    active_bits: tuple[int, ...]

    @property
    def active_mask(self) -> int:
        return sum(1 << bit for bit in self.active_bits)


def make_transfer_structures(dimension: int) -> tuple[TransferStructure, ...]:
    if dimension == 4:
        nibble_sets = [(nibble,) for nibble in range(16)]
    elif dimension == 12:
        nibble_sets = list(combinations(range(16), 3))[:STRUCTURES_PER_DIMENSION]
    else:
        raise ValueError("E70 supports active dimensions 4 and 12")
    return tuple(
        TransferStructure(
            index=index,
            dimension=dimension,
            structure_id=f"d{dimension}_cube_{index:02d}",
            active_bits=tuple(
                bit
                for nibble in nibbles
                for bit in range(4 * nibble, 4 * nibble + 4)
            ),
        )
        for index, nibbles in enumerate(nibble_sets)
    )


def variable_dimension_supports(
    active_bits: tuple[int, ...],
    rounds: int,
    combination_cap: int = SUPPORT_COMBINATION_CAP,
) -> tuple[frozenset[int], ...]:
    if combination_cap < 1:
        raise ValueError("combination_cap must be positive")
    variable_by_bit = {bit: variable for variable, bit in enumerate(active_bits)}
    state: list[set[int]] = [
        {0, 1 << variable_by_bit[bit]} if bit in variable_by_bit else {0}
        for bit in range(64)
    ]
    player = np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)], dtype=np.int64
    )
    for round_index in range(1, rounds + 1):
        keyed_state = [support | {0} for support in state]
        after_sbox: list[set[int]] = [set() for _ in range(64)]
        for nibble in range(16):
            inputs = keyed_state[4 * nibble : 4 * nibble + 4]
            for output_bit in range(4):
                after_sbox[4 * nibble + output_bit] = _bounded_sbox_output_support(
                    inputs,
                    output_bit,
                    round_index=round_index,
                    nibble=nibble,
                    combination_cap=combination_cap,
                )
        after_player: list[set[int]] = [set() for _ in range(64)]
        for bit, support in enumerate(after_sbox):
            after_player[int(player[bit])] = support
        state = after_player
    return tuple(frozenset(support) for support in state)


def _bounded_sbox_output_support(
    inputs: list[set[int]],
    output_bit: int,
    *,
    round_index: int,
    nibble: int,
    combination_cap: int,
) -> set[int]:
    output: set[int] = set()
    for local_term in PRESENT_SBOX_ANF[output_bit]:
        product = {0}
        for input_bit in range(4):
            if not local_term & (1 << input_bit):
                continue
            candidate_count = len(product) * len(inputs[input_bit])
            if candidate_count > combination_cap:
                raise SupportGrowthCapExceeded(
                    round_index=round_index,
                    nibble=nibble,
                    output_bit=output_bit,
                    local_term=local_term,
                    input_bit=input_bit,
                    candidate_count=candidate_count,
                    cap=combination_cap,
                )
            product = {
                left | right for left in product for right in inputs[input_bit]
            }
        output.update(product)
    return output


def compatible_prefix_features(
    supports_by_round: dict[int, tuple[frozenset[int], ...]],
    active_dimension: int,
) -> np.ndarray:
    features = np.empty((64, 39), dtype=np.float64)
    universe = float(1 << active_dimension)
    for output_bit in range(64):
        values: list[float] = []
        for rounds in (1, 2, 3):
            support = supports_by_round[rounds][output_bit]
            size = len(support)
            values.extend((size / universe, size / universe, size / universe, size / universe))
            degree_counts = np.zeros(9, dtype=np.float64)
            for monomial in support:
                degree_counts[min(monomial.bit_count(), 8)] += 1.0
            degree_counts /= max(1.0, float(degree_counts.sum()))
            values.extend(degree_counts.tolist())
        features[output_bit] = np.asarray(values, dtype=np.float64)
    return features


def build_dimension_labels(
    config: ActiveDimensionTransferConfig,
    structures: tuple[TransferStructure, ...],
) -> dict[str, Any]:
    keys = make_keys(count=config.witness_keys, seed=config.key_seed)
    round_keys = present_round_key_matrix(keys, rounds=config.rounds)
    labels = np.full((len(structures), 64), -1, dtype=np.int8)
    witness_key = np.full((len(structures), 64), -1, dtype=np.int16)
    witness_offset = np.zeros((len(structures), 64), dtype=np.uint64)
    prefix = np.zeros((len(structures), 64, 39), dtype=np.float64)
    support_sizes: list[int] = []
    provider_cap_events: list[dict[str, Any]] = []
    completed_structures = 0
    for structure in structures:
        try:
            supports = {
                rounds: variable_dimension_supports(structure.active_bits, rounds)
                for rounds in (1, 2, 3, 4)
            }
        except SupportGrowthCapExceeded as error:
            provider_cap_events.append(
                {
                    "structure_index": structure.index,
                    "structure_id": structure.structure_id,
                    **error.details,
                }
            )
            continue
        completed_structures += 1
        prefix[structure.index] = compatible_prefix_features(
            {rounds: supports[rounds] for rounds in (1, 2, 3)},
            structure.dimension,
        )
        support_sizes.extend(len(support) for support in supports[4])
        full_cube = (1 << structure.dimension) - 1
        assignments = _cube_assignments(structure.active_bits)
        negative_bits: dict[int, tuple[int, int]] = {}
        rng = random.Random(
            config.offset_seed
            + structure.dimension * 1000
            + sum((index + 1) * bit for index, bit in enumerate(structure.active_bits))
        )
        for _ in range(config.offsets_per_structure):
            offset = rng.getrandbits(64) & ~structure.active_mask
            ciphertexts = _encrypt_present_words(
                assignments ^ np.uint64(offset), round_keys
            )
            parity_words = np.bitwise_xor.reduce(ciphertexts, axis=1)
            for key_index, parity_word in enumerate(parity_words):
                word = int(parity_word)
                for output_bit in range(64):
                    if output_bit not in negative_bits and word & (1 << output_bit):
                        negative_bits[output_bit] = (key_index, offset)
        for output_bit in range(64):
            if full_cube not in supports[4][output_bit]:
                labels[structure.index, output_bit] = 1
            elif output_bit in negative_bits:
                labels[structure.index, output_bit] = 0
                witness_key[structure.index, output_bit] = negative_bits[output_bit][0]
                witness_offset[structure.index, output_bit] = np.uint64(
                    negative_bits[output_bit][1]
                )
    return {
        "labels": labels,
        "witness_key_indices": witness_key,
        "witness_offsets": witness_offset,
        "prefix_features": prefix,
        "keys": keys,
        "support_size_min": min(support_sizes, default=None),
        "support_size_max": max(support_sizes, default=None),
        "provider_complete": completed_structures == len(structures),
        "provider_cap_events": provider_cap_events,
        "completed_structures": completed_structures,
    }


def build_transfer_rows(labels: np.ndarray, dimension: int, attempts: int) -> dict[str, Any]:
    edges, rectangles = _select_checkerboards(
        labels=labels,
        structure_indices=tuple(range(labels.shape[0])),
        attempts=attempts,
        seed=70000 + dimension,
    )
    rows = [
        {
            "dimension": dimension,
            "structure_index": structure,
            "output_bit": output_bit,
            "label": int(labels[structure, output_bit]),
            "split": "validation",
        }
        for structure, output_bit in sorted(edges)
    ]
    balance_rows = [
        {
            "split": "validation",
            "structure_index": row["structure_index"],
            "mask_index": row["output_bit"],
            "label": row["label"],
        }
        for row in rows
    ]
    return {
        "rows": rows,
        "rectangles": len(rectangles),
        "metrics": {
            "rows": len(rows),
            "positive": sum(row["label"] == 1 for row in rows),
            "negative": sum(row["label"] == 0 for row in rows),
            "structures": len({row["structure_index"] for row in rows}),
            "output_bits": len({row["output_bit"] for row in rows}),
        },
        "balance": checkerboard_balance(balance_rows),
    }


def scalar_validate_negatives(
    config: ActiveDimensionTransferConfig,
    structures: tuple[TransferStructure, ...],
    data: dict[str, Any],
    sample_count: int = 16,
) -> dict[str, int | bool]:
    negatives = np.argwhere(data["labels"] == 0)
    if not len(negatives):
        return {"checked": 0, "passed": 0, "all_pass": False}
    indices = np.linspace(
        0, len(negatives) - 1, num=min(sample_count, len(negatives)), dtype=np.int64
    )
    passed = 0
    for selected in indices:
        structure_index, output_bit = negatives[int(selected)]
        key_index = int(data["witness_key_indices"][structure_index, output_bit])
        offset = int(data["witness_offsets"][structure_index, output_bit])
        parity = _scalar_parity_word(
            structures[int(structure_index)].active_bits,
            config.rounds,
            int(data["keys"][key_index]),
            offset,
        )
        passed += int(bool(parity & (1 << int(output_bit))))
    return {
        "checked": int(len(indices)),
        "passed": passed,
        "all_pass": passed == len(indices),
    }


def load_transfer_sources(
    e43_root: Path,
    e65_root: Path,
    seed0_root: Path,
    seed1_root: Path,
) -> dict[str, Any]:
    e43_structures = json.loads(
        (e43_root / "structures.json").read_text(encoding="utf-8")
    )["structures"]
    e65_gate = json.loads((e65_root / "gate.json").read_text(encoding="utf-8"))
    with (e65_root / "features.csv").open(encoding="utf-8", newline="") as handle:
        e65_features = list(csv.DictReader(handle))
    seed0_gate = json.loads((seed0_root / "gate.json").read_text(encoding="utf-8"))
    seed1_gate = json.loads((seed1_root / "gate.json").read_text(encoding="utf-8"))
    checkpoint_roots = {0: seed0_root / "checkpoints", 1: seed1_root / "checkpoints"}
    hashes = {
        "e43_structures": _sha256(e43_root / "structures.json"),
        "e65_gate": _sha256(e65_root / "gate.json"),
        "e65_features": _sha256(e65_root / "features.csv"),
        "seed0_gate": _sha256(seed0_root / "gate.json"),
        "seed1_gate": _sha256(seed1_root / "gate.json"),
    }
    for seed, root in checkpoint_roots.items():
        for mode in ("independent", "true", "corrupted"):
            hashes[f"seed{seed}_{mode}_checkpoint"] = _sha256(
                root / f"profile_{mode}_seed{seed}.pt"
            )
    return {
        "e43_structures": e43_structures,
        "e65_gate": e65_gate,
        "e65_features": e65_features,
        "seed0_gate": seed0_gate,
        "seed1_gate": seed1_gate,
        "checkpoint_roots": checkpoint_roots,
        "hashes": hashes,
    }


def validate_transfer_sources(sources: dict[str, Any]) -> dict[str, bool | float]:
    replay_errors = []
    cache: dict[int, np.ndarray] = {}
    for row in sources["e65_features"]:
        structure_index = int(row["structure_index"])
        if structure_index not in cache:
            active_bits = tuple(
                int(bit) for bit in sources["e43_structures"][structure_index]["active_bits"]
            )
            supports = {
                rounds: variable_dimension_supports(active_bits, rounds)
                for rounds in (1, 2, 3)
            }
            cache[structure_index] = compatible_prefix_features(supports, 8)
        output_bit = int(row["output_bit"])
        expected = np.asarray(
            [float(row[f"anf_prefix_{column:02d}"]) for column in range(39)]
        )
        replay_errors.append(
            float(np.max(np.abs(cache[structure_index][output_bit] - expected)))
        )
    replay_error = max(replay_errors, default=math.inf)
    return {
        "e65_run_id_matches": sources["e65_gate"].get("run_id") == E65_RUN_ID,
        "e65_status_pass": sources["e65_gate"].get("status") == "pass",
        "seed0_decision_confirmed": sources["seed0_gate"].get("decision")
        == "innovation2_present_profile_operator_neural_gain_attributed",
        "seed1_run_id_matches": sources["seed1_gate"].get("run_id") == E68_RUN_ID,
        "seed1_decision_confirmed": sources["seed1_gate"].get("decision")
        == "innovation2_present_profile_operator_two_seed_confirmed",
        "all_source_hashes_present": all(
            len(value) == 64 for value in sources["hashes"].values()
        ),
        "eight_bit_prefix_replay_at_most_1e12": replay_error <= 1e-12,
        "eight_bit_prefix_replay_max_abs_error": replay_error,
    }


def evaluate_zero_shot_models(
    sources: dict[str, Any],
    dimension_data: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    true_player = np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)], dtype=np.int64
    )
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    reports: dict[str, dict[str, float]] = {}
    for dimension, payload in dimension_data.items():
        rows = payload["transfer"]["rows"]
        features = torch.from_numpy(payload["data"]["prefix_features"].astype(np.float32))
        labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
        dimension_reports: dict[str, float] = {}
        for seed in (0, 1):
            for mode in ("independent", "true", "corrupted"):
                player = corrupted_player if mode == "corrupted" else true_player
                model = PresentBalanceProfileOperator(
                    PresentBalanceProfileOperatorSpec(
                        input_dim=39,
                        hidden_dim=32,
                        steps=2,
                        dropout=0.10,
                        relation_mode=mode,
                    ),
                    torch.from_numpy(player),
                )
                state = torch.load(
                    sources["checkpoint_roots"][seed]
                    / f"profile_{mode}_seed{seed}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    logits = model(features).numpy()
                scores = np.asarray(
                    [logits[row["structure_index"], row["output_bit"]] for row in rows]
                )
                dimension_reports[f"seed{seed}_{mode}_auc"] = _safe_auc(labels, scores)
        ridge = _fit_e65_ridge_to_transfer(sources["e65_features"], payload, rows)
        dimension_reports["e65_ridge_auc"] = _safe_auc(labels, ridge)
        reports[str(dimension)] = dimension_reports
    return reports


def _fit_e65_ridge_to_transfer(
    e65_rows: list[dict[str, str]], payload: dict[str, Any], target_rows: list[dict[str, Any]]
) -> np.ndarray:
    if not target_rows:
        return np.empty(0, dtype=np.float64)
    train = [row for row in e65_rows if row["split"] == "train"]
    train_x = np.asarray(
        [
            [float(row[f"anf_prefix_{column:02d}"]) for column in range(39)]
            for row in train
        ],
        dtype=np.float64,
    )
    train_y = np.asarray([int(row["label"]) for row in train], dtype=np.float64)
    target_x = np.asarray(
        [
            payload["data"]["prefix_features"][
                row["structure_index"], row["output_bit"]
            ]
            for row in target_rows
        ],
        dtype=np.float64,
    )
    return fit_train_only_ridge(train_x, train_y, target_x, 1e-3)["validation_scores"]


def adjudicate_active_dimension_transfer(
    config: ActiveDimensionTransferConfig,
    source_checks: dict[str, bool | float],
    dimension_data: dict[int, dict[str, Any]],
    transfer_reports: dict[str, dict[str, float]],
) -> dict[str, Any]:
    boolean_source_checks = {
        key: value for key, value in source_checks.items() if isinstance(value, bool)
    }
    label_checks = {}
    for dimension in ACTIVE_DIMENSIONS:
        payload = dimension_data[dimension]
        labels = payload["data"]["labels"]
        metrics = payload["transfer"]["metrics"]
        balance = payload["transfer"]["balance"]
        scalar = payload["scalar_validation"]
        label_checks.update(
            {
                f"d{dimension}_provider_complete": bool(
                    payload["data"]["provider_complete"]
                ),
                f"d{dimension}_raw_each_class_at_least_128": int(np.sum(labels == 1))
                >= 128
                and int(np.sum(labels == 0)) >= 128,
                f"d{dimension}_matched_each_class_at_least_40": metrics["positive"]
                >= 40
                and metrics["negative"] >= 40,
                f"d{dimension}_matched_structures_at_least_8": metrics["structures"]
                >= 8,
                f"d{dimension}_matched_output_bits_at_least_16": metrics["output_bits"]
                >= 16,
                f"d{dimension}_structure_balance": balance[
                    "maximum_structure_class_delta"
                ]
                == 0,
                f"d{dimension}_output_balance": balance["maximum_mask_class_delta"]
                == 0,
                f"d{dimension}_negative_scalar_replay": bool(scalar["all_pass"]),
            }
        )
    model_checks = {}
    deltas = []
    for dimension in ACTIVE_DIMENSIONS:
        report = transfer_reports[str(dimension)]
        model_checks[f"d{dimension}_mean_true_auc_at_least_0p60"] = (
            report["seed0_true_auc"] + report["seed1_true_auc"]
        ) / 2.0 >= 0.60
        for seed in (0, 1):
            true_auc = report[f"seed{seed}_true_auc"]
            model_checks[f"d{dimension}_seed{seed}_true_auc_at_least_0p55"] = (
                true_auc >= 0.55
            )
            deltas.append(
                {
                    "true_minus_independent": true_auc
                    - report[f"seed{seed}_independent_auc"],
                    "true_minus_corrupted": true_auc
                    - report[f"seed{seed}_corrupted_auc"],
                    "true_minus_ridge": true_auc - report["e65_ridge_auc"],
                }
            )
    mean_deltas = {
        key: float(np.mean([row[key] for row in deltas])) for key in deltas[0]
    }
    model_checks.update(
        {
            "mean_true_minus_independent_at_least_0p03": mean_deltas[
                "true_minus_independent"
            ]
            >= 0.03,
            "mean_true_minus_corrupted_at_least_0p03": mean_deltas[
                "true_minus_corrupted"
            ]
            >= 0.03,
            "mean_true_minus_ridge_at_least_0p02": mean_deltas["true_minus_ridge"]
            >= 0.02,
        }
    )
    if not all(boolean_source_checks.values()):
        status = "fail"
        decision = "innovation2_present_active_dimension_transfer_protocol_invalid"
        action = "repair source, prefix compatibility, or checkpoint protocol"
    elif not all(label_checks.values()):
        status = "hold"
        decision = "innovation2_present_active_dimension_transfer_labels_not_ready"
        action = "stop transfer interpretation; strict 4/12-bit labels are too narrow"
    elif not all(model_checks.values()):
        status = "hold"
        decision = "innovation2_present_active_dimension_zero_shot_not_confirmed"
        action = "retain E68 in-domain evidence; do not fine-tune on new dimensions"
    else:
        status = "pass"
        decision = "innovation2_present_active_dimension_zero_shot_confirmed"
        action = "prepare a unified dimension-conditioned profile operator attribution"
    dimension_metrics = {
        str(dimension): {
            "raw_positive": int(np.sum(dimension_data[dimension]["data"]["labels"] == 1)),
            "raw_negative": int(np.sum(dimension_data[dimension]["data"]["labels"] == 0)),
            "raw_unknown": int(np.sum(dimension_data[dimension]["data"]["labels"] < 0)),
            "matched": dimension_data[dimension]["transfer"]["metrics"],
            "scalar_validation": dimension_data[dimension]["scalar_validation"],
            "provider_complete": dimension_data[dimension]["data"][
                "provider_complete"
            ],
            "provider_cap_events": dimension_data[dimension]["data"][
                "provider_cap_events"
            ],
            "completed_structures": dimension_data[dimension]["data"][
                "completed_structures"
            ],
            "transfer": transfer_reports[str(dimension)],
        }
        for dimension in ACTIVE_DIMENSIONS
    }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": boolean_source_checks,
        "label_checks": label_checks,
        "model_checks": model_checks,
        "metrics": {
            "dimensions": dimension_metrics,
            "mean_deltas": mean_deltas,
        },
        "claim_scope": (
            "strict-label and no-training zero-shot active-dimension transfer audit "
            "for the two-seed PRESENT-80 r4 unit-profile operator; no high-round, "
            "new-attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "dimension_conditioned_training": status == "pass",
            "remote_scale": False,
        },
    }


def result_rows_for_transfer(config: ActiveDimensionTransferConfig, gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dimension in ACTIVE_DIMENSIONS:
        metrics = gate["metrics"]["dimensions"][str(dimension)]
        for seed in (0, 1):
            transfer = metrics["transfer"]
            rows.append(
                {
                    "run_id": config.run_id,
                    "task": "innovation2_present_active_dimension_zero_shot_transfer",
                    "active_dimension": dimension,
                    "seed": seed,
                    "raw_positive": metrics["raw_positive"],
                    "raw_negative": metrics["raw_negative"],
                    "raw_unknown": metrics["raw_unknown"],
                    "matched_rows": metrics["matched"]["rows"],
                    "provider_complete": metrics["provider_complete"],
                    "completed_structures": metrics["completed_structures"],
                    "true_auc": transfer[f"seed{seed}_true_auc"],
                    "independent_auc": transfer[f"seed{seed}_independent_auc"],
                    "corrupted_auc": transfer[f"seed{seed}_corrupted_auc"],
                    "ridge_auc": transfer["e65_ridge_auc"],
                    "status": gate["status"],
                    "decision": gate["decision"],
                    "training_performed": False,
                }
            )
    return rows


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def serializable_config(config: ActiveDimensionTransferConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ACTIVE_DIMENSIONS",
    "SUPPORT_COMBINATION_CAP",
    "ActiveDimensionTransferConfig",
    "SupportGrowthCapExceeded",
    "adjudicate_active_dimension_transfer",
    "build_dimension_labels",
    "build_transfer_rows",
    "compatible_prefix_features",
    "evaluate_zero_shot_models",
    "load_transfer_sources",
    "make_transfer_structures",
    "result_rows_for_transfer",
    "scalar_validate_negatives",
    "serializable_config",
    "validate_transfer_sources",
    "variable_dimension_supports",
]
