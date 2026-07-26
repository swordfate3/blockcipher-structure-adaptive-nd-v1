from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import torch

from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
    gift64_runtime_structure,
    present_runtime_structure,
    rectangle80_runtime_structure,
    skinny64_runtime_structure,
    uknit64_runtime_structure,
)


RUN_ID = "i1_runtime_spn_source_topology_diversity_d1_20260726"
_EXPECTED_STRUCTURES = {
    "present": {"factory": "present", "rounds": 1, "round_start": 0},
    "gift": {"factory": "gift", "rounds": 1, "round_start": 0},
    "rectangle": {"factory": "rectangle", "rounds": 1, "round_start": 0},
    "skinny": {"factory": "skinny", "rounds": 1, "round_start": 0},
    "uknit": {"factory": "uknit", "rounds": 10, "round_start": 0},
    "dialga": {"factory": "dialga", "rounds": 4, "round_start": 0},
}
_EXPECTED_GENERATOR = {
    "name": "sparse_elementary_row_column_v1",
    "seed_material": "source_gf2_topology_sha256",
    "mutation_counts": [4, 8, 16, 32],
    "seeds": [0, 1, 2, 3],
    "alternate_row_column": True,
    "dialga_width_lift": 2,
    "require_cross_half_edge": True,
}
_EXPECTED_FEATURES = {
    "name": "normalized_gf2_topology_v1",
    "power_rank_exponents": [0, 1, 2, 3],
    "distance": "root_mean_square_euclidean",
}
_EXPECTED_GATES = {
    "minimum_unique_signature_fraction": 0.9,
    "required_changed_from_source_fraction": 1.0,
    "minimum_overall_median_relative_improvement": 0.1,
    "minimum_cipher_median_relative_improvement": 0.1,
    "minimum_improved_cipher_count": 4,
    "required_uknit_median_relative_improvement": 0.1,
    "required_dialga_median_relative_improvement": 0.1,
    "maximum_holdout_signature_collisions": 0,
    "distance_epsilon": 1e-12,
}
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TransitionRecord:
    cipher: str
    transition_index: int
    matrix: torch.Tensor
    cell_membership: torch.Tensor
    bit_role: torch.Tensor

    @property
    def block_bits(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def transition_id(self) -> str:
        return f"{self.cipher}:t{self.transition_index}"


@dataclass(frozen=True)
class TopologyFeatures:
    vector: torch.Tensor
    signature: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateRecord:
    public: Mapping[str, Any]
    features: TopologyFeatures
    control_features: TopologyFeatures


def load_and_validate_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"D1 config unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("D1 config must be a JSON object")
    expected = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "audit": {"training_rows": 0, "optimizer_steps": 0, "remote": False},
        "structures": _EXPECTED_STRUCTURES,
        "generator": _EXPECTED_GENERATOR,
        "features": _EXPECTED_FEATURES,
        "gates": _EXPECTED_GATES,
    }
    if payload != expected:
        raise ValueError("D1 config does not match the frozen zero-training contract")
    return payload


def build_real_transition_panel(config: Mapping[str, Any]) -> list[TransitionRecord]:
    if config.get("structures") != _EXPECTED_STRUCTURES:
        raise ValueError("D1 structure panel does not match the frozen contract")
    structures = {
        "present": present_runtime_structure(rounds=1),
        "gift": gift64_runtime_structure(rounds=1),
        "rectangle": rectangle80_runtime_structure(rounds=1),
        "skinny": skinny64_runtime_structure(rounds=1),
        "uknit": uknit64_runtime_structure(rounds=10, round_start=0),
        "dialga": dialga128_runtime_structure(rounds=4, round_start=0),
    }
    panel: list[TransitionRecord] = []
    for cipher, structure in structures.items():
        for transition_index in range(structure.rounds):
            panel.append(
                TransitionRecord(
                    cipher=cipher,
                    transition_index=transition_index,
                    matrix=structure.linear_matrices[transition_index].clone(),
                    cell_membership=structure.cell_membership.clone(),
                    bit_role=structure.bit_role.clone(),
                )
            )
    return panel


def gf2_rank(matrix: torch.Tensor) -> int:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu")
    if values.ndim != 2:
        raise ValueError("GF(2) rank requires a two-dimensional matrix")
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("GF(2) rank requires a binary matrix")
    return _gf2_rank_rows(_matrix_rows(values))


def lift_transition(
    matrix: torch.Tensor,
    cell_membership: torch.Tensor,
    bit_role: torch.Tensor,
    *,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu")
    membership = torch.as_tensor(cell_membership, dtype=torch.long, device="cpu")
    roles = torch.as_tensor(bit_role, dtype=torch.long, device="cpu")
    if factor <= 1:
        raise ValueError("topology lift factor must be greater than one")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("topology lift requires a square matrix")
    if membership.numel() != values.shape[0] or roles.shape != membership.shape:
        raise ValueError("topology lift cell coordinates do not match matrix width")
    cells = int(torch.max(membership)) + 1
    lifted = torch.block_diag(*[values for _ in range(factor)]).to(torch.uint8)
    lifted_membership = torch.cat(
        [membership + copy_index * cells for copy_index in range(factor)]
    )
    lifted_roles = roles.repeat(factor)
    return lifted, lifted_membership, lifted_roles


def mutate_invertible_matrix(
    matrix: torch.Tensor,
    *,
    mutation_count: int,
    seed: int,
    half_width: int | None,
) -> torch.Tensor:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu").clone()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("topology mutation requires a square matrix")
    width = int(values.shape[0])
    if mutation_count <= 0:
        raise ValueError("topology mutation count must be positive")
    if gf2_rank(values) != width:
        raise ValueError("topology mutation source must be invertible")
    if half_width is not None and (half_width <= 0 or 2 * half_width != width):
        raise ValueError("cross-half topology mutation requires two equal halves")

    generator = torch.Generator().manual_seed(int(seed))
    for operation_index in range(mutation_count):
        if half_width is not None and operation_index < 2:
            target = operation_index % half_width
            source = half_width + operation_index % half_width
        else:
            indices = torch.randperm(width, generator=generator)[:2]
            target, source = (int(indices[0]), int(indices[1]))
        if operation_index % 2 == 0:
            values[target] ^= values[source]
        else:
            values[:, target] ^= values[:, source]

    if gf2_rank(values) != width:
        raise AssertionError("elementary GF(2) operations must preserve full rank")
    if torch.equal(values, matrix):
        raise ValueError("topology mutation unexpectedly returned its source")
    return values


def cell_relabel_matrix(
    matrix: torch.Tensor,
    cell_membership: torch.Tensor,
    bit_role: torch.Tensor,
    *,
    cell_permutation: Sequence[int],
) -> torch.Tensor:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu")
    membership = torch.as_tensor(cell_membership, dtype=torch.long, device="cpu")
    roles = torch.as_tensor(bit_role, dtype=torch.long, device="cpu")
    cells = int(torch.max(membership)) + 1
    permutation = tuple(int(value) for value in cell_permutation)
    if sorted(permutation) != list(range(cells)):
        raise ValueError("cell relabeling must permute every cell exactly once")
    lookup = {
        (int(cell), int(role)): bit_index
        for bit_index, (cell, role) in enumerate(zip(membership, roles, strict=True))
    }
    bit_permutation = torch.empty(values.shape[0], dtype=torch.long)
    for bit_index, (cell, role) in enumerate(zip(membership, roles, strict=True)):
        bit_permutation[bit_index] = lookup[(permutation[int(cell)], int(role))]
    relabeled = torch.empty_like(values)
    relabeled[bit_permutation[:, None], bit_permutation[None, :]] = values
    return relabeled


def topology_features(
    matrix: torch.Tensor,
    cell_membership: torch.Tensor,
    bit_role: torch.Tensor,
    *,
    power_exponents: Sequence[int],
) -> TopologyFeatures:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu")
    membership = torch.as_tensor(cell_membership, dtype=torch.long, device="cpu")
    roles = torch.as_tensor(bit_role, dtype=torch.long, device="cpu")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("topology features require a square matrix")
    width = int(values.shape[0])
    if membership.numel() != width or roles.shape != membership.shape:
        raise ValueError("topology feature coordinates do not match matrix width")
    cells = int(torch.max(membership)) + 1
    if width != 4 * cells:
        raise ValueError("topology features require four-bit cells")

    row_weights = torch.sum(values, dim=1).to(torch.long)
    column_weights = torch.sum(values, dim=0).to(torch.long)
    row_histogram = torch.bincount(row_weights, minlength=width + 1)
    column_histogram = torch.bincount(column_weights, minlength=width + 1)
    targets, sources = torch.nonzero(values, as_tuple=True)
    edge_count = int(targets.numel())
    target_cells = membership[targets]
    source_cells = membership[sources]
    cell_pair_counts = torch.bincount(
        target_cells * cells + source_cells,
        minlength=cells * cells,
    ).reshape(cells, cells)
    target_source_counts = torch.sum(cell_pair_counts > 0, dim=1).to(torch.long)
    source_target_counts = torch.sum(cell_pair_counts > 0, dim=0).to(torch.long)
    target_source_histogram = torch.bincount(target_source_counts, minlength=cells + 1)
    source_target_histogram = torch.bincount(source_target_counts, minlength=cells + 1)
    cell_pair_histogram = torch.bincount(
        cell_pair_counts.reshape(-1).to(torch.long), minlength=17
    )
    role_pair_counts = torch.bincount(roles[targets] * 4 + roles[sources], minlength=16)
    same_cell_edges = int(torch.sum(target_cells == source_cells))
    same_role_edges = int(torch.sum(roles[targets] == roles[sources]))

    is_permutation = bool(
        torch.all(row_weights == 1) and torch.all(column_weights == 1)
    )
    cycle_mass = torch.zeros(width + 1, dtype=torch.long)
    if is_permutation:
        target_by_source = torch.argmax(values, dim=0).tolist()
        visited = [False] * width
        for start in range(width):
            if visited[start]:
                continue
            current = start
            length = 0
            while not visited[current]:
                visited[current] = True
                length += 1
                current = int(target_by_source[current])
            cycle_mass[length] += length

    rows = _matrix_rows(values)
    power_rank_values: list[int] = []
    current = rows
    expected_exponents = tuple(int(value) for value in power_exponents)
    if expected_exponents != tuple(range(len(expected_exponents))):
        raise ValueError("power-rank exponents must be consecutive from zero")
    identity_rows = [1 << index for index in range(width)]
    for _ in expected_exponents:
        power_rank_values.append(
            _gf2_rank_rows(
                [row ^ identity for row, identity in zip(current, identity_rows)]
            )
        )
        current = _square_rows(current)

    raw = {
        "width": width,
        "cells": cells,
        "edge_count": edge_count,
        "row_weight_histogram": row_histogram.tolist(),
        "column_weight_histogram": column_histogram.tolist(),
        "target_source_cell_histogram": target_source_histogram.tolist(),
        "source_target_cell_histogram": source_target_histogram.tolist(),
        "cell_pair_edge_histogram": cell_pair_histogram.tolist(),
        "role_pair_edge_counts": role_pair_counts.tolist(),
        "same_cell_edges": same_cell_edges,
        "same_role_edges": same_role_edges,
        "is_permutation": is_permutation,
        "permutation_cycle_mass": cycle_mass.tolist(),
        "power_plus_identity_ranks": power_rank_values,
    }
    edge_denominator = max(edge_count, 1)
    vector = torch.cat(
        [
            row_histogram.to(torch.float64) / width,
            column_histogram.to(torch.float64) / width,
            target_source_histogram.to(torch.float64) / cells,
            source_target_histogram.to(torch.float64) / cells,
            cell_pair_histogram.to(torch.float64) / (cells * cells),
            role_pair_counts.to(torch.float64) / edge_denominator,
            torch.tensor(
                [
                    same_cell_edges / edge_denominator,
                    same_role_edges / edge_denominator,
                    float(is_permutation),
                ],
                dtype=torch.float64,
            ),
            cycle_mass.to(torch.float64) / width,
            torch.tensor(power_rank_values, dtype=torch.float64) / width,
        ]
    )
    signature = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TopologyFeatures(vector=vector, signature=signature, raw=raw)


def run_topology_diversity_audit(
    config: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    panel = build_real_transition_panel(config)
    power_exponents = tuple(config["features"]["power_rank_exponents"])
    _emit(progress_callback, "real_panel_loaded", transition_rows=len(panel))
    real_features = {
        row.transition_id: topology_features(
            row.matrix,
            row.cell_membership,
            row.bit_role,
            power_exponents=power_exponents,
        )
        for row in panel
    }
    candidates = _build_candidate_bank(
        panel,
        config=config,
        power_exponents=power_exponents,
        progress_callback=progress_callback,
    )
    repeated_candidates = _build_candidate_bank(
        panel,
        config=config,
        power_exponents=power_exponents,
        progress_callback=None,
    )
    manifest_hash = _candidate_manifest_hash(candidates)
    repeated_manifest_hash = _candidate_manifest_hash(repeated_candidates)

    epsilon = float(config["gates"]["distance_epsilon"])
    results: list[dict[str, Any]] = []
    fold_candidate_counts: dict[str, int] = {}
    fold_control_counts: dict[str, int] = {}
    holdout_source_leaks: list[str] = []
    for holdout in panel:
        pool = [
            candidate
            for candidate in candidates
            if candidate.public["block_bits"] == holdout.block_bits
            and candidate.public["source_cipher"] != holdout.cipher
        ]
        fold_candidate_counts[holdout.transition_id] = len(pool)
        fold_control_counts[holdout.transition_id] = len(pool)
        if any(row.public["source_cipher"] == holdout.cipher for row in pool):
            holdout_source_leaks.append(holdout.transition_id)
        holdout_features = real_features[holdout.transition_id]
        candidate_distances = [
            _feature_distance(holdout_features, row.features) for row in pool
        ]
        control_distances = [
            _feature_distance(holdout_features, row.control_features) for row in pool
        ]
        nearest_candidate = min(candidate_distances, default=math.inf)
        nearest_control = min(control_distances, default=math.inf)
        if nearest_control <= epsilon:
            relative_improvement = 0.0
        else:
            relative_improvement = (nearest_control - nearest_candidate) / max(
                nearest_control, epsilon
            )
        collisions = sum(
            row.features.signature == holdout_features.signature for row in pool
        )
        results.append(
            {
                "run_id": RUN_ID,
                "holdout_cipher": holdout.cipher,
                "holdout_transition": holdout.transition_index,
                "holdout_transition_id": holdout.transition_id,
                "block_bits": holdout.block_bits,
                "candidate_count": len(pool),
                "control_count": len(pool),
                "nearest_synthetic_distance": nearest_candidate,
                "nearest_relabel_control_distance": nearest_control,
                "relative_distance_improvement": relative_improvement,
                "holdout_signature_collisions": collisions,
            }
        )

    candidate_rows = [dict(row.public) for row in candidates]
    candidate_signatures = [row.features.signature for row in candidates]
    unique_signature_fraction = len(set(candidate_signatures)) / max(
        len(candidate_signatures), 1
    )
    changed_fraction = sum(
        row.public["changed_from_source"] is True for row in candidates
    ) / max(len(candidates), 1)
    cipher_medians = {
        cipher: median(
            row["relative_distance_improvement"]
            for row in results
            if row["holdout_cipher"] == cipher
        )
        for cipher in _EXPECTED_STRUCTURES
    }
    overall_median = median(row["relative_distance_improvement"] for row in results)
    holdout_collisions = sum(
        int(row["holdout_signature_collisions"]) for row in results
    )
    threshold = float(config["gates"]["minimum_cipher_median_relative_improvement"])
    improved_cipher_count = sum(value >= threshold for value in cipher_medians.values())

    protocol_checks = {
        "six_ciphers_and_eighteen_transition_rows": (
            len(panel) == 18 and len({row.cipher for row in panel}) == 6
        ),
        "real_matrices_full_rank": all(
            gf2_rank(row.matrix) == row.block_bits for row in panel
        ),
        "synthetic_matrices_full_rank": all(
            row.public["full_rank"] is True for row in candidates
        ),
        "holdout_source_exclusion": not holdout_source_leaks,
        "candidate_control_counts_match": fold_candidate_counts == fold_control_counts,
        "relabel_controls_match_source_features": all(
            row.public["control_features_match_source"] is True for row in candidates
        ),
        "dialga_candidates_are_non_dialga_lifts_with_cross_half_edges": all(
            row.public["source_cipher"] != "dialga"
            and row.public["lifted_from_64"] is True
            and row.public["cross_half_edge"] is True
            for row in candidates
            if row.public["block_bits"] == 128
        ),
        "all_synthetic_candidates_change_source_signature": all(
            row.public["changed_from_source"] is True for row in candidates
        ),
        "candidate_manifest_deterministic": manifest_hash == repeated_manifest_hash,
        "zero_training_and_remote_contract": config["audit"]
        == {"training_rows": 0, "optimizer_steps": 0, "remote": False},
    }
    validation_errors = [
        key for key, value in protocol_checks.items() if value is not True
    ]
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not validation_errors else "fail",
        "checks": protocol_checks,
        "errors": validation_errors,
        "real_transition_rows": len(panel),
        "candidate_rows": len(candidates),
        "manifest_sha256": manifest_hash,
        "repeated_manifest_sha256": repeated_manifest_hash,
    }
    gates = config["gates"]
    research_checks = {
        "unique_signature_fraction": unique_signature_fraction
        >= float(gates["minimum_unique_signature_fraction"]),
        "changed_from_source_fraction": changed_fraction
        >= float(gates["required_changed_from_source_fraction"]),
        "overall_median_relative_improvement": overall_median
        >= float(gates["minimum_overall_median_relative_improvement"]),
        "minimum_improved_cipher_count": improved_cipher_count
        >= int(gates["minimum_improved_cipher_count"]),
        "uknit_median_relative_improvement": cipher_medians["uknit"]
        >= float(gates["required_uknit_median_relative_improvement"]),
        "dialga_median_relative_improvement": cipher_medians["dialga"]
        >= float(gates["required_dialga_median_relative_improvement"]),
        "holdout_signature_collisions": holdout_collisions
        <= int(gates["maximum_holdout_signature_collisions"]),
    }
    research_passed = all(research_checks.values())
    if validation["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_source_topology_diversity_invalid"
        next_action = "repair only failed protocol invariants and rerun D1"
    elif research_passed:
        status = "pass"
        decision = "innovation1_runtime_spn_source_topology_diversity_feasible"
        next_action = (
            "prepare D2 synthetic-cipher signal and disk-data readiness; do not "
            "start neural training from topology coverage alone"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_source_topology_diversity_not_ready"
        next_action = (
            "stop synthetic scaling and inspect the failed frozen coverage gate; "
            "do not tune D1 after result reveal"
        )
    gate = {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "metrics": {
            "unique_signature_fraction": unique_signature_fraction,
            "changed_from_source_fraction": changed_fraction,
            "overall_median_relative_improvement": overall_median,
            "cipher_median_relative_improvement": cipher_medians,
            "improved_cipher_count": improved_cipher_count,
            "holdout_signature_collisions": holdout_collisions,
        },
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
        "next_action": next_action,
        "claim_scope": (
            "zero-training topology-space feasibility only; not differential-signal, "
            "neural-training, whole-cipher holdout, universal-SPN, attack, SOTA or "
            "breakthrough evidence"
        ),
        "blocked_actions": [
            "launch D2 training before a separate signal/data readiness pass",
            "change D1 seeds, mutations, features or thresholds after result reveal",
            "treat known-cipher leave-one-out folds as a new unseen-cipher result",
            "modify or duplicate the running C3 protocol",
        ],
    }
    summary = {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
        "real_transition_rows": len(panel),
        "candidate_rows": len(candidates),
        "metrics": gate["metrics"],
        "next_action": next_action,
        "claim_scope": gate["claim_scope"],
    }
    _emit(progress_callback, "audit_adjudicated", status=status, decision=decision)
    return {
        "candidates": candidate_rows,
        "results": results,
        "validation": validation,
        "gate": gate,
        "summary": summary,
    }


def write_audit_artifacts(payload: Mapping[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "candidates.jsonl", payload["candidates"])
    _write_jsonl(output_root / "results.jsonl", payload["results"])
    for name in ("validation", "gate", "summary"):
        _write_json(output_root / f"{name}.json", payload[name])


def _build_candidate_bank(
    panel: Sequence[TransitionRecord],
    *,
    config: Mapping[str, Any],
    power_exponents: Sequence[int],
    progress_callback: ProgressCallback | None,
) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    source_rows = [row for row in panel if row.block_bits == 64]
    for source in source_rows:
        for target_width in (64, 128):
            if target_width == 64:
                base_matrix = source.matrix
                membership = source.cell_membership
                roles = source.bit_role
                lifted = False
                half_width = None
            else:
                base_matrix, membership, roles = lift_transition(
                    source.matrix,
                    source.cell_membership,
                    source.bit_role,
                    factor=int(config["generator"]["dialga_width_lift"]),
                )
                lifted = True
                half_width = 64
            source_features = topology_features(
                base_matrix,
                membership,
                roles,
                power_exponents=power_exponents,
            )
            generator_source_sha256 = _topology_seed_material_sha256(
                base_matrix,
                membership,
                roles,
            )
            cells = int(torch.max(membership)) + 1
            for mutation_count in config["generator"]["mutation_counts"]:
                for seed in config["generator"]["seeds"]:
                    combined_seed = _stable_seed(
                        generator_source_sha256,
                        target_width,
                        mutation_count,
                        seed,
                    )
                    mutated = mutate_invertible_matrix(
                        base_matrix,
                        mutation_count=int(mutation_count),
                        seed=combined_seed,
                        half_width=half_width,
                    )
                    features = topology_features(
                        mutated,
                        membership,
                        roles,
                        power_exponents=power_exponents,
                    )
                    cell_permutation = _deterministic_cell_permutation(
                        cells,
                        seed=combined_seed ^ 0x5A17,
                    )
                    control_matrix = cell_relabel_matrix(
                        base_matrix,
                        membership,
                        roles,
                        cell_permutation=cell_permutation,
                    )
                    control_features = topology_features(
                        control_matrix,
                        membership,
                        roles,
                        power_exponents=power_exponents,
                    )
                    cross_half = (
                        True
                        if half_width is None
                        else bool(torch.any(mutated[:half_width, half_width:]))
                        or bool(torch.any(mutated[half_width:, :half_width]))
                    )
                    candidate_id = f"{source.transition_id}:w{target_width}:m{mutation_count}:s{seed}"
                    public = {
                        "run_id": RUN_ID,
                        "candidate_id": candidate_id,
                        "source_cipher": source.cipher,
                        "source_transition": source.transition_index,
                        "block_bits": target_width,
                        "lifted_from_64": lifted,
                        "mutation_count": int(mutation_count),
                        "seed": int(seed),
                        "generator_source_sha256": generator_source_sha256,
                        "matrix_sha256": _matrix_sha256(mutated),
                        "source_signature": source_features.signature,
                        "synthetic_signature": features.signature,
                        "control_signature": control_features.signature,
                        "full_rank": gf2_rank(mutated) == target_width,
                        "changed_from_source": features.signature
                        != source_features.signature,
                        "control_features_match_source": (
                            control_features.signature == source_features.signature
                            and torch.equal(
                                control_features.vector, source_features.vector
                            )
                        ),
                        "cross_half_edge": cross_half,
                    }
                    candidates.append(
                        CandidateRecord(
                            public=public,
                            features=features,
                            control_features=control_features,
                        )
                    )
        _emit(
            progress_callback,
            "source_candidates_built",
            source_transition=source.transition_id,
            candidate_rows=len(candidates),
        )
    return candidates


def _feature_distance(left: TopologyFeatures, right: TopologyFeatures) -> float:
    if left.vector.shape != right.vector.shape:
        raise ValueError("topology feature distances require the same block width")
    return float(torch.sqrt(torch.mean(torch.square(left.vector - right.vector))))


def _candidate_manifest_hash(candidates: Sequence[CandidateRecord]) -> str:
    canonical = json.dumps(
        [dict(row.public) for row in candidates],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _matrix_sha256(matrix: torch.Tensor) -> str:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu").contiguous()
    return hashlib.sha256(bytes(values.reshape(-1).tolist())).hexdigest()


def _topology_seed_material_sha256(
    matrix: torch.Tensor,
    cell_membership: torch.Tensor,
    bit_role: torch.Tensor,
) -> str:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu").contiguous()
    membership = torch.as_tensor(
        cell_membership, dtype=torch.long, device="cpu"
    ).contiguous()
    roles = torch.as_tensor(bit_role, dtype=torch.long, device="cpu").contiguous()
    payload = {
        "matrix": values.tolist(),
        "cell_membership": membership.tolist(),
        "bit_role": roles.tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_seed(*values: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _deterministic_cell_permutation(cells: int, *, seed: int) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(cells, generator=generator)
    identity = torch.arange(cells)
    if torch.equal(permutation, identity):
        permutation = torch.roll(identity, shifts=1)
    return tuple(int(value) for value in permutation)


def _matrix_rows(matrix: torch.Tensor) -> list[int]:
    rows: list[int] = []
    for row in matrix.tolist():
        value = 0
        for column, bit in enumerate(row):
            if bit:
                value |= 1 << column
        rows.append(value)
    return rows


def _gf2_rank_rows(rows: Sequence[int]) -> int:
    basis: dict[int, int] = {}
    for original in rows:
        value = int(original)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)


def _square_rows(rows: Sequence[int]) -> list[int]:
    squared: list[int] = []
    for row in rows:
        value = int(row)
        output = 0
        while value:
            low_bit = value & -value
            index = low_bit.bit_length() - 1
            output ^= int(rows[index])
            value ^= low_bit
        squared.append(output)
    return squared


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "RUN_ID",
    "TopologyFeatures",
    "TransitionRecord",
    "build_real_transition_panel",
    "cell_relabel_matrix",
    "gf2_rank",
    "lift_transition",
    "load_and_validate_config",
    "mutate_invertible_matrix",
    "run_topology_diversity_audit",
    "topology_features",
    "write_audit_artifacts",
]
