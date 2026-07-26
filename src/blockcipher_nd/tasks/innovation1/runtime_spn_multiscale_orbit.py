from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    runtime_spn_structure_from_truth_bits,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
    gift64_runtime_structure,
    present_runtime_structure,
    rectangle80_runtime_structure,
    skinny64_runtime_structure,
    uknit64_runtime_structure,
)


RUN_ID = "i1_runtime_spn_multiscale_orbit_basis_c4_20260726"
_EXPECTED_STRUCTURES = {
    "present": {"factory": "present", "rounds": 1, "round_start": 0},
    "gift": {"factory": "gift", "rounds": 1, "round_start": 0},
    "rectangle": {"factory": "rectangle", "rounds": 1, "round_start": 0},
    "skinny": {"factory": "skinny", "rounds": 1, "round_start": 0},
    "uknit": {"factory": "uknit", "rounds": 10, "round_start": 0},
    "dialga": {"factory": "dialga", "rounds": 4, "round_start": 0},
}
_EXPECTED_ORBIT = {
    "depths": [0, 1, 2, 4, 8],
    "traversal": "reverse_runtime_window_cyclic_v1",
    "probe_basis": "all_unit_bits",
    "corruption_seed": 20260724,
    "cell_relabeling": "rotate_cells_by_one",
}
_EXPECTED_CONTROLS = {
    "corrupted_topology": True,
    "no_topology": True,
    "repeat_last_for_heterogeneous": True,
    "rotate_window_for_heterogeneous": True,
}
_EXPECTED_GATES = {
    "minimum_unique_exact_views_per_cipher": 3,
    "minimum_new_multihop_views_per_cipher": 1,
    "minimum_exact_corrupted_support_distance_per_cipher": 0.25,
    "minimum_exact_no_topology_support_distance_per_cipher": 0.25,
    "minimum_heterogeneous_repeat_last_support_distance": 0.10,
    "minimum_heterogeneous_rotated_window_support_distance": 0.10,
    "required_cipher_count": 6,
    "required_heterogeneous_ciphers": ["uknit", "dialga"],
}
ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_and_validate_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"C4 config unreadable: {path}: {exc}") from exc
    expected = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "audit": {"training_rows": 0, "optimizer_steps": 0, "remote": False},
        "structures": _EXPECTED_STRUCTURES,
        "orbit": _EXPECTED_ORBIT,
        "controls": _EXPECTED_CONTROLS,
        "gates": _EXPECTED_GATES,
    }
    if payload != expected:
        raise ValueError("C4 config does not match the frozen zero-training contract")
    return payload


def build_structure_panel(
    config: Mapping[str, Any],
) -> dict[str, RuntimeSpnStructure]:
    if config.get("structures") != _EXPECTED_STRUCTURES:
        raise ValueError("C4 structure panel does not match the frozen contract")
    return {
        "present": present_runtime_structure(rounds=1),
        "gift": gift64_runtime_structure(rounds=1),
        "rectangle": rectangle80_runtime_structure(rounds=1),
        "skinny": skinny64_runtime_structure(rounds=1),
        "uknit": uknit64_runtime_structure(rounds=10, round_start=0),
        "dialga": dialga128_runtime_structure(rounds=4, round_start=0),
    }


def exact_inverse_orbit(
    inverse_matrices: torch.Tensor,
    *,
    depths: Sequence[int],
) -> torch.Tensor:
    matrices = torch.as_tensor(inverse_matrices, dtype=torch.uint8, device="cpu")
    requested = tuple(int(depth) for depth in depths)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("inverse orbit requires rounds x width x width matrices")
    if matrices.shape[0] == 0:
        raise ValueError("inverse orbit requires at least one runtime transition")
    if not torch.all((matrices == 0) | (matrices == 1)):
        raise ValueError("inverse orbit matrices must be binary")
    if not requested or requested[0] != 0:
        raise ValueError("inverse orbit depths must start at zero")
    if any(left >= right for left, right in zip(requested, requested[1:])):
        raise ValueError("inverse orbit depths must be strictly increasing")

    width = int(matrices.shape[1])
    current = torch.eye(width, dtype=torch.uint8)
    captured = {0: current.clone()}
    requested_set = set(requested)
    rounds = int(matrices.shape[0])
    for step in range(1, requested[-1] + 1):
        inverse = matrices[(-step) % rounds]
        current = _gf2_matmul(inverse, current)
        if step in requested_set:
            captured[step] = current.clone()
    return torch.stack([captured[depth] for depth in requested])


def support_jaccard_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_binary = torch.as_tensor(left, dtype=torch.bool, device="cpu")
    right_binary = torch.as_tensor(right, dtype=torch.bool, device="cpu")
    if left_binary.shape != right_binary.shape:
        raise ValueError("support distance requires equal tensor shapes")
    union = int(torch.count_nonzero(left_binary | right_binary))
    if union == 0:
        return 0.0
    difference = int(torch.count_nonzero(left_binary ^ right_binary))
    return difference / union


def run_multiscale_orbit_audit(
    config: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    panel = build_structure_panel(config)
    results = _build_result_rows(
        panel,
        config=config,
        progress_callback=progress_callback,
    )
    repeated_results = _build_result_rows(
        panel,
        config=config,
        progress_callback=None,
    )
    manifest_hash = _manifest_sha256(results)
    repeated_manifest_hash = _manifest_sha256(repeated_results)
    required_heterogeneous = set(config["gates"]["required_heterogeneous_ciphers"])

    protocol_checks = {
        "six_frozen_structures": (
            len(results) == int(config["gates"]["required_cipher_count"])
            and {row["cipher"] for row in results} == set(_EXPECTED_STRUCTURES)
        ),
        "depth_zero_identity": all(row["depth_zero_identity"] for row in results),
        "all_views_binary_and_full_rank": all(
            row["all_views_binary_and_full_rank"] for row in results
        ),
        "depth_one_matches_runtime_exact_inverse": all(
            row["depth_one_matches_runtime_exact_inverse"] for row in results
        ),
        "corruption_deterministic_and_distinct": all(
            row["corruption_deterministic_and_distinct"] for row in results
        ),
        "heterogeneous_controls_scoped_and_distinct": (
            {row["cipher"] for row in results if row["heterogeneous"]}
            == required_heterogeneous
            and all(row["heterogeneous_controls_valid"] for row in results)
        ),
        "cell_relabel_equivariance": all(
            row["cell_relabel_equivariant"] for row in results
        ),
        "complete_unit_basis_and_finite_metrics": all(
            row["probe_count"] == row["block_bits"] and row["metrics_finite"]
            for row in results
        ),
        "manifest_deterministic": manifest_hash == repeated_manifest_hash,
        "zero_training_and_remote_contract": config["audit"]
        == {"training_rows": 0, "optimizer_steps": 0, "remote": False},
    }
    errors = [name for name, passed in protocol_checks.items() if passed is not True]
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not errors else "fail",
        "checks": protocol_checks,
        "errors": errors,
        "result_rows": len(results),
        "manifest_sha256": manifest_hash,
        "repeated_manifest_sha256": repeated_manifest_hash,
    }

    gates = config["gates"]
    research_checks = {
        "all_ciphers_have_unique_multiscale_views": all(
            row["unique_exact_views"]
            >= int(gates["minimum_unique_exact_views_per_cipher"])
            for row in results
        ),
        "all_ciphers_add_a_new_multihop_view": all(
            row["new_multihop_views"]
            >= int(gates["minimum_new_multihop_views_per_cipher"])
            for row in results
        ),
        "all_ciphers_separate_corrupted_topology": all(
            row["exact_corrupted_support_distance"]
            >= float(gates["minimum_exact_corrupted_support_distance_per_cipher"])
            for row in results
        ),
        "all_ciphers_separate_no_topology": all(
            row["exact_no_topology_support_distance"]
            >= float(gates["minimum_exact_no_topology_support_distance_per_cipher"])
            for row in results
        ),
        "heterogeneous_ciphers_separate_repeat_last": all(
            row["exact_repeat_last_support_distance"]
            >= float(gates["minimum_heterogeneous_repeat_last_support_distance"])
            for row in results
            if row["heterogeneous"]
        ),
        "heterogeneous_ciphers_separate_rotated_window": all(
            row["exact_rotated_window_support_distance"]
            >= float(gates["minimum_heterogeneous_rotated_window_support_distance"])
            for row in results
            if row["heterogeneous"]
        ),
    }
    research_passed = all(research_checks.values())
    if validation["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_multiscale_orbit_basis_invalid"
        next_action = "repair only failed C4 protocol invariants and rerun"
    elif research_passed:
        status = "pass"
        decision = "innovation1_runtime_spn_multiscale_orbit_basis_feasible"
        next_action = (
            "after C3 completes, preregister C5 same-budget local neural diagnostic; "
            "C4 alone does not authorize training"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_multiscale_orbit_basis_not_ready"
        next_action = (
            "close the fixed multiscale orbit basis and keep C3 as the only active "
            "training route; do not tune C4 after result reveal"
        )

    gate = {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "metrics": {
            row["cipher"]: {
                "unique_exact_views": row["unique_exact_views"],
                "new_multihop_views": row["new_multihop_views"],
                "exact_corrupted_support_distance": row[
                    "exact_corrupted_support_distance"
                ],
                "exact_no_topology_support_distance": row[
                    "exact_no_topology_support_distance"
                ],
                "exact_repeat_last_support_distance": row[
                    "exact_repeat_last_support_distance"
                ],
                "exact_rotated_window_support_distance": row[
                    "exact_rotated_window_support_distance"
                ],
            }
            for row in results
        },
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
        "next_action": next_action,
        "claim_scope": (
            "zero-training exact-operator representation feasibility only; not "
            "differential signal, neural learnability, unseen-cipher transfer, "
            "nonlinear composability, attack, SOTA or breakthrough evidence"
        ),
        "blocked_actions": [
            "implement or train C5 before C3 completion and a separate plan",
            "change C4 depths, controls, metrics, seeds or thresholds after reveal",
            "treat six known structures as an unseen-cipher result",
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
        "result_rows": len(results),
        "metrics": gate["metrics"],
        "next_action": next_action,
        "claim_scope": gate["claim_scope"],
    }
    _emit(progress_callback, "audit_adjudicated", status=status, decision=decision)
    return {
        "results": results,
        "validation": validation,
        "gate": gate,
        "summary": summary,
    }


def write_audit_artifacts(payload: Mapping[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "results.jsonl", payload["results"])
    for name in ("validation", "gate", "summary"):
        _write_json(output_root / f"{name}.json", payload[name])


def _build_result_rows(
    panel: Mapping[str, RuntimeSpnStructure],
    *,
    config: Mapping[str, Any],
    progress_callback: ProgressCallback | None,
) -> list[dict[str, Any]]:
    depths = tuple(int(value) for value in config["orbit"]["depths"])
    corruption_seed = int(config["orbit"]["corruption_seed"])
    rows: list[dict[str, Any]] = []
    for cipher, structure in panel.items():
        exact = exact_inverse_orbit(
            structure.inverse_linear_matrices,
            depths=depths,
        )
        corrupted_structure = structure.corrupted(seed=corruption_seed)
        repeated_corrupted = structure.corrupted(seed=corruption_seed)
        corrupted = exact_inverse_orbit(
            corrupted_structure.inverse_linear_matrices,
            depths=depths,
        )
        identity = torch.eye(structure.block_bits, dtype=torch.uint8)
        no_topology = identity[None].repeat(len(depths), 1, 1)
        linear_transition_count = len(
            {_matrix_sha256(matrix) for matrix in structure.linear_matrices}
        )
        heterogeneous = linear_transition_count > 1

        repeat_last_distance: float | None = None
        rotated_distance: float | None = None
        heterogeneous_controls_valid = not heterogeneous
        control_views: list[torch.Tensor] = [exact, corrupted, no_topology]
        if heterogeneous:
            repeat_last = structure.repeat_last_transition()
            rotated = _rotate_runtime_window(structure)
            repeat_last_orbit = exact_inverse_orbit(
                repeat_last.inverse_linear_matrices,
                depths=depths,
            )
            rotated_orbit = exact_inverse_orbit(
                rotated.inverse_linear_matrices,
                depths=depths,
            )
            repeat_last_distance = support_jaccard_distance(
                exact[1:], repeat_last_orbit[1:]
            )
            rotated_distance = support_jaccard_distance(exact[1:], rotated_orbit[1:])
            heterogeneous_controls_valid = not torch.equal(
                structure.linear_matrices, repeat_last.linear_matrices
            ) and not torch.equal(structure.linear_matrices, rotated.linear_matrices)
            control_views.extend((repeat_last_orbit, rotated_orbit))

        view_signatures = [_matrix_sha256(view) for view in exact]
        anchor_signatures = set(view_signatures[:2])
        new_multihop_views = len(
            {
                signature
                for signature in view_signatures[2:]
                if signature not in anchor_signatures
            }
        )
        relabeled, bit_permutation = structure.relabel_cells(
            tuple((cell + 1) % structure.cells for cell in range(structure.cells))
        )
        relabeled_orbit = exact_inverse_orbit(
            relabeled.inverse_linear_matrices,
            depths=depths,
        )
        restored_orbit = relabeled_orbit[
            :, bit_permutation[:, None], bit_permutation[None, :]
        ]
        distance_values = [
            support_jaccard_distance(exact[1:], corrupted[1:]),
            support_jaccard_distance(exact[1:], no_topology[1:]),
        ]
        if repeat_last_distance is not None:
            distance_values.append(repeat_last_distance)
        if rotated_distance is not None:
            distance_values.append(rotated_distance)
        all_views_binary_and_full_rank = all(
            _views_binary_and_full_rank(views) for views in control_views
        )
        row = {
            "run_id": RUN_ID,
            "cipher": cipher,
            "block_bits": structure.block_bits,
            "rounds": structure.rounds,
            "linear_transition_count": linear_transition_count,
            "heterogeneous": heterogeneous,
            "depths": list(depths),
            "probe_basis": config["orbit"]["probe_basis"],
            "probe_count": structure.block_bits,
            "structure_linear_sha256": _tensor_sha256(structure.linear_matrices),
            "exact_orbit_sha256": _tensor_sha256(exact),
            "corrupted_orbit_sha256": _tensor_sha256(corrupted),
            "no_topology_orbit_sha256": _tensor_sha256(no_topology),
            "unique_exact_views": len(set(view_signatures)),
            "new_multihop_views": new_multihop_views,
            "exact_corrupted_support_distance": distance_values[0],
            "exact_no_topology_support_distance": distance_values[1],
            "exact_repeat_last_support_distance": repeat_last_distance,
            "exact_rotated_window_support_distance": rotated_distance,
            "depth_zero_identity": torch.equal(exact[0], identity),
            "all_views_binary_and_full_rank": all_views_binary_and_full_rank,
            "depth_one_matches_runtime_exact_inverse": torch.equal(
                exact[1], structure.inverse_linear_matrices[-1]
            ),
            "corruption_deterministic_and_distinct": (
                torch.equal(
                    corrupted_structure.linear_matrices,
                    repeated_corrupted.linear_matrices,
                )
                and not torch.equal(
                    structure.linear_matrices,
                    corrupted_structure.linear_matrices,
                )
            ),
            "heterogeneous_controls_valid": heterogeneous_controls_valid,
            "cell_relabel_equivariant": torch.equal(restored_orbit, exact),
            "metrics_finite": all(math.isfinite(value) for value in distance_values),
        }
        rows.append(row)
        _emit(
            progress_callback,
            "cipher_orbit_audited",
            cipher=cipher,
            unique_exact_views=row["unique_exact_views"],
            new_multihop_views=row["new_multihop_views"],
        )
    return rows


def _rotate_runtime_window(structure: RuntimeSpnStructure) -> RuntimeSpnStructure:
    return runtime_spn_structure_from_truth_bits(
        structure.cell_membership,
        structure.bit_role,
        torch.roll(structure.sbox_truth_bits, shifts=1, dims=0),
        torch.roll(structure.linear_matrices, shifts=1, dims=0),
    )


def _gf2_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    product = left.to(torch.int16) @ right.to(torch.int16)
    return torch.remainder(product, 2).to(torch.uint8)


def _views_binary_and_full_rank(views: torch.Tensor) -> bool:
    return bool(torch.all((views == 0) | (views == 1))) and all(
        _gf2_rank(view) == view.shape[0] for view in views
    )


def _gf2_rank(matrix: torch.Tensor) -> int:
    rows: list[int] = []
    for row in torch.as_tensor(matrix, dtype=torch.uint8, device="cpu").tolist():
        value = 0
        for column, bit in enumerate(row):
            if bit:
                value |= 1 << column
        rows.append(value)
    basis: dict[int, int] = {}
    for original in rows:
        value = original
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)


def _matrix_sha256(matrix: torch.Tensor) -> str:
    return _tensor_sha256(torch.as_tensor(matrix, dtype=torch.uint8, device="cpu"))


def _tensor_sha256(values: torch.Tensor) -> str:
    tensor = torch.as_tensor(values, dtype=torch.uint8, device="cpu").contiguous()
    shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    payload = bytes(tensor.reshape(-1).tolist())
    return hashlib.sha256(shape + b":" + payload).hexdigest()


def _manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        [dict(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
    "build_structure_panel",
    "exact_inverse_orbit",
    "load_and_validate_config",
    "run_multiscale_orbit_audit",
    "support_jaccard_distance",
    "write_audit_artifacts",
]
