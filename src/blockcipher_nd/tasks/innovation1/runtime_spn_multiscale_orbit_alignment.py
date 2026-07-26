from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from blockcipher_nd.models.structure.spn.runtime_structure import (
    LoadedRuntimeSpnDescriptor,
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_multiscale_orbit import (
    run_multiscale_orbit_panel_audit,
    write_audit_artifacts,
)


RUN_ID = "i1_runtime_spn_multiscale_orbit_protocol_alignment_c4p_20260726"
_EXPECTED_STRUCTURES = {
    "present": {
        "descriptor_path": "configs/runtime/spn/present64.json",
        "rounds": 2,
        "round_start": 0,
        "protocol_rounds": 7,
    },
    "gift": {
        "descriptor_path": "configs/runtime/spn/gift64.json",
        "rounds": 2,
        "round_start": 0,
        "protocol_rounds": 6,
    },
    "rectangle": {
        "descriptor_path": "configs/runtime/spn/rectangle64.json",
        "rounds": 2,
        "round_start": 0,
        "protocol_rounds": 6,
    },
    "skinny": {
        "descriptor_path": "configs/runtime/spn/skinny64.json",
        "rounds": 2,
        "round_start": 0,
        "protocol_rounds": 7,
    },
    "uknit": {
        "descriptor_path": "configs/runtime/spn/uknit64.json",
        "rounds": 2,
        "round_start": 3,
        "protocol_rounds": 5,
    },
    "dialga": {
        "descriptor_path": "configs/runtime/spn/dialga128.json",
        "rounds": 2,
        "round_start": 2,
        "protocol_rounds": 4,
    },
}
_EXPECTED_ORBIT = {
    "depths": [0, 1, 2, 4, 8],
    "traversal": "reverse_loaded_runtime_window_cyclic_v1",
    "semantics": "periodic_topology_operator_power_not_literal_round_state",
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


def load_and_validate_alignment_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"C4-P config unreadable: {path}: {exc}") from exc
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
        raise ValueError("C4-P config does not match the frozen alignment contract")
    return payload


def build_aligned_structure_panel(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[str, RuntimeSpnStructure], dict[str, dict[str, Any]]]:
    if config.get("structures") != _EXPECTED_STRUCTURES:
        raise ValueError("C4-P structure panel does not match the frozen contract")
    panel: dict[str, RuntimeSpnStructure] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for cipher, item in config["structures"].items():
        loaded = load_runtime_spn_descriptor(
            project_root / item["descriptor_path"],
            rounds=int(item["rounds"]),
            round_start=int(item["round_start"]),
        )
        panel[cipher] = loaded.structure
        metadata[cipher] = _descriptor_metadata(loaded, item, project_root)
    return panel, metadata


def run_multiscale_orbit_alignment_audit(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    panel, metadata = build_aligned_structure_panel(
        config,
        project_root=project_root,
    )
    homogeneous = {"present", "gift", "rectangle", "skinny"}
    heterogeneous = {"uknit", "dialga"}
    checks = {
        "descriptor_windows_match_joint_protocol": all(
            structure.rounds == 2
            and metadata[cipher]["loaded_rounds"] == 2
            and metadata[cipher]["round_start"]
            == int(config["structures"][cipher]["round_start"])
            for cipher, structure in panel.items()
        ),
        "periodic_operator_semantics_frozen": config["orbit"]["semantics"]
        == "periodic_topology_operator_power_not_literal_round_state",
        "homogeneous_windows_repeat_exactly": all(
            torch.equal(
                panel[cipher].linear_matrices[0],
                panel[cipher].linear_matrices[1],
            )
            for cipher in homogeneous
        ),
        "heterogeneous_windows_have_two_distinct_transitions": all(
            not torch.equal(
                panel[cipher].linear_matrices[0],
                panel[cipher].linear_matrices[1],
            )
            for cipher in heterogeneous
        ),
    }
    return run_multiscale_orbit_panel_audit(
        panel=panel,
        config=config,
        run_id=RUN_ID,
        decisions={
            "invalid": (
                "innovation1_runtime_spn_multiscale_orbit_protocol_alignment_invalid"
            ),
            "pass": (
                "innovation1_runtime_spn_multiscale_orbit_protocol_alignment_supported"
            ),
            "hold": (
                "innovation1_runtime_spn_multiscale_orbit_protocol_alignment_not_supported"
            ),
        },
        next_actions={
            "invalid": "repair only failed C4-P protocol invariants and rerun",
            "pass": (
                "wait for C3 completion, then preregister the parameter-shape-matched "
                "C5 local neural diagnostic using these exact periodic windows"
            ),
            "hold": (
                "close the fixed two-transition periodic-orbit candidate; do not "
                "tune depths, windows, thresholds, samples or model size"
            ),
        },
        claim_scope=(
            "zero-training protocol-aligned periodic topology-operator feasibility "
            "only; not literal partial decryption, differential signal, neural "
            "learnability, complete round-window use, transfer, attack, SOTA or "
            "breakthrough evidence"
        ),
        blocked_actions=[
            "implement or train C5 before C3 completion and a separate plan",
            "describe O4 or O8 as literal earlier-round states",
            "change C4-P depths, windows, controls, metrics or thresholds after reveal",
            "modify or duplicate the running C3 protocol",
        ],
        progress_callback=progress_callback,
        row_metadata=metadata,
        extra_protocol_checks=checks,
    )


def _descriptor_metadata(
    loaded: LoadedRuntimeSpnDescriptor,
    item: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    return {
        "descriptor_path": str(loaded.path.relative_to(project_root.resolve())),
        "descriptor_sha256": loaded.sha256,
        "round_start": loaded.round_start,
        "loaded_rounds": loaded.structure.rounds,
        "available_descriptor_rounds": loaded.available_rounds,
        "protocol_rounds": int(item["protocol_rounds"]),
        "orbit_semantics": _EXPECTED_ORBIT["semantics"],
    }


__all__ = [
    "RUN_ID",
    "build_aligned_structure_panel",
    "load_and_validate_alignment_config",
    "run_multiscale_orbit_alignment_audit",
    "write_audit_artifacts",
]
