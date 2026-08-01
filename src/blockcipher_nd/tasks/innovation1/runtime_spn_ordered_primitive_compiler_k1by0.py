from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    EXPERT_CONTRACT,
    CompiledSpnProgram,
    compile_ordered_primitive_program,
    permute_program_target_bindings,
    program_exactly_replays,
    replay_ordered_primitive_program,
    rotate_program_stages,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.runtime_spn_structure_program_pretrain_k1bw import (
    load_structures,
    structure_variants,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_runtime_spn_ordered_primitive_compiler_k1by0_20260801"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_ordered_primitive_compiler_k1by0_20260801.json"
)
CONTROL_NAMES = (
    "exact_replay",
    "joint_cell_relabel",
    "wrong_order_when_distinct",
    "wrong_target_binding",
)


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_ordered_primitive_compiler_k1by0"
    ):
        raise ValueError("K1-BY0 experiment identity drifted")
    source = config.get("source", {})
    source_config_path = ROOT / str(source.get("config", ""))
    source_config = _read_json(source_config_path)
    if _file_sha256(source_config_path) != source.get("digests", {}).get("config"):
        raise ValueError("K1-BY0 K1-BX0 config digest drifted")
    if config.get("structures") != source_config.get("structures"):
        raise ValueError("K1-BY0 changed the K1-BX0 structure panel")
    expected_audit = {
        "control_seeds": [11, 23, 37, 53],
        "training_steps": 0,
        "ciphertext_rows": 0,
        "device": "cpu",
        "execution": "local_deterministic_compiler_audit",
    }
    if config.get("audit") != expected_audit:
        raise ValueError("K1-BY0 audit protocol drifted")
    if config.get("controls") != list(CONTROL_NAMES[1:]):
        raise ValueError("K1-BY0 control panel drifted")
    gates = config.get("gates", {})
    expected_gates = {
        "protocol_errors_max": 0,
        "exact_replay_required": True,
        "relabel_semantic_digest_required": True,
        "applicable_wrong_order_rejected": True,
        "wrong_target_binding_rejected": True,
        "expert_contract_equal_across_structures": True,
        "cipher_identity_input": False,
        "remote_scale": "no",
    }
    if gates != expected_gates:
        raise ValueError("K1-BY0 gate contract drifted")
    return config


def load_source_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, bool]]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {
        name: source_root / name
        for name in ("gate.json", "results.jsonl", "validation.json")
    }
    checks = {
        f"k1bx0_{name}_digest_exact": path.is_file()
        and _file_sha256(path) == source["digests"][name]
        for name, path in paths.items()
    }
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    checks["k1bx0_expected_hold_exact"] = (
        gate.get("status") == source["required_status"]
        and gate.get("decision") == source["required_decision"]
        and not gate.get("failed_protocol_checks")
        and validation.get("status") == "pass"
    )
    return gate, checks


def audit_programs(
    config: Mapping[str, Any],
    structures: Mapping[str, RuntimeSpnStructure],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for cipher, structure in structures.items():
        program = compile_ordered_primitive_program(structure)
        replay = replay_ordered_primitive_program(program)
        full_replay_exact = _structures_equal(replay, structure)
        rows.append(
            {
                "cipher_key": cipher,
                "control": "exact_replay",
                "control_seed": None,
                "applicable": True,
                "semantic_digest_equal": True,
                "semantic_digest_changed": False,
                "replay_exact": full_replay_exact,
                "passed": full_replay_exact,
                "source_semantic_sha256": program.semantic_sha256,
                "control_semantic_sha256": program.semantic_sha256,
            }
        )
        for control_seed in config["audit"]["control_seeds"]:
            relabeled, semantic_ids, _variants = structure_variants(
                structure,
                seed=int(control_seed),
            )
            relabeled_program = compile_ordered_primitive_program(
                relabeled,
                semantic_cell_ids=semantic_ids,
            )
            relabel_digest_equal = (
                relabeled_program.semantic_sha256 == program.semantic_sha256
            )
            relabel_replay_exact = program_exactly_replays(
                relabeled_program,
                relabeled,
            )
            rows.append(
                {
                    "cipher_key": cipher,
                    "control": "joint_cell_relabel",
                    "control_seed": int(control_seed),
                    "applicable": True,
                    "semantic_digest_equal": relabel_digest_equal,
                    "semantic_digest_changed": not relabel_digest_equal,
                    "replay_exact": relabel_replay_exact,
                    "passed": relabel_digest_equal and relabel_replay_exact,
                    "source_semantic_sha256": program.semantic_sha256,
                    "control_semantic_sha256": relabeled_program.semantic_sha256,
                }
            )

            wrong_binding = permute_program_target_bindings(
                program,
                seed=int(control_seed),
            )
            binding_digest_changed = (
                wrong_binding.semantic_sha256 != program.semantic_sha256
            )
            binding_replay_exact = program_exactly_replays(
                wrong_binding,
                structure,
            )
            rows.append(
                {
                    "cipher_key": cipher,
                    "control": "wrong_target_binding",
                    "control_seed": int(control_seed),
                    "applicable": True,
                    "semantic_digest_equal": not binding_digest_changed,
                    "semantic_digest_changed": binding_digest_changed,
                    "replay_exact": binding_replay_exact,
                    "passed": binding_digest_changed and not binding_replay_exact,
                    "source_semantic_sha256": program.semantic_sha256,
                    "control_semantic_sha256": wrong_binding.semantic_sha256,
                }
            )

        wrong_order = rotate_program_stages(program)
        order_applicable = len(set(program.stage_content_sha256s)) > 1
        order_digest_changed = wrong_order.semantic_sha256 != program.semantic_sha256
        rotated_structure = structure.rotate_transitions()
        order_replay_exact = program_exactly_replays(
            wrong_order,
            rotated_structure,
        )
        order_passed = (
            order_digest_changed and order_replay_exact
            if order_applicable
            else not order_digest_changed and order_replay_exact
        )
        rows.append(
            {
                "cipher_key": cipher,
                "control": "wrong_order_when_distinct",
                "control_seed": None,
                "applicable": order_applicable,
                "semantic_digest_equal": not order_digest_changed,
                "semantic_digest_changed": order_digest_changed,
                "replay_exact": order_replay_exact,
                "passed": order_passed,
                "source_semantic_sha256": program.semantic_sha256,
                "control_semantic_sha256": wrong_order.semantic_sha256,
            }
        )
        manifests[cipher] = _program_manifest(program, structure)
    return rows, manifests


def adjudicate(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    programs: Mapping[str, Any],
    *,
    protocol_checks: Mapping[str, bool],
) -> dict[str, Any]:
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    by_control = {
        control: [row for row in rows if row["control"] == control]
        for control in CONTROL_NAMES
    }
    applicable_order = [
        row for row in by_control["wrong_order_when_distinct"] if row["applicable"]
    ]
    research_checks = {
        "all_exact_replays": bool(by_control["exact_replay"])
        and all(bool(row["passed"]) for row in by_control["exact_replay"]),
        "all_relabels_semantically_equal": bool(
            by_control["joint_cell_relabel"]
        )
        and all(bool(row["passed"]) for row in by_control["joint_cell_relabel"]),
        "applicable_wrong_orders_rejected": bool(applicable_order)
        and all(bool(row["passed"]) for row in applicable_order),
        "all_wrong_bindings_rejected": bool(
            by_control["wrong_target_binding"]
        )
        and all(bool(row["passed"]) for row in by_control["wrong_target_binding"]),
        "expert_contract_equal_across_structures": len(
            {
                json.dumps(value["expert_contract"], sort_keys=True)
                for value in programs.values()
            }
        )
        == 1,
    }
    failed_research = sorted(
        name for name, passed in research_checks.items() if not passed
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by0_compiler_protocol_invalid"
    elif failed_research:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by0_ordered_compiler_not_ready"
    else:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by0_ordered_compiler_ready"
    expert_totals = {
        expert: sum(
            int(program["expert_usage"][expert]) for program in programs.values()
        )
        for expert in EXPERT_CONTRACT
    }
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "protocol_checks": dict(protocol_checks),
        "research_checks": research_checks,
        "control_summary": {
            control: {
                "rows": len(selected),
                "applicable_rows": sum(bool(row["applicable"]) for row in selected),
                "passed_rows": sum(bool(row["passed"]) for row in selected),
            }
            for control, selected in by_control.items()
        },
        "applicable_wrong_order_ciphers": sorted(
            str(row["cipher_key"]) for row in applicable_order
        ),
        "expert_contract": EXPERT_CONTRACT,
        "expert_usage_totals": expert_totals,
        "training_steps": 0,
        "ciphertext_rows": 0,
        "remote_scale": "no",
        "claim_scope": (
            "Local deterministic ordered-primitive compiler audit over seven "
            "runtime descriptors; not learned-expert training, differential AUC, "
            "unseen-cipher neural transfer, formal scale, attack or SOTA evidence."
        ),
        "next_action": (
            "If pass, preregister K1-BY1 small local Runtime-E4 readiness with "
            "ordered compiler routing, wrong-order, wrong-binding and no-structure "
            "controls. If hold, repair only the uncovered primitive or replay defect."
        ),
    }


def run_audit(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("K1-BY0 output root already contains artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    progress = output_root / "progress.jsonl"
    _append_jsonl(progress, {"event": "run_start", "run_id": RUN_ID, "time": time.time()})
    _source_gate, source_checks = load_source_authority(
        config,
        project_root=project_root,
    )
    structures, descriptor_manifest = load_structures(
        config,
        project_root=project_root,
    )
    rows, programs = audit_programs(config, structures)
    expected_rows = len(structures) * (
        2 + 2 * len(config["audit"]["control_seeds"])
    )
    protocol_checks = {
        **source_checks,
        "seven_descriptors_loaded": len(structures) == 7,
        "all_two_transition_windows": all(
            structure.rounds == 2 for structure in structures.values()
        ),
        "expected_result_rows": len(rows) == expected_rows,
        "zero_training_steps": config["audit"]["training_steps"] == 0,
        "zero_ciphertext_rows": config["audit"]["ciphertext_rows"] == 0,
        "cipher_identity_absent": all(
            contract["uses_cipher_identity"] is False
            for contract in EXPERT_CONTRACT.values()
        ),
        "all_rows_finite_boolean": all(
            isinstance(row["passed"], bool)
            and isinstance(row["replay_exact"], bool)
            for row in rows
        ),
    }
    gate = adjudicate(
        config,
        rows,
        programs,
        protocol_checks=protocol_checks,
    )
    validation = {
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "result_rows": len(rows),
        "expected_rows": expected_rows,
        "programs": len(programs),
        "errors": gate["failed_protocol_checks"],
        "training_steps": 0,
        "ciphertext_rows": 0,
    }
    preflight = {
        "run_id": RUN_ID,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _file_sha256(CONFIG_PATH),
        "source_checks": source_checks,
        "descriptor_manifest": descriptor_manifest,
        "device": config["audit"]["device"],
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "control_summary": gate["control_summary"],
        "applicable_wrong_order_ciphers": gate[
            "applicable_wrong_order_ciphers"
        ],
        "expert_usage_totals": gate["expert_usage_totals"],
        "claim_scope": gate["claim_scope"],
        "next_action": gate["next_action"],
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_json(output_root / "programs.json", programs)
    _write_jsonl(output_root / "results.jsonl", rows)
    _write_csv(output_root / "results.csv", rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_jsonl(
        progress,
        {
            "event": "run_done",
            "status": gate["status"],
            "decision": gate["decision"],
            "time": time.time(),
        },
    )
    return {
        "preflight": preflight,
        "programs": programs,
        "results": rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _program_manifest(
    program: CompiledSpnProgram,
    structure: RuntimeSpnStructure,
) -> dict[str, Any]:
    return {
        "block_bits": program.block_bits,
        "cells": program.cells,
        "rounds": program.rounds,
        "semantic_sha256": program.semantic_sha256,
        "source_window_sha256": structure.window_sha256(),
        "stage_content_sha256s": program.stage_content_sha256s,
        "expert_contract": EXPERT_CONTRACT,
        "expert_usage": program.expert_usage,
        "stages": [
            {
                "ordinal": ordinal,
                "source_stage": stage.source_stage,
                "content_sha256": stage.content_sha256,
                "sboxes": [
                    {
                        "cell": item.cell,
                        "expert": item.expert,
                        "truth_sha256": hashlib.sha256(
                            bytes(item.truth_bits)
                        ).hexdigest(),
                    }
                    for item in stage.sboxes
                ],
                "linear_cells": [
                    {
                        "target_cell": item.target_cell,
                        "expert": item.expert,
                        "edges": item.edges,
                    }
                    for item in stage.linear_cells
                ],
            }
            for ordinal, stage in enumerate(program.stages)
        ],
    }


def _structures_equal(
    left: RuntimeSpnStructure,
    right: RuntimeSpnStructure,
) -> bool:
    return all(
        torch.equal(first, second)
        for first, second in (
            (left.cell_membership, right.cell_membership),
            (left.bit_role, right.bit_role),
            (left.sbox_truth_bits, right.sbox_truth_bits),
            (left.linear_matrices, right.linear_matrices),
            (left.inverse_linear_matrices, right.inverse_linear_matrices),
        )
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("K1-BY0 requires result rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "CONFIG_PATH",
    "CONTROL_NAMES",
    "ROOT",
    "RUN_ID",
    "adjudicate",
    "audit_programs",
    "load_and_validate_config",
    "load_source_authority",
    "run_audit",
]
