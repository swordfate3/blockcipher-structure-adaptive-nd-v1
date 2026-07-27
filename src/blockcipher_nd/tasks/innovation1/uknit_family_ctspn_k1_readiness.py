from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from typing import Any, Mapping, Sequence

import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.canonical_components import (
    compile_canonical_linear_schedule,
)
from blockcipher_nd.registry.model_factory import build_model


RUN_ID = "i1_uknit_family_ctspn_linear_schedule_k1_readiness_20260727"
K0_RUN_ID = "i1_uknit_family_canonical_component_factorization_k0_20260727"
K0_DECISION = "innovation1_uknit_family_canonical_component_factorization_supported"
ANCHOR_MODEL = "runtime_spn_e4_equivariant_true"
CANDIDATE_MODEL = "runtime_spn_ct_k1_canonical_true"
CORRUPTED_MODEL = "runtime_spn_ct_k1_canonical_corrupted"
INDEPENDENT_MODEL = "runtime_spn_ct_k1_canonical_independent"
ANCHOR_PARAMETERS = 442466
MAX_PARAMETER_RELATIVE_DELTA = 0.01
RELABEL_LOGIT_TOLERANCE = 1e-6

CIPHER_PROTOCOLS = {
    "uknit64": {
        "rounds": 5,
        "block_bits": 64,
        "runtime_round_start": 3,
        "validation_key": int("1" * 32, 16),
    },
    "dialga128": {
        "rounds": 4,
        "block_bits": 128,
        "runtime_round_start": 2,
        "validation_key": int("1" * 64, 16),
    },
}


def build_ctspn_k1_readiness(
    *,
    run_id: str,
    tasks: Sequence[Mapping[str, Any]],
    k0_gate: Mapping[str, Any],
    k0_validation: Mapping[str, Any],
    present_gate: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    models: dict[tuple[str, int, str], torch.nn.Module] = {}
    build_errors: list[str] = []
    for task in tasks:
        try:
            cipher = str(task["cipher_key"])
            seed = int(task["seed"])
            model_key = str(task["model_key"])
            model = _build_task_model(task)
            models[(cipher, seed, model_key)] = model
            manifests.append(_task_manifest(task, model))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            build_errors.append(f"{task.get('network', '<unknown>')}: {exc}")

    grouped = _group_tasks(tasks)
    protocol_checks = _protocol_checks(
        grouped=grouped,
        manifests=manifests,
        build_errors=build_errors,
        k0_gate=k0_gate,
        k0_validation=k0_validation,
    )
    structural_evidence: dict[str, Any] = {}
    if protocol_checks["eight_row_frozen_panel"] and not build_errors:
        try:
            structural_evidence = _structural_evidence(grouped, models)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            build_errors.append(f"structural_evidence: {exc}")

    evidence_checks = {
        "candidate_parameter_budget": _candidate_parameter_budget(manifests),
        "same_candidate_geometry_across_widths": _same_candidate_geometry(models),
        "canonical_unit_inverse_exact": bool(
            structural_evidence.get("canonical_unit_inverse_exact")
        ),
        "canonical_edge_relation_exact": bool(
            structural_evidence.get("canonical_edge_relation_exact")
        ),
        "cell_relabel_logit_invariant": bool(
            structural_evidence.get("cell_relabel_logit_invariant")
        ),
        "control_fingerprints_deterministic_and_distinct": bool(
            structural_evidence.get(
                "control_fingerprints_deterministic_and_distinct"
            )
        ),
        "strict_control_state_dict_load": bool(
            structural_evidence.get("strict_control_state_dict_load")
        ),
        "control_factors_survive_checkpoint_load": bool(
            structural_evidence.get("control_factors_survive_checkpoint_load")
        ),
        "same_checkpoint_controls_change_logits": bool(
            structural_evidence.get("same_checkpoint_controls_change_logits")
        ),
        "cross_width_strict_state_dict_load": bool(
            structural_evidence.get("cross_width_strict_state_dict_load")
        ),
        "no_cipher_identity_tensor": bool(
            structural_evidence.get("no_cipher_identity_tensor")
        ),
        "anchor_sbox_path_zero_contribution": bool(
            structural_evidence.get("anchor_sbox_path_zero_contribution")
        ),
        "forward_shapes_and_finite": bool(
            structural_evidence.get("forward_shapes_and_finite")
        ),
    }
    all_checks = {**protocol_checks, **evidence_checks}
    implementation_ready = not build_errors and all(all_checks.values())
    present_adjudicated = _present_adjudicated(present_gate)
    optimizer_step_authorized = implementation_ready and present_adjudicated
    if not implementation_ready:
        decision = "innovation1_uknit_family_ctspn_k1_readiness_failed"
        next_action = (
            "Repair only the failed CT-SPN implementation or protocol invariant and "
            "rerun this zero-training readiness gate unchanged."
        )
    elif not present_adjudicated:
        decision = (
            "innovation1_uknit_family_ctspn_k1_readiness_passed_waiting_present"
        )
        next_action = (
            "Keep K1 at zero optimizer steps until the existing PRESENT formal seed1 "
            "monitor retrieves and adjudicates a local result; then rerun only the "
            "launch interlock before the frozen eight-row local diagnostic."
        )
    else:
        decision = "innovation1_uknit_family_ctspn_k1_training_authorized"
        next_action = (
            "Run the frozen eight-row local K1 diagnostic, evaluate all controls from "
            "the selected candidate checkpoint without retraining, and apply the "
            "per-seed per-cipher advance gate."
        )
    gate = {
        "run_id": run_id,
        "task": "innovation1_uknit_family_ctspn_k1_zero_training_readiness",
        "status": "pass" if implementation_ready else "fail",
        "decision": decision,
        "implementation_ready": implementation_ready,
        "present_formal_seed1_adjudicated": present_adjudicated,
        "optimizer_step_authorized": optimizer_step_authorized,
        "training_rows": 0,
        "optimizer_steps": 0,
        "expected_plan_rows": 8,
        "manifest_rows": len(manifests),
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "build_errors": build_errors,
        "structural_evidence": structural_evidence,
        "claim_scope": (
            "zero-training CT-SPN implementation readiness only; no neural efficacy, "
            "attack, remote-scale or arbitrary-SPN claim"
        ),
        "next_action": next_action,
    }
    return manifests, gate


def _build_task_model(task: Mapping[str, Any]) -> torch.nn.Module:
    cipher = str(task["cipher_key"])
    protocol = CIPHER_PROTOCOLS[cipher]
    block_bits = int(protocol["block_bits"])
    pairs = int(task["pairs_per_sample"])
    seed_offset = 0 if cipher == "uknit64" else 1000
    model_offset = 0 if task["model_key"] == ANCHOR_MODEL else 100
    torch.manual_seed(20260727 + seed_offset + model_offset + int(task["seed"]))
    model = build_model(
        str(task["model_key"]),
        input_bits=pairs * 2 * block_bits,
        hidden_bits=64,
        pair_bits=2 * block_bits,
        structure="SPN",
        model_options=dict(task["model_options"]),
    )
    return model


def _task_manifest(
    task: Mapping[str, Any], model: torch.nn.Module
) -> dict[str, Any]:
    metadata = model_metadata(model)
    return {
        "cipher_key": task["cipher_key"],
        "seed": int(task["seed"]),
        "network": task.get("architecture", str(task["model_key"])),
        "model": task["model_key"],
        "rounds": int(task["rounds"]),
        "samples_per_class": int(task["samples_per_class"]),
        "validation_samples_per_class": max(
            8, int(task["samples_per_class"]) // 2
        ),
        "pairs_per_sample": int(task["pairs_per_sample"]),
        "input_difference": int(task["input_difference"]),
        "train_key": int(task["train_key"]),
        "validation_key": int(task["validation_key"]),
        "negative_mode": task["negative_mode"],
        "loss": task["loss"],
        "optimizer": task["optimizer"],
        "learning_rate": float(task["learning_rate"]),
        "weight_decay": float(task["weight_decay"]),
        "target_epochs": int(task["target_epochs"]),
        "model_options": deepcopy(task["model_options"]),
        **metadata,
        "state_dict_geometry": _state_dict_geometry(model),
    }


def _group_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, dict[str, Mapping[str, Any]]]]:
    grouped: dict[str, dict[int, dict[str, Mapping[str, Any]]]] = {}
    for task in tasks:
        cipher = str(task.get("cipher_key"))
        seed = int(task.get("seed", -1))
        model = str(task.get("model_key"))
        grouped.setdefault(cipher, {}).setdefault(seed, {})[model] = task
    return grouped


def _protocol_checks(
    *,
    grouped: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    manifests: Sequence[Mapping[str, Any]],
    build_errors: Sequence[str],
    k0_gate: Mapping[str, Any],
    k0_validation: Mapping[str, Any],
) -> dict[str, bool]:
    expected_models = {ANCHOR_MODEL, CANDIDATE_MODEL}
    panel = (
        set(grouped) == set(CIPHER_PROTOCOLS)
        and all(set(grouped[cipher]) == {0, 1} for cipher in CIPHER_PROTOCOLS)
        and all(
            set(grouped[cipher][seed]) == expected_models
            for cipher in CIPHER_PROTOCOLS
            for seed in (0, 1)
        )
    )
    flat_tasks = [
        task
        for by_seed in grouped.values()
        for by_model in by_seed.values()
        for task in by_model.values()
    ]
    return {
        "k0_gate_and_validation_pass": (
            k0_gate.get("run_id") == K0_RUN_ID
            and k0_gate.get("status") == "pass"
            and k0_gate.get("decision") == K0_DECISION
            and k0_validation.get("run_id") == K0_RUN_ID
            and k0_validation.get("status") == "pass"
        ),
        "eight_row_frozen_panel": panel and len(flat_tasks) == 8,
        "all_models_build": len(manifests) == 8 and not build_errors,
        "frozen_data_protocol": panel
        and all(
            int(task["rounds"]) == int(CIPHER_PROTOCOLS[str(task["cipher_key"])]["rounds"])
            and int(task["samples_per_class"]) == 2048
            and task.get("validation_samples_total") is None
            and int(task["pairs_per_sample"]) == 4
            and int(task["input_difference"]) == 0x40
            and int(task["train_key"]) == 0
            and int(task["validation_key"])
            == int(CIPHER_PROTOCOLS[str(task["cipher_key"])]["validation_key"])
            and task["negative_mode"] == "encrypted_random_plaintexts"
            and task["sample_structure"] == "independent_pairs"
            for task in flat_tasks
        ),
        "frozen_training_protocol": panel
        and all(
            task["loss"] == "mse"
            and task["optimizer"] == "adam"
            and float(task["learning_rate"]) == 1e-4
            and float(task["weight_decay"]) == 1e-5
            and int(task["target_epochs"]) == 10
            and task["checkpoint_metric"] == "val_auc"
            and bool(task["restore_best_checkpoint"])
            for task in flat_tasks
        ),
        "runtime_windows_exact": panel
        and all(
            int(task["model_options"]["runtime_round_start"])
            == int(CIPHER_PROTOCOLS[str(task["cipher_key"])]["runtime_round_start"])
            and int(task["model_options"]["runtime_rounds"]) == 2
            for task in flat_tasks
        ),
        "anchor_sbox_disabled_in_plan": panel
        and all(
            float(grouped[cipher][seed][ANCHOR_MODEL]["model_options"].get(
                "sbox_context_scale", -1.0
            ))
            == 0.0
            for cipher in CIPHER_PROTOCOLS
            for seed in (0, 1)
        ),
    }


def _structural_evidence(
    grouped: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    models: Mapping[tuple[str, int, str], torch.nn.Module],
) -> dict[str, Any]:
    per_cipher: dict[str, Any] = {}
    for cipher, protocol in CIPHER_PROTOCOLS.items():
        candidate = models[(cipher, 0, CANDIDATE_MODEL)]
        anchor = models[(cipher, 0, ANCHOR_MODEL)]
        candidate.eval()
        anchor.eval()
        unit_inverse, edge_relation = _unit_checks(candidate)
        relabel_error = _cell_relabel_error(candidate)
        controls = _control_evidence(
            grouped[cipher][0][CANDIDATE_MODEL], candidate
        )
        anchor_sbox_zero = _anchor_sbox_zero_contribution(anchor, int(protocol["block_bits"]))
        probe = _binary_probe(
            batch=2,
            pairs=4,
            block_bits=int(protocol["block_bits"]),
            seed=20260727 + int(protocol["block_bits"]),
        ).flip(-1).reshape(2, -1)
        with torch.no_grad():
            output = candidate(probe)
        per_cipher[cipher] = {
            "canonical_unit_inverse_exact": unit_inverse,
            "canonical_edge_relation_exact": edge_relation,
            "cell_relabel_max_logit_error": relabel_error,
            "cell_relabel_logit_invariant": relabel_error <= RELABEL_LOGIT_TOLERANCE,
            "anchor_sbox_path_zero_contribution": anchor_sbox_zero,
            "forward_output_shape": list(output.shape),
            "forward_output_finite": bool(torch.isfinite(output).all()),
            "controls": controls,
        }

    uknit = models[("uknit64", 0, CANDIDATE_MODEL)]
    dialga = models[("dialga128", 0, CANDIDATE_MODEL)]
    dialga_factor_before = dialga.canonical_factor_manifest_sha256
    cross_width_load = dialga.load_state_dict(uknit.state_dict(), strict=True)
    cross_width_ok = (
        not cross_width_load.missing_keys
        and not cross_width_load.unexpected_keys
        and dialga.canonical_factor_manifest_sha256 == dialga_factor_before
    )
    return {
        "per_cipher": per_cipher,
        "canonical_unit_inverse_exact": all(
            row["canonical_unit_inverse_exact"] for row in per_cipher.values()
        ),
        "canonical_edge_relation_exact": all(
            row["canonical_edge_relation_exact"] for row in per_cipher.values()
        ),
        "cell_relabel_logit_invariant": all(
            row["cell_relabel_logit_invariant"] for row in per_cipher.values()
        ),
        "control_fingerprints_deterministic_and_distinct": all(
            row["controls"]["fingerprints_deterministic_and_distinct"]
            for row in per_cipher.values()
        ),
        "strict_control_state_dict_load": all(
            row["controls"]["strict_state_dict_load"] for row in per_cipher.values()
        ),
        "control_factors_survive_checkpoint_load": all(
            row["controls"]["factors_survive_checkpoint_load"]
            for row in per_cipher.values()
        ),
        "same_checkpoint_controls_change_logits": all(
            row["controls"]["control_logits_noncollapsed"]
            for row in per_cipher.values()
        ),
        "cross_width_strict_state_dict_load": cross_width_ok,
        "no_cipher_identity_tensor": _no_cipher_identity_tensor(uknit)
        and _no_cipher_identity_tensor(dialga),
        "anchor_sbox_path_zero_contribution": all(
            row["anchor_sbox_path_zero_contribution"] for row in per_cipher.values()
        ),
        "forward_shapes_and_finite": all(
            row["forward_output_shape"] == [2, 1]
            and row["forward_output_finite"]
            for row in per_cipher.values()
        ),
    }


def _unit_checks(model: torch.nn.Module) -> tuple[bool, bool]:
    structure = model.runtime_structure
    schedule = model.canonical_schedule
    units = torch.eye(structure.block_bits, dtype=torch.float32)
    inverse_exact = True
    edge_exact = True
    targets = schedule.canonical_edge_index[0]
    sources = schedule.canonical_edge_index[1]
    for round_index in range(structure.rounds):
        canonical_output, canonical_input, native_input = schedule.transition(
            units, round_index
        )
        inverse_exact = inverse_exact and torch.equal(
            native_input, structure.exact_inverse(units, round_index)
        )
        reconstructed = torch.zeros_like(canonical_output)
        for target in range(structure.block_bits):
            source_indices = sources[targets == target]
            reconstructed[:, target] = torch.remainder(
                canonical_input[:, source_indices].sum(dim=-1), 2.0
            )
        edge_exact = edge_exact and torch.equal(reconstructed, canonical_output)
    return inverse_exact, edge_exact


def _cell_relabel_error(model: torch.nn.Module) -> float:
    structure = model.runtime_structure
    permutation = tuple(reversed(range(structure.cells)))
    relabeled, bit_permutation = structure.relabel_cells(permutation)
    pairs = _binary_probe(
        batch=3,
        pairs=4,
        block_bits=structure.block_bits,
        seed=20260727 + structure.block_bits,
    )
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    relabeled_schedule = compile_canonical_linear_schedule(relabeled)
    with torch.no_grad():
        expected = model.backbone(
            pairs, structure, model.canonical_schedule, relation_mode="true"
        )
        observed = model.backbone(
            relabeled_pairs, relabeled, relabeled_schedule, relation_mode="true"
        )
    return float(torch.max(torch.abs(expected - observed)))


def _control_evidence(
    task: Mapping[str, Any], correct_model: torch.nn.Module
) -> dict[str, Any]:
    options = dict(task["model_options"])
    variants = {
        "correct": (CANDIDATE_MODEL, {}),
        "repeat_last": (
            CANDIDATE_MODEL,
            {"runtime_structure_window_control": "repeat_last"},
        ),
        "rotated": (
            CANDIDATE_MODEL,
            {"canonical_schedule_control": "rotated"},
        ),
        "corrupted": (
            CORRUPTED_MODEL,
            {"topology_corruption_seed": 20260727},
        ),
        "independent": (INDEPENDENT_MODEL, {}),
    }
    block_bits = int(CIPHER_PROTOCOLS[str(task["cipher_key"])]["block_bits"])
    controls: dict[str, torch.nn.Module] = {}
    fingerprints: dict[str, str] = {}
    repeated_fingerprints: dict[str, str] = {}
    checkpoint = correct_model.state_dict()
    strict_load = True
    factors_survive = True
    output_hashes: dict[str, str] = {}
    probe = _binary_probe(
        batch=3, pairs=4, block_bits=block_bits, seed=20260727 + block_bits
    ).flip(-1).reshape(3, -1)
    for role, (model_key, overrides) in variants.items():
        role_options = {**options, **overrides}
        control = build_model(
            model_key,
            input_bits=8 * block_bits,
            hidden_bits=64,
            pair_bits=2 * block_bits,
            structure="SPN",
            model_options=role_options,
        )
        controls[role] = control
        before = _control_fingerprint(control)
        loaded = control.load_state_dict(checkpoint, strict=True)
        after = _control_fingerprint(control)
        strict_load = strict_load and not loaded.missing_keys and not loaded.unexpected_keys
        factors_survive = factors_survive and before == after
        fingerprints[role] = after
        rebuilt = build_model(
            model_key,
            input_bits=8 * block_bits,
            hidden_bits=64,
            pair_bits=2 * block_bits,
            structure="SPN",
            model_options=role_options,
        )
        repeated_fingerprints[role] = _control_fingerprint(rebuilt)
        control.eval()
        with torch.no_grad():
            output_hashes[role] = _tensor_sha256(control(probe))
    return {
        "fingerprints": fingerprints,
        "repeated_fingerprints": repeated_fingerprints,
        "output_sha256s": output_hashes,
        "fingerprints_deterministic_and_distinct": (
            fingerprints == repeated_fingerprints
            and len(set(fingerprints.values())) == len(fingerprints)
        ),
        "strict_state_dict_load": strict_load,
        "factors_survive_checkpoint_load": factors_survive,
        "control_logits_noncollapsed": len(set(output_hashes.values())) >= 4,
    }


def _anchor_sbox_zero_contribution(model: torch.nn.Module, block_bits: int) -> bool:
    model.zero_grad(set_to_none=True)
    model.train()
    probe = _binary_probe(
        batch=2, pairs=4, block_bits=block_bits, seed=20260727 + block_bits
    ).flip(-1).reshape(2, -1)
    model(probe).sum().backward()
    sbox_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "sbox_encoder" in name
    ]
    zero = bool(sbox_parameters) and all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in sbox_parameters
    )
    model.zero_grad(set_to_none=True)
    model.eval()
    return zero


def _candidate_parameter_budget(manifests: Sequence[Mapping[str, Any]]) -> bool:
    counts = {
        int(row["trainable_parameter_count"])
        for row in manifests
        if row.get("model") == CANDIDATE_MODEL
    }
    return len(counts) == 1 and all(
        abs(count - ANCHOR_PARAMETERS) / ANCHOR_PARAMETERS
        <= MAX_PARAMETER_RELATIVE_DELTA
        for count in counts
    )


def _same_candidate_geometry(
    models: Mapping[tuple[str, int, str], torch.nn.Module]
) -> bool:
    geometries = {
        _geometry_sha256(models[(cipher, seed, CANDIDATE_MODEL)])
        for cipher in CIPHER_PROTOCOLS
        for seed in (0, 1)
        if (cipher, seed, CANDIDATE_MODEL) in models
    }
    return len(geometries) == 1 and len(models) == 8


def _no_cipher_identity_tensor(model: torch.nn.Module) -> bool:
    forbidden = {"cipher", "descriptor", "fingerprint", "round_count"}
    tensor_names = [
        name for name, _ in (*model.named_parameters(), *model.named_buffers())
    ]
    forward_parameters = tuple(inspect.signature(model.backbone.forward).parameters)
    return not any(
        forbidden.intersection(name.lower().replace(".", "_").split("_"))
        for name in tensor_names
    ) and not forbidden.intersection(forward_parameters)


def _present_adjudicated(gate: Mapping[str, Any] | None) -> bool:
    if gate is None:
        return False
    run_id = str(gate.get("run_id", ""))
    decision = str(gate.get("decision", ""))
    return (
        "rtg3b_present80_one_to_one_formal_1000000_seed1" in run_id
        and gate.get("status") in {"pass", "fail"}
        and "launch_authorized" not in decision
    )


def _control_fingerprint(model: torch.nn.Module) -> str:
    payload = {
        "relation_mode": model.relation_mode,
        "runtime_structure_mode": model.runtime_structure_mode,
        "runtime_structure_window_control": model.runtime_structure_window_control,
        "runtime_structure_window_sha256": model.runtime_structure_window_sha256,
        "canonical_schedule_control": model.canonical_schedule_control,
        "canonical_factor_manifest_sha256": model.canonical_factor_manifest_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _state_dict_geometry(model: torch.nn.Module) -> list[list[Any]]:
    return [[name, list(tensor.shape)] for name, tensor in model.state_dict().items()]


def _geometry_sha256(model: torch.nn.Module) -> str:
    return hashlib.sha256(
        json.dumps(_state_dict_geometry(model), separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_sha256(values: torch.Tensor) -> str:
    return hashlib.sha256(
        values.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _binary_probe(
    *, batch: int, pairs: int, block_bits: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        2,
        (batch, pairs, 2, block_bits),
        generator=generator,
        dtype=torch.float32,
    )


__all__ = [
    "ANCHOR_MODEL",
    "CANDIDATE_MODEL",
    "CORRUPTED_MODEL",
    "INDEPENDENT_MODEL",
    "K0_DECISION",
    "K0_RUN_ID",
    "RUN_ID",
    "build_ctspn_k1_readiness",
]
