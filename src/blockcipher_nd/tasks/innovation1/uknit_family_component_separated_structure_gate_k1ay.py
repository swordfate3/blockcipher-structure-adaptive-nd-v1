from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.structure_conditioned_gate import (
    LINEAR_SUMMARY_DIM,
    SBOX_SUMMARY_DIM,
    STRUCTURE_SUMMARY_DIM,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
    MISMATCH_CONDITIONS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_channel_orientation_k1ax import (
    load_and_validate_config as load_k1ax_config,
    load_authority as load_k1ax_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    build_candidate as build_k1aw_candidate,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_component_separated_structure_gate_k1ay_readiness_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_component_separated_structure_gate_"
    "k1ay_readiness_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "2967e49aeec3895d59820927b56d5fa6f2c55249ae60daa42da72395b89bdc97"
)
MODEL_KEY = "runtime_spn_ct_k1ay_component_separated_structure_gate_true"
EXPECTED_REPLICAS = (0, 1)
EXPECTED_PARAMETER_COUNT = 219_764
EXPECTED_STATE_ENTRIES = 55
EXPECTED_RESULT_ROWS = 12
EXPECTED_CONTROL_ROWS = 36


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AY config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AY identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_component_separated_structure_gate_k1ay_readiness"
    ):
        raise ValueError("K1-AY experiment name drifted")
    if config.get("model") != {
        "model_key": MODEL_KEY,
        "summary_dim": STRUCTURE_SUMMARY_DIM,
        "summary_sbox_dim": SBOX_SUMMARY_DIM,
        "summary_linear_dim": LINEAR_SUMMARY_DIM,
        "gate_hidden_dim": 12,
        "gate_output_dim": 2,
        "edge_channel_index": 0,
        "transition_channel_index": 1,
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "expected_state_dict_entries": EXPECTED_STATE_ENTRIES,
    }:
        raise ValueError("K1-AY model contract drifted")
    if config.get("audit") != {
        "replicas": list(EXPECTED_REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "mismatch_conditions": list(MISMATCH_CONDITIONS),
        "rows_per_split": 32,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "expected_control_rows": EXPECTED_CONTROL_ROWS,
        "training_performed": False,
        "optimizer_steps": 0,
        "device": "cpu",
    }:
        raise ValueError("K1-AY audit protocol drifted")
    if config.get("gates") != {
        "disabled_k1aw_logit_replay_tolerance": 0.0,
        "irrelevant_component_jacobian_tolerance": 0.0,
        "irrelevant_mismatch_gate_delta_tolerance": 0.0,
        "minimum_relevant_component_jacobian_l2": 1e-6,
        "minimum_relevant_mismatch_gate_delta": 1e-6,
        "minimum_full_mismatch_gate_delta": 1e-6,
        "minimum_enabled_logit_delta": 1e-8,
        "require_strict_state_dict_load": True,
        "require_state_immutable": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AY gate contract drifted")
    return config


def build_candidate(
    cipher: Mapping[str, Any],
    anchor_model: Mapping[str, Any],
    candidate_model: Mapping[str, Any],
) -> torch.nn.Module:
    options = {
        "runtime_structure_path": str(cipher["runtime_structure_path"]),
        "runtime_round_start": int(cipher["runtime_round_start"]),
        "runtime_rounds": int(cipher["runtime_rounds"]),
        "pair_embedding_dim": int(anchor_model["pair_embedding_dim"]),
        "transition_value_dim": int(anchor_model["transition_value_dim"]),
        "virtual_projection_slots": int(anchor_model["virtual_projection_slots"]),
        "dropout": float(anchor_model["dropout"]),
        "residual_gate_initial_effective": float(
            anchor_model["residual_gate_initial_effective"]
        ),
        "transition_gate_initial_effective": float(
            anchor_model["transition_gate_initial_effective"]
        ),
        "structure_gate_hidden_dim": int(candidate_model["gate_hidden_dim"]),
    }
    return build_model(
        MODEL_KEY,
        input_bits=int(cipher["input_bits"]),
        hidden_bits=int(anchor_model["hidden_bits"]),
        pair_bits=int(cipher["pair_bits"]),
        structure="SPN",
        model_options=options,
    )


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    Mapping[tuple[str, int, str], Any],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor | None]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {name: source_root / name for name in source["digests"]}
    source_gate = _read_json(paths["gate.json"])
    source_validation = _read_json(paths["validation.json"])
    source_results = _read_jsonl(paths["results.jsonl"])
    source_manifest = _read_json(paths["checkpoint_manifest.json"])
    source_summaries = _read_json(paths["structure_summaries.json"])
    k1ax_config = load_k1ax_config(project_root / str(source["config"]))
    (
        readiness,
        k1av,
        dataset_rows,
        datasets,
        structures,
        controls,
        summary_rows,
        checkpoints,
        inherited,
    ) = load_k1ax_authority(
        k1ax_config,
        project_root=project_root,
        device=device,
    )
    checks = {
        "source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        ),
        "source_gate_supports_component_separation": (
            source_gate.get("run_id") == source["run_id"]
            and source_gate.get("status") == "pass"
            and source_gate.get("decision") == source["required_decision"]
            and not source_gate.get("failed_protocol_checks")
            and source_gate.get("routing_results", {}).get("sbox_aligned_panels") == 8
            and source_gate.get("routing_results", {}).get("linear_aligned_panels")
            == 8
            and source_gate.get("remote_scale") == "no"
        ),
        "source_validation_passes": (
            source_validation.get("status") == "pass"
            and not source_validation.get("errors")
        ),
        "source_forty_eight_rows_complete": len(source_results) == 48,
        "source_manifest_binds_two_checkpoints": (
            source_manifest.get("status") == "pass"
            and len(source_manifest.get("entries", [])) == 2
        ),
        "source_three_structure_summaries_complete": len(
            source_summaries.get("rows", [])
        )
        == 3,
        **{f"inherited_{name}": bool(value) for name, value in inherited.items()},
    }
    return (
        readiness,
        k1av,
        dataset_rows,
        datasets,
        structures,
        controls,
        summary_rows,
        checkpoints,
        checks,
    )


def audit_candidate_geometry(
    *,
    readiness_config: Mapping[str, Any],
    k1av_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    models = {
        cipher: build_candidate(
            cipher_configs[cipher], readiness_config["model"], config["model"]
        )
        for cipher in EXPECTED_CIPHERS
    }
    source = build_k1aw_candidate(
        cipher_configs[EXPECTED_CIPHERS[0]],
        readiness_config["model"],
        k1av_config["model"],
    )
    geometries = {
        cipher: tuple(
            (name, tuple(value.shape))
            for name, value in model.state_dict().items()
        )
        for cipher, model in models.items()
    }
    source_geometry = tuple(
        (name, tuple(value.shape)) for name, value in source.state_dict().items()
    )
    parameter_counts = {
        cipher: int(model_metadata(model)["trainable_parameter_count"])
        for cipher, model in models.items()
    }
    state_entries = {cipher: len(model.state_dict()) for cipher, model in models.items()}
    parameter_names = set(next(iter(models.values())).state_dict())
    forbidden_tokens = ("cipher", "expert", "router", "adapter", "head.")
    checks = {
        "candidate_parameter_count_exact": set(parameter_counts.values())
        == {EXPECTED_PARAMETER_COUNT},
        "candidate_state_entries_exact": set(state_entries.values())
        == {EXPECTED_STATE_ENTRIES},
        "candidate_geometry_identical_across_ciphers": len(set(geometries.values()))
        == 1,
        "candidate_geometry_exactly_matches_k1aw": all(
            geometry == source_geometry for geometry in geometries.values()
        ),
        "shared_34_to_12_to_2_tensors_preserved": all(
            model.backbone.structure_gate.summary_dim == STRUCTURE_SUMMARY_DIM
            and model.backbone.structure_gate.hidden_dim == 12
            and model.backbone.structure_gate.output_dim == 2
            and tuple(model.backbone.structure_gate.network[0].weight.shape)
            == (12, 34)
            and tuple(model.backbone.structure_gate.network[2].weight.shape)
            == (2, 12)
            for model in models.values()
        ),
        "component_separation_declared": all(
            model.structure_gate_component_separated is True
            for model in models.values()
        ),
        "no_cipher_expert_router_adapter_or_independent_head_parameters": not any(
            token in name.lower()
            for name in parameter_names
            for token in forbidden_tokens
        ),
        "models_declare_no_cipher_identity": all(
            model.uses_cipher_identity is False
            and model.structure_gate_uses_cipher_identity is False
            and model.structure_gate_shared is True
            for model in models.values()
        ),
    }
    return checks, {
        "parameter_counts": parameter_counts,
        "state_entries": state_entries,
        "shared_geometry": list(next(iter(geometries.values()))),
        "k1aw_geometry": list(source_geometry),
    }


def evaluate_panels(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    k1av_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, Any],
    structure_controls: Mapping[str, Mapping[str, torch.Tensor | None]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    source_config = load_k1ax_config(ROOT / str(config["source"]["config"]))
    k1aw_config_path = ROOT / str(source_config["source"]["config"])
    k1aw_config = _read_json(k1aw_config_path)
    replica_seeds = {
        int(row["replica"]): {
            cipher: int(seed) for cipher, seed in row["dataset_seeds"].items()
        }
        for row in k1aw_config["replicas"]
    }
    results: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    loads: list[dict[str, Any]] = []
    for replica in EXPECTED_REPLICAS:
        source = build_k1aw_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            readiness_config["model"],
            k1av_config["model"],
        ).to(device)
        source.load_state_dict(checkpoints[replica]["state_dict"], strict=True)
        candidate = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            readiness_config["model"],
            config["model"],
        ).to(device)
        incompatible = candidate.load_state_dict(
            checkpoints[replica]["state_dict"], strict=True
        )
        source_state_sha256 = tensor_mapping_sha256(source.state_dict())
        candidate_state_sha256 = tensor_mapping_sha256(candidate.state_dict())
        loads.append(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "checkpoint_sha256": checkpoints[replica]["sha256"],
                "source_state_dict_sha256": source_state_sha256,
                "candidate_state_dict_sha256": candidate_state_sha256,
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "strict_load_exact": (
                    not incompatible.missing_keys
                    and not incompatible.unexpected_keys
                    and candidate_state_sha256 == source_state_sha256
                    and candidate_state_sha256
                    == checkpoints[replica]["state_dict_sha256"]
                ),
            }
        )
        source.eval()
        candidate.eval()
        for cipher in EXPECTED_CIPHERS:
            structure = structures[cipher]
            summaries = structure_controls[cipher]
            correct_summary = summaries["correct_descriptor"]
            if correct_summary is None:
                raise RuntimeError("K1-AY correct summary is unavailable")
            gradients = _component_gradient_metrics(
                candidate, structure, correct_summary
            )
            correct_gates = _path_gate_values(
                candidate, structure, correct_summary
            )
            mismatch_gates = {}
            for condition in MISMATCH_CONDITIONS:
                summary = summaries[condition]
                if summary is None:
                    raise RuntimeError("K1-AY mismatch summary is unavailable")
                mismatch_gates[condition] = _path_gate_values(
                    candidate, structure, summary
                )
            seed = replica_seeds[replica][cipher]
            for split in FRESH_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                count = int(config["audit"]["rows_per_split"])
                features = torch.as_tensor(
                    np.array(dataset.features[:count], copy=True),
                    dtype=torch.float32,
                    device=device,
                )
                state_before = tensor_mapping_sha256(candidate.state_dict())
                with torch.inference_mode():
                    source_logits = source.logits_with_runtime(
                        features,
                        structure,
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=correct_summary,
                        dual_path_enabled=True,
                    )
                    compatibility_logits = candidate.logits_with_runtime(
                        features,
                        structure,
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=correct_summary,
                        dual_path_enabled=True,
                        component_separation_enabled=False,
                    )
                    separated_logits = candidate.logits_with_runtime(
                        features,
                        structure,
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=correct_summary,
                        dual_path_enabled=True,
                        component_separation_enabled=True,
                    )
                    mismatch_logits = {
                        condition: candidate.logits_with_runtime(
                            features,
                            structure,
                            apply_sboxes=True,
                            transition_branch_enabled=True,
                            gate_summary=summaries[condition],
                            dual_path_enabled=True,
                            component_separation_enabled=True,
                        )
                        for condition in MISMATCH_CONDITIONS
                    }
                state_immutable = (
                    tensor_mapping_sha256(candidate.state_dict()) == state_before
                )
                results.append(
                    {
                        "run_id": RUN_ID,
                        "replica": replica,
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "rows_inspected": count,
                        "dataset_sha256": differential_dataset_sha256(dataset),
                        "checkpoint_sha256": checkpoints[replica]["sha256"],
                        "state_dict_sha256": candidate_state_sha256,
                        "correct_edge_gate": correct_gates["edge"],
                        "correct_transition_gate": correct_gates["transition"],
                        "disabled_k1aw_max_abs_logit_delta": _max_abs_delta(
                            compatibility_logits, source_logits
                        ),
                        "enabled_max_abs_logit_delta": _max_abs_delta(
                            separated_logits, compatibility_logits
                        ),
                        **gradients,
                        "all_gate_values_finite_bounded": all(
                            math.isfinite(float(value)) and -1.0 < float(value) < 1.0
                            for gates in [correct_gates, *mismatch_gates.values()]
                            for value in gates.values()
                        ),
                        "runtime_structure_held_correct": True,
                        "state_immutable": state_immutable,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
                for condition in MISMATCH_CONDITIONS:
                    controls.append(
                        {
                            "run_id": RUN_ID,
                            "replica": replica,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "rows_inspected": count,
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "checkpoint_sha256": checkpoints[replica]["sha256"],
                            "edge_gate_delta": abs(
                                correct_gates["edge"]
                                - mismatch_gates[condition]["edge"]
                            ),
                            "transition_gate_delta": abs(
                                correct_gates["transition"]
                                - mismatch_gates[condition]["transition"]
                            ),
                            "max_abs_logit_delta": _max_abs_delta(
                                separated_logits, mismatch_logits[condition]
                            ),
                            "runtime_structure_held_correct": True,
                            "state_immutable": state_immutable,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return results, controls, loads


def adjudicate(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    geometry_checks: Mapping[str, bool],
    results: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    loads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gates = config["gates"]
    expected_results = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    expected_controls = {
        (replica, cipher, split, condition)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
        for condition in MISMATCH_CONDITIONS
    }
    observed_results = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        for row in results
    }
    observed_controls = {
        (
            int(row["replica"]),
            str(row["cipher_key"]),
            str(row["split"]),
            str(row["condition"]),
        )
        for row in controls
    }
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        **{name: bool(value) for name, value in geometry_checks.items()},
        "twelve_result_rows_complete": len(results) == EXPECTED_RESULT_ROWS
        and observed_results == expected_results,
        "thirty_six_control_rows_complete": len(controls) == EXPECTED_CONTROL_ROWS
        and observed_controls == expected_controls,
        "two_strict_checkpoint_loads_exact": len(loads) == 2
        and {int(row["replica"]) for row in loads} == set(EXPECTED_REPLICAS)
        and all(row.get("strict_load_exact") is True for row in loads),
        "disabled_mode_exactly_replays_k1aw": all(
            float(row.get("disabled_k1aw_max_abs_logit_delta", math.inf))
            <= float(gates["disabled_k1aw_logit_replay_tolerance"])
            for row in results
        ),
        "all_states_immutable": all(
            row.get("state_immutable") is True for row in [*results, *controls]
        ),
        "all_rows_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            for row in [*results, *controls]
        ),
    }
    relevant_jacobian = float(gates["minimum_relevant_component_jacobian_l2"])
    irrelevant_jacobian = float(gates["irrelevant_component_jacobian_tolerance"])
    relevant_delta = float(gates["minimum_relevant_mismatch_gate_delta"])
    irrelevant_delta = float(gates["irrelevant_mismatch_gate_delta_tolerance"])
    full_delta = float(gates["minimum_full_mismatch_gate_delta"])
    logit_delta = float(gates["minimum_enabled_logit_delta"])
    research_checks = {
        "all_gate_values_finite_and_bounded": all(
            row.get("all_gate_values_finite_bounded") is True for row in results
        ),
        "edge_reads_linear_component": all(
            float(row.get("edge_linear_summary_jacobian_l2", -math.inf))
            >= relevant_jacobian
            for row in results
        ),
        "edge_ignores_sbox_component_exactly": all(
            float(row.get("edge_sbox_summary_jacobian_l2", math.inf))
            <= irrelevant_jacobian
            for row in results
        ),
        "transition_reads_sbox_component": all(
            float(row.get("transition_sbox_summary_jacobian_l2", -math.inf))
            >= relevant_jacobian
            for row in results
        ),
        "transition_ignores_linear_component_exactly": all(
            float(row.get("transition_linear_summary_jacobian_l2", math.inf))
            <= irrelevant_jacobian
            for row in results
        ),
        "sbox_mismatch_isolated_to_transition_gate": all(
            float(row["edge_gate_delta"]) <= irrelevant_delta
            and float(row["transition_gate_delta"]) >= relevant_delta
            for row in controls
            if row["condition"] == "sbox_only_mismatch"
        ),
        "linear_mismatch_isolated_to_edge_gate": all(
            float(row["transition_gate_delta"]) <= irrelevant_delta
            and float(row["edge_gate_delta"]) >= relevant_delta
            for row in controls
            if row["condition"] == "linear_only_mismatch"
        ),
        "full_mismatch_changes_both_gates": all(
            float(row["edge_gate_delta"]) >= full_delta
            and float(row["transition_gate_delta"]) >= full_delta
            for row in controls
            if row["condition"] == "full_mismatch"
        ),
        "component_separation_changes_logits": all(
            float(row.get("enabled_max_abs_logit_delta", -math.inf)) >= logit_delta
            for row in results
        ),
        "each_descriptor_component_changes_logits": all(
            float(row.get("max_abs_logit_delta", -math.inf)) >= logit_delta
            for row in controls
        ),
    }
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_research = [name for name, passed in research_checks.items() if not passed]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1ay_protocol_invalid"
        next_action = (
            "Repair only the failed source, geometry, strict-load, replay, row or "
            "immutability binding and replay K1-AY unchanged."
        )
    elif failed_research:
        status = "hold"
        decision = "innovation1_uknit_family_k1ay_component_separation_not_ready"
        next_action = (
            "Repair only the failed component relevance, exact isolation or "
            "observable-response mechanism; do not train or scale."
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_k1ay_component_separation_runtime_ready"
        next_action = (
            "Preregister K1-AZ as one same-budget local comparison against K1-AW, "
            "changing only component-separated summary connectivity."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "next_training_authorized": status == "pass",
        "remote_scale": "no",
        "maximum_disabled_k1aw_logit_replay_delta": max(
            float(row["disabled_k1aw_max_abs_logit_delta"]) for row in results
        ),
        "minimum_edge_linear_summary_jacobian_l2": min(
            float(row["edge_linear_summary_jacobian_l2"]) for row in results
        ),
        "maximum_edge_sbox_summary_jacobian_l2": max(
            float(row["edge_sbox_summary_jacobian_l2"]) for row in results
        ),
        "minimum_transition_sbox_summary_jacobian_l2": min(
            float(row["transition_sbox_summary_jacobian_l2"]) for row in results
        ),
        "maximum_transition_linear_summary_jacobian_l2": max(
            float(row["transition_linear_summary_jacobian_l2"]) for row in results
        ),
        "minimum_sbox_mismatch_transition_gate_delta": min(
            float(row["transition_gate_delta"])
            for row in controls
            if row["condition"] == "sbox_only_mismatch"
        ),
        "maximum_sbox_mismatch_edge_gate_delta": max(
            float(row["edge_gate_delta"])
            for row in controls
            if row["condition"] == "sbox_only_mismatch"
        ),
        "minimum_linear_mismatch_edge_gate_delta": min(
            float(row["edge_gate_delta"])
            for row in controls
            if row["condition"] == "linear_only_mismatch"
        ),
        "maximum_linear_mismatch_transition_gate_delta": max(
            float(row["transition_gate_delta"])
            for row in controls
            if row["condition"] == "linear_only_mismatch"
        ),
        "minimum_enabled_logit_delta": min(
            float(row["enabled_max_abs_logit_delta"]) for row in results
        ),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Zero-training local readiness over two frozen K1-AW checkpoints and "
            "twelve fixed panels; not AUC evidence, formal scale, an attack, "
            "unseen-cipher transfer, arbitrary-SPN generalization, or SOTA evidence."
        ),
    }


def run_readiness(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    _require_fresh_output_root(output_root)
    output_root.mkdir(parents=True)
    _append_progress(output_root / "progress.jsonl", "run_start", run_id=RUN_ID)
    (
        readiness,
        k1av,
        dataset_rows,
        datasets,
        structures,
        structure_controls,
        summary_rows,
        checkpoints,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-AY source binding failed: {source_checks}")
    geometry_checks, geometry = audit_candidate_geometry(
        readiness_config=readiness,
        k1av_config=k1av,
        config=config,
    )
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "geometry_checks": geometry_checks,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    _write_json(
        output_root / "structure_summaries.json",
        {"run_id": RUN_ID, "status": "pass", "rows": summary_rows},
    )
    results, controls, loads = evaluate_panels(
        config=config,
        readiness_config=readiness,
        k1av_config=k1av,
        datasets=datasets,
        structures=structures,
        structure_controls=structure_controls,
        checkpoints=checkpoints,
        device=device,
    )
    gate = adjudicate(
        config=config,
        source_checks=source_checks,
        geometry_checks=geometry_checks,
        results=results,
        controls=controls,
        loads=loads,
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "source_run_id": config["source"]["run_id"],
        "entries": [
            {
                key: value
                for key, value in checkpoints[replica].items()
                if key != "state_dict"
            }
            for replica in EXPECTED_REPLICAS
        ],
        "strict_loads": loads,
    }
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(results),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "control_rows": len(controls),
        "expected_control_rows": EXPECTED_CONTROL_ROWS,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "maximum_disabled_k1aw_logit_replay_delta": gate[
            "maximum_disabled_k1aw_logit_replay_delta"
        ],
        "minimum_edge_linear_summary_jacobian_l2": gate[
            "minimum_edge_linear_summary_jacobian_l2"
        ],
        "maximum_edge_sbox_summary_jacobian_l2": gate[
            "maximum_edge_sbox_summary_jacobian_l2"
        ],
        "minimum_transition_sbox_summary_jacobian_l2": gate[
            "minimum_transition_sbox_summary_jacobian_l2"
        ],
        "maximum_transition_linear_summary_jacobian_l2": gate[
            "maximum_transition_linear_summary_jacobian_l2"
        ],
        "next_training_authorized": gate["next_training_authorized"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", results)
    _write_jsonl(output_root / "controls.jsonl", controls)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(output_root / "geometry.json", geometry)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(results),
        control_rows=len(controls),
    )
    return {
        "preflight": preflight,
        "results": results,
        "controls": controls,
        "checkpoint_manifest": checkpoint_manifest,
        "structure_summaries": summary_rows,
        "geometry": geometry,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _component_gradient_metrics(
    model: torch.nn.Module,
    structure: Any,
    summary: torch.Tensor,
) -> dict[str, float | bool]:
    descriptor = summary.detach().clone().to(torch.float32).requires_grad_(True)
    edge_gate, transition_gate = model.effective_path_gates(
        structure,
        summary=descriptor,
        dual_path_enabled=True,
        component_separation_enabled=True,
    )
    edge_gradient = torch.autograd.grad(
        edge_gate, descriptor, retain_graph=True
    )[0]
    transition_gradient = torch.autograd.grad(transition_gate, descriptor)[0]
    return {
        "edge_sbox_summary_jacobian_l2": float(
            torch.linalg.vector_norm(edge_gradient[:SBOX_SUMMARY_DIM]).detach()
        ),
        "edge_linear_summary_jacobian_l2": float(
            torch.linalg.vector_norm(edge_gradient[SBOX_SUMMARY_DIM:]).detach()
        ),
        "transition_sbox_summary_jacobian_l2": float(
            torch.linalg.vector_norm(
                transition_gradient[:SBOX_SUMMARY_DIM]
            ).detach()
        ),
        "transition_linear_summary_jacobian_l2": float(
            torch.linalg.vector_norm(
                transition_gradient[SBOX_SUMMARY_DIM:]
            ).detach()
        ),
        "all_component_gradients_finite": bool(torch.isfinite(edge_gradient).all())
        and bool(torch.isfinite(transition_gradient).all()),
    }


def _path_gate_values(
    model: torch.nn.Module,
    structure: Any,
    summary: torch.Tensor,
) -> dict[str, float]:
    edge, transition = model.effective_path_gates(
        structure,
        summary=summary,
        dual_path_enabled=True,
        component_separation_enabled=True,
    )
    return {
        "edge": float(edge.detach()),
        "transition": float(transition.detach()),
    }


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": event, "time": time.time(), **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(
        (path / name).exists()
        for name in ("preflight.json", "results.jsonl", "gate.json")
    ):
        raise ValueError("K1-AY output already exists")


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_CONTROL_ROWS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_STATE_ENTRIES",
    "MODEL_KEY",
    "ROOT",
    "RUN_ID",
    "adjudicate",
    "audit_candidate_geometry",
    "build_candidate",
    "evaluate_panels",
    "load_and_validate_config",
    "load_authority",
    "run_readiness",
]
