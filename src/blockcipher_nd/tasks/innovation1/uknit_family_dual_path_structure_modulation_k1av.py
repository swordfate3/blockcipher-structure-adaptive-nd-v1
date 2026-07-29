from __future__ import annotations

import hashlib
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
    runtime_structure_summary,
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
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1as import (
    build_candidate as build_k1as_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
    MISMATCH_CONDITIONS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_gate_identifiability_k1au import (
    EXPECTED_REPLICAS,
    REPLICA_DATASET_SEEDS,
    load_and_validate_config as load_k1au_config,
    load_authority as load_k1au_authority,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_dual_path_structure_modulation_k1av_readiness_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_dual_path_structure_modulation_"
    "k1av_readiness_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "409c75bb80cbbba736cbc6121d2fc53b9b0072eb11416ea932e8472a51d2f0a7"
)
MODEL_KEY = "runtime_spn_ct_k1av_dual_path_structure_gate_true"
OUTPUT_WEIGHT_KEY = "backbone.structure_gate.network.2.weight"
EXPECTED_PARAMETER_COUNT = 219_764
EXPECTED_STATE_ENTRIES = 55
EXPECTED_RESULT_ROWS = 12
EXPECTED_CONTROL_ROWS = 36


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AV config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AV identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_dual_path_structure_modulation_k1av_readiness"
    ):
        raise ValueError("K1-AV experiment name drifted")
    if config.get("model") != {
        "model_key": MODEL_KEY,
        "initialization_seed": 83,
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
        raise ValueError("K1-AV model contract drifted")
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
        raise ValueError("K1-AV audit protocol drifted")
    if config.get("gates") != {
        "disabled_k1at_logit_replay_tolerance": 0.0,
        "cell_relabel_summary_tolerance": 0.0,
        "minimum_component_jacobian_l2": 1e-6,
        "maximum_cross_channel_parameter_jacobian_l2": 0.0,
        "minimum_relevant_descriptor_gate_delta": 1e-6,
        "minimum_enabled_logit_delta": 1e-8,
        "require_no_cipher_identity_or_experts": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AV gates drifted")
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


def migrate_k1at_state(
    model: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    target_state = model.state_dict()
    if set(target_state) != set(source_state):
        raise ValueError("K1-AV state keys differ from K1-AT")
    migrated = {name: value.detach().clone() for name, value in target_state.items()}
    expanded_keys: list[str] = []
    for name, source_value in source_state.items():
        if name == OUTPUT_WEIGHT_KEY:
            if tuple(source_value.shape) != (1, 12):
                raise ValueError("K1-AT output projection shape drifted")
            if tuple(migrated[name].shape) != (2, 12):
                raise ValueError("K1-AV output projection shape drifted")
            migrated[name][1].copy_(source_value[0])
            expanded_keys.append(name)
            continue
        if migrated[name].shape != source_value.shape:
            raise ValueError(f"K1-AV unexpected shape change: {name}")
        migrated[name].copy_(source_value)
    model.load_state_dict(migrated, strict=True)
    loaded = model.state_dict()
    return {
        "expanded_keys": expanded_keys,
        "only_final_projection_expanded": expanded_keys == [OUTPUT_WEIGHT_KEY],
        "transition_row_exact": torch.equal(
            loaded[OUTPUT_WEIGHT_KEY][1], source_state[OUTPUT_WEIGHT_KEY][0]
        ),
        "new_edge_row_finite_nonzero": bool(
            torch.isfinite(loaded[OUTPUT_WEIGHT_KEY][0]).all()
        )
        and float(torch.linalg.vector_norm(loaded[OUTPUT_WEIGHT_KEY][0])) > 0.0,
        "loaded_state_dict_sha256": tensor_mapping_sha256(loaded),
        "edge_output_row_sha256": _tensor_sha256(loaded[OUTPUT_WEIGHT_KEY][0]),
        "transition_output_row_sha256": _tensor_sha256(
            loaded[OUTPUT_WEIGHT_KEY][1]
        ),
    }


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[tuple[str, int, str], Any],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor | None]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {name: source_root / name for name in source["digests"]}
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    results = _read_jsonl(paths["results.jsonl"])
    controls = _read_jsonl(paths["controls.jsonl"])
    manifest = _read_json(paths["checkpoint_manifest.json"])
    summary = _read_json(paths["summary.json"])
    k1au_config = load_k1au_config(project_root / str(source["config"]))
    (
        readiness_config,
        k1as_config,
        datasets,
        structures,
        structure_controls,
        checkpoints,
        checkpoint_rows,
        inherited,
    ) = load_k1au_authority(
        k1au_config,
        project_root=project_root,
        device=device,
    )
    checks = {
        "k1au_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        ),
        "k1au_gate_authorizes_k1av": (
            gate.get("run_id") == source["run_id"]
            and gate.get("status") == "pass"
            and gate.get("decision") == source["required_decision"]
            and gate.get("representation_preserved_through_hidden") is True
            and gate.get("final_scalar_mapping_stable") is False
            and not gate.get("failed_protocol_checks")
            and gate.get("remote_scale") == "no"
        ),
        "k1au_validation_passes": (
            validation.get("run_id") == source["run_id"]
            and validation.get("status") == "pass"
            and not validation.get("errors")
        ),
        "k1au_result_and_control_rows_bound": len(results) == 6
        and len(controls) == 36,
        "k1au_manifest_binds_two_checkpoints": manifest.get("status") == "pass"
        and len(manifest.get("entries", [])) == 2,
        "k1au_summary_matches_decision": summary.get("decision")
        == source["required_decision"],
        **{f"k1au_{name}": bool(value) for name, value in inherited.items()},
    }
    return (
        readiness_config,
        k1as_config,
        datasets,
        structures,
        structure_controls,
        checkpoints,
        checkpoint_rows,
        checks,
    )


def build_structure_report(
    *,
    structures: Mapping[str, Any],
    structure_controls: Mapping[str, Mapping[str, torch.Tensor | None]],
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        correct = structure_controls[cipher]["correct_descriptor"]
        if correct is None:
            raise RuntimeError("K1-AV correct structure summary is unavailable")
        structure = structures[cipher]
        relabeled, _ = structure.relabel_cells(tuple(reversed(range(structure.cells))))
        relabeled_summary = runtime_structure_summary(relabeled)
        mismatch_hashes = {}
        fixed_width = True
        for condition in MISMATCH_CONDITIONS:
            mismatch = structure_controls[cipher][condition]
            if mismatch is None:
                raise RuntimeError("K1-AV mismatch summary is unavailable")
            fixed_width &= mismatch.shape == (STRUCTURE_SUMMARY_DIM,)
            mismatch_hashes[condition] = _tensor_sha256(mismatch)
        relabel_delta = float(torch.max(torch.abs(correct - relabeled_summary)))
        rows.append(
            {
                "run_id": RUN_ID,
                "cipher_key": cipher,
                "correct_summary": correct.tolist(),
                "correct_summary_sha256": _tensor_sha256(correct),
                "mismatch_summary_sha256s": mismatch_hashes,
                "fixed_width": fixed_width,
                "cell_relabel_max_abs_delta": relabel_delta,
                "cell_relabel_exact": relabel_delta == 0.0,
            }
        )
    checks = {
        "three_structure_rows_complete": len(rows) == 3,
        "all_summaries_fixed_width": all(row["fixed_width"] for row in rows),
        "cell_relabeling_invariant": all(
            float(row["cell_relabel_max_abs_delta"]) <= tolerance for row in rows
        ),
    }
    return rows, checks


def audit_candidate_geometry(
    *,
    readiness_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    models = {}
    for cipher in EXPECTED_CIPHERS:
        with torch.random.fork_rng():
            torch.manual_seed(int(config["model"]["initialization_seed"]))
            models[cipher] = build_candidate(
                cipher_configs[cipher], readiness_config["model"], config["model"]
            )
    geometries = {
        cipher: tuple(
            (name, tuple(value.shape))
            for name, value in model.state_dict().items()
        )
        for cipher, model in models.items()
    }
    state_hashes = {
        cipher: tensor_mapping_sha256(model.state_dict())
        for cipher, model in models.items()
    }
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
        "candidate_geometry_identical_across_ciphers": len(set(geometries.values())) == 1,
        "candidate_initial_state_identical_across_ciphers": len(set(state_hashes.values())) == 1,
        "one_shared_34_to_12_to_2_network": all(
            model.backbone.structure_gate.summary_dim == STRUCTURE_SUMMARY_DIM
            and model.backbone.structure_gate.hidden_dim == 12
            and model.backbone.structure_gate.output_dim == 2
            and tuple(model.backbone.structure_gate.network[2].weight.shape) == (2, 12)
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
        "state_hashes": state_hashes,
        "shared_geometry": list(next(iter(geometries.values()))),
    }


def evaluate_panels(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    k1as_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, Any],
    structure_controls: Mapping[str, Mapping[str, torch.Tensor | None]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    results: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    for replica in EXPECTED_REPLICAS:
        source = build_k1as_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            readiness_config["model"],
            k1as_config["model"],
        ).to(device)
        source.load_state_dict(checkpoints[replica]["state_dict"], strict=True)
        with torch.random.fork_rng():
            torch.manual_seed(int(config["model"]["initialization_seed"]))
            candidate = build_candidate(
                cipher_configs[EXPECTED_CIPHERS[0]],
                readiness_config["model"],
                config["model"],
            ).to(device)
        migration = migrate_k1at_state(candidate, checkpoints[replica]["state_dict"])
        migrations.append(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "checkpoint_sha256": checkpoints[replica]["sha256"],
                **migration,
            }
        )
        source.eval()
        candidate.eval()
        state_sha256 = tensor_mapping_sha256(candidate.state_dict())
        for cipher in EXPECTED_CIPHERS:
            structure = structures[cipher]
            correct_summary = structure_controls[cipher]["correct_descriptor"]
            if correct_summary is None:
                raise RuntimeError("K1-AV correct summary is unavailable")
            gradient_metrics = _gradient_metrics(candidate, structure, correct_summary)
            correct_gates = _path_gate_values(candidate, structure, correct_summary)
            mismatch_gates = {}
            for condition in MISMATCH_CONDITIONS:
                mismatch_summary = structure_controls[cipher][condition]
                if mismatch_summary is None:
                    raise RuntimeError("K1-AV mismatch summary is unavailable")
                mismatch_gates[condition] = _path_gate_values(
                    candidate, structure, mismatch_summary
                )
            seed = int(REPLICA_DATASET_SEEDS[replica][cipher])
            for split in FRESH_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                count = int(config["audit"]["rows_per_split"])
                features = torch.as_tensor(
                    np.array(dataset.features[:count], copy=True),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.inference_mode():
                    source_logits = source.logits_with_runtime(
                        features,
                        structure,
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=correct_summary,
                        structure_gate_enabled=True,
                    )
                    compatibility_logits = candidate.logits_with_runtime(
                        features,
                        structure,
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=correct_summary,
                        dual_path_enabled=False,
                    )
                    correct_logits = candidate.logits_with_runtime(
                        features,
                        structure,
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=correct_summary,
                        dual_path_enabled=True,
                    )
                    mismatch_logits = {
                        condition: candidate.logits_with_runtime(
                            features,
                            structure,
                            apply_sboxes=True,
                            transition_branch_enabled=True,
                            gate_summary=structure_controls[cipher][condition],
                            dual_path_enabled=True,
                        )
                        for condition in MISMATCH_CONDITIONS
                    }
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
                        "state_dict_sha256": state_sha256,
                        "correct_edge_gate": correct_gates["edge"],
                        "correct_transition_gate": correct_gates["transition"],
                        "disabled_k1at_max_abs_logit_delta": _max_abs_delta(
                            compatibility_logits, source_logits
                        ),
                        "enabled_max_abs_logit_delta": _max_abs_delta(
                            correct_logits, compatibility_logits
                        ),
                        **gradient_metrics,
                        "all_gate_values_finite_bounded": all(
                            math.isfinite(float(value)) and -1.0 < float(value) < 1.0
                            for gates in [correct_gates, *mismatch_gates.values()]
                            for value in gates.values()
                        ),
                        "state_immutable": tensor_mapping_sha256(
                            candidate.state_dict()
                        )
                        == state_sha256,
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
                                correct_logits, mismatch_logits[condition]
                            ),
                            "runtime_structure_held_correct": True,
                            "state_immutable": tensor_mapping_sha256(
                                candidate.state_dict()
                            )
                            == state_sha256,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return results, controls, migrations


def adjudicate(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    geometry_checks: Mapping[str, bool],
    structure_checks: Mapping[str, bool],
    results: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    migrations: Sequence[Mapping[str, Any]],
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
        "two_checkpoint_migrations_complete": len(migrations) == 2
        and {int(row["replica"]) for row in migrations}
        == set(EXPECTED_REPLICAS),
        "only_final_projection_expanded": all(
            row.get("only_final_projection_expanded") is True
            and row.get("transition_row_exact") is True
            and row.get("new_edge_row_finite_nonzero") is True
            for row in migrations
        ),
        "disabled_mode_exactly_replays_k1at": all(
            float(row.get("disabled_k1at_max_abs_logit_delta", math.inf))
            <= float(gates["disabled_k1at_logit_replay_tolerance"])
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
    minimum_component = float(gates["minimum_component_jacobian_l2"])
    maximum_cross = float(gates["maximum_cross_channel_parameter_jacobian_l2"])
    minimum_gate_delta = float(gates["minimum_relevant_descriptor_gate_delta"])
    minimum_logit_delta = float(gates["minimum_enabled_logit_delta"])
    research_checks = {
        **{name: bool(value) for name, value in structure_checks.items()},
        "all_gate_values_finite_and_bounded": all(
            row.get("all_gate_values_finite_bounded") is True for row in results
        ),
        "edge_gate_has_linear_summary_sensitivity": all(
            float(row.get("edge_linear_summary_jacobian_l2", -math.inf))
            >= minimum_component
            for row in results
        ),
        "transition_gate_has_sbox_summary_sensitivity": all(
            float(row.get("transition_sbox_summary_jacobian_l2", -math.inf))
            >= minimum_component
            for row in results
        ),
        "edge_gate_wired_only_to_output_row_zero": all(
            float(row.get("edge_own_row_parameter_jacobian_l2", -math.inf))
            >= minimum_component
            and float(row.get("edge_cross_row_parameter_jacobian_l2", math.inf))
            <= maximum_cross
            for row in results
        ),
        "transition_gate_wired_only_to_output_row_one": all(
            float(row.get("transition_own_row_parameter_jacobian_l2", -math.inf))
            >= minimum_component
            and float(
                row.get("transition_cross_row_parameter_jacobian_l2", math.inf)
            )
            <= maximum_cross
            for row in results
        ),
        "linear_mismatch_changes_edge_gate": all(
            float(row["edge_gate_delta"]) >= minimum_gate_delta
            for row in controls
            if row["condition"] == "linear_only_mismatch"
        ),
        "sbox_mismatch_changes_transition_gate": all(
            float(row["transition_gate_delta"]) >= minimum_gate_delta
            for row in controls
            if row["condition"] == "sbox_only_mismatch"
        ),
        "full_mismatch_changes_both_gates": all(
            float(row["edge_gate_delta"]) >= minimum_gate_delta
            and float(row["transition_gate_delta"]) >= minimum_gate_delta
            for row in controls
            if row["condition"] == "full_mismatch"
        ),
        "dual_path_enabled_changes_logits": all(
            float(row.get("enabled_max_abs_logit_delta", -math.inf))
            >= minimum_logit_delta
            for row in results
        ),
        "all_descriptor_controls_change_logits": all(
            float(row.get("max_abs_logit_delta", -math.inf)) >= minimum_logit_delta
            for row in controls
        ),
    }
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_research = [name for name, passed in research_checks.items() if not passed]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1av_protocol_invalid"
        next_action = (
            "Repair only the failed source, geometry, migration, replay, row or "
            "state binding and replay K1-AV unchanged."
        )
    elif failed_research:
        status = "hold"
        decision = "innovation1_uknit_family_k1av_dual_path_modulation_not_ready"
        next_action = (
            "Repair only the failed channel wiring, sensitivity, descriptor or "
            "equivariance mechanism; do not train or scale."
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_k1av_dual_path_modulation_runtime_ready"
        next_action = (
            "Open one separately preregistered K1-AW same-budget local training "
            "comparison against the frozen K1-AT single-scalar anchor."
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
        "minimum_edge_linear_summary_jacobian_l2": min(
            float(row["edge_linear_summary_jacobian_l2"]) for row in results
        ),
        "minimum_transition_sbox_summary_jacobian_l2": min(
            float(row["transition_sbox_summary_jacobian_l2"]) for row in results
        ),
        "maximum_cross_channel_parameter_jacobian_l2": max(
            max(
                float(row["edge_cross_row_parameter_jacobian_l2"]),
                float(row["transition_cross_row_parameter_jacobian_l2"]),
            )
            for row in results
        ),
        "minimum_linear_mismatch_edge_gate_delta": min(
            float(row["edge_gate_delta"])
            for row in controls
            if row["condition"] == "linear_only_mismatch"
        ),
        "minimum_sbox_mismatch_transition_gate_delta": min(
            float(row["transition_gate_delta"])
            for row in controls
            if row["condition"] == "sbox_only_mismatch"
        ),
        "minimum_enabled_logit_delta": min(
            float(row["enabled_max_abs_logit_delta"]) for row in results
        ),
        "maximum_disabled_k1at_logit_replay_delta": max(
            float(row["disabled_k1at_max_abs_logit_delta"]) for row in results
        ),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Zero-training local readiness over two K1-AT checkpoints and twelve "
            "deterministic panels; not AUC evidence, formal scale, an attack, "
            "unseen-cipher transfer, or SOTA evidence."
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
        readiness_config,
        k1as_config,
        datasets,
        structures,
        structure_controls,
        checkpoints,
        checkpoint_rows,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-AV source binding failed: {source_checks}")
    structure_rows, structure_checks = build_structure_report(
        structures=structures,
        structure_controls=structure_controls,
        tolerance=float(config["gates"]["cell_relabel_summary_tolerance"]),
    )
    geometry_checks, geometry_report = audit_candidate_geometry(
        readiness_config=readiness_config,
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
        "structure_checks": structure_checks,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    results, controls, migrations = evaluate_panels(
        config=config,
        readiness_config=readiness_config,
        k1as_config=k1as_config,
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
        structure_checks=structure_checks,
        results=results,
        controls=controls,
        migrations=migrations,
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "source_run_id": config["source"]["run_id"],
        "source_entries": checkpoint_rows,
        "migrations": migrations,
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
        "maximum_disabled_k1at_logit_replay_delta": gate[
            "maximum_disabled_k1at_logit_replay_delta"
        ],
        "minimum_edge_linear_summary_jacobian_l2": gate[
            "minimum_edge_linear_summary_jacobian_l2"
        ],
        "minimum_transition_sbox_summary_jacobian_l2": gate[
            "minimum_transition_sbox_summary_jacobian_l2"
        ],
        "maximum_cross_channel_parameter_jacobian_l2": gate[
            "maximum_cross_channel_parameter_jacobian_l2"
        ],
        "next_training_authorized": gate["next_training_authorized"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", results)
    _write_jsonl(output_root / "controls.jsonl", controls)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(
        output_root / "structure_summaries.json",
        {"run_id": RUN_ID, "status": "pass", "rows": structure_rows},
    )
    _write_json(output_root / "geometry.json", geometry_report)
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
        "structure_summaries": structure_rows,
        "geometry": geometry_report,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _gradient_metrics(
    model: torch.nn.Module,
    structure: Any,
    summary: torch.Tensor,
) -> dict[str, float | bool]:
    descriptor = summary.detach().clone().to(torch.float32).requires_grad_(True)
    edge_gate, transition_gate = model.effective_path_gates(
        structure,
        summary=descriptor,
        dual_path_enabled=True,
    )
    output_weight = model.backbone.structure_gate.network[2].weight
    edge_summary_gradient = torch.autograd.grad(
        edge_gate, descriptor, retain_graph=True
    )[0]
    transition_summary_gradient = torch.autograd.grad(
        transition_gate, descriptor, retain_graph=True
    )[0]
    edge_weight_gradient = torch.autograd.grad(
        edge_gate, output_weight, retain_graph=True
    )[0]
    transition_weight_gradient = torch.autograd.grad(
        transition_gate, output_weight
    )[0]
    tensors = (
        edge_summary_gradient,
        transition_summary_gradient,
        edge_weight_gradient,
        transition_weight_gradient,
    )
    return {
        "edge_linear_summary_jacobian_l2": float(
            torch.linalg.vector_norm(edge_summary_gradient[SBOX_SUMMARY_DIM:]).detach()
        ),
        "transition_sbox_summary_jacobian_l2": float(
            torch.linalg.vector_norm(
                transition_summary_gradient[:SBOX_SUMMARY_DIM]
            ).detach()
        ),
        "edge_own_row_parameter_jacobian_l2": float(
            torch.linalg.vector_norm(edge_weight_gradient[0]).detach()
        ),
        "edge_cross_row_parameter_jacobian_l2": float(
            torch.linalg.vector_norm(edge_weight_gradient[1]).detach()
        ),
        "transition_own_row_parameter_jacobian_l2": float(
            torch.linalg.vector_norm(transition_weight_gradient[1]).detach()
        ),
        "transition_cross_row_parameter_jacobian_l2": float(
            torch.linalg.vector_norm(transition_weight_gradient[0]).detach()
        ),
        "all_channel_gradients_finite": all(
            bool(torch.isfinite(tensor).all()) for tensor in tensors
        ),
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
    )
    return {
        "edge": float(edge.detach()),
        "transition": float(transition.detach()),
    }


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach())


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


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
        raise ValueError("K1-AV output already exists")


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
    "build_structure_report",
    "evaluate_panels",
    "load_and_validate_config",
    "load_authority",
    "migrate_k1at_state",
    "run_readiness",
]
