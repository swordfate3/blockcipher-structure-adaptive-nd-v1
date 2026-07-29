from __future__ import annotations

import inspect
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
    hybrid_structure_summary,
    runtime_structure_summary,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_path_contribution_k1ar import (
    CHECKPOINT_FAMILIES,
    FRESH_SPLITS,
    REPLICAS,
    load_and_validate_config as load_k1ar_config,
    load_authority as load_k1ar_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    build_runtime_model,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_inverse_norm_k1aq import (
    load_and_validate_config as load_k1aq_config,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_structure_derived_gate_k1as_"
    "readiness_replay_fix_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_structure_derived_gate_k1as_readiness_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "f53f1b2bd684912f4ae48da3d59bb930aca943928c1717e0765ac7bd32e34b7d"
)
MODEL_KEY = "runtime_spn_ct_k1as_structure_gate_true"
EXPECTED_PARAMETER_COUNT = 219_752
EXPECTED_STATE_ENTRIES = 55
EXPECTED_MISSING_KEYS = {
    "backbone.structure_gate.network.0.weight",
    "backbone.structure_gate.network.0.bias",
    "backbone.structure_gate.network.2.weight",
}
EXPECTED_PANELS = 24


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AS config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AS identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_structure_derived_gate_k1as_readiness"
    ):
        raise ValueError("K1-AS experiment name drifted")
    model = config.get("model", {})
    if model != {
        "model_key": MODEL_KEY,
        "initialization_seed": 73,
        "summary_dim": STRUCTURE_SUMMARY_DIM,
        "summary_sbox_dim": SBOX_SUMMARY_DIM,
        "summary_linear_dim": LINEAR_SUMMARY_DIM,
        "gate_hidden_dim": 12,
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "expected_state_dict_entries": EXPECTED_STATE_ENTRIES,
        "global_transition_gate_initial_effective": 0.05,
    }:
        raise ValueError("K1-AS model contract drifted")
    audit = config.get("audit", {})
    if audit != {
        "checkpoint_families": list(CHECKPOINT_FAMILIES),
        "replicas": list(REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "expected_panels": EXPECTED_PANELS,
        "rows_per_panel": 32,
        "training_performed": False,
        "optimizer_steps": 0,
        "data_generation": False,
        "device": "cpu",
    }:
        raise ValueError("K1-AS audit contract drifted")
    if config.get("controls", {}).get("conditions") != [
        "correct_descriptor",
        "full_mismatch",
        "sbox_only_mismatch",
        "linear_only_mismatch",
        "descriptor_disabled",
    ]:
        raise ValueError("K1-AS control conditions drifted")
    if config.get("controls", {}).get("mismatch_order") != {
        "uknit64": "midori64",
        "midori64": "dialga128",
        "dialga128": "uknit64",
    }:
        raise ValueError("K1-AS mismatch order drifted")
    gates = config.get("gates", {})
    if gates != {
        "disabled_logit_replay_tolerance": 0.0,
        "cell_relabel_summary_tolerance": 0.0,
        "minimum_mismatch_gate_delta": 1e-6,
        "minimum_observable_logit_delta": 1e-8,
        "require_nonzero_finite_gradient": True,
        "require_no_cipher_identity_or_experts": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AS gates drifted")
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
    dict[tuple[str, int, str], Any],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, Any],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {name: source_root / name for name in source["digests"]}
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    manifest = _read_json(paths["checkpoint_manifest.json"])
    direct_checks = {
        "k1ar_source_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        ),
        "k1ar_gate_authorizes_k1as": (
            gate.get("run_id") == source["run_id"]
            and gate.get("status") == "pass"
            and gate.get("decision") == source["required_decision"]
            and gate.get("heterogeneous_transition_demand_supported") is True
            and not gate.get("failed_protocol_checks")
            and gate.get("remote_scale") == "no"
        ),
        "k1ar_validation_passes": (
            validation.get("run_id") == source["run_id"]
            and validation.get("status") == "pass"
            and not validation.get("errors")
        ),
        "k1ar_manifest_binds_four_sources": (
            manifest.get("run_id") == source["run_id"]
            and manifest.get("status") == "pass"
            and len(manifest.get("source_entries", [])) == 4
        ),
    }
    k1ar_config = load_k1ar_config()
    readiness, datasets, checkpoints, _controls, _dataset_rows, inherited = (
        load_k1ar_authority(
            k1ar_config,
            project_root=project_root,
            device=device,
        )
    )
    checks = {
        **direct_checks,
        **{f"k1ar_{name}": bool(value) for name, value in inherited.items()},
    }
    return readiness, datasets, checkpoints, manifest, checks


def build_structure_controls(
    *,
    readiness: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    structures = {
        cipher: build_runtime_model(
            cipher_configs[cipher], readiness["model"]
        ).runtime_structure
        for cipher in EXPECTED_CIPHERS
    }
    summaries = {
        cipher: runtime_structure_summary(structure)
        for cipher, structure in structures.items()
    }
    rows: list[dict[str, Any]] = []
    relabel_exact = True
    all_finite_bounded = True
    all_hybrids_have_fixed_width = True
    mismatch_order = config["controls"]["mismatch_order"]
    for cipher in EXPECTED_CIPHERS:
        other = str(mismatch_order[cipher])
        structure = structures[cipher]
        relabeled, _ = structure.relabel_cells(tuple(reversed(range(structure.cells))))
        relabeled_summary = runtime_structure_summary(relabeled)
        sbox_mismatch = hybrid_structure_summary(
            sbox_structure=structures[other],
            linear_structure=structure,
        )
        linear_mismatch = hybrid_structure_summary(
            sbox_structure=structure,
            linear_structure=structures[other],
        )
        controls = {
            "correct_descriptor": summaries[cipher],
            "full_mismatch": summaries[other],
            "sbox_only_mismatch": sbox_mismatch,
            "linear_only_mismatch": linear_mismatch,
        }
        relabel_delta = float(torch.max(torch.abs(summaries[cipher] - relabeled_summary)))
        relabel_exact &= relabel_delta <= float(
            config["gates"]["cell_relabel_summary_tolerance"]
        )
        all_hybrids_have_fixed_width &= all(
            value.shape == (STRUCTURE_SUMMARY_DIM,) for value in controls.values()
        )
        all_finite_bounded &= all(
            bool(torch.isfinite(value).all())
            and bool(torch.all((value >= 0.0) & (value <= 1.0)))
            for value in controls.values()
        )
        rows.append(
            {
                "run_id": RUN_ID,
                "cipher_key": cipher,
                "mismatch_cipher_key": other,
                "summary": summaries[cipher].tolist(),
                "full_mismatch_summary": summaries[other].tolist(),
                "sbox_only_mismatch_summary": sbox_mismatch.tolist(),
                "linear_only_mismatch_summary": linear_mismatch.tolist(),
                "cell_relabel_max_abs_delta": relabel_delta,
                "cell_relabel_exact": relabel_delta == 0.0,
            }
        )
    checks = {
        "all_structure_summaries_finite_bounded": all_finite_bounded,
        "all_structure_summaries_fixed_width": all_hybrids_have_fixed_width,
        "all_cell_relabels_exact": relabel_exact,
        "summary_function_accepts_only_runtime_structure": tuple(
            inspect.signature(runtime_structure_summary).parameters
        )
        == ("structure",),
    }
    return structures, rows, checks


def audit_candidate_geometry(
    *,
    readiness: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    models = {}
    for cipher in EXPECTED_CIPHERS:
        with torch.random.fork_rng():
            torch.manual_seed(int(config["model"]["initialization_seed"]))
            models[cipher] = build_candidate(
                cipher_configs[cipher], readiness["model"], config["model"]
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
        "one_shared_gate_network_only": all(
            sum("structure_gate.network" in name for name, _ in model.named_parameters())
            == 3
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
    report = {
        "parameter_counts": parameter_counts,
        "state_entries": state_entries,
        "state_hashes": state_hashes,
        "shared_geometry": list(next(iter(geometries.values()))),
    }
    return checks, report


def evaluate_panels(
    *,
    config: Mapping[str, Any],
    readiness: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    checkpoints: Mapping[str, Mapping[int, Mapping[str, Any]]],
    structures: Mapping[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    k1aq_config = load_k1aq_config()
    replica_configs = {
        int(row["replica"]): row for row in k1aq_config["replicas"]
    }
    summaries = {
        cipher: runtime_structure_summary(structures[cipher])
        for cipher in EXPECTED_CIPHERS
    }
    mismatch_order = config["controls"]["mismatch_order"]
    rows: list[dict[str, Any]] = []
    for family in CHECKPOINT_FAMILIES:
        for replica in REPLICAS:
            with torch.random.fork_rng():
                torch.manual_seed(int(config["model"]["initialization_seed"]))
                candidate = build_candidate(
                    cipher_configs[EXPECTED_CIPHERS[0]],
                    readiness["model"],
                    config["model"],
                ).to(device)
            source_model = build_runtime_model(
                cipher_configs[EXPECTED_CIPHERS[0]], readiness["model"]
            ).to(device)
            checkpoint = checkpoints[family][replica]
            source_model.load_state_dict(checkpoint["state_dict"], strict=True)
            incompatibility = candidate.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            missing_keys = set(incompatibility.missing_keys)
            unexpected_keys = set(incompatibility.unexpected_keys)
            source_model.eval()
            candidate.eval()
            for cipher in EXPECTED_CIPHERS:
                other = str(mismatch_order[cipher])
                structure = structures[cipher]
                control_summaries = {
                    "correct_descriptor": summaries[cipher],
                    "full_mismatch": summaries[other],
                    "sbox_only_mismatch": hybrid_structure_summary(
                        sbox_structure=structures[other],
                        linear_structure=structure,
                    ),
                    "linear_only_mismatch": hybrid_structure_summary(
                        sbox_structure=structure,
                        linear_structure=structures[other],
                    ),
                }
                gradient_metrics = _gate_gradient_metrics(
                    candidate, structure, control_summaries["correct_descriptor"]
                )
                gate_values = {
                    name: float(
                        candidate.effective_transition_gate(
                            structure,
                            summary=summary,
                            enabled=True,
                        ).detach()
                    )
                    for name, summary in control_summaries.items()
                }
                gate_values["descriptor_disabled"] = float(
                    candidate.effective_transition_gate(
                        structure,
                        enabled=False,
                    ).detach()
                )
                seed = int(replica_configs[replica]["dataset_seeds"][cipher])
                for split in FRESH_SPLITS:
                    dataset = datasets[(cipher, seed, split)]
                    count = int(config["audit"]["rows_per_panel"])
                    features = torch.as_tensor(
                        np.array(dataset.features[:count], copy=True),
                        dtype=torch.float32,
                        device=device,
                    )
                    state_before = tensor_mapping_sha256(candidate.state_dict())
                    with torch.inference_mode():
                        source_logits = source_model.logits_with_runtime(
                            features,
                            structure,
                            apply_sboxes=True,
                            transition_branch_enabled=True,
                        )
                        logits = {
                            name: candidate.logits_with_runtime(
                                features,
                                structure,
                                apply_sboxes=True,
                                transition_branch_enabled=True,
                                gate_summary=summary,
                                structure_gate_enabled=True,
                            )
                            for name, summary in control_summaries.items()
                        }
                        logits["descriptor_disabled"] = candidate.logits_with_runtime(
                            features,
                            structure,
                            apply_sboxes=True,
                            transition_branch_enabled=True,
                            structure_gate_enabled=False,
                        )
                    state_after = tensor_mapping_sha256(candidate.state_dict())
                    correct_logits = logits["correct_descriptor"]
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "checkpoint_family": family,
                            "replica": replica,
                            "cipher_key": cipher,
                            "mismatch_cipher_key": other,
                            "seed": seed,
                            "split": split,
                            "rows_inspected": count,
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "checkpoint_sha256": checkpoint["sha256"],
                            "checkpoint_missing_keys": sorted(missing_keys),
                            "checkpoint_unexpected_keys": sorted(unexpected_keys),
                            "disabled_source_max_abs_logit_delta": _max_abs_delta(
                                logits["descriptor_disabled"], source_logits
                            ),
                            "full_mismatch_gate_delta": abs(
                                gate_values["correct_descriptor"]
                                - gate_values["full_mismatch"]
                            ),
                            "sbox_only_mismatch_gate_delta": abs(
                                gate_values["correct_descriptor"]
                                - gate_values["sbox_only_mismatch"]
                            ),
                            "linear_only_mismatch_gate_delta": abs(
                                gate_values["correct_descriptor"]
                                - gate_values["linear_only_mismatch"]
                            ),
                            "full_mismatch_max_abs_logit_delta": _max_abs_delta(
                                correct_logits, logits["full_mismatch"]
                            ),
                            "sbox_only_mismatch_max_abs_logit_delta": _max_abs_delta(
                                correct_logits, logits["sbox_only_mismatch"]
                            ),
                            "linear_only_mismatch_max_abs_logit_delta": _max_abs_delta(
                                correct_logits, logits["linear_only_mismatch"]
                            ),
                            "gate_values": gate_values,
                            **gradient_metrics,
                            "state_sha256_before": state_before,
                            "state_sha256_after": state_after,
                            "state_immutable": state_before == state_after,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def adjudicate(
    *,
    source_checks: Mapping[str, bool],
    geometry_checks: Mapping[str, bool],
    structure_checks: Mapping[str, bool],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        (family, replica, cipher, split)
        for family in CHECKPOINT_FAMILIES
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    observed_keys = {
        (
            str(row.get("checkpoint_family")),
            int(row.get("replica", -1)),
            str(row.get("cipher_key")),
            str(row.get("split")),
        )
        for row in rows
    }
    replay_tolerance = float(config["gates"]["disabled_logit_replay_tolerance"])
    gate_delta = float(config["gates"]["minimum_mismatch_gate_delta"])
    logit_delta = float(config["gates"]["minimum_observable_logit_delta"])
    protocol_checks = {
        **{name: bool(value) for name, value in source_checks.items()},
        **{name: bool(value) for name, value in geometry_checks.items()},
        "result_panels_complete": len(rows) == EXPECTED_PANELS
        and observed_keys == expected_keys,
        "old_checkpoints_load_only_new_gate_parameters_missing": all(
            set(row.get("checkpoint_missing_keys", ())) == EXPECTED_MISSING_KEYS
            and not row.get("checkpoint_unexpected_keys")
            for row in rows
        ),
        "descriptor_disabled_exactly_replays_source": all(
            float(row.get("disabled_source_max_abs_logit_delta", math.inf))
            <= replay_tolerance
            for row in rows
        ),
        "all_states_immutable": all(row.get("state_immutable") is True for row in rows),
        "all_rows_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            for row in rows
        ),
    }
    research_checks = {
        **{name: bool(value) for name, value in structure_checks.items()},
        "all_gate_values_finite_and_bounded": all(
            all(math.isfinite(float(value)) and -1.0 < float(value) < 1.0 for value in row.get("gate_values", {}).values())
            for row in rows
        ),
        "shared_gate_has_nonzero_finite_gradient": all(
            math.isfinite(float(row.get("structure_gate_gradient_norm", math.nan)))
            and float(row.get("structure_gate_gradient_norm", 0.0)) > 0.0
            and row.get("all_structure_gate_gradients_finite") is True
            for row in rows
        ),
        "all_full_mismatch_gate_deltas_observable": all(
            float(row.get("full_mismatch_gate_delta", -math.inf)) >= gate_delta
            for row in rows
        ),
        "all_sbox_mismatch_gate_deltas_observable": all(
            float(row.get("sbox_only_mismatch_gate_delta", -math.inf)) >= gate_delta
            for row in rows
        ),
        "all_linear_mismatch_gate_deltas_observable": all(
            float(row.get("linear_only_mismatch_gate_delta", -math.inf)) >= gate_delta
            for row in rows
        ),
        "all_full_mismatch_logits_observable": all(
            float(row.get("full_mismatch_max_abs_logit_delta", -math.inf)) >= logit_delta
            for row in rows
        ),
        "all_sbox_mismatch_logits_observable": all(
            float(row.get("sbox_only_mismatch_max_abs_logit_delta", -math.inf)) >= logit_delta
            for row in rows
        ),
        "all_linear_mismatch_logits_observable": all(
            float(row.get("linear_only_mismatch_max_abs_logit_delta", -math.inf)) >= logit_delta
            for row in rows
        ),
    }
    failed_protocol_checks = [name for name, passed in protocol_checks.items() if not passed]
    failed_research_checks = [name for name, passed in research_checks.items() if not passed]
    if failed_protocol_checks:
        status = "fail"
        decision = "innovation1_uknit_family_k1as_protocol_invalid"
        next_action = "Repair only the failed source, geometry, or exact replay check."
    elif failed_research_checks:
        status = "hold"
        decision = "innovation1_uknit_family_k1as_structure_gate_not_ready"
        next_action = (
            "Audit summary identifiability or initialization without training; "
            "do not open K1-AT."
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_k1as_structure_gate_runtime_ready"
        next_action = (
            "Open one local K1-AT 2048/class/cipher, 4-pair, replica0/1, "
            "10-epoch comparison against the K1-AO equal-loss anchor."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": failed_protocol_checks,
        "failed_research_checks": failed_research_checks,
        "remote_scale": "no",
        "next_training_authorized": status == "pass",
        "next_action": next_action,
        "blocked_actions": [
            "K1-AQ loss-scale tuning or PCGrad",
            "16 pairs, more samples, epochs, width, seeds, or remote GPU",
            "cipher IDs, per-cipher heads, adapters, MoE, or experts",
        ],
        "claim_scope": (
            "Zero-training local readiness over 24 deterministic source panels; "
            "not AUC evidence, formal scale, an attack, transfer, or SOTA evidence."
        ),
    }


def run_readiness(
    *,
    config_path: Path = CONFIG_PATH,
    output_root: Path,
    device: str = "cpu",
    project_root: Path = ROOT,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"K1-AS output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    _append_progress(output_root / "progress.jsonl", "run_started")
    config = load_and_validate_config(config_path)
    readiness, datasets, checkpoints, source_manifest, source_checks = load_authority(
        config,
        project_root=project_root,
        device=device,
    )
    structures, summary_rows, structure_checks = build_structure_controls(
        readiness=readiness,
        config=config,
    )
    geometry_checks, geometry_report = audit_candidate_geometry(
        readiness=readiness,
        config=config,
    )
    preflight = {
        "run_id": RUN_ID,
        "status": "pass" if all(source_checks.values()) and all(geometry_checks.values()) else "fail",
        "config_sha256": file_sha256(config_path),
        "source_checks": source_checks,
        "geometry_checks": geometry_checks,
        "structure_checks": structure_checks,
        "geometry": geometry_report,
        "source_checkpoint_manifest": source_manifest,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_json(
        output_root / "structure_summaries.json",
        {"run_id": RUN_ID, "rows": summary_rows, "checks": structure_checks},
    )
    rows = evaluate_panels(
        config=config,
        readiness=readiness,
        datasets=datasets,
        checkpoints=checkpoints,
        structures=structures,
        device=device,
    )
    gate = adjudicate(
        source_checks=source_checks,
        geometry_checks=geometry_checks,
        structure_checks=structure_checks,
        rows=rows,
        config=config,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "result_rows": len(rows),
        "expected_rows": EXPECTED_PANELS,
        "optimizer_steps": 0,
        "errors": gate["failed_protocol_checks"],
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "failed_research_checks": gate["failed_research_checks"],
        "next_training_authorized": gate["next_training_authorized"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    return {
        "preflight": preflight,
        "results": rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _gate_gradient_metrics(
    model: torch.nn.Module,
    structure: Any,
    summary: torch.Tensor,
) -> dict[str, Any]:
    parameters = tuple(model.backbone.structure_gate.parameters())
    gate = model.effective_transition_gate(
        structure,
        summary=summary,
        enabled=True,
    )
    gradients = torch.autograd.grad(gate, parameters, allow_unused=False)
    squared_norm = sum(torch.sum(torch.square(gradient)) for gradient in gradients)
    return {
        "structure_gate_gradient_norm": float(torch.sqrt(squared_norm).detach()),
        "all_structure_gate_gradients_finite": all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "structure_gate_gradient_tensor_count": len(gradients),
    }


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, **fields: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "event": event,
                    "timestamp_unix": time.time(),
                    **fields,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_MISSING_KEYS",
    "EXPECTED_PANELS",
    "RUN_ID",
    "adjudicate",
    "audit_candidate_geometry",
    "build_candidate",
    "build_structure_controls",
    "evaluate_panels",
    "load_and_validate_config",
    "load_authority",
    "run_readiness",
]
