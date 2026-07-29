from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
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
    _optimizer_step_range,
    derive_structure_controls,
    load_and_validate_config as load_k1at_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1ay import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_ENTRIES,
    build_candidate,
    load_and_validate_config as load_k1ay_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    build_candidate as build_k1aw_candidate,
    migrate_k1at_state,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1aw import (
    CONTROL_CONDITIONS,
    EXPECTED_BATCHES_PER_CIPHER,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_STEPS_PER_EPOCH,
    EXPECTED_STEPS_PER_REPLICA,
    evaluate_descriptor_auc,
    load_and_validate_config as load_k1aw_config,
    load_sources as load_k1aw_sources,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_component_separated_structure_gate_k1az_"
    "2048_replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_component_separated_structure_gate_"
    "k1az_2048_replica0_replica1_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "c61b20f208fc608c6e5cf8f4eb9ec6110dc90dae4a22756020600ef1281b9675"
)
EXPECTED_REPLICAS = (0, 1)
MACRO_IMPROVEMENT = 0.0
NO_HARM_MARGIN = -0.005
MISMATCH_MARGIN = 0.001
MINIMUM_PASSING_MISMATCH_PANELS = 10


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AZ config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AZ identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_component_separated_structure_gate_k1az"
    ):
        raise ValueError("K1-AZ experiment name drifted")
    if config.get("model") != {
        "model_key": "runtime_spn_ct_k1ay_component_separated_structure_gate_true",
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "expected_state_dict_entries": EXPECTED_STATE_ENTRIES,
        "component_separation_enabled": True,
        "initial_state": "exact_k1aw_pretraining_state",
    }:
        raise ValueError("K1-AZ model contract drifted")
    if config.get("replicas") != [
        {
            "replica": 0,
            "initialization_seed": 30,
            "dataset_seeds": {"uknit64": 3, "midori64": 6, "dialga128": 0},
        },
        {
            "replica": 1,
            "initialization_seed": 31,
            "dataset_seeds": {"uknit64": 4, "midori64": 7, "dialga128": 1},
        },
    ]:
        raise ValueError("K1-AZ replica binding drifted")
    if config.get("training") != {
        "samples_per_class_per_cipher": 2048,
        "fresh_samples_per_class_per_cipher": 1024,
        "pairs_per_sample": 4,
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "equal_batches_per_cipher_per_epoch": EXPECTED_BATCHES_PER_CIPHER,
        "optimizer_steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "optimizer_steps_total_per_replica": EXPECTED_STEPS_PER_REPLICA,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "checkpoint_metric": "minimum_cross_key_auc_across_ciphers",
        "negative_mode": "encrypted_random_plaintexts",
        "execution": "local_diagnostic",
    }:
        raise ValueError("K1-AZ training protocol drifted")
    controls = config.get("controls", {})
    if controls.get("mismatch_order") != {
        "uknit64": "midori64",
        "midori64": "dialga128",
        "dialga128": "uknit64",
    }:
        raise ValueError("K1-AZ mismatch order drifted")
    if {
        "splits": controls.get("splits"),
        "conditions": controls.get("conditions"),
        "expected_rows": controls.get("expected_rows"),
        "optimizer_steps": controls.get("optimizer_steps"),
    } != {
        "splits": list(FRESH_SPLITS),
        "conditions": list(CONTROL_CONDITIONS),
        "expected_rows": EXPECTED_EVALUATION_ROWS,
        "optimizer_steps": 0,
    }:
        raise ValueError("K1-AZ control protocol drifted")
    if config.get("gates") != {
        "cross_key_macro_improvement_per_replica": MACRO_IMPROVEMENT,
        "per_panel_no_harm_margin": NO_HARM_MARGIN,
        "correct_minus_mismatch_margin": MISMATCH_MARGIN,
        "minimum_passing_panels_per_mismatch": MINIMUM_PASSING_MISMATCH_PANELS,
        "require_each_cipher_replica_split_covered": True,
        "dual_path_disabled_is_supporting_control": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AZ gate drifted")
    return config


def load_sources(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int, str], DiskDifferentialDataset],
    dict[tuple[int, str, str], float],
    dict[str, bool],
]:
    readiness = config["readiness"]
    readiness_root = project_root / str(readiness["root"])
    readiness_paths = {
        name: readiness_root / name for name in readiness["digests"]
    }
    readiness_gate = _read_json(readiness_paths["gate.json"])
    readiness_validation = _read_json(readiness_paths["validation.json"])
    readiness_results = _read_jsonl(readiness_paths["results.jsonl"])
    readiness_controls = _read_jsonl(readiness_paths["controls.jsonl"])
    readiness_manifest = _read_json(readiness_paths["checkpoint_manifest.json"])
    readiness_geometry = _read_json(readiness_paths["geometry.json"])
    readiness_summaries = _read_json(readiness_paths["structure_summaries.json"])
    k1ay_config = load_k1ay_config(project_root / str(readiness["config"]))

    anchor = config["same_budget_anchor"]
    anchor_root = project_root / str(anchor["root"])
    anchor_paths = {name: anchor_root / name for name in anchor["digests"]}
    anchor_gate = _read_json(anchor_paths["gate.json"])
    anchor_validation = _read_json(anchor_paths["validation.json"])
    anchor_results = _read_jsonl(anchor_paths["results.jsonl"])
    anchor_controls = _read_jsonl(anchor_paths["controls.jsonl"])
    anchor_manifest = _read_json(anchor_paths["checkpoint_manifest.json"])
    k1aw_config = load_k1aw_config(project_root / str(anchor["config"]))
    (
        base_readiness,
        k1as_config,
        k1av_config,
        dataset_rows,
        datasets,
        _k1at_anchors,
        inherited,
    ) = load_k1aw_sources(k1aw_config, project_root=project_root)
    anchors = _extract_correct_anchors(anchor_controls)
    expected_anchor_keys = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    checks = {
        "k1ay_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == readiness["digests"][name]
            for name, path in readiness_paths.items()
        ),
        "k1ay_gate_authorizes_k1az": (
            readiness_gate.get("run_id") == readiness["run_id"]
            and readiness_gate.get("status") == "pass"
            and readiness_gate.get("decision") == readiness["required_decision"]
            and readiness_gate.get("next_training_authorized") is True
            and not readiness_gate.get("failed_protocol_checks")
            and not readiness_gate.get("failed_research_checks")
            and readiness_gate.get("remote_scale") == "no"
        ),
        "k1ay_validation_passes": (
            readiness_validation.get("run_id") == readiness["run_id"]
            and readiness_validation.get("status") == "pass"
            and not readiness_validation.get("errors")
        ),
        "k1ay_twelve_results_and_thirty_six_controls_bound": (
            len(readiness_results) == 12 and len(readiness_controls) == 36
        ),
        "k1ay_two_strict_loads_bound": (
            readiness_manifest.get("status") == "pass"
            and len(readiness_manifest.get("strict_loads", [])) == 2
            and all(
                row.get("strict_load_exact") is True
                for row in readiness_manifest.get("strict_loads", [])
            )
        ),
        "k1ay_geometry_exact": set(readiness_geometry["parameter_counts"].values())
        == {EXPECTED_PARAMETER_COUNT}
        and set(readiness_geometry["state_entries"].values())
        == {EXPECTED_STATE_ENTRIES},
        "k1ay_three_structure_summaries_bound": len(
            readiness_summaries.get("rows", [])
        )
        == 3,
        "k1aw_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == anchor["digests"][name]
            for name, path in anchor_paths.items()
        ),
        "k1aw_gate_is_valid_same_budget_hold": (
            anchor_gate.get("run_id") == anchor["run_id"]
            and anchor_gate.get("status") == "hold"
            and anchor_gate.get("decision") == anchor["required_decision"]
            and not anchor_gate.get("failed_protocol_checks")
        ),
        "k1aw_validation_passes": (
            anchor_validation.get("run_id") == anchor["run_id"]
            and anchor_validation.get("status") == "pass"
            and not anchor_validation.get("errors")
        ),
        "k1aw_two_training_rows_and_checkpoints_bound": len(anchor_results) == 2
        and len(anchor_manifest.get("entries", [])) == 2,
        "k1aw_sixty_controls_bound": len(anchor_controls) == 60,
        "k1aw_twelve_correct_descriptor_anchors_exact": set(anchors)
        == expected_anchor_keys,
        "k1aw_anchor_aucs_finite": all(math.isfinite(value) for value in anchors.values()),
        "readiness_config_model_key_exact": k1ay_config["model"]["model_key"]
        == config["model"]["model_key"],
        **{f"k1aw_source_{name}": bool(value) for name, value in inherited.items()},
    }
    return (
        base_readiness,
        k1as_config,
        k1av_config,
        dataset_rows,
        datasets,
        anchors,
        checks,
    )


def train_shared_replicas(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    k1as_config: Mapping[str, Any],
    k1av_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    structures: Mapping[str, Any],
    structure_controls: Mapping[str, Mapping[str, torch.Tensor | None]],
    output_root: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    anchor_model_config = readiness_config["model"]
    result_rows: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    history_rows: list[dict[str, Any]] = []
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        initialization_seed = int(replica_config["initialization_seed"])
        with torch.random.fork_rng():
            torch.manual_seed(initialization_seed)
            k1at_initial = build_k1as_candidate(
                cipher_configs[EXPECTED_CIPHERS[0]],
                anchor_model_config,
                k1as_config["model"],
            ).to(device)
        with torch.random.fork_rng():
            torch.manual_seed(initialization_seed)
            k1aw_initial = build_k1aw_candidate(
                cipher_configs[EXPECTED_CIPHERS[0]],
                anchor_model_config,
                k1av_config["model"],
            ).to(device)
        migration = migrate_k1at_state(k1aw_initial, k1at_initial.state_dict())
        if not all(
            (
                migration["only_final_projection_expanded"],
                migration["transition_row_exact"],
                migration["new_edge_row_finite_nonzero"],
            )
        ):
            raise ValueError(f"K1-AZ K1-AW initialization failed: {migration}")
        k1aw_initial_state = {
            name: value.detach().clone()
            for name, value in k1aw_initial.state_dict().items()
        }
        model = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            anchor_model_config,
            {"gate_hidden_dim": 12},
        ).to(device)
        incompatible = model.load_state_dict(k1aw_initial_state, strict=True)
        initial_state_sha256 = tensor_mapping_sha256(model.state_dict())
        k1aw_initial_state_sha256 = tensor_mapping_sha256(k1aw_initial_state)
        initial_alignment = {
            "k1aw_migration": migration,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "k1aw_initial_state_sha256": k1aw_initial_state_sha256,
            "candidate_initial_state_sha256": initial_state_sha256,
            "exact_k1aw_initial_state": (
                not incompatible.missing_keys
                and not incompatible.unexpected_keys
                and initial_state_sha256 == k1aw_initial_state_sha256
            ),
        }
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        step_count = 0
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_min_auc = -math.inf
        best_mean_auc = -math.inf
        best_aucs: dict[str, float] = {}

        for epoch in range(1, EXPECTED_EPOCHS + 1):
            model.train()
            permutations = {
                cipher: np.random.default_rng(
                    initialization_seed * 1000
                    + epoch * 10
                    + list(EXPECTED_CIPHERS).index(cipher)
                ).permutation(4096)
                for cipher in EXPECTED_CIPHERS
            }
            loss_sums = {cipher: 0.0 for cipher in EXPECTED_CIPHERS}
            batch_counts = {cipher: 0 for cipher in EXPECTED_CIPHERS}
            for batch_index in range(EXPECTED_BATCHES_PER_CIPHER):
                for cipher in EXPECTED_CIPHERS:
                    seed = int(replica_config["dataset_seeds"][cipher])
                    dataset = datasets[(cipher, seed, "train_seen")]
                    indices = permutations[cipher][
                        batch_index * EXPECTED_BATCH_SIZE : (batch_index + 1)
                        * EXPECTED_BATCH_SIZE
                    ]
                    features = torch.as_tensor(
                        np.array(dataset.features[indices], copy=True),
                        dtype=torch.float32,
                        device=device,
                    )
                    labels = torch.as_tensor(
                        np.array(dataset.labels[indices], copy=True),
                        dtype=torch.float32,
                        device=device,
                    ).reshape(-1, 1)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.logits_with_runtime(
                        features,
                        structures[cipher],
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                        gate_summary=structure_controls[cipher]["correct_descriptor"],
                        dual_path_enabled=True,
                        component_separation_enabled=True,
                    )
                    loss = nn.functional.mse_loss(torch.sigmoid(logits), labels)
                    loss.backward()
                    optimizer.step()
                    step_count += 1
                    loss_sums[cipher] += float(loss.detach().cpu())
                    batch_counts[cipher] += 1

            validation_aucs = {
                cipher: evaluate_descriptor_auc(
                    model=model,
                    dataset=datasets[
                        (
                            cipher,
                            int(replica_config["dataset_seeds"][cipher]),
                            "cross_key_validation",
                        )
                    ],
                    structure=structures[cipher],
                    summary=structure_controls[cipher]["correct_descriptor"],
                    dual_path_enabled=True,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )[0]
                for cipher in EXPECTED_CIPHERS
            }
            min_auc = min(validation_aucs.values())
            mean_auc = float(np.mean(tuple(validation_aucs.values())))
            if min_auc > best_min_auc or (
                min_auc == best_min_auc and mean_auc > best_mean_auc
            ):
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_epoch = epoch
                best_min_auc = min_auc
                best_mean_auc = mean_auc
                best_aucs = dict(validation_aucs)
            history_rows.append(
                {
                    "run_id": RUN_ID,
                    "replica": replica,
                    "epoch": epoch,
                    "optimizer_steps": step_count,
                    **{
                        f"train_loss_{cipher}": loss_sums[cipher]
                        / batch_counts[cipher]
                        for cipher in EXPECTED_CIPHERS
                    },
                    **{
                        f"cross_key_auc_{cipher}": validation_aucs[cipher]
                        for cipher in EXPECTED_CIPHERS
                    },
                    "minimum_cross_key_auc": min_auc,
                    "mean_cross_key_auc": mean_auc,
                }
            )
            _append_progress(
                output_root / "progress.jsonl",
                "epoch_done",
                replica=replica,
                epoch=epoch,
                optimizer_steps=step_count,
                minimum_cross_key_auc=min_auc,
                mean_cross_key_auc=mean_auc,
            )

        if best_state is None:
            raise RuntimeError("K1-AZ failed to select a checkpoint")
        model.load_state_dict(best_state, strict=True)
        checkpoint_path = checkpoint_root / f"replica{replica}_best.pt"
        torch.save(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "best_epoch": best_epoch,
                "state_dict": best_state,
                "initial_state_sha256": initial_state_sha256,
                "k1aw_initial_state_sha256": k1aw_initial_state_sha256,
            },
            checkpoint_path,
        )
        checkpoint_sha256 = file_sha256(checkpoint_path)
        state_dict_sha256 = tensor_mapping_sha256(best_state)
        checkpoints[replica] = {
            "replica": replica,
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "state_dict_sha256": state_dict_sha256,
            "best_epoch": best_epoch,
            "best_minimum_cross_key_auc": best_min_auc,
            "best_mean_cross_key_auc": best_mean_auc,
            "best_cross_key_aucs": best_aucs,
            "initial_state_sha256": initial_state_sha256,
            "k1aw_initial_state_sha256": k1aw_initial_state_sha256,
            "state_dict": best_state,
        }
        optimizer_min, optimizer_max = _optimizer_step_range(optimizer)
        metadata = model_metadata(model)
        result_rows.append(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "initialization_seed": initialization_seed,
                "dataset_seeds": dict(replica_config["dataset_seeds"]),
                "initial_alignment": initial_alignment,
                "initial_state_sha256": initial_state_sha256,
                "selected_checkpoint": {
                    "epoch": best_epoch,
                    "minimum_cross_key_auc": best_min_auc,
                    "mean_cross_key_auc": best_mean_auc,
                    "cross_key_aucs": best_aucs,
                    "checkpoint_sha256": checkpoint_sha256,
                    "state_dict_sha256": state_dict_sha256,
                },
                "training": {
                    "epochs": EXPECTED_EPOCHS,
                    "optimizer_steps": step_count,
                    "optimizer_state_step_min": optimizer_min,
                    "optimizer_state_step_max": optimizer_max,
                    "one_shared_optimizer": True,
                    "equal_batches_per_cipher": all(
                        row["equal_batches_per_cipher_per_epoch"]
                        == EXPECTED_BATCHES_PER_CIPHER
                        for row in (config["training"],)
                    ),
                    "correct_summary_precomputed_once_per_cipher": True,
                    "component_separation_enabled": True,
                    "loss": "mse",
                    "learning_rate": float(config["training"]["learning_rate"]),
                    "weight_decay": float(config["training"]["weight_decay"]),
                },
                "trainable_parameter_count": int(metadata["trainable_parameter_count"]),
                "state_dict_entries": len(model.state_dict()),
                "uses_cipher_identity": bool(model.uses_cipher_identity),
                "structure_gate_uses_cipher_identity": bool(
                    model.structure_gate_uses_cipher_identity
                ),
                "structure_gate_shared": bool(model.structure_gate_shared),
                "negative_mode": "encrypted_random_plaintexts",
                "pairs_per_sample": 4,
                "samples_per_class_per_cipher": 2048,
                "validation_samples_per_class_per_cipher": 1024,
            }
        )
    return result_rows, checkpoints, history_rows


def evaluate_same_checkpoint_panels(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    anchors: Mapping[tuple[int, str, str], float],
    structures: Mapping[str, Any],
    structure_controls: Mapping[str, Mapping[str, torch.Tensor | None]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    rows: list[dict[str, Any]] = []
    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        model = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            readiness_config["model"],
            {"gate_hidden_dim": 12},
        ).to(device)
        model.load_state_dict(checkpoints[replica]["state_dict"], strict=True)
        model.eval()
        state_sha256 = tensor_mapping_sha256(model.state_dict())
        for cipher in EXPECTED_CIPHERS:
            seed = int(replica_config["dataset_seeds"][cipher])
            for split in FRESH_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                condition_aucs: dict[str, float] = {}
                probability_hashes: dict[str, str] = {}
                gate_values: dict[str, dict[str, float]] = {}
                state_before = tensor_mapping_sha256(model.state_dict())
                for condition in CONTROL_CONDITIONS:
                    dual_enabled = condition != "dual_path_disabled"
                    summary = (
                        structure_controls[cipher]["correct_descriptor"]
                        if condition == "dual_path_disabled"
                        else structure_controls[cipher][condition]
                    )
                    auc, probabilities = evaluate_descriptor_auc(
                        model=model,
                        dataset=dataset,
                        structure=structures[cipher],
                        summary=summary,
                        dual_path_enabled=dual_enabled,
                        batch_size=EXPECTED_BATCH_SIZE,
                        device=device,
                    )
                    condition_aucs[condition] = auc
                    probability_hashes[condition] = _array_sha256(probabilities)
                    gate_values[condition] = _path_gate_values(
                        model,
                        structures[cipher],
                        summary,
                        dual_path_enabled=dual_enabled,
                    )
                state_after = tensor_mapping_sha256(model.state_dict())
                anchor_auc = float(anchors[(replica, cipher, split)])
                correct_auc = condition_aucs["correct_descriptor"]
                for condition in CONTROL_CONDITIONS:
                    summary = (
                        structure_controls[cipher]["correct_descriptor"]
                        if condition == "dual_path_disabled"
                        else structure_controls[cipher][condition]
                    )
                    if summary is None:
                        raise ValueError("K1-AZ summary cannot be absent")
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "replica": replica,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "auc": condition_aucs[condition],
                            "k1aw_anchor_auc": anchor_auc,
                            "correct_minus_k1aw_auc": correct_auc - anchor_auc,
                            "correct_minus_condition_auc": (
                                0.0
                                if condition == "correct_descriptor"
                                else correct_auc - condition_aucs[condition]
                            ),
                            "effective_edge_gate": gate_values[condition]["edge"],
                            "effective_transition_gate": gate_values[condition][
                                "transition"
                            ],
                            "runtime_structure_cipher_key": cipher,
                            "runtime_structure_held_correct": True,
                            "descriptor_summary_sha256": _tensor_sha256(summary),
                            "dual_path_enabled": condition != "dual_path_disabled",
                            "component_separation_enabled": True,
                            "rows": int(dataset.features.shape[0]),
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "probabilities_sha256": probability_hashes[condition],
                            "checkpoint_sha256": checkpoints[replica]["sha256"],
                            "state_dict_sha256": state_sha256,
                            "state_immutable_across_controls": state_before
                            == state_after
                            == state_sha256,
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def adjudicate_training(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    structure_checks: Mapping[str, bool],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_panels = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in evaluation_rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    protocol_checks = {
        "training_config_digest_exact": file_sha256(CONFIG_PATH)
        == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "all_structure_controls_valid": bool(structure_checks)
        and all(structure_checks.values()),
        "two_training_rows_complete": len(training_rows) == 2
        and {int(row["replica"]) for row in training_rows} == set(EXPECTED_REPLICAS),
        "candidate_geometry_and_exact_k1aw_initial_state": all(
            int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
            and int(row.get("state_dict_entries", -1)) == EXPECTED_STATE_ENTRIES
            and row.get("uses_cipher_identity") is False
            and row.get("structure_gate_uses_cipher_identity") is False
            and row.get("structure_gate_shared") is True
            and row.get("initial_alignment", {}).get("exact_k1aw_initial_state")
            is True
            for row in training_rows
        ),
        "ten_epochs_and_1920_steps_each": all(
            int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("optimizer_steps", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and int(row.get("training", {}).get("optimizer_state_step_min", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and int(row.get("training", {}).get("optimizer_state_step_max", -1))
            == EXPECTED_STEPS_PER_REPLICA
            for row in training_rows
        ),
        "one_shared_optimizer_equal_batches_and_component_separation": all(
            row.get("training", {}).get("one_shared_optimizer") is True
            and row.get("training", {}).get("equal_batches_per_cipher") is True
            and row.get("training", {}).get(
                "correct_summary_precomputed_once_per_cipher"
            )
            is True
            and row.get("training", {}).get("component_separation_enabled") is True
            for row in training_rows
        ),
        "two_checkpoints_complete": set(checkpoints) == set(EXPECTED_REPLICAS)
        and all(
            Path(str(checkpoints[replica]["path"])).is_file()
            and file_sha256(Path(str(checkpoints[replica]["path"])))
            == checkpoints[replica]["sha256"]
            for replica in EXPECTED_REPLICAS
        ),
        "sixty_evaluation_rows_complete": len(evaluation_rows)
        == EXPECTED_EVALUATION_ROWS
        and set(grouped) == expected_panels
        and all(
            set(conditions) == set(CONTROL_CONDITIONS)
            for conditions in grouped.values()
        ),
        "evaluation_zero_step_same_checkpoint_immutable": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("state_immutable_across_controls") is True
            and row.get("strict_state_dict_load") is True
            and row.get("component_separation_enabled") is True
            for row in evaluation_rows
        ),
        "runtime_structure_held_correct_for_all_controls": all(
            row.get("runtime_structure_held_correct") is True
            and row.get("runtime_structure_cipher_key") == row.get("cipher_key")
            for row in evaluation_rows
        ),
        "all_gate_values_finite_bounded": all(
            math.isfinite(float(row.get("effective_edge_gate", math.nan)))
            and abs(float(row.get("effective_edge_gate", math.nan))) < 1.0
            and math.isfinite(float(row.get("effective_transition_gate", math.nan)))
            and abs(float(row.get("effective_transition_gate", math.nan))) < 1.0
            for row in evaluation_rows
        ),
    }

    panel_results: dict[str, dict[str, Any]] = {}
    no_harm_checks: dict[str, bool] = {}
    mismatch_pass_panels: dict[str, list[tuple[int, str, str]]] = {
        condition: [] for condition in MISMATCH_CONDITIONS
    }
    for replica, cipher, split in sorted(expected_panels):
        conditions = grouped.get((replica, cipher, split), {})
        correct = conditions.get("correct_descriptor", {})
        correct_auc = float(correct.get("auc", math.nan))
        anchor_auc = float(correct.get("k1aw_anchor_auc", math.nan))
        prefix = f"replica{replica}_{cipher}_{split}"
        no_harm_checks[f"{prefix}_no_harm_vs_k1aw"] = (
            correct_auc - anchor_auc >= NO_HARM_MARGIN
        )
        margins = {}
        aucs = {}
        for condition in CONTROL_CONDITIONS:
            auc = float(conditions.get(condition, {}).get("auc", math.nan))
            aucs[condition] = auc
            margins[condition] = correct_auc - auc
            if (
                condition in MISMATCH_CONDITIONS
                and margins[condition] >= MISMATCH_MARGIN
            ):
                mismatch_pass_panels[condition].append((replica, cipher, split))
        panel_results[prefix] = {
            "correct_auc": correct_auc,
            "k1aw_anchor_auc": anchor_auc,
            "correct_minus_k1aw": correct_auc - anchor_auc,
            "condition_aucs": aucs,
            "correct_minus_condition": margins,
            "no_harm_pass": no_harm_checks[f"{prefix}_no_harm_vs_k1aw"],
        }

    macro_results: dict[str, Any] = {}
    macro_checks: dict[str, bool] = {}
    for replica in EXPECTED_REPLICAS:
        candidate_values = [
            float(
                grouped[(replica, cipher, "cross_key_validation")][
                    "correct_descriptor"
                ]["auc"]
            )
            for cipher in EXPECTED_CIPHERS
        ]
        anchor_values = [
            float(
                grouped[(replica, cipher, "cross_key_validation")][
                    "correct_descriptor"
                ]["k1aw_anchor_auc"]
            )
            for cipher in EXPECTED_CIPHERS
        ]
        candidate_macro = float(np.mean(candidate_values))
        anchor_macro = float(np.mean(anchor_values))
        improvement = candidate_macro - anchor_macro
        macro_checks[f"replica{replica}_cross_key_macro_retains_k1aw"] = (
            improvement >= MACRO_IMPROVEMENT
        )
        macro_results[f"replica{replica}"] = {
            "candidate_cross_key_macro_auc": candidate_macro,
            "k1aw_cross_key_macro_auc": anchor_macro,
            "improvement": improvement,
            "pass": improvement >= MACRO_IMPROVEMENT,
        }

    mismatch_results: dict[str, Any] = {}
    mismatch_checks: dict[str, bool] = {}
    for condition, panels in mismatch_pass_panels.items():
        ciphers = {cipher for _, cipher, _ in panels}
        replicas = {replica for replica, _, _ in panels}
        splits = {split for _, _, split in panels}
        count_pass = len(panels) >= MINIMUM_PASSING_MISMATCH_PANELS
        coverage_pass = (
            ciphers == set(EXPECTED_CIPHERS)
            and replicas == set(EXPECTED_REPLICAS)
            and splits == set(FRESH_SPLITS)
        )
        mismatch_checks[f"{condition}_at_least_10_of_12"] = count_pass
        mismatch_checks[f"{condition}_covers_every_axis"] = coverage_pass
        mismatch_results[condition] = {
            "passing_panels": len(panels),
            "expected_panels": 12,
            "minimum_required": MINIMUM_PASSING_MISMATCH_PANELS,
            "covered_ciphers": sorted(ciphers),
            "covered_replicas": sorted(replicas),
            "covered_splits": sorted(splits),
            "count_pass": count_pass,
            "coverage_pass": coverage_pass,
        }

    research_checks = {**macro_checks, **no_harm_checks, **mismatch_checks}
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_research = [name for name, passed in research_checks.items() if not passed]
    macro_pass = all(macro_checks.values())
    no_harm_pass = all(no_harm_checks.values())
    mismatch_pass = all(mismatch_checks.values())
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1az_protocol_invalid"
        next_action = (
            "Repair only the failed source, initialization, data, step, checkpoint, "
            "runtime or evaluation binding and replay K1-AZ unchanged."
        )
    elif macro_pass and no_harm_pass and mismatch_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1az_component_separated_training_supported"
        )
        next_action = (
            "Retain component separation as the local family candidate and open a "
            "separate 65536/class/cipher remote cache/resume readiness audit; do "
            "not launch remotely before that gate passes."
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_k1az_component_separated_training_not_supported"
        )
        if not mismatch_pass:
            next_action = (
                "Freeze both K1-AZ checkpoints and audit which summary dimensions "
                "or residual responses preserve mismatch invariance; do not add "
                "pairs, data, epochs, experts or remote scale."
            )
        elif not no_harm_pass:
            next_action = (
                "Compare the harmed cipher's epoch-wise K1-AZ and K1-AW path "
                "contributions before changing another model variable."
            )
        else:
            next_action = (
                "Compare epoch-wise K1-AZ and K1-AW trajectories; both replicas did "
                "not retain macro AUC, so do not increase scale."
            )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "panel_results": panel_results,
        "macro_results": macro_results,
        "mismatch_results": mismatch_results,
        "cross_key_macro_retention_both_replicas": macro_pass,
        "per_panel_no_harm_all": no_harm_pass,
        "descriptor_mismatch_gate_all": mismatch_pass,
        "dual_path_disabled_role": "supporting_same_checkpoint_control",
        "remote_scale": "no",
        "claim_scope": (
            "Local 2048/class/cipher, four-pair, two-replica diagnostic only; "
            "not formal scale, an attack, arbitrary-SPN generalization, unseen-"
            "cipher transfer, or SOTA evidence."
        ),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
    }


def run_training(
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
        k1av_config,
        dataset_rows,
        datasets,
        anchors,
        source_checks,
    ) = load_sources(config, project_root=project_root)
    k1aw_config = load_k1aw_config(
        project_root / str(config["same_budget_anchor"]["config"])
    )
    k1at_config = load_k1at_config(
        project_root / str(k1aw_config["same_budget_anchor"]["config"])
    )
    structures, structure_controls, summary_rows, structure_checks = (
        derive_structure_controls(
            readiness_config=readiness_config,
            config=k1at_config,
        )
    )
    if not all(source_checks.values()) or not all(structure_checks.values()):
        raise ValueError(
            f"K1-AZ preflight failed: source={source_checks}, "
            f"structure={structure_checks}"
        )
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "structure_checks": structure_checks,
        "training": dict(config["training"]),
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    _write_json(
        output_root / "structure_summaries.json",
        {"run_id": RUN_ID, "rows": summary_rows},
    )
    training_rows, checkpoints_with_states, history_rows = train_shared_replicas(
        config=config,
        readiness_config=readiness_config,
        k1as_config=k1as_config,
        k1av_config=k1av_config,
        datasets=datasets,
        structures=structures,
        structure_controls=structure_controls,
        output_root=output_root,
        device=device,
    )
    evaluation_rows = evaluate_same_checkpoint_panels(
        config=config,
        readiness_config=readiness_config,
        datasets=datasets,
        anchors=anchors,
        structures=structures,
        structure_controls=structure_controls,
        checkpoints=checkpoints_with_states,
        device=device,
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {
                key: value
                for key, value in checkpoints_with_states[replica].items()
                if key != "state_dict"
            }
            for replica in EXPECTED_REPLICAS
        ],
    }
    gate = adjudicate_training(
        config=config,
        source_checks=source_checks,
        structure_checks=structure_checks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoints=checkpoints_with_states,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "training_rows": len(training_rows),
        "expected_training_rows": 2,
        "evaluation_rows": len(evaluation_rows),
        "expected_evaluation_rows": EXPECTED_EVALUATION_ROWS,
        "optimizer_steps_per_replica": {
            str(row["replica"]): row["training"]["optimizer_steps"]
            for row in training_rows
        },
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "cross_key_macro_retention_both_replicas": gate[
            "cross_key_macro_retention_both_replicas"
        ],
        "per_panel_no_harm_all": gate["per_panel_no_harm_all"],
        "descriptor_mismatch_gate_all": gate["descriptor_mismatch_gate_all"],
        "macro_results": gate["macro_results"],
        "mismatch_results": gate["mismatch_results"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", training_rows)
    _write_jsonl(output_root / "controls.jsonl", evaluation_rows)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_history_csv(output_root / "history.csv", history_rows)
    _write_comparison_csv(output_root / "comparison.csv", evaluation_rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        training_rows=len(training_rows),
        evaluation_rows=len(evaluation_rows),
    )
    return {
        "preflight": preflight,
        "results": training_rows,
        "controls": evaluation_rows,
        "checkpoint_manifest": checkpoint_manifest,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _extract_correct_anchors(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], float]:
    anchors = {}
    for row in rows:
        if row.get("condition") != "correct_descriptor":
            continue
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        if key in anchors:
            raise ValueError(f"duplicate K1-AW anchor: {key}")
        anchors[key] = float(row["auc"])
    return anchors


def _path_gate_values(
    model: torch.nn.Module,
    structure: Any,
    summary: torch.Tensor | None,
    *,
    dual_path_enabled: bool,
) -> dict[str, float]:
    if summary is None:
        raise ValueError("K1-AZ path gate summary cannot be absent")
    edge, transition = model.effective_path_gates(
        structure,
        summary=summary,
        dual_path_enabled=dual_path_enabled,
        component_separation_enabled=True,
    )
    return {"edge": float(edge.detach()), "transition": float(transition.detach())}


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
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


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("K1-AZ history rows are empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    fields = (
        "replica",
        "cipher_key",
        "seed",
        "split",
        "condition",
        "auc",
        "k1aw_anchor_auc",
        "correct_minus_k1aw_auc",
        "correct_minus_condition_auc",
        "effective_edge_gate",
        "effective_transition_gate",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


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
        raise ValueError("K1-AZ output already exists")


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "ROOT",
    "RUN_ID",
    "adjudicate_training",
    "evaluate_same_checkpoint_panels",
    "load_and_validate_config",
    "load_sources",
    "run_training",
    "train_shared_replicas",
]
