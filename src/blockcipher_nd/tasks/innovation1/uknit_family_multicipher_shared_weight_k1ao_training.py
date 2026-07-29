from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    RUN_ID as READINESS_RUN_ID,
    adjudicate_readiness,
    audit_shared_runtime,
    audit_source_datasets,
    build_runtime_model,
    load_and_validate_config as load_readiness_config,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_multicipher_shared_weight_k1ao_"
    "2048_replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_multicipher_shared_weight_k1ao_"
    "2048_replica0_replica1_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "d6de41297fbd917c94587c9f32495d1f83e28be3000fc8bb6eeb4585f41327b8"
)
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
CONTROL_CONDITIONS = (
    "correct_runtime",
    "wrong_sbox_same_checkpoint",
    "transition_branch_off_same_checkpoint",
)
EXPECTED_REPLICAS = (0, 1)
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_BATCHES_PER_CIPHER = 64
EXPECTED_STEPS_PER_EPOCH = 192
EXPECTED_STEPS_PER_REPLICA = 1_920
EXPECTED_EVALUATION_ROWS = 36
ANCHOR_RETENTION_MARGIN = -0.010
SEMANTIC_MARGIN = 0.005
BRANCH_MARGIN = 0.005


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("K1-AO training config must be a JSON object")
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AO training config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AO training identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_multicipher_shared_weight_k1ao"
    ):
        raise ValueError("K1-AO training experiment name drifted")
    replicas = config.get("replicas", [])
    if replicas != [
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
        raise ValueError("K1-AO replica binding drifted")
    expected_training = {
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
    }
    if config.get("training") != expected_training:
        raise ValueError("K1-AO training protocol drifted")
    if config.get("evaluation") != {
        "splits": list(FRESH_SPLITS),
        "conditions": list(CONTROL_CONDITIONS),
        "expected_rows": EXPECTED_EVALUATION_ROWS,
        "optimizer_steps": 0,
    }:
        raise ValueError("K1-AO evaluation protocol drifted")
    if config.get("gates") != {
        "anchor_retention_margin": ANCHOR_RETENTION_MARGIN,
        "correct_minus_wrong_sbox_margin": SEMANTIC_MARGIN,
        "correct_minus_branch_off_margin": BRANCH_MARGIN,
        "require_every_cipher_replica_split": True,
        "allow_macro_average_to_rescue_failure": False,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AO gate drifted")
    return config


def load_sources(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int, str], DiskDifferentialDataset],
    dict[tuple[str, int, str], float],
    dict[str, bool],
]:
    readiness = config["readiness"]
    readiness_root = project_root / str(readiness["root"])
    readiness_paths = {
        name: readiness_root / name for name in readiness["digests"]
    }
    readiness_digests = {
        name: file_sha256(path) for name, path in readiness_paths.items()
    }
    readiness_gate = _read_json(readiness_paths["gate.json"])
    readiness_validation = _read_json(readiness_paths["validation.json"])
    readiness_config = load_readiness_config(project_root / str(readiness["config"]))
    dataset_rows, datasets, source_checks = audit_source_datasets(
        readiness_config,
        project_root=project_root,
    )
    _runtime_rows, _result_rows, runtime_checks = audit_shared_runtime(
        readiness_config,
        datasets,
    )
    replay_gate = adjudicate_readiness(
        config=readiness_config,
        source_checks=source_checks,
        runtime_checks=runtime_checks,
    )
    anchors, anchor_checks = _load_anchors(config, project_root=project_root)
    checks = {
        "readiness_artifact_digests_exact": readiness_digests
        == readiness["digests"],
        "readiness_gate_pass_exact": (
            readiness_gate.get("run_id") == READINESS_RUN_ID
            and readiness_gate.get("status") == "pass"
            and readiness_gate.get("decision")
            == "innovation1_uknit_family_k1ao_shared_weight_runtime_ready"
            and not readiness_gate.get("failed_protocol_checks")
            and not readiness_gate.get("failed_evidence_checks")
        ),
        "readiness_validation_pass_exact": (
            readiness_validation.get("run_id") == READINESS_RUN_ID
            and readiness_validation.get("status") == "pass"
            and not readiness_validation.get("errors")
        ),
        "readiness_replay_pass_exact": replay_gate.get("status") == "pass",
        "eighteen_datasets_rebound": len(datasets) == 18,
        **{f"readiness_{name}": value for name, value in source_checks.items()},
        **anchor_checks,
    }
    return readiness_config, dataset_rows, datasets, anchors, checks


def train_shared_replicas(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    output_root: Path,
    device: str = "cpu",
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    model_config = readiness_config["model"]
    result_rows: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    history_rows: list[dict[str, Any]] = []
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        initialization_seed = int(replica_config["initialization_seed"])
        torch.manual_seed(initialization_seed)
        model = build_runtime_model(
            cipher_configs[EXPECTED_CIPHERS[0]],
            model_config,
        ).to(device)
        correct_structures = {
            cipher_key: build_runtime_model(cipher_configs[cipher_key], model_config)
            .runtime_structure
            for cipher_key in EXPECTED_CIPHERS
        }
        initial_state_sha256 = tensor_mapping_sha256(model.state_dict())
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
                cipher_key: np.random.default_rng(
                    initialization_seed * 1000
                    + epoch * 10
                    + list(EXPECTED_CIPHERS).index(cipher_key)
                ).permutation(4096)
                for cipher_key in EXPECTED_CIPHERS
            }
            loss_sums = {cipher_key: 0.0 for cipher_key in EXPECTED_CIPHERS}
            batch_counts = {cipher_key: 0 for cipher_key in EXPECTED_CIPHERS}
            for batch_index in range(EXPECTED_BATCHES_PER_CIPHER):
                for cipher_key in EXPECTED_CIPHERS:
                    seed = int(replica_config["dataset_seeds"][cipher_key])
                    dataset = datasets[(cipher_key, seed, "train_seen")]
                    indices = permutations[cipher_key][
                        batch_index * EXPECTED_BATCH_SIZE :
                        (batch_index + 1) * EXPECTED_BATCH_SIZE
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
                        correct_structures[cipher_key],
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                    )
                    loss = nn.functional.mse_loss(torch.sigmoid(logits), labels)
                    loss.backward()
                    optimizer.step()
                    step_count += 1
                    loss_sums[cipher_key] += float(loss.detach().cpu())
                    batch_counts[cipher_key] += 1

            validation_aucs = {
                cipher_key: evaluate_runtime_auc(
                    model=model,
                    dataset=datasets[
                        (
                            cipher_key,
                            int(replica_config["dataset_seeds"][cipher_key]),
                            "cross_key_validation",
                        )
                    ],
                    structure=correct_structures[cipher_key],
                    apply_sboxes=True,
                    transition_branch_enabled=True,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )[0]
                for cipher_key in EXPECTED_CIPHERS
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
            history_row = {
                "run_id": RUN_ID,
                "replica": replica,
                "epoch": epoch,
                "optimizer_steps": step_count,
                "minimum_cross_key_auc": min_auc,
                "mean_cross_key_auc": mean_auc,
                **{
                    f"{cipher_key}_train_loss": (
                        loss_sums[cipher_key] / batch_counts[cipher_key]
                    )
                    for cipher_key in EXPECTED_CIPHERS
                },
                **{
                    f"{cipher_key}_cross_key_auc": validation_aucs[cipher_key]
                    for cipher_key in EXPECTED_CIPHERS
                },
            }
            history_rows.append(history_row)
            _append_progress(
                output_root / "progress.jsonl",
                "epoch_done",
                **history_row,
            )

        if best_state is None:
            raise RuntimeError("K1-AO shared training did not select a checkpoint")
        model.load_state_dict(best_state, strict=True)
        selected_state_sha256 = tensor_mapping_sha256(model.state_dict())
        checkpoint_path = checkpoint_root / f"replica{replica}_best.pt"
        torch.save(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "initialization_seed": initialization_seed,
                "dataset_seeds": dict(replica_config["dataset_seeds"]),
                "best_epoch": best_epoch,
                "best_minimum_cross_key_auc": best_min_auc,
                "best_mean_cross_key_auc": best_mean_auc,
                "best_cross_key_aucs": best_aucs,
                "optimizer_steps": step_count,
                "state_dict": best_state,
            },
            checkpoint_path,
        )
        optimizer_steps = _optimizer_step_range(optimizer)
        checkpoint = {
            "run_id": RUN_ID,
            "replica": replica,
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "state_dict_sha256": selected_state_sha256,
            "strict_state_dict_load": True,
            "best_epoch": best_epoch,
            "optimizer_steps": step_count,
        }
        checkpoints[replica] = {
            **checkpoint,
            "state_dict": deepcopy(best_state),
        }
        result_rows.append(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "model": "runtime_spn_ct_k1ak_sbox_transition_true",
                "shared_ciphers": list(EXPECTED_CIPHERS),
                "initialization_seed": initialization_seed,
                "dataset_seeds": dict(replica_config["dataset_seeds"]),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "initial_state_sha256": initial_state_sha256,
                "selected_state_sha256": selected_state_sha256,
                "best_epoch": best_epoch,
                "metrics": {
                    "minimum_cross_key_auc": best_min_auc,
                    "mean_cross_key_auc": best_mean_auc,
                    "cross_key_auc_by_cipher": best_aucs,
                },
                "training": {
                    "epochs": EXPECTED_EPOCHS,
                    "batch_size": EXPECTED_BATCH_SIZE,
                    "batches_per_cipher_per_epoch": EXPECTED_BATCHES_PER_CIPHER,
                    "optimizer": "adam",
                    "loss": "mse",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                    "optimizer_steps": step_count,
                    "optimizer_state_step_min": optimizer_steps[0],
                    "optimizer_state_step_max": optimizer_steps[1],
                    "checkpoint_metric": "minimum_cross_key_auc_across_ciphers",
                    "selected_checkpoint": "best",
                    "checkpoint_output": str(checkpoint_path),
                    "one_shared_optimizer": True,
                    "equal_batches_per_cipher": True,
                },
                "negative_mode": "encrypted_random_plaintexts",
                "pairs_per_sample": 4,
                "samples_per_class_per_cipher": 2048,
                "validation_samples_per_class_per_cipher": 1024,
            }
        )
    return result_rows, checkpoints, history_rows


def evaluate_same_checkpoint_panel(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    anchors: Mapping[tuple[str, int, str], float],
    checkpoints: Mapping[int, Mapping[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    model_config = readiness_config["model"]
    correct_structures = {
        cipher_key: build_runtime_model(cipher_configs[cipher_key], model_config)
        .runtime_structure
        for cipher_key in EXPECTED_CIPHERS
    }
    wrong_structures = {
        cipher_key: build_runtime_model(
            cipher_configs[cipher_key],
            model_config,
            wrong_sbox=True,
        ).runtime_structure
        for cipher_key in EXPECTED_CIPHERS
    }
    rows: list[dict[str, Any]] = []
    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        model = build_runtime_model(
            cipher_configs[EXPECTED_CIPHERS[0]],
            model_config,
        ).to(device)
        model.load_state_dict(checkpoints[replica]["state_dict"], strict=True)
        state_sha256 = tensor_mapping_sha256(model.state_dict())
        for cipher_key in EXPECTED_CIPHERS:
            seed = int(replica_config["dataset_seeds"][cipher_key])
            for split in FRESH_SPLITS:
                dataset = datasets[(cipher_key, seed, split)]
                condition_aucs: dict[str, float] = {}
                condition_probability_hashes: dict[str, str] = {}
                state_before = tensor_mapping_sha256(model.state_dict())
                for condition in CONTROL_CONDITIONS:
                    structure = (
                        wrong_structures[cipher_key]
                        if condition == "wrong_sbox_same_checkpoint"
                        else correct_structures[cipher_key]
                    )
                    transition_branch_enabled = (
                        condition != "transition_branch_off_same_checkpoint"
                    )
                    auc, probabilities = evaluate_runtime_auc(
                        model=model,
                        dataset=dataset,
                        structure=structure,
                        apply_sboxes=True,
                        transition_branch_enabled=transition_branch_enabled,
                        batch_size=EXPECTED_BATCH_SIZE,
                        device=device,
                    )
                    condition_aucs[condition] = auc
                    condition_probability_hashes[condition] = _array_sha256(
                        probabilities
                    )
                state_after = tensor_mapping_sha256(model.state_dict())
                anchor_auc = float(anchors[(cipher_key, seed, split)])
                correct_auc = condition_aucs["correct_runtime"]
                for condition in CONTROL_CONDITIONS:
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "replica": replica,
                            "cipher_key": cipher_key,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "auc": condition_aucs[condition],
                            "anchor_auc": anchor_auc,
                            "correct_minus_anchor_auc": correct_auc - anchor_auc,
                            "correct_minus_condition_auc": (
                                0.0
                                if condition == "correct_runtime"
                                else correct_auc - condition_aucs[condition]
                            ),
                            "rows": int(dataset.features.shape[0]),
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "probabilities_sha256": condition_probability_hashes[
                                condition
                            ],
                            "checkpoint_sha256": checkpoints[replica]["sha256"],
                            "state_dict_sha256": state_sha256,
                            "state_immutable_across_controls": (
                                state_before == state_after == state_sha256
                            ),
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def evaluate_runtime_auc(
    *,
    model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    structure: Any,
    apply_sboxes: bool,
    transition_branch_enabled: bool,
    batch_size: int,
    device: str,
) -> tuple[float, np.ndarray]:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset.labels), batch_size):
            features = torch.as_tensor(
                np.array(dataset.features[start : start + batch_size], copy=True),
                dtype=torch.float32,
                device=device,
            )
            logits = model.logits_with_runtime(
                features,
                structure,
                apply_sboxes=apply_sboxes,
                transition_branch_enabled=transition_branch_enabled,
            )
            outputs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    probabilities = np.concatenate(outputs).astype(np.float64, copy=False)
    auc = binary_auc(
        np.asarray(dataset.labels, dtype=np.float32),
        probabilities,
    )
    return float(auc), probabilities


def adjudicate_training(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_panels = {
        (replica, cipher_key, split)
        for replica in EXPECTED_REPLICAS
        for cipher_key in EXPECTED_CIPHERS
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
        "two_training_rows_complete": len(training_rows) == 2
        and {int(row["replica"]) for row in training_rows} == set(EXPECTED_REPLICAS),
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
        "one_shared_optimizer_equal_batches": all(
            row.get("training", {}).get("one_shared_optimizer") is True
            and row.get("training", {}).get("equal_batches_per_cipher") is True
            for row in training_rows
        ),
        "two_checkpoints_complete": set(checkpoints) == set(EXPECTED_REPLICAS)
        and all(
            Path(str(checkpoints[replica]["path"])).is_file()
            and file_sha256(Path(str(checkpoints[replica]["path"])))
            == checkpoints[replica]["sha256"]
            for replica in EXPECTED_REPLICAS
        ),
        "thirty_six_evaluation_rows_complete": (
            len(evaluation_rows) == EXPECTED_EVALUATION_ROWS
            and set(grouped) == expected_panels
            and all(set(rows) == set(CONTROL_CONDITIONS) for rows in grouped.values())
        ),
        "evaluation_has_zero_optimizer_steps": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            for row in evaluation_rows
        ),
        "same_checkpoint_state_immutable": all(
            row.get("state_immutable_across_controls") is True
            and row.get("strict_state_dict_load") is True
            for row in evaluation_rows
        ),
    }
    research_checks: dict[str, bool] = {}
    panel_results: dict[str, Any] = {}
    for replica, cipher_key, split in sorted(expected_panels):
        rows = grouped.get((replica, cipher_key, split), {})
        correct = rows.get("correct_runtime", {})
        wrong = rows.get("wrong_sbox_same_checkpoint", {})
        branch_off = rows.get("transition_branch_off_same_checkpoint", {})
        correct_auc = float(correct.get("auc", math.nan))
        anchor_auc = float(correct.get("anchor_auc", math.nan))
        wrong_auc = float(wrong.get("auc", math.nan))
        branch_off_auc = float(branch_off.get("auc", math.nan))
        retention = correct_auc - anchor_auc >= ANCHOR_RETENTION_MARGIN
        semantic = correct_auc - wrong_auc >= SEMANTIC_MARGIN
        branch = correct_auc - branch_off_auc >= BRANCH_MARGIN
        prefix = f"replica{replica}_{cipher_key}_{split}"
        research_checks[f"{prefix}_retains_anchor"] = retention
        research_checks[f"{prefix}_beats_wrong_sbox"] = semantic
        research_checks[f"{prefix}_beats_branch_off"] = branch
        panel_results[prefix] = {
            "correct_auc": correct_auc,
            "anchor_auc": anchor_auc,
            "wrong_sbox_auc": wrong_auc,
            "branch_off_auc": branch_off_auc,
            "correct_minus_anchor": correct_auc - anchor_auc,
            "correct_minus_wrong_sbox": correct_auc - wrong_auc,
            "correct_minus_branch_off": correct_auc - branch_off_auc,
            "retention_pass": retention,
            "semantic_pass": semantic,
            "branch_pass": branch,
        }
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_research = [name for name, passed in research_checks.items() if not passed]
    retention_pass = all(
        value for name, value in research_checks.items() if name.endswith("retains_anchor")
    )
    semantic_pass = all(
        value for name, value in research_checks.items() if name.endswith("beats_wrong_sbox")
    )
    branch_pass = all(
        value for name, value in research_checks.items() if name.endswith("beats_branch_off")
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1ao_shared_training_protocol_invalid"
        next_action = "Repair only the failed protocol binding and rerun without interpreting AUC."
    elif retention_pass and semantic_pass and branch_pass:
        status = "pass"
        decision = "innovation1_uknit_family_k1ao_shared_semantic_training_supported"
        next_action = (
            "Prepare a separate 65536/class/cipher remote-readiness audit with "
            "disk-backed caches; do not launch until that audit passes."
        )
    elif retention_pass:
        status = "hold"
        decision = "innovation1_uknit_family_k1ao_shared_weights_operator_insensitive"
        next_action = (
            "Keep the retained signal but stop scale; audit whether the K1-AK "
            "transition summary is semantically identifiable across all three runtimes."
        )
    elif semantic_pass and branch_pass:
        status = "hold"
        decision = "innovation1_uknit_family_k1ao_shared_capacity_or_gradient_conflict"
        next_action = (
            "Measure per-cipher gradient conflict at the selected checkpoints before "
            "considering one minimal conflict-aware optimizer; do not add experts."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1ao_shared_training_retention_and_semantics_failed"
        next_action = (
            "Discard current shared training and return to representation design; "
            "do not increase pairs, samples, epochs, width, or use remote GPUs."
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
        "retention_all_panels": retention_pass,
        "wrong_sbox_margin_all_panels": semantic_pass,
        "branch_margin_all_panels": branch_pass,
        "remote_scale": "no",
        "claim_scope": (
            "Local 2048/class/cipher two-replica diagnostic only; not formal "
            "scale, an attack, arbitrary-SPN generalization, or SOTA evidence."
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
    readiness_config, dataset_rows, datasets, anchors, source_checks = load_sources(
        config,
        project_root=project_root,
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-AO training source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "training": dict(config["training"]),
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)

    training_rows, checkpoints_with_states, history_rows = train_shared_replicas(
        config=config,
        readiness_config=readiness_config,
        datasets=datasets,
        output_root=output_root,
        device=device,
    )
    evaluation_rows = evaluate_same_checkpoint_panel(
        config=config,
        readiness_config=readiness_config,
        datasets=datasets,
        anchors=anchors,
        checkpoints=checkpoints_with_states,
        device=device,
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {key: value for key, value in checkpoints_with_states[replica].items() if key != "state_dict"}
            for replica in EXPECTED_REPLICAS
        ],
    }
    gate = adjudicate_training(
        config=config,
        source_checks=source_checks,
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
        "retention_all_panels": gate["retention_all_panels"],
        "wrong_sbox_margin_all_panels": gate[
            "wrong_sbox_margin_all_panels"
        ],
        "branch_margin_all_panels": gate["branch_margin_all_panels"],
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


def _load_anchors(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[tuple[str, int, str], float], dict[str, bool]]:
    anchors: dict[tuple[str, int, str], float] = {}
    digests_exact = True
    for cipher_key, source in config["anchor_sources"].items():
        path = project_root / str(source["path"])
        digests_exact &= file_sha256(path) == source["sha256"]
        for row in _read_jsonl(path):
            key = (
                str(row.get("cipher_key")),
                int(row.get("seed", -1)),
                str(row.get("split")),
            )
            if (
                key[0] == cipher_key
                and key[2] in FRESH_SPLITS
                and row.get("condition") == source["condition"]
            ):
                if key in anchors:
                    raise ValueError(f"duplicate K1-AO anchor: {key}")
                anchors[key] = float(row["auc"])
    expected = {
        ("uknit64", seed, split)
        for seed in (3, 4)
        for split in FRESH_SPLITS
    } | {
        ("midori64", seed, split)
        for seed in (6, 7)
        for split in FRESH_SPLITS
    } | {
        ("dialga128", seed, split)
        for seed in (0, 1)
        for split in FRESH_SPLITS
    }
    return anchors, {
        "three_anchor_artifact_digests_exact": digests_exact,
        "twelve_fresh_anchor_rows_exact": set(anchors) == expected,
        "all_anchor_aucs_finite": all(math.isfinite(value) for value in anchors.values()),
    }


def _optimizer_step_range(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    steps = []
    for state in optimizer.state.values():
        if "step" in state:
            step = state["step"]
            steps.append(int(step.item() if torch.is_tensor(step) else step))
    if not steps:
        return 0, 0
    return min(steps), max(steps)


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

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
        raise ValueError("K1-AO history rows are empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "replica",
        "cipher_key",
        "seed",
        "split",
        "condition",
        "auc",
        "anchor_auc",
        "correct_minus_anchor_auc",
        "correct_minus_condition_auc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


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
        raise ValueError("K1-AO training output already exists")


__all__ = [
    "ANCHOR_RETENTION_MARGIN",
    "BRANCH_MARGIN",
    "CONFIG_PATH",
    "CONTROL_CONDITIONS",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_STEPS_PER_REPLICA",
    "FRESH_SPLITS",
    "RUN_ID",
    "SEMANTIC_MARGIN",
    "adjudicate_training",
    "evaluate_runtime_auc",
    "evaluate_same_checkpoint_panel",
    "load_and_validate_config",
    "load_sources",
    "run_training",
    "train_shared_replicas",
]
