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
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    build_runtime_model,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao_training import (
    CONTROL_CONDITIONS,
    EXPECTED_BATCHES_PER_CIPHER,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_STEPS_PER_REPLICA,
    FRESH_SPLITS,
    RUN_ID as BASELINE_RUN_ID,
    evaluate_runtime_auc,
    evaluate_same_checkpoint_panel,
    load_and_validate_config as load_baseline_config,
    load_sources,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_multicipher_inverse_norm_k1aq_"
    "2048_replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_multicipher_inverse_norm_k1aq_"
    "2048_replica0_replica1_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "8e2a10050ec647531516c13e0f4db32dc28973c22bf5d2adb7cdb4dde6a57ea0"
)
REPLICAS = (0, 1)
TARGET_CIPHERS = ("uknit64", "midori64")
EXPECTED_EVALUATION_ROWS = 36
TARGET_IMPROVEMENT = 0.010
TARGET_PANELS_MIN = 6
NO_HARM_FLOOR = -0.010
SEMANTIC_MARGIN = 0.005
BRANCH_PANELS_MIN = 11
RETENTION_MARGIN = -0.010


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("K1-AQ config must be a JSON object")
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AQ config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AQ identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_multicipher_inverse_norm_k1aq"
    ):
        raise ValueError("K1-AQ experiment name drifted")
    expected_training = {
        "samples_per_class_per_cipher": 2048,
        "fresh_samples_per_class_per_cipher": 1024,
        "pairs_per_sample": 4,
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "equal_batches_per_cipher_per_epoch": EXPECTED_BATCHES_PER_CIPHER,
        "optimizer_steps_per_epoch": 192,
        "optimizer_steps_total_per_replica": EXPECTED_STEPS_PER_REPLICA,
        "loss": "mse",
        "loss_scaling": "fixed_inverse_k1ap_median_gradient_norm_geomean_one",
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "checkpoint_metric": "minimum_cross_key_auc_across_ciphers",
        "negative_mode": "encrypted_random_plaintexts",
        "execution": "local_diagnostic",
    }
    if config.get("training") != expected_training:
        raise ValueError("K1-AQ training protocol drifted")
    if config.get("evaluation") != {
        "splits": list(FRESH_SPLITS),
        "conditions": list(CONTROL_CONDITIONS),
        "expected_rows": EXPECTED_EVALUATION_ROWS,
        "optimizer_steps": 0,
    }:
        raise ValueError("K1-AQ evaluation protocol drifted")
    if config.get("gates") != {
        "target_ciphers": list(TARGET_CIPHERS),
        "target_improvement_min": TARGET_IMPROVEMENT,
        "target_panels_improved_min": TARGET_PANELS_MIN,
        "all_correct_panels_baseline_delta_min": NO_HARM_FLOOR,
        "correct_minus_wrong_sbox_margin": SEMANTIC_MARGIN,
        "branch_pass_count_min": BRANCH_PANELS_MIN,
        "full_anchor_retention_margin": RETENTION_MARGIN,
        "remote_scale": "no",
    }:
        raise ValueError("K1-AQ gates drifted")
    if [int(row["replica"]) for row in config.get("replicas", [])] != [0, 1]:
        raise ValueError("K1-AQ replicas drifted")
    for replica in config["replicas"]:
        scales = [float(value) for value in replica["loss_scales"].values()]
        if set(replica["loss_scales"]) != set(EXPECTED_CIPHERS):
            raise ValueError("K1-AQ loss-scale ciphers drifted")
        if not math.isclose(math.prod(scales), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("K1-AQ loss-scale geometric mean drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int, str], DiskDifferentialDataset],
    dict[tuple[str, int, str], float],
    list[dict[str, Any]],
    dict[str, bool],
]:
    gradient = config["gradient_authority"]
    gradient_root = project_root / str(gradient["root"])
    gradient_paths = {
        name: gradient_root / name for name in gradient["digests"]
    }
    baseline = config["baseline"]
    baseline_root = project_root / str(baseline["root"])
    baseline_paths = {
        name: baseline_root / name for name in baseline["digests"]
    }
    gradient_gate = _read_json(gradient_paths["gate.json"])
    gradient_validation = _read_json(gradient_paths["validation.json"])
    gradient_rows = _read_jsonl(gradient_paths["results.jsonl"])
    baseline_gate = _read_json(baseline_paths["gate.json"])
    baseline_rows = _read_jsonl(baseline_paths["controls.jsonl"])

    baseline_config = load_baseline_config()
    readiness, dataset_rows, datasets, anchors, baseline_source_checks = load_sources(
        baseline_config,
        project_root=project_root,
    )
    checks = {
        "three_gradient_authority_digests_exact": all(
            file_sha256(gradient_paths[name]) == digest
            for name, digest in gradient["digests"].items()
        ),
        "two_baseline_digests_exact": all(
            file_sha256(baseline_paths[name]) == digest
            for name, digest in baseline["digests"].items()
        ),
        "k1ap_authorizes_fixed_normalization": (
            gradient_gate.get("status") == "pass"
            and gradient_gate.get("decision")
            == "innovation1_uknit_family_k1ap_stable_gradient_norm_imbalance_supported"
            and gradient_gate.get("stable_gradient_norm_imbalance") is True
            and not gradient_gate.get("stable_conflict_pairs")
            and not gradient_gate.get("failed_protocol_checks")
        ),
        "k1ap_validation_passes": (
            gradient_validation.get("status") == "pass"
            and gradient_validation.get("summary_rows") == 72
            and gradient_validation.get("optimizer_steps") == 0
            and not gradient_validation.get("errors")
        ),
        "k1ao_baseline_is_valid_hold": (
            baseline_gate.get("run_id") == BASELINE_RUN_ID
            and baseline_gate.get("status") == "hold"
            and not baseline_gate.get("failed_protocol_checks")
        ),
        "k1ao_baseline_has_complete_36_rows": _complete_control_panels(
            baseline_rows
        ),
        "loss_scales_rederived_from_k1ap": _validate_loss_scales(
            config, gradient_rows
        ),
        **{f"k1ao_{name}": value for name, value in baseline_source_checks.items()},
    }
    return readiness, dataset_rows, datasets, anchors, baseline_rows, checks


def scaled_mse_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_loss = nn.functional.mse_loss(torch.sigmoid(logits), labels)
    return raw_loss, raw_loss * scale


def train_scaled_replicas(
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
        loss_scales = {
            key: float(value) for key, value in replica_config["loss_scales"].items()
        }
        torch.manual_seed(initialization_seed)
        model = build_runtime_model(
            cipher_configs[EXPECTED_CIPHERS[0]], model_config
        ).to(device)
        structures = {
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
            raw_loss_sums = {cipher_key: 0.0 for cipher_key in EXPECTED_CIPHERS}
            scaled_loss_sums = {cipher_key: 0.0 for cipher_key in EXPECTED_CIPHERS}
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
                        structures[cipher_key],
                        apply_sboxes=True,
                        transition_branch_enabled=True,
                    )
                    raw_loss, scaled_loss = scaled_mse_loss(
                        logits, labels, scale=loss_scales[cipher_key]
                    )
                    scaled_loss.backward()
                    optimizer.step()
                    step_count += 1
                    raw_loss_sums[cipher_key] += float(raw_loss.detach().cpu())
                    scaled_loss_sums[cipher_key] += float(scaled_loss.detach().cpu())

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
                    structure=structures[cipher_key],
                    apply_sboxes=True,
                    transition_branch_enabled=True,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )[0]
                for cipher_key in EXPECTED_CIPHERS
            }
            minimum_auc = min(validation_aucs.values())
            mean_auc = float(np.mean(tuple(validation_aucs.values())))
            if minimum_auc > best_min_auc or (
                minimum_auc == best_min_auc and mean_auc > best_mean_auc
            ):
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_epoch = epoch
                best_min_auc = minimum_auc
                best_mean_auc = mean_auc
                best_aucs = dict(validation_aucs)
            history_row = {
                "run_id": RUN_ID,
                "replica": replica,
                "epoch": epoch,
                "optimizer_steps": step_count,
                "minimum_cross_key_auc": minimum_auc,
                "mean_cross_key_auc": mean_auc,
                **{
                    f"{cipher_key}_raw_train_loss": (
                        raw_loss_sums[cipher_key] / EXPECTED_BATCHES_PER_CIPHER
                    )
                    for cipher_key in EXPECTED_CIPHERS
                },
                **{
                    f"{cipher_key}_scaled_train_loss": (
                        scaled_loss_sums[cipher_key] / EXPECTED_BATCHES_PER_CIPHER
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
                output_root / "progress.jsonl", "epoch_done", **history_row
            )

        if best_state is None:
            raise RuntimeError("K1-AQ did not select a checkpoint")
        model.load_state_dict(best_state, strict=True)
        state_sha256 = tensor_mapping_sha256(model.state_dict())
        checkpoint_path = checkpoint_root / f"replica{replica}_best.pt"
        torch.save(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "initialization_seed": initialization_seed,
                "dataset_seeds": dict(replica_config["dataset_seeds"]),
                "loss_scales": loss_scales,
                "best_epoch": best_epoch,
                "best_minimum_cross_key_auc": best_min_auc,
                "best_mean_cross_key_auc": best_mean_auc,
                "best_cross_key_aucs": best_aucs,
                "optimizer_steps": step_count,
                "state_dict": best_state,
            },
            checkpoint_path,
        )
        optimizer_step_min, optimizer_step_max = _optimizer_step_range(optimizer)
        checkpoint = {
            "run_id": RUN_ID,
            "replica": replica,
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "state_dict_sha256": state_sha256,
            "strict_state_dict_load": True,
            "best_epoch": best_epoch,
            "optimizer_steps": step_count,
        }
        checkpoints[replica] = {**checkpoint, "state_dict": deepcopy(best_state)}
        result_rows.append(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "model": "runtime_spn_ct_k1ak_sbox_transition_true",
                "shared_ciphers": list(EXPECTED_CIPHERS),
                "initialization_seed": initialization_seed,
                "dataset_seeds": dict(replica_config["dataset_seeds"]),
                "loss_scales": loss_scales,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "initial_state_sha256": initial_state_sha256,
                "selected_state_sha256": state_sha256,
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
                    "loss": "fixed_inverse_norm_scaled_mse",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                    "optimizer_steps": step_count,
                    "optimizer_state_step_min": optimizer_step_min,
                    "optimizer_state_step_max": optimizer_step_max,
                    "one_shared_optimizer": True,
                    "equal_batches_per_cipher": True,
                    "unchanged_sequential_batch_order": True,
                },
                "negative_mode": "encrypted_random_plaintexts",
                "pairs_per_sample": 4,
                "samples_per_class_per_cipher": 2048,
            }
        )
    return result_rows, checkpoints, history_rows


def adjudicate(
    *,
    source_checks: Mapping[str, bool],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate = _group_controls(evaluation_rows)
    baseline = _group_controls(baseline_rows)
    expected_panels = {
        (replica, cipher_key, split)
        for replica in REPLICAS
        for cipher_key in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    protocol_checks = {
        "training_config_digest_exact": file_sha256(CONFIG_PATH)
        == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "two_training_rows_complete": len(training_rows) == 2
        and {int(row["replica"]) for row in training_rows} == set(REPLICAS),
        "ten_epochs_and_1920_steps_each": all(
            int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("optimizer_steps", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and int(row.get("training", {}).get("optimizer_state_step_min", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and int(row.get("training", {}).get("optimizer_state_step_max", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and row.get("training", {}).get("unchanged_sequential_batch_order") is True
            for row in training_rows
        ),
        "two_checkpoints_complete": set(checkpoints) == set(REPLICAS)
        and all(
            Path(str(checkpoints[replica]["path"])).is_file()
            and file_sha256(Path(str(checkpoints[replica]["path"])))
            == checkpoints[replica]["sha256"]
            for replica in REPLICAS
        ),
        "candidate_36_rows_complete": len(evaluation_rows) == 36
        and set(candidate) == expected_panels
        and all(set(rows) == set(CONTROL_CONDITIONS) for rows in candidate.values()),
        "baseline_36_rows_complete": len(baseline_rows) == 36
        and set(baseline) == expected_panels
        and all(set(rows) == set(CONTROL_CONDITIONS) for rows in baseline.values()),
        "evaluation_zero_step_and_immutable": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("state_immutable_across_controls") is True
            for row in evaluation_rows
        ),
    }
    panel_results = {}
    target_improved_count = 0
    no_harm_count = 0
    semantic_count = 0
    branch_count = 0
    retention_count = 0
    for panel in sorted(expected_panels):
        replica, cipher_key, split = panel
        candidate_rows = candidate[panel]
        baseline_rows_by_condition = baseline[panel]
        correct = float(candidate_rows["correct_runtime"]["auc"])
        baseline_correct = float(baseline_rows_by_condition["correct_runtime"]["auc"])
        wrong = float(candidate_rows["wrong_sbox_same_checkpoint"]["auc"])
        branch = float(
            candidate_rows["transition_branch_off_same_checkpoint"]["auc"]
        )
        anchor = float(candidate_rows["correct_runtime"]["anchor_auc"])
        delta = correct - baseline_correct
        target_improved = cipher_key in TARGET_CIPHERS and delta >= TARGET_IMPROVEMENT
        no_harm = delta >= NO_HARM_FLOOR
        semantic = correct - wrong >= SEMANTIC_MARGIN
        branch_pass = correct - branch >= SEMANTIC_MARGIN
        retention = correct - anchor >= RETENTION_MARGIN
        target_improved_count += int(target_improved)
        no_harm_count += int(no_harm)
        semantic_count += int(semantic)
        branch_count += int(branch_pass)
        retention_count += int(retention)
        panel_results[f"replica{replica}_{cipher_key}_{split}"] = {
            "candidate_correct_auc": correct,
            "baseline_correct_auc": baseline_correct,
            "candidate_minus_baseline": delta,
            "independent_anchor_auc": anchor,
            "candidate_minus_anchor": correct - anchor,
            "wrong_sbox_auc": wrong,
            "correct_minus_wrong_sbox": correct - wrong,
            "branch_off_auc": branch,
            "correct_minus_branch_off": correct - branch,
            "target_improvement_pass": target_improved,
            "no_harm_pass": no_harm,
            "semantic_pass": semantic,
            "branch_pass": branch_pass,
            "retention_pass": retention,
        }
    advance_gate = (
        target_improved_count >= TARGET_PANELS_MIN
        and no_harm_count == 12
        and semantic_count == 12
        and branch_count >= BRANCH_PANELS_MIN
    )
    full_support = advance_gate and retention_count == 12 and branch_count == 12
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1aq_protocol_invalid"
        next_action = "Repair only the failed binding or count invariant and rerun."
    elif full_support:
        status = "pass"
        decision = "innovation1_uknit_family_k1aq_full_shared_training_supported"
        next_action = (
            "Prepare a separate 65536/class/cipher remote-readiness audit with "
            "disk-backed caches; do not launch until readiness passes."
        )
    elif advance_gate:
        status = "pass"
        decision = "innovation1_uknit_family_k1aq_inverse_norm_partial_recovery_supported"
        next_action = (
            "Keep fixed inverse-norm scaling as a supported local component, but "
            "audit the remaining failed retention/branch panels before any scale-up."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1aq_inverse_norm_scaling_not_supported"
        next_action = (
            "Stop optimizer scaling and return to transition representation design; "
            "do not tune scales, pairs, samples, epochs, width, or remote GPUs."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol,
        "panel_results": panel_results,
        "target_improved_count": target_improved_count,
        "target_improved_required": TARGET_PANELS_MIN,
        "no_harm_count": no_harm_count,
        "semantic_pass_count": semantic_count,
        "branch_pass_count": branch_count,
        "retention_pass_count": retention_count,
        "advance_gate": advance_gate,
        "full_support_gate": full_support,
        "remote_scale": "no",
        "next_action": next_action,
        "blocked_actions": [
            "PCGrad, dynamic weighting, MoE, or experts",
            "16-pair, larger data, more epochs, width, or remote execution",
            "post-hoc tuning of the frozen loss scales",
        ],
        "claim_scope": (
            "Local 2048/class/cipher two-replica fixed-scale diagnostic only; "
            "not formal scale, an attack, arbitrary-SPN proof, or SOTA evidence."
        ),
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
    _append_progress(output_root / "progress.jsonl", "run_start")
    readiness, dataset_rows, datasets, anchors, baseline_rows, source_checks = (
        load_authority(config, project_root=project_root)
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-AQ source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "training": dict(config["training"]),
        "loss_scales": {
            str(row["replica"]): dict(row["loss_scales"])
            for row in config["replicas"]
        },
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)

    training_rows, checkpoints, history_rows = train_scaled_replicas(
        config=config,
        readiness_config=readiness,
        datasets=datasets,
        output_root=output_root,
        device=device,
    )
    evaluation_rows = evaluate_same_checkpoint_panel(
        config=config,
        readiness_config=readiness,
        datasets=datasets,
        anchors=anchors,
        checkpoints=checkpoints,
        device=device,
    )
    for row in evaluation_rows:
        row["run_id"] = RUN_ID
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {key: value for key, value in checkpoints[replica].items() if key != "state_dict"}
            for replica in REPLICAS
        ],
    }
    gate = adjudicate(
        source_checks=source_checks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        baseline_rows=baseline_rows,
        checkpoints=checkpoints,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "optimizer_steps_per_replica": {
            str(row["replica"]): row["training"]["optimizer_steps"]
            for row in training_rows
        },
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "target_improved_count": gate["target_improved_count"],
        "no_harm_count": gate["no_harm_count"],
        "semantic_pass_count": gate["semantic_pass_count"],
        "branch_pass_count": gate["branch_pass_count"],
        "retention_pass_count": gate["retention_pass_count"],
        "advance_gate": gate["advance_gate"],
        "full_support_gate": gate["full_support_gate"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", training_rows)
    _write_jsonl(output_root / "controls.jsonl", evaluation_rows)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_history_csv(output_root / "history.csv", history_rows)
    _write_comparison_csv(
        output_root / "comparison.csv", gate["panel_results"]
    )
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
        "results": training_rows,
        "controls": evaluation_rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _validate_loss_scales(
    config: Mapping[str, Any], gradient_rows: Sequence[Mapping[str, Any]]
) -> bool:
    norms = {
        (int(row["replica"]), str(row["cipher_key"])): float(
            row["median_gradient_norm"]
        )
        for row in gradient_rows
        if row.get("metric_type") == "gradient_norm"
        and row.get("condition") == "correct_runtime"
        and row.get("parameter_group") == "all_trainable"
    }
    if len(norms) != 6:
        return False
    for replica in config["replicas"]:
        replica_id = int(replica["replica"])
        source_norms = {
            key: float(value)
            for key, value in replica["source_median_gradient_norms"].items()
        }
        if any(
            not math.isclose(
                source_norms[cipher], norms[(replica_id, cipher)], rel_tol=0.0, abs_tol=1e-12
            )
            for cipher in EXPECTED_CIPHERS
        ):
            return False
        target = math.prod(source_norms.values()) ** (1.0 / 3.0)
        for cipher in EXPECTED_CIPHERS:
            expected = target / source_norms[cipher]
            if not math.isclose(
                float(replica["loss_scales"][cipher]),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
    return True


def _complete_control_panels(rows: Sequence[Mapping[str, Any]]) -> bool:
    grouped = _group_controls(rows)
    expected = {
        (replica, cipher, split)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    return (
        len(rows) == 36
        and set(grouped) == expected
        and all(set(conditions) == set(CONTROL_CONDITIONS) for conditions in grouped.values())
    )


def _group_controls(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], dict[str, Mapping[str, Any]]]:
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    return grouped


def _optimizer_step_range(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    steps = []
    for state in optimizer.state.values():
        if "step" in state:
            step = state["step"]
            steps.append(int(step.item() if torch.is_tensor(step) else step))
    return (min(steps), max(steps)) if steps else (0, 0)


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
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_csv(path: Path, panels: Mapping[str, Mapping[str, Any]]) -> None:
    fieldnames = ["panel", *next(iter(panels.values())).keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel, values in sorted(panels.items()):
            writer.writerow({"panel": panel, **values})


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    row = {"run_id": RUN_ID, "event": event, "time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _require_fresh_output_root(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"K1-AQ output root already exists: {path}")


__all__ = [
    "CONFIG_PATH",
    "RUN_ID",
    "adjudicate",
    "load_and_validate_config",
    "load_authority",
    "run_training",
    "scaled_mse_loss",
    "train_scaled_replicas",
]
