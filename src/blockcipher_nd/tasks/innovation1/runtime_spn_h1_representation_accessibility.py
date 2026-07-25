from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    EXPECTED_SEEDS,
    EXPECTED_SOURCES,
    RelationModeRuntimeE4,
    _load_structures,
    _plain_spec,
    config_sha256,
    load_and_validate_holdout_config,
)
from blockcipher_nd.training.metrics import binary_auc
from blockcipher_nd.training.runtime_spn_joint import _to_runtime_coordinates


CHECKPOINT_ROLES = ("h1", "a2", "a3")
ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_and_validate_h1_representation_accessibility_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("H1-A4 config schema_version must be 1")
    source = config.get("source", {})
    audit = config.get("audit", {})
    gate = config.get("gate", {})
    if config_sha256(project_root / source["h1_config_path"]) != source.get(
        "h1_config_sha256"
    ):
        raise ValueError("H1-A4 H1 config hash drifted")
    if tuple(source.get("checkpoints", {})) != CHECKPOINT_ROLES:
        raise ValueError("H1-A4 checkpoint role order drifted")
    expected_sources = {
        "h1": (
            "innovation1_runtime_spn_rectangle_holdout_not_supported",
            "correct",
            "mean_loss",
        ),
        "a2": (
            "innovation1_runtime_spn_h1_gradient_equalization_partial",
            "candidate",
            "representation_l2_equalized",
        ),
        "a3": (
            "innovation1_runtime_spn_h1_equalized_pcgrad_partial",
            "candidate",
            "representation_l2_equalized_pcgrad_fixed_order",
        ),
    }
    for role, (decision, checkpoint_role, combination) in expected_sources.items():
        row = source["checkpoints"][role]
        if (
            row.get("required_decision") != decision
            or row.get("role") != checkpoint_role
            or row.get("gradient_combination") != combination
            or tuple(sorted(row.get("files", {}))) != ("0", "1")
            or tuple(sorted(row.get("sha256", {}))) != ("0", "1")
        ):
            raise ValueError(f"H1-A4 checkpoint source drifted: {role}")
    expected_audit = {
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "seeds": [0, 1],
        "source_ciphers": list(EXPECTED_SOURCES),
        "split": "validation",
        "rows_per_cipher": 2048,
        "rows_per_class": 1024,
        "representation_width": 384,
        "extraction_batch_size": 256,
        "device": "cpu",
        "probe": "stratified_two_fold_closed_form_ridge",
        "ridge_lambda": 0.01,
        "probe_label_encoding": "minus_one_plus_one",
        "probe_standardization": "fit_fold_only",
        "label_shuffle_control": ("a3_skinny_only_deterministic_fit_label_permutation"),
        "label_shuffle_seed": 260726,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"H1-A4 audit field drifted: {key}")
    expected_gate = {
        "probe_auc_floor": 0.55,
        "classifier_gap_floor": 0.02,
        "label_shuffle_auc_tolerance": 0.05,
        "required_seeds": [0, 1],
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"H1-A4 gate field drifted: {key}")
    return config


def run_h1_representation_accessibility_audit(
    *,
    config: dict[str, Any],
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source = config["source"]
    h1_config = load_and_validate_holdout_config(
        project_root / source["h1_config_path"]
    )
    structures = _load_structures(h1_config)
    metrics: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    checkpoint_checks: list[dict[str, Any]] = []
    cache_checks: list[dict[str, Any]] = []

    for checkpoint_role in CHECKPOINT_ROLES:
        source_row = source["checkpoints"][checkpoint_role]
        output_root = project_root / source_row["output_root"]
        gate = _read_json(output_root / "gate.json")
        gate_valid = bool(
            gate.get("decision") == source_row["required_decision"]
            and gate.get("protocol_valid") is True
        )
        for seed in EXPECTED_SEEDS:
            seed_key = str(seed)
            checkpoint_path = output_root / source_row["files"][seed_key]
            checkpoint_sha256 = _file_sha256(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint_valid = bool(
                checkpoint_sha256 == source_row["sha256"][seed_key]
                and checkpoint.get("seed") == seed
                and checkpoint.get("role") == source_row["role"]
                and checkpoint.get("config_sha256") == source_row["config_sha256"]
                and checkpoint.get("gradient_combination", "mean_loss")
                == source_row["gradient_combination"]
                and tuple(checkpoint.get("checkpoint_selection_tasks", ()))
                == EXPECTED_SOURCES
                and checkpoint.get("holdout_cipher") == "rectangle80"
            )
            checkpoint_checks.append(
                {
                    "checkpoint_role": checkpoint_role,
                    "seed": seed,
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha256,
                    "gate_valid": gate_valid,
                    "valid": checkpoint_valid and gate_valid,
                }
            )
            model = RelationModeRuntimeE4(_plain_spec(h1_config["model"]), "true")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(torch.device(config["audit"]["device"]))
            model.eval()

            for task in EXPECTED_SOURCES:
                _emit(
                    progress_callback,
                    "representation_audit_start",
                    checkpoint_role=checkpoint_role,
                    seed=seed,
                    task=task,
                )
                features, labels, cache_valid = _load_validation_cache(
                    h1_config=h1_config,
                    task=task,
                    seed=seed,
                    project_root=project_root,
                )
                cache_checks.append(
                    {
                        "checkpoint_role": checkpoint_role,
                        "seed": seed,
                        "task": task,
                        "rows": int(labels.shape[0]),
                        "valid": cache_valid,
                    }
                )
                representation, probabilities = _extract_representation_and_scores(
                    model=model,
                    structure=structures[task],
                    features=features,
                    batch_size=int(config["audit"]["extraction_batch_size"]),
                )
                centroid = class_centroid_geometry(representation, labels)
                probe_scores = stratified_two_fold_ridge_scores(
                    representation,
                    labels,
                    ridge_lambda=float(config["audit"]["ridge_lambda"]),
                )
                shared_auc = binary_auc(labels, probabilities)
                probe_auc = binary_auc(labels, probe_scores)
                metrics.append(
                    {
                        "row_kind": "representation_accessibility",
                        "checkpoint_role": checkpoint_role,
                        "seed": seed,
                        "task": task,
                        "shared_classifier_auc": shared_auc,
                        "closed_form_probe_auc": probe_auc,
                        "probe_gain": probe_auc - shared_auc,
                        **centroid,
                        "rows": int(labels.shape[0]),
                        "positive_rows": int(np.sum(labels == 1)),
                        "negative_rows": int(np.sum(labels == 0)),
                        "representation_width": int(representation.shape[1]),
                        "representation_finite": bool(
                            np.isfinite(representation).all()
                        ),
                        "neural_optimizer_steps": 0,
                    }
                )
                if checkpoint_role == "a3" and task == "skinny64":
                    shuffled_scores = stratified_two_fold_ridge_scores(
                        representation,
                        labels,
                        ridge_lambda=float(config["audit"]["ridge_lambda"]),
                        permute_fit_labels=True,
                        permutation_seed=(
                            int(config["audit"]["label_shuffle_seed"]) + seed
                        ),
                    )
                    controls.append(
                        {
                            "row_kind": "label_shuffle_control",
                            "checkpoint_role": checkpoint_role,
                            "seed": seed,
                            "task": task,
                            "closed_form_probe_auc": binary_auc(
                                labels,
                                shuffled_scores,
                            ),
                            "rows": int(labels.shape[0]),
                        }
                    )
                _emit(
                    progress_callback,
                    "representation_audit_done",
                    checkpoint_role=checkpoint_role,
                    seed=seed,
                    task=task,
                    shared_auc=shared_auc,
                    probe_auc=probe_auc,
                )

    expected_rows = len(CHECKPOINT_ROLES) * len(EXPECTED_SEEDS) * len(EXPECTED_SOURCES)
    checks = {
        "h1_config_hash_matches": config_sha256(project_root / source["h1_config_path"])
        == source["h1_config_sha256"],
        "checkpoint_and_gate_sources_valid": all(
            row["valid"] for row in checkpoint_checks
        ),
        "source_validation_caches_valid": all(row["valid"] for row in cache_checks),
        "metric_row_count": len(metrics) == expected_rows,
        "control_row_count": len(controls) == len(EXPECTED_SEEDS),
        "representations_finite": all(row["representation_finite"] for row in metrics),
        "representation_width_exact": {row["representation_width"] for row in metrics}
        == {config["audit"]["representation_width"]},
        "rows_and_classes_exact": all(
            row["rows"] == config["audit"]["rows_per_cipher"]
            and row["positive_rows"] == config["audit"]["rows_per_class"]
            and row["negative_rows"] == config["audit"]["rows_per_class"]
            for row in metrics
        ),
        "metrics_finite": all(
            np.isfinite(row[key])
            for row in metrics
            for key in (
                "shared_classifier_auc",
                "closed_form_probe_auc",
                "centroid_distance",
                "within_class_rms",
                "centroid_separation_ratio",
            )
        ),
        "neural_optimizer_steps_zero": True,
        "rectangle_rows_loaded_zero": True,
    }
    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "checkpoint_checks": checkpoint_checks,
        "cache_checks": cache_checks,
        "expected_metric_rows": expected_rows,
        "actual_metric_rows": len(metrics),
        "neural_optimizer_steps": 0,
        "rectangle_rows_loaded": 0,
    }
    return {
        "config": config,
        "metrics": metrics,
        "controls": controls,
        "validation": validation,
    }


def stratified_two_fold_ridge_scores(
    representations: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_lambda: float,
    permute_fit_labels: bool = False,
    permutation_seed: int = 0,
) -> np.ndarray:
    x = np.asarray(representations, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("representation and labels must form a row-aligned matrix")
    if ridge_lambda <= 0.0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("ridge probe requires positive lambda and binary labels")
    folds = stratified_two_fold_indices(y)
    scores = np.empty(len(y), dtype=np.float64)
    for eval_fold, eval_indices in enumerate(folds):
        fit_indices = folds[1 - eval_fold]
        fit_x = x[fit_indices]
        eval_x = x[eval_indices]
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        fit_x = (fit_x - mean) / scale
        eval_x = (eval_x - mean) / scale
        fit_x = np.column_stack((fit_x, np.ones(len(fit_x))))
        eval_x = np.column_stack((eval_x, np.ones(len(eval_x))))
        fit_y = y[fit_indices].astype(np.float64) * 2.0 - 1.0
        if permute_fit_labels:
            rng = np.random.default_rng(permutation_seed + eval_fold * 1009)
            fit_y = fit_y[rng.permutation(len(fit_y))]
        regularizer = np.eye(fit_x.shape[1], dtype=np.float64) * ridge_lambda
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(
            fit_x.T @ fit_x + regularizer,
            fit_x.T @ fit_y,
        )
        scores[eval_indices] = eval_x @ weights
    return scores


def stratified_two_fold_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("stratified folds require a binary label vector")
    by_class = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) % 2 for indices in by_class):
        raise ValueError("each class must contain an even number of rows")
    folds = []
    for parity in (0, 1):
        fold = np.sort(np.concatenate([indices[parity::2] for indices in by_class]))
        folds.append(fold)
    return folds[0], folds[1]


def class_centroid_geometry(
    representations: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(representations, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - x.mean(axis=0)) / scale
    means = [standardized[y == label].mean(axis=0) for label in (0, 1)]
    centroid_distance = float(np.linalg.norm(means[1] - means[0]))
    residuals = np.concatenate(
        [standardized[y == label] - means[label] for label in (0, 1)],
        axis=0,
    )
    within_rms = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    return {
        "centroid_distance": centroid_distance,
        "within_class_rms": within_rms,
        "centroid_separation_ratio": centroid_distance / max(within_rms, 1e-12),
    }


def adjudicate_h1_representation_accessibility(
    payload: dict[str, Any],
) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    a3_skinny = {
        str(seed): next(
            row
            for row in payload["metrics"]
            if row["checkpoint_role"] == "a3"
            and row["seed"] == seed
            and row["task"] == "skinny64"
        )
        for seed in EXPECTED_SEEDS
    }
    controls = {
        str(row["seed"]): row["closed_form_probe_auc"] for row in payload["controls"]
    }
    per_seed = {}
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        row = a3_skinny[key]
        checks = {
            "probe_auc_floor": row["closed_form_probe_auc"]
            >= gate_config["probe_auc_floor"],
            "classifier_gap": row["probe_gain"] >= gate_config["classifier_gap_floor"],
            "label_shuffle_near_chance": abs(controls[key] - 0.5)
            <= gate_config["label_shuffle_auc_tolerance"],
        }
        per_seed[key] = {
            "shared_classifier_auc": row["shared_classifier_auc"],
            "closed_form_probe_auc": row["closed_form_probe_auc"],
            "probe_gain": row["probe_gain"],
            "centroid_separation_ratio": row["centroid_separation_ratio"],
            "label_shuffle_probe_auc": controls[key],
            "checks": checks,
            "classifier_bottleneck": all(checks.values()),
        }
    protocol_valid = payload["validation"]["status"] == "pass"
    bottleneck = protocol_valid and all(
        row["classifier_bottleneck"] for row in per_seed.values()
    )
    representation_weak = protocol_valid and all(
        not row["checks"]["probe_auc_floor"] for row in per_seed.values()
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_runtime_spn_h1_representation_accessibility_invalid"
        next_action = "repair the exact checkpoint, cache, fold or metric failure"
    elif bottleneck:
        status = "pass"
        decision = "innovation1_runtime_spn_h1_shared_classifier_bottleneck_supported"
        next_action = (
            "preregister one no-cipher-ID structure-conditioned shared readout; "
            "keep Runtime-E4, source rows and RECTANGLE holdout frozen"
        )
    elif representation_weak:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_shared_representation_weak"
        next_action = (
            "redesign the shared structure primitive; do not change the optimizer "
            "or add a larger classifier"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_accessibility_mixed"
        next_action = (
            "run a deterministic per-mode representation audit before choosing "
            "a shared-readout or representation redesign"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "per_seed": per_seed,
        "shared_classifier_bottleneck_supported": bottleneck,
        "shared_representation_weak": representation_weak,
        "neural_optimizer_steps": 0,
        "rectangle_rows_loaded": 0,
        "claim_scope": (
            "frozen local H1/A2/A3 source-validation representation audit; "
            "closed-form probe accessibility is not neural-distinguisher performance"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "add another optimizer treatment or remote scale",
            "train on or select with RECTANGLE",
            "claim probe AUC as a new neural distinguisher result",
            "add cipher IDs, cipher-specific experts or target heads",
        ],
    }


def write_h1_representation_accessibility_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = payload["metrics"] + payload["controls"]
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_csv(output_root / "representation_metrics.csv", payload["metrics"])
    _write_csv(output_root / "label_shuffle_controls.csv", payload["controls"])
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    render_h1_representation_accessibility_svg(
        payload,
        gate,
        output_root / "curves.svg",
    )


def render_h1_representation_accessibility_svg(
    payload: dict[str, Any],
    gate: dict[str, Any],
    output: Path,
) -> None:
    display = {
        "gift64": "GIFT",
        "skinny64": "SKINNY",
        "uknit64": "uKNIT",
        "dialga128": "Dialga",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for column, seed in enumerate(EXPECTED_SEEDS):
            a3_rows = [
                row
                for row in payload["metrics"]
                if row["checkpoint_role"] == "a3" and row["seed"] == seed
            ]
            a3_by_task = {row["task"]: row for row in a3_rows}
            y = np.arange(len(EXPECTED_SOURCES))
            shared = axes[0, column].barh(
                y - 0.18,
                [
                    a3_by_task[task]["shared_classifier_auc"]
                    for task in EXPECTED_SOURCES
                ],
                height=0.34,
                color="#7F8C8D",
                label="冻结共享分类头",
            )
            probe = axes[0, column].barh(
                y + 0.18,
                [
                    a3_by_task[task]["closed_form_probe_auc"]
                    for task in EXPECTED_SOURCES
                ],
                height=0.34,
                color="#0072B2",
                label="闭式两折线性探针",
            )
            axes[0, column].bar_label(shared, fmt="%.3f", padding=2, fontsize=8)
            axes[0, column].bar_label(probe, fmt="%.3f", padding=2, fontsize=8)
            axes[0, column].axvline(0.5, color="#34495E")
            axes[0, column].axvline(0.55, color="#7B2CBF", linestyle="--")
            axes[0, column].set_xlim(0.4, 1.0)
            axes[0, column].set_yticks(y, [display[task] for task in EXPECTED_SOURCES])
            axes[0, column].set_xlabel("验证 AUC")
            axes[0, column].set_title(f"seed{seed}：A3逐密码表示可达性")
            axes[0, column].legend(frameon=False, loc="lower right")

            skinny_rows = [
                row
                for role in CHECKPOINT_ROLES
                for row in payload["metrics"]
                if row["checkpoint_role"] == role
                and row["seed"] == seed
                and row["task"] == "skinny64"
            ]
            labels = ["H1", "A2", "A3"]
            shared_skinny = axes[1, column].barh(
                np.arange(3) - 0.18,
                [row["shared_classifier_auc"] for row in skinny_rows],
                height=0.34,
                color="#7F8C8D",
                label="冻结共享分类头",
            )
            probe_skinny = axes[1, column].barh(
                np.arange(3) + 0.18,
                [row["closed_form_probe_auc"] for row in skinny_rows],
                height=0.34,
                color="#009E73",
                label="闭式两折线性探针",
            )
            axes[1, column].bar_label(shared_skinny, fmt="%.3f", padding=2, fontsize=8)
            axes[1, column].bar_label(probe_skinny, fmt="%.3f", padding=2, fontsize=8)
            control = next(
                row["closed_form_probe_auc"]
                for row in payload["controls"]
                if row["seed"] == seed
            )
            axes[1, column].axvline(0.5, color="#34495E")
            axes[1, column].axvline(control, color="#D55E00", linestyle=":")
            axes[1, column].axvline(0.55, color="#7B2CBF", linestyle="--")
            axes[1, column].set_xlim(0.4, 0.75)
            axes[1, column].set_yticks(np.arange(3), labels)
            axes[1, column].set_xlabel("SKINNY 验证 AUC")
            axes[1, column].set_title(
                f"seed{seed}：SKINNY历次检查点（打乱探针 {control:.3f}）"
            )
            axes[1, column].legend(frameon=False, loc="lower right")
        fig.suptitle(
            "创新1 H1-A4：冻结 Runtime-E4 表示可达性审计\n"
            "闭式探针只诊断共享分类头；不训练神经网络、不加载 RECTANGLE",
            fontsize=17,
            y=0.985,
        )
        fig.text(
            0.5,
            0.025,
            f"裁决：{_decision_chinese(gate['decision'])}",
            ha="center",
            fontsize=12,
        )
        fig.subplots_adjust(
            left=0.11,
            right=0.98,
            top=0.86,
            bottom=0.1,
            wspace=0.30,
            hspace=0.42,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="svg", bbox_inches="tight")
        plt.close(fig)


def _load_validation_cache(
    *,
    h1_config: dict[str, Any],
    task: str,
    seed: int,
    project_root: Path,
) -> tuple[np.ndarray, np.ndarray, bool]:
    root = (
        project_root
        / h1_config["training"]["cache_source_root"]
        / f"seed{seed}"
        / task
        / "validation"
    )
    features = np.load(root / "features.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    metadata = _read_json(root / "metadata.json")
    protocol = next(row for row in h1_config["protocols"] if row["name"] == task)
    training = h1_config["training"]
    valid = bool(
        features.shape[0] == labels.shape[0] == 2048
        and metadata.get("samples_total") == 2048
        and metadata.get("samples_per_class")
        == training["validation_samples_per_class"]
        and metadata.get("positive_rows") == 1024
        and metadata.get("negative_rows") == 1024
        and metadata.get("pairs_per_sample") == training["pairs_per_sample"]
        and metadata.get("negative_mode") == training["negative_mode"]
        and metadata.get("sample_structure") == training["sample_structure"]
        and metadata.get("feature_encoding") == training["feature_encoding"]
        and metadata.get("rounds") == protocol["rounds"]
        and metadata.get("input_difference") == int(protocol["input_difference"], 0)
        and metadata.get("seed") == seed + 10_000
    )
    return features, labels, valid


def _extract_representation_and_scores(
    *,
    model: RelationModeRuntimeE4,
    structure: Any,
    features: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    representations = []
    probabilities = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.as_tensor(
                np.asarray(features[start:stop]).copy(),
                dtype=torch.float32,
                device=device,
            )
            runtime = _to_runtime_coordinates(batch, structure.block_bits)
            representation = model.backbone.encode(
                runtime,
                structure,
                relation_mode=model.relation_mode,
            )
            logits = model.backbone.classifier(representation).squeeze(1)
            representations.append(representation.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(representations), np.concatenate(probabilities)


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_h1_shared_classifier_bottleneck_supported": (
            "SKINNY表示可线性读取但共享头未利用，下一步设计无密码ID的结构条件读出"
        ),
        "innovation1_runtime_spn_h1_shared_representation_weak": (
            "SKINNY表示本身信号弱，下一步重设计共享结构原语"
        ),
        "innovation1_runtime_spn_h1_accessibility_mixed": (
            "表示可达性跨seed不一致，先做确定性表示模式审计"
        ),
        "innovation1_runtime_spn_h1_representation_accessibility_invalid": "协议无效",
    }.get(decision, decision)


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "adjudicate_h1_representation_accessibility",
    "class_centroid_geometry",
    "load_and_validate_h1_representation_accessibility_config",
    "run_h1_representation_accessibility_audit",
    "stratified_two_fold_indices",
    "stratified_two_fold_ridge_scores",
    "write_h1_representation_accessibility_artifacts",
]
