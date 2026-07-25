from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    EXPECTED_SEEDS,
    EXPECTED_SOURCES,
    RelationModeRuntimeE4,
    config_sha256,
    load_and_validate_holdout_config,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.training.optim import compute_loss, make_loss


GRADIENT_VIEWS = (
    "representation_backbone",
    "classifier",
    "all_parameters",
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_and_validate_h1_source_gradient_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("H1 source gradient audit schema_version must be 1")
    source = payload.get("source", {})
    audit = payload.get("audit", {})
    gate = payload.get("gate", {})
    expected_source = {
        "required_decision": "innovation1_runtime_spn_rectangle_holdout_not_supported",
        "checkpoint_role": "correct",
        "expected_failing_seeds": [1],
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"H1 source gradient source field {key} drifted")
    expected_audit = {
        "seeds": [0, 1],
        "source_ciphers": list(EXPECTED_SOURCES),
        "split": "train",
        "rows_per_cipher": 4096,
        "batch_size": 256,
        "loss": "mse",
        "device": "cpu",
        "gradient_views": list(GRADIENT_VIEWS),
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"H1 source gradient audit field {key} drifted")
    expected_gate = {
        "primary_gradient_view": "representation_backbone",
        "maximum_task_gradient_share": 0.5,
        "maximum_norm_to_other_median_ratio": 2.0,
        "conflict_cosine": -0.1,
        "stable_conflict_cosine": 0.0,
        "source_auc_range_floor": 0.3,
        "weak_source_auc_ceiling": 0.5,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"H1 source gradient gate field {key} drifted")

    source_config_path = project_root / source["config_path"]
    source_config = load_and_validate_holdout_config(source_config_path)
    if config_sha256(source_config_path) != source.get("config_sha256"):
        raise ValueError("H1 source gradient source config hash drifted")
    if source_config["run_id"] not in source["output_root"]:
        raise ValueError("H1 source gradient source output root drifted")
    return payload


def run_h1_source_gradient_audit(
    *,
    config: dict[str, Any],
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source = config["source"]
    source_config_path = project_root / source["config_path"]
    source_config = load_and_validate_holdout_config(source_config_path)
    source_root = project_root / source["output_root"]
    source_gate = _read_json(source_root / "gate.json")
    source_validation = _read_json(source_root / "validation.json")
    source_metrics = _read_json(source_root / "source-metrics.json")
    structures = _load_source_structures(source_config, project_root)
    gradient_vectors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    gradient_norms: list[dict[str, Any]] = []
    source_auc_rows: list[dict[str, Any]] = []
    checkpoint_checks: list[dict[str, Any]] = []
    cache_checks: list[dict[str, Any]] = []

    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        checkpoint_path = source_root / "checkpoints" / f"seed{seed}-correct.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_valid = all(
            (
                checkpoint.get("seed") == seed,
                checkpoint.get("role") == source["checkpoint_role"],
                checkpoint.get("config_sha256") == source["config_sha256"],
                checkpoint.get("holdout_cipher") == "rectangle80",
                tuple(checkpoint.get("checkpoint_selection_tasks", ()))
                == EXPECTED_SOURCES,
            )
        )
        checkpoint_checks.append(
            {
                "seed": seed,
                "path": str(checkpoint_path),
                "sha256": _file_sha256(checkpoint_path),
                "valid": checkpoint_valid,
            }
        )
        model = RelationModeRuntimeE4(_model_spec(source_config["model"]), "true")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(torch.device(config["audit"]["device"]))
        gradient_vectors[seed_key] = {}

        for task in EXPECTED_SOURCES:
            _emit(progress_callback, "source_gradient_start", seed=seed, task=task)
            cache_root = (
                project_root
                / source_config["training"]["cache_source_root"]
                / f"seed{seed}"
                / task
                / config["audit"]["split"]
            )
            features, labels, cache_valid = _load_and_validate_cache(
                cache_root,
                source_config=source_config,
                task_name=task,
                seed=seed,
                expected_rows=int(config["audit"]["rows_per_cipher"]),
            )
            cache_checks.append(
                {
                    "seed": seed,
                    "task": task,
                    "path": str(cache_root),
                    "rows": int(labels.shape[0]),
                    "valid": cache_valid,
                }
            )
            vectors = _mean_loss_gradients(
                model=model,
                structure=structures[task],
                features=features,
                labels=labels,
                batch_size=int(config["audit"]["batch_size"]),
                loss_name=config["audit"]["loss"],
            )
            gradient_vectors[seed_key][task] = vectors
            for view, vector in vectors.items():
                gradient_norms.append(
                    {
                        "seed": seed,
                        "task": task,
                        "view": view,
                        "l2_norm": float(torch.linalg.vector_norm(vector)),
                        "finite": bool(torch.isfinite(vector).all()),
                    }
                )
            auc = float(source_metrics[seed_key]["correct"][task]["auc"])
            source_auc_rows.append(
                {"seed": seed, "task": task, "validation_auc": auc}
            )
            _emit(progress_callback, "source_gradient_done", seed=seed, task=task)

    gradient_norms = _add_norm_shares(gradient_norms)
    gradient_cosines = pairwise_gradient_cosines(gradient_vectors)
    expected_cosines = len(EXPECTED_SEEDS) * len(GRADIENT_VIEWS) * 6
    source_failing_seeds = sorted(
        int(seed)
        for seed, row in source_gate.get("per_seed", {}).items()
        if not row.get("pass")
    )
    checks = {
        "source_gate_matches": source_gate.get("decision")
        == source["required_decision"],
        "source_protocol_valid": source_gate.get("protocol_valid") is True,
        "source_validation_passed": source_validation.get("status") == "pass",
        "failing_seed_contract": source_failing_seeds
        == source["expected_failing_seeds"],
        "checkpoints_valid": all(row["valid"] for row in checkpoint_checks),
        "caches_valid": all(row["valid"] for row in cache_checks),
        "gradients_finite": all(row["finite"] for row in gradient_norms),
        "cosine_row_count": len(gradient_cosines) == expected_cosines,
        "source_auc_row_count": len(source_auc_rows)
        == len(EXPECTED_SEEDS) * len(EXPECTED_SOURCES),
        "training_or_optimizer_steps_zero": True,
        "rectangle_rows_loaded_zero": True,
    }
    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "checkpoint_checks": checkpoint_checks,
        "cache_checks": cache_checks,
        "expected_cosine_rows": expected_cosines,
        "actual_cosine_rows": len(gradient_cosines),
        "source_failing_seeds": source_failing_seeds,
        "training_or_optimizer_steps": 0,
        "rectangle_rows_loaded": 0,
    }
    return {
        "config": config,
        "gradient_norms": gradient_norms,
        "gradient_cosines": gradient_cosines,
        "source_auc": source_auc_rows,
        "validation": validation,
    }


def pairwise_gradient_cosines(
    gradients: dict[str, dict[str, dict[str, torch.Tensor]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_key, by_task in gradients.items():
        for view in GRADIENT_VIEWS:
            for task_a, task_b in itertools.combinations(EXPECTED_SOURCES, 2):
                vector_a = by_task[task_a][view]
                vector_b = by_task[task_b][view]
                norm_a = float(torch.linalg.vector_norm(vector_a))
                norm_b = float(torch.linalg.vector_norm(vector_b))
                cosine = None
                if norm_a > 0.0 and norm_b > 0.0:
                    cosine = float(torch.dot(vector_a, vector_b) / (norm_a * norm_b))
                rows.append(
                    {
                        "seed": int(seed_key),
                        "view": view,
                        "task_a": task_a,
                        "task_b": task_b,
                        "cosine": cosine,
                    }
                )
    return rows


def adjudicate_h1_source_gradient_audit(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    primary_view = gate_config["primary_gradient_view"]
    per_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        norms = [
            row
            for row in payload["gradient_norms"]
            if row["seed"] == seed and row["view"] == primary_view
        ]
        norms.sort(key=lambda row: row["l2_norm"], reverse=True)
        largest = norms[0]
        other_norms = [row["l2_norm"] for row in norms[1:]]
        ratio = largest["l2_norm"] / max(float(np.median(other_norms)), 1e-12)
        cosines = [
            row
            for row in payload["gradient_cosines"]
            if row["seed"] == seed and row["view"] == primary_view
        ]
        conflicts = [
            row
            for row in cosines
            if row["cosine"] is not None
            and row["cosine"] <= gate_config["conflict_cosine"]
        ]
        auc_rows = [row for row in payload["source_auc"] if row["seed"] == seed]
        auc_by_task = {row["task"]: row["validation_auc"] for row in auc_rows}
        auc_range = max(auc_by_task.values()) - min(auc_by_task.values())
        per_seed[str(seed)] = {
            "largest_gradient_task": largest["task"],
            "largest_gradient_share": largest["norm_share"],
            "largest_to_other_median_ratio": ratio,
            "dialga_gradient_share": next(
                row["norm_share"] for row in norms if row["task"] == "dialga128"
            ),
            "gradient_imbalance": (
                largest["norm_share"]
                >= gate_config["maximum_task_gradient_share"]
                or ratio >= gate_config["maximum_norm_to_other_median_ratio"]
            ),
            "conflict_pairs": [
                {
                    "tasks": [row["task_a"], row["task_b"]],
                    "cosine": row["cosine"],
                }
                for row in conflicts
            ],
            "source_auc": auc_by_task,
            "source_auc_range": auc_range,
            "weak_sources": sorted(
                task
                for task, auc in auc_by_task.items()
                if auc <= gate_config["weak_source_auc_ceiling"]
            ),
        }

    stable_conflicts = _stable_conflict_pairs(
        payload["gradient_cosines"],
        view=primary_view,
        threshold=float(gate_config["stable_conflict_cosine"]),
    )
    failing_seeds = config["source"]["expected_failing_seeds"]
    failing_auc_imbalanced = all(
        per_seed[str(seed)]["source_auc_range"]
        >= gate_config["source_auc_range_floor"]
        for seed in failing_seeds
    )
    failing_gradient_imbalance = failing_auc_imbalanced and any(
        per_seed[str(seed)]["gradient_imbalance"] for seed in failing_seeds
    )
    failing_conflict = failing_auc_imbalanced and any(
        per_seed[str(seed)]["conflict_pairs"] for seed in failing_seeds
    )
    protocol_valid = payload["validation"]["status"] == "pass"
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_runtime_spn_h1_source_gradient_audit_invalid"
        next_action = "repair the exact checkpoint, cache or source-evidence mismatch"
    elif failing_gradient_imbalance:
        status = "pass"
        decision = "innovation1_runtime_spn_h1_source_gradient_imbalance_supported"
        next_action = (
            "preregister one same-budget per-task gradient-normalization gate; "
            "keep the model, source rows, target holdout and task sampling fixed"
        )
    elif failing_conflict and stable_conflicts:
        status = "pass"
        decision = "innovation1_runtime_spn_h1_stable_source_gradient_conflict_supported"
        next_action = (
            "preregister one same-budget parameter-matched gradient-conflict "
            "treatment; keep task weights and all structures fixed"
        )
    elif failing_conflict:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_seed_specific_gradient_conflict_only"
        next_action = (
            "audit fixed representation geometry before training because the "
            "conflict is not stable across seeds"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_representation_alignment_priority"
        next_action = (
            "run a no-training per-cipher representation geometry and shared "
            "classifier accessibility audit; do not alter the optimizer"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "primary_gradient_view": primary_view,
        "per_seed": per_seed,
        "stable_conflict_pairs": stable_conflicts,
        "failing_seed_auc_imbalance": failing_auc_imbalanced,
        "failing_seed_gradient_imbalance": failing_gradient_imbalance,
        "failing_seed_gradient_conflict": failing_conflict,
        "training_or_optimizer_steps": 0,
        "rectangle_rows_loaded": 0,
        "claim_scope": (
            "frozen H1 source-checkpoint gradient diagnostic only; endpoint "
            "gradients do not reconstruct the complete training trajectory"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "add MoE, Adapter, FiLM, typed GNN or cipher identity routing",
            "train or load RECTANGLE target rows",
            "change samples, epochs, labels, negatives or remote scale",
            "claim unseen-cipher support from this audit",
        ],
    }


def write_h1_source_gradient_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result_rows = [
        {"row_kind": "gradient_norm", **row} for row in payload["gradient_norms"]
    ] + [
        {"row_kind": "gradient_cosine", **row}
        for row in payload["gradient_cosines"]
    ] + [
        {"row_kind": "source_auc", **row} for row in payload["source_auc"]
    ]
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    _write_csv(output_root / "gradient_norms.csv", payload["gradient_norms"])
    _write_csv(output_root / "gradient_cosines.csv", payload["gradient_cosines"])
    _write_csv(output_root / "source_auc.csv", payload["source_auc"])
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
    render_h1_source_gradient_svg(payload, gate, output_root / "curves.svg")


def render_h1_source_gradient_svg(
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
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
        image_handle = None
        for row_index, seed in enumerate(EXPECTED_SEEDS):
            norms = [
                row
                for row in payload["gradient_norms"]
                if row["seed"] == seed
                and row["view"] == gate["primary_gradient_view"]
            ]
            norm_by_task = {row["task"]: row["norm_share"] for row in norms}
            gradient_bars = axes[row_index, 0].barh(
                range(len(EXPECTED_SOURCES)),
                [norm_by_task[name] for name in EXPECTED_SOURCES],
                color=colors,
            )
            axes[row_index, 0].axvline(0.5, color="#7B2CBF", linestyle="--")
            axes[row_index, 0].set_xlim(0.0, 1.0)
            axes[row_index, 0].bar_label(
                gradient_bars,
                fmt="%.3f",
                padding=3,
                fontsize=9,
            )
            axes[row_index, 0].set_yticks(
                range(len(EXPECTED_SOURCES)),
                [display[name] for name in EXPECTED_SOURCES],
            )
            axes[row_index, 0].set_xlabel("表示主干梯度范数占比")
            axes[row_index, 0].set_title(f"seed{seed}：任务梯度占比")

            matrix = _cosine_matrix(
                payload["gradient_cosines"],
                seed=seed,
                view=gate["primary_gradient_view"],
            )
            image_handle = axes[row_index, 1].imshow(
                matrix,
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu_r",
            )
            for y in range(len(EXPECTED_SOURCES)):
                for x in range(len(EXPECTED_SOURCES)):
                    value = matrix[y, x]
                    axes[row_index, 1].text(
                        x,
                        y,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if abs(value) > 0.55 else "black",
                    )
            axes[row_index, 1].set_xticks(
                range(len(EXPECTED_SOURCES)),
                [display[name] for name in EXPECTED_SOURCES],
                rotation=25,
                ha="right",
            )
            axes[row_index, 1].set_yticks(
                range(len(EXPECTED_SOURCES)),
                [display[name] for name in EXPECTED_SOURCES],
            )
            axes[row_index, 1].set_title(f"seed{seed}：表示主干梯度余弦")

            auc_rows = [row for row in payload["source_auc"] if row["seed"] == seed]
            auc_by_task = {row["task"]: row["validation_auc"] for row in auc_rows}
            auc_bars = axes[row_index, 2].barh(
                range(len(EXPECTED_SOURCES)),
                [auc_by_task[name] for name in EXPECTED_SOURCES],
                color=colors,
            )
            axes[row_index, 2].axvline(0.5, color="#34495E", linewidth=1.2)
            axes[row_index, 2].set_xlim(0.4, 1.02)
            axes[row_index, 2].bar_label(
                auc_bars,
                fmt="%.3f",
                padding=3,
                fontsize=9,
            )
            axes[row_index, 2].set_yticks(
                range(len(EXPECTED_SOURCES)),
                [display[name] for name in EXPECTED_SOURCES],
            )
            axes[row_index, 2].set_xlabel("H1 源验证 AUC")
            axes[row_index, 2].set_title(f"seed{seed}：逐密码验证信号")

        fig.suptitle(
            "创新1 H1-A1：四源共享 Runtime-E4 的梯度主导与冲突审计\n"
            "冻结 H1 检查点与每密码 4096 行训练切片；不训练、不加载 RECTANGLE",
            fontsize=17,
            y=0.985,
        )
        if image_handle is not None:
            colorbar_axis = fig.add_axes((0.39, 0.045, 0.25, 0.022))
            colorbar = fig.colorbar(
                image_handle,
                cax=colorbar_axis,
                orientation="horizontal",
            )
            colorbar.set_label("梯度余弦相似度")
        fig.text(
            0.79,
            0.035,
            f"裁决：{_decision_chinese(gate['decision'])}",
            ha="center",
            fontsize=11,
        )
        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            top=0.88,
            bottom=0.13,
            wspace=0.34,
            hspace=0.45,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="svg", bbox_inches="tight")
        plt.close(fig)


def _load_source_structures(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, RuntimeSpnStructure]:
    return {
        item["name"]: load_runtime_spn_descriptor(
            project_root / item["runtime_structure_path"],
            rounds=int(config["model"]["runtime_rounds"]),
            round_start=int(item["runtime_round_start"]),
        ).structure
        for item in config["protocols"]
        if item["name"] in EXPECTED_SOURCES
    }


def _model_spec(model: dict[str, Any]) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=int(model["hidden_dim"]),
        pair_embedding_dim=int(model["pair_embedding_dim"]),
        processor_steps=int(model["processor_steps"]),
        dropout=float(model["dropout"]),
        sbox_context_mode=model["sbox_context_mode"],
        cell_input_mode=model["cell_input_mode"],
        round_window_mode=model["round_window_mode"],
    )


def _load_and_validate_cache(
    cache_root: Path,
    *,
    source_config: dict[str, Any],
    task_name: str,
    seed: int,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    features = np.load(cache_root / "features.npy", mmap_mode="r")
    labels = np.load(cache_root / "labels.npy", mmap_mode="r")
    metadata = _read_json(cache_root / "metadata.json")
    protocol = next(
        item for item in source_config["protocols"] if item["name"] == task_name
    )
    training = source_config["training"]
    valid = bool(
        features.shape[0] == labels.shape[0] == expected_rows
        and metadata.get("samples_total") == expected_rows
        and metadata.get("samples_per_class") == training["samples_per_class"]
        and metadata.get("pairs_per_sample") == training["pairs_per_sample"]
        and metadata.get("negative_mode") == training["negative_mode"]
        and metadata.get("sample_structure") == training["sample_structure"]
        and metadata.get("feature_encoding") == training["feature_encoding"]
        and metadata.get("rounds") == protocol["rounds"]
        and metadata.get("input_difference") == int(protocol["input_difference"], 0)
        and metadata.get("seed") == seed
    )
    return features, labels, valid


def _mean_loss_gradients(
    *,
    model: RelationModeRuntimeE4,
    structure: RuntimeSpnStructure,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    loss_name: str,
) -> dict[str, torch.Tensor]:
    model.eval()
    model.zero_grad(set_to_none=True)
    loss_fn = make_loss(loss_name)
    device = next(model.parameters()).device
    total_rows = int(labels.shape[0])
    pair_bits = 2 * structure.block_bits
    if features.shape[1] % pair_bits:
        raise ValueError("H1 source cache contains incomplete ciphertext pairs")
    for start in range(0, total_rows, batch_size):
        stop = min(start + batch_size, total_rows)
        batch_features = torch.as_tensor(
            np.asarray(features[start:stop]).copy(),
            dtype=torch.float32,
            device=device,
        )
        batch_labels = torch.as_tensor(
            np.asarray(labels[start:stop]).copy(),
            dtype=torch.float32,
            device=device,
        )
        runtime_features = batch_features.reshape(
            batch_features.shape[0], -1, 2, structure.block_bits
        ).flip(-1)
        logits = model(runtime_features, structure).squeeze(1)
        batch_loss = compute_loss(loss_fn, logits, batch_labels, loss_name)
        (batch_loss * ((stop - start) / total_rows)).backward()
    named_parameters = list(model.named_parameters())
    return {
        view: _flatten_gradient_view(named_parameters, view) for view in GRADIENT_VIEWS
    }


def _flatten_gradient_view(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    view: str,
) -> torch.Tensor:
    def included(name: str) -> bool:
        is_classifier = name.startswith("backbone.classifier.")
        if view == "representation_backbone":
            return not is_classifier
        if view == "classifier":
            return is_classifier
        if view == "all_parameters":
            return True
        raise ValueError(f"unknown gradient view: {view}")

    chunks = []
    for name, parameter in named_parameters:
        if not included(name):
            continue
        gradient = parameter.grad
        chunks.append(
            torch.zeros_like(parameter, device="cpu").reshape(-1)
            if gradient is None
            else gradient.detach().cpu().reshape(-1)
        )
    if not chunks:
        raise ValueError(f"gradient view has no parameters: {view}")
    return torch.cat(chunks)


def _add_norm_shares(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = {
        (seed, view): sum(
            row["l2_norm"]
            for row in rows
            if row["seed"] == seed and row["view"] == view
        )
        for seed in EXPECTED_SEEDS
        for view in GRADIENT_VIEWS
    }
    return [
        {
            **row,
            "norm_share": row["l2_norm"] / max(totals[(row["seed"], row["view"])], 1e-12),
        }
        for row in rows
    ]


def _stable_conflict_pairs(
    rows: list[dict[str, Any]],
    *,
    view: str,
    threshold: float,
) -> list[dict[str, Any]]:
    stable = []
    for task_a, task_b in itertools.combinations(EXPECTED_SOURCES, 2):
        values = [
            row["cosine"]
            for row in rows
            if row["view"] == view
            and {row["task_a"], row["task_b"]} == {task_a, task_b}
        ]
        if len(values) == len(EXPECTED_SEEDS) and all(
            value is not None and value < threshold for value in values
        ):
            stable.append(
                {"tasks": [task_a, task_b], "per_seed_cosines": values}
            )
    return stable


def _cosine_matrix(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    view: str,
) -> np.ndarray:
    size = len(EXPECTED_SOURCES)
    matrix = np.eye(size, dtype=np.float64)
    indices = {task: index for index, task in enumerate(EXPECTED_SOURCES)}
    for row in rows:
        if row["seed"] != seed or row["view"] != view:
            continue
        a = indices[row["task_a"]]
        b = indices[row["task_b"]]
        value = np.nan if row["cosine"] is None else row["cosine"]
        matrix[a, b] = matrix[b, a] = value
    return matrix


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_h1_source_gradient_imbalance_supported": (
            "源任务梯度不平衡，开放同预算归一化门"
        ),
        "innovation1_runtime_spn_h1_stable_source_gradient_conflict_supported": (
            "存在跨种子稳定梯度冲突，开放同预算冲突处理门"
        ),
        "innovation1_runtime_spn_h1_seed_specific_gradient_conflict_only": (
            "仅单种子冲突，先做表示对齐审计"
        ),
        "innovation1_runtime_spn_h1_representation_alignment_priority": (
            "未支持梯度机制，转无训练表示对齐审计"
        ),
        "innovation1_runtime_spn_h1_source_gradient_audit_invalid": "协议无效",
    }.get(decision, decision)


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
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "GRADIENT_VIEWS",
    "adjudicate_h1_source_gradient_audit",
    "load_and_validate_h1_source_gradient_config",
    "pairwise_gradient_cosines",
    "render_h1_source_gradient_svg",
    "run_h1_source_gradient_audit",
    "write_h1_source_gradient_artifacts",
]
