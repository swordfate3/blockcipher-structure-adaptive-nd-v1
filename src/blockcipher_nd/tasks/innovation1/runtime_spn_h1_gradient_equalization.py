from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    EXPECTED_SEEDS,
    EXPECTED_SOURCES,
    HOLDOUT_CIPHER,
    RelationModeRuntimeE4,
    _evaluate_target,
    _load_source_tasks,
    _load_structures,
    _load_target_validation,
    _plain_spec,
    _training_config,
    load_and_validate_holdout_config,
)
from blockcipher_nd.training.runtime_spn_joint import train_runtime_spn_joint
from blockcipher_nd.training.types import ProgressCallback


EXPECTED_TARGET_EVALUATIONS = (
    "candidate_correct",
    "candidate_corrupted_target",
    "candidate_no_topology_target",
)


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_h1_gradient_equalization_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("H1-A2 config schema_version must be 1")
    source = config.get("source", {})
    candidate = config.get("candidate", {})
    gate = config.get("gate", {})
    required_source = {
        "h1_required_decision": "innovation1_runtime_spn_rectangle_holdout_not_supported",
        "a1_required_decision": (
            "innovation1_runtime_spn_h1_source_gradient_imbalance_supported"
        ),
    }
    for key, expected in required_source.items():
        if source.get(key) != expected:
            raise ValueError(f"H1-A2 source field {key} drifted")
    required_candidate = {
        "gradient_combination": "representation_l2_equalized",
        "representation_parameters": "all_except_shared_classifier",
        "classifier_gradient_combination": "raw_arithmetic_mean",
        "task_sampling": "unchanged_equal_one_batch_per_task",
        "seeds": [0, 1],
        "expected_parameter_count": 442466,
    }
    for key, expected in required_candidate.items():
        if candidate.get(key) != expected:
            raise ValueError(f"H1-A2 candidate field {key} drifted")
    expected_target = {
        "candidate_correct": {"structure": "correct", "relation_mode": "true"},
        "candidate_corrupted_target": {
            "structure": "corrupted",
            "relation_mode": "true",
        },
        "candidate_no_topology_target": {
            "structure": "correct",
            "relation_mode": "independent",
        },
    }
    if config.get("target_evaluations") != expected_target:
        raise ValueError("H1-A2 target evaluations drifted")
    required_gate = {
        "target_auc_floor": 0.55,
        "target_topology_margin": 0.005,
        "seed0_anchor_retention_tolerance": 0.02,
        "seed1_anchor_improvement": 0.0,
        "source_macro_retention_tolerance": 0.01,
        "partial_margin_improvement": 0.005,
        "required_seeds": [0, 1],
    }
    for key, expected in required_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"H1-A2 gate field {key} drifted")

    h1_path = project_root / source["h1_config_path"]
    a1_path = project_root / source["a1_config_path"]
    h1 = load_and_validate_holdout_config(h1_path)
    if config_sha256(h1_path) != source.get("h1_config_sha256"):
        raise ValueError("H1-A2 H1 config hash drifted")
    if config_sha256(a1_path) != source.get("a1_config_sha256"):
        raise ValueError("H1-A2 A1 config hash drifted")
    if h1["run_id"] not in source["h1_output_root"]:
        raise ValueError("H1-A2 H1 output root drifted")
    return config


def run_h1_gradient_equalization(
    *,
    config: dict[str, Any],
    config_sha256_value: str,
    output_root: Path,
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source = config["source"]
    h1_config = load_and_validate_holdout_config(
        project_root / source["h1_config_path"]
    )
    h1_root = project_root / source["h1_output_root"]
    h1_gate = _read_json(h1_root / "gate.json")
    a1_gate = _read_json(project_root / source["a1_gate_path"])
    structures = _load_structures(h1_config)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, dict[str, Any]] = {}

    for seed in EXPECTED_SEEDS:
        tasks = _load_source_tasks(
            h1_config,
            seed=seed,
            structures=structures,
            progress_callback=progress_callback,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = RelationModeRuntimeE4(_plain_spec(h1_config["model"]), "true")
        _emit(progress_callback, "candidate_train_start", seed=seed)
        result = train_runtime_spn_joint(
            model,
            tasks,
            _training_config(h1_config["training"], seed),
            progress_callback=(
                None
                if progress_callback is None
                else lambda event, payload, seed=seed: progress_callback(
                    event,
                    {"seed": seed, **payload},
                )
            ),
            gradient_combination=config["candidate"]["gradient_combination"],
        )
        checkpoint_path = checkpoint_root / f"seed{seed}-candidate.pt"
        checkpoint = {
            "state_dict": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "seed": seed,
            "role": "candidate",
            "config_sha256": config_sha256_value,
            "source_h1_config_sha256": source["h1_config_sha256"],
            "gradient_combination": config["candidate"]["gradient_combination"],
            "best_epoch": result.metadata["best_epoch"],
            "checkpoint_selection_tasks": list(EXPECTED_SOURCES),
            "holdout_cipher": HOLDOUT_CIPHER,
        }
        torch.save(checkpoint, checkpoint_path)
        candidates[str(seed)] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "train_metrics": result.train_metrics,
            "validation_metrics": result.validation_metrics,
            "history": result.history,
            "metadata": result.metadata,
            "gradient_diagnostics": result.gradient_diagnostics,
        }
        _emit(
            progress_callback,
            "candidate_train_done",
            seed=seed,
            best_epoch=result.metadata["best_epoch"],
        )

    all_candidates_trained = len(candidates) == len(EXPECTED_SEEDS)
    for seed in EXPECTED_SEEDS:
        _emit(progress_callback, "target_validation_load_start", seed=seed)
        target = _load_target_validation(
            h1_config,
            seed=seed,
            progress_callback=progress_callback,
        )
        checkpoint = torch.load(
            candidates[str(seed)]["checkpoint_path"],
            map_location="cpu",
            weights_only=True,
        )
        evaluations: dict[str, Any] = {}
        for name, intervention in config["target_evaluations"].items():
            model = RelationModeRuntimeE4(
                _plain_spec(h1_config["model"]),
                intervention["relation_mode"],
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            structure = structures[HOLDOUT_CIPHER]
            if intervention["structure"] == "corrupted":
                structure = structure.corrupted()
            metrics = _evaluate_target(model, target, structure, h1_config["training"])
            evaluations[name] = {
                "metrics": metrics,
                "checkpoint_sha256": candidates[str(seed)]["checkpoint_sha256"],
                "optimizer_steps": 0,
                "target_head_trained": False,
            }
            _emit(
                progress_callback,
                "target_evaluation_done",
                seed=seed,
                evaluation=name,
                auc=metrics["auc"],
            )
        candidates[str(seed)]["target_evaluations"] = evaluations

    return _assemble_payload(
        config=config,
        config_sha256_value=config_sha256_value,
        h1_config=h1_config,
        h1_gate=h1_gate,
        a1_gate=a1_gate,
        h1_root=h1_root,
        candidates=candidates,
        all_candidates_trained=all_candidates_trained,
    )


def adjudicate_h1_gradient_equalization(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    per_seed: dict[str, Any] = {}
    full_pass = payload["validation"]["status"] == "pass"
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        target = payload["candidate_target_auc"][key]
        anchor_target = payload["anchor_target_auc"][key]
        source_macro = payload["candidate_source_macro_auc"][key]
        anchor_macro = payload["anchor_source_macro_auc"][key]
        correct = target["candidate_correct"]
        target_margins = {
            name: correct - target[name]
            for name in (
                "candidate_corrupted_target",
                "candidate_no_topology_target",
            )
        }
        anchor_worst_margin = min(
            anchor_target["candidate_correct"]
            - anchor_target["candidate_corrupted_target"],
            anchor_target["candidate_correct"]
            - anchor_target["candidate_no_topology_target"],
        )
        candidate_worst_margin = min(target_margins.values())
        retention_floor = (
            anchor_target["candidate_correct"]
            - gate_config["seed0_anchor_retention_tolerance"]
            if seed == 0
            else anchor_target["candidate_correct"]
            + gate_config["seed1_anchor_improvement"]
        )
        checks = {
            "target_auc_floor": correct >= gate_config["target_auc_floor"],
            "target_topology_margins": all(
                margin >= gate_config["target_topology_margin"]
                for margin in target_margins.values()
            ),
            "anchor_retention_or_improvement": correct >= retention_floor,
            "source_macro_retained": source_macro
            >= anchor_macro - gate_config["source_macro_retention_tolerance"],
        }
        seed_pass = all(checks.values())
        full_pass = full_pass and seed_pass
        per_seed[key] = {
            "candidate_target_auc": correct,
            "anchor_target_auc": anchor_target["candidate_correct"],
            "target_auc_delta": correct - anchor_target["candidate_correct"],
            "target_margins": target_margins,
            "worst_margin_improvement": candidate_worst_margin
            - anchor_worst_margin,
            "candidate_source_macro_auc": source_macro,
            "anchor_source_macro_auc": anchor_macro,
            "source_macro_delta": source_macro - anchor_macro,
            "checks": checks,
            "pass": seed_pass,
        }
    seed1_partial = (
        per_seed["1"]["worst_margin_improvement"]
        >= gate_config["partial_margin_improvement"]
    )
    if payload["validation"]["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_h1_gradient_equalization_protocol_invalid"
        next_action = "repair the exact candidate, cache, checkpoint or leakage failure"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_h1_gradient_equalization_supported"
        next_action = (
            "preregister a second independent whole-cipher holdout with the "
            "same gradient-equalized Runtime-E4 and controls"
        )
    elif seed1_partial:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_gradient_equalization_partial"
        next_action = (
            "retain gradient equalization and preregister one parameter-matched "
            "stable-conflict removal gate; do not add architecture or scale"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_gradient_equalization_not_supported"
        next_action = (
            "stop optimizer modification and run a no-training per-cipher "
            "representation geometry and classifier accessibility audit"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": payload["validation"]["status"] == "pass",
        "full_pass": full_pass,
        "seed1_partial_margin_improvement": seed1_partial,
        "per_seed": per_seed,
        "claim_scope": (
            "local 2048/class/source optimizer-only RECTANGLE whole-cipher "
            "holdout diagnostic; not formal scale, universality, attack or SOTA"
        ),
        "training_or_optimizer_steps_on_target": 0,
        "target_training_rows": 0,
        "next_action": next_action,
        "blocked_actions": [
            "add MoE, Adapter, FiLM, typed GNN, PCGrad or cipher identity routing",
            "train or select on RECTANGLE",
            "change samples, epochs, labels, negatives or remote scale",
            "claim universal adaptation from one holdout cipher",
        ],
    }


def write_h1_gradient_equalization_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["rows"]),
        encoding="utf-8",
    )
    _write_csv(output_root / "history.csv", payload["history"])
    _write_csv(output_root / "gradient_scales.csv", payload["gradient_scales"])
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "source-metrics.json", payload["candidate_source_auc"])
    _write_json(output_root / "target-metrics.json", payload["candidate_target_auc"])
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
    render_h1_gradient_equalization_svg(payload, gate, output_root / "curves.svg")


def render_h1_gradient_equalization_svg(
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
    target_labels = (
        ("candidate_correct", "A2正确结构"),
        ("candidate_corrupted_target", "A2损坏结构"),
        ("candidate_no_topology_target", "A2无拓扑"),
        ("anchor", "H1原始锚点"),
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#7F8C8D")
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
        for column, seed in enumerate(EXPECTED_SEEDS):
            key = str(seed)
            target = payload["candidate_target_auc"][key]
            target_values = [
                payload["anchor_target_auc"][key]["candidate_correct"]
                if name == "anchor"
                else target[name]
                for name, _ in target_labels
            ]
            target_bars = axes[0, column].barh(
                range(len(target_labels)),
                target_values,
                color=colors,
            )
            axes[0, column].axvline(0.5, color="#34495E", linewidth=1.2)
            axes[0, column].axvline(0.55, color="#7B2CBF", linestyle="--")
            axes[0, column].set_xlim(0.48, 0.75)
            axes[0, column].set_yticks(
                range(len(target_labels)),
                [label for _, label in target_labels],
            )
            axes[0, column].bar_label(target_bars, fmt="%.4f", padding=3)
            axes[0, column].set_xlabel("未见 RECTANGLE 验证 AUC")
            axes[0, column].set_title(f"seed{seed}：零微调目标结果")

            candidate_auc = payload["candidate_source_auc"][key]
            anchor_auc = payload["anchor_source_auc"][key]
            y = np.arange(len(EXPECTED_SOURCES))
            candidate_bars = axes[1, column].barh(
                y - 0.18,
                [candidate_auc[name] for name in EXPECTED_SOURCES],
                height=0.34,
                color="#0072B2",
                label="A2梯度归一化",
            )
            anchor_bars = axes[1, column].barh(
                y + 0.18,
                [anchor_auc[name] for name in EXPECTED_SOURCES],
                height=0.34,
                color="#AAB7B8",
                label="H1原始锚点",
            )
            axes[1, column].axvline(0.5, color="#34495E", linewidth=1.2)
            axes[1, column].set_xlim(0.4, 1.0)
            axes[1, column].set_yticks(
                y,
                [display[name] for name in EXPECTED_SOURCES],
            )
            axes[1, column].set_xlabel("四源验证 AUC")
            axes[1, column].set_title(f"seed{seed}：逐密码源验证对比")
            axes[1, column].legend(frameon=False, loc="lower right")
            axes[1, column].bar_label(
                candidate_bars,
                fmt="%.3f",
                padding=2,
                fontsize=8,
            )
            axes[1, column].bar_label(
                anchor_bars,
                fmt="%.3f",
                padding=2,
                fontsize=8,
            )

        fig.suptitle(
            "创新1 H1-A2：每任务表示梯度归一化的 RECTANGLE 整密码留出\n"
            "仅改变四源表示梯度组合；RECTANGLE 不参与训练、选模或微调",
            fontsize=17,
            y=0.985,
        )
        fig.text(
            0.5,
            0.03,
            f"裁决：{_decision_chinese(gate['decision'])}",
            ha="center",
            fontsize=12,
        )
        fig.subplots_adjust(
            left=0.12,
            right=0.98,
            top=0.86,
            bottom=0.1,
            wspace=0.32,
            hspace=0.42,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="svg", bbox_inches="tight")
        plt.close(fig)


def _assemble_payload(
    *,
    config: dict[str, Any],
    config_sha256_value: str,
    h1_config: dict[str, Any],
    h1_gate: dict[str, Any],
    a1_gate: dict[str, Any],
    h1_root: Path,
    candidates: dict[str, dict[str, Any]],
    all_candidates_trained: bool,
) -> dict[str, Any]:
    anchor_source_metrics = _read_json(h1_root / "source-metrics.json")
    anchor_target = _read_json(h1_root / "target-metrics.json")
    candidate_source_auc: dict[str, dict[str, float]] = {}
    anchor_source_auc: dict[str, dict[str, float]] = {}
    candidate_source_macro: dict[str, float] = {}
    anchor_source_macro: dict[str, float] = {}
    candidate_target: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    gradient_scales: list[dict[str, Any]] = []
    protocol_by_name = {item["name"]: item for item in h1_config["protocols"]}
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        candidate = candidates[key]
        candidate_source_auc[key] = {
            name: float(candidate["validation_metrics"][name]["auc"])
            for name in EXPECTED_SOURCES
        }
        anchor_source_auc[key] = {
            name: float(anchor_source_metrics[key]["correct"][name]["auc"])
            for name in EXPECTED_SOURCES
        }
        candidate_source_macro[key] = float(
            np.mean(list(candidate_source_auc[key].values()))
        )
        anchor_source_macro[key] = float(np.mean(list(anchor_source_auc[key].values())))
        candidate_target[key] = {
            name: float(candidate["target_evaluations"][name]["metrics"]["auc"])
            for name in EXPECTED_TARGET_EVALUATIONS
        }
        for name in EXPECTED_SOURCES:
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "source_validation",
                    "seed": seed,
                    "cipher": name,
                    "cipher_display_name": protocol_by_name[name]["display_name"],
                    "rounds": protocol_by_name[name]["rounds"],
                    "role": "gradient_equalized_candidate",
                    "parameter_count": candidate["parameter_count"],
                    "samples_per_class": h1_config["training"]["samples_per_class"],
                    "validation_samples_per_class": h1_config["training"][
                        "validation_samples_per_class"
                    ],
                    "pairs_per_sample": h1_config["training"]["pairs_per_sample"],
                    "negative_mode": h1_config["training"]["negative_mode"],
                    "metrics": {
                        "train": candidate["train_metrics"][name],
                        "validation": candidate["validation_metrics"][name],
                    },
                    "anchor_validation_auc": anchor_source_auc[key][name],
                    "checkpoint": candidate["checkpoint_path"],
                    "config_sha256": config_sha256_value,
                }
            )
        for name in EXPECTED_TARGET_EVALUATIONS:
            evaluation = candidate["target_evaluations"][name]
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "holdout_target",
                    "seed": seed,
                    "evaluation": name,
                    "cipher": HOLDOUT_CIPHER,
                    "cipher_display_name": protocol_by_name[HOLDOUT_CIPHER][
                        "display_name"
                    ],
                    "rounds": protocol_by_name[HOLDOUT_CIPHER]["rounds"],
                    "parameter_count": candidate["parameter_count"],
                    "training_samples_per_class": 0,
                    "validation_samples_per_class": h1_config["training"][
                        "validation_samples_per_class"
                    ],
                    "pairs_per_sample": h1_config["training"]["pairs_per_sample"],
                    "negative_mode": h1_config["training"]["negative_mode"],
                    "checkpoint": candidate["checkpoint_path"],
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "metrics": {"validation": evaluation["metrics"]},
                    "anchor_auc": anchor_target[key][name],
                    "optimizer_steps": 0,
                    "target_head_trained": False,
                    "config_sha256": config_sha256_value,
                }
            )
        history.extend({"seed": seed, **row} for row in candidate["history"])
        diagnostics = candidate["gradient_diagnostics"]
        for task in EXPECTED_SOURCES:
            gradient_scales.append(
                {
                    "seed": seed,
                    "task": task,
                    "mean_raw_representation_gradient_l2": diagnostics[
                        "task_representation_gradient_mean_l2"
                    ][task],
                    "mean_applied_scale": diagnostics["task_gradient_scale_mean"][
                        task
                    ],
                    "observations": diagnostics[
                        "task_gradient_scale_observations"
                    ][task],
                    "conflict_projections": diagnostics[
                        "task_conflict_projection_counts"
                    ][task],
                }
            )
    checkpoint_hashes_match = all(
        len(
            {
                candidates[str(seed)]["target_evaluations"][name][
                    "checkpoint_sha256"
                ]
                for name in EXPECTED_TARGET_EVALUATIONS
            }
        )
        == 1
        for seed in EXPECTED_SEEDS
    )
    checks = {
        "h1_gate_matches": h1_gate.get("decision")
        == config["source"]["h1_required_decision"],
        "a1_gate_matches": a1_gate.get("decision")
        == config["source"]["a1_required_decision"],
        "all_candidates_trained_before_target": all_candidates_trained,
        "two_checkpoints_exist": all(
            Path(candidates[str(seed)]["checkpoint_path"]).is_file()
            for seed in EXPECTED_SEEDS
        ),
        "parameter_count_matches": {
            candidates[str(seed)]["parameter_count"] for seed in EXPECTED_SEEDS
        }
        == {config["candidate"]["expected_parameter_count"]},
        "gradient_combination_exact": all(
            candidates[str(seed)]["metadata"]["gradient_combination"]
            == config["candidate"]["gradient_combination"]
            for seed in EXPECTED_SEEDS
        ),
        "gradient_scales_observed": all(
            row["observations"] > 0
            and np.isfinite(row["mean_applied_scale"])
            and row["mean_applied_scale"] > 0.0
            for row in gradient_scales
        ),
        "candidate_same_checkpoint_counterfactuals": checkpoint_hashes_match,
        "source_only_checkpoint_selection": all(
            tuple(candidates[str(seed)]["metadata"]["task_names"])
            == EXPECTED_SOURCES
            and candidates[str(seed)]["metadata"]["selected_checkpoint"] == "best"
            for seed in EXPECTED_SEEDS
        ),
        "target_optimizer_steps_zero": all(
            candidates[str(seed)]["target_evaluations"][name]["optimizer_steps"] == 0
            for seed in EXPECTED_SEEDS
            for name in EXPECTED_TARGET_EVALUATIONS
        ),
        "target_head_never_trained": all(
            not candidates[str(seed)]["target_evaluations"][name][
                "target_head_trained"
            ]
            for seed in EXPECTED_SEEDS
            for name in EXPECTED_TARGET_EVALUATIONS
        ),
        "strict_negative_mode": h1_config["training"]["negative_mode"]
        == "encrypted_random_plaintexts",
        "result_rows": len(rows) == 14,
    }
    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "result_rows": len(rows),
        "checkpoint_count": len(EXPECTED_SEEDS),
        "parameter_counts": sorted(
            {candidates[str(seed)]["parameter_count"] for seed in EXPECTED_SEEDS}
        ),
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
        "checkpoint_selection_tasks": list(EXPECTED_SOURCES),
    }
    return {
        "config": config,
        "candidate_source_auc": candidate_source_auc,
        "anchor_source_auc": anchor_source_auc,
        "candidate_source_macro_auc": candidate_source_macro,
        "anchor_source_macro_auc": anchor_source_macro,
        "candidate_target_auc": candidate_target,
        "anchor_target_auc": anchor_target,
        "rows": rows,
        "history": history,
        "gradient_scales": gradient_scales,
        "validation": validation,
    }


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_h1_gradient_equalization_supported": (
            "双seed整密码留出通过，开放第二独立留出设计"
        ),
        "innovation1_runtime_spn_h1_gradient_equalization_partial": (
            "归因边际改善但未全过，保留归一化并只开放冲突处理门"
        ),
        "innovation1_runtime_spn_h1_gradient_equalization_not_supported": (
            "梯度归一化未修复整密码泛化，转无训练表示对齐审计"
        ),
        "innovation1_runtime_spn_h1_gradient_equalization_protocol_invalid": (
            "协议无效"
        ),
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
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "adjudicate_h1_gradient_equalization",
    "config_sha256",
    "load_and_validate_h1_gradient_equalization_config",
    "render_h1_gradient_equalization_svg",
    "run_h1_gradient_equalization",
    "write_h1_gradient_equalization_artifacts",
]
