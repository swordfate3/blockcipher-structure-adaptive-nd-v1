from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_h1_gradient_equalization import (
    config_sha256,
    run_h1_gradient_equalization,
    write_h1_gradient_equalization_artifacts,
)
from blockcipher_nd.training.types import ProgressCallback


EXPECTED_SOURCES = ("gift64", "skinny64", "uknit64", "dialga128")
EXPECTED_SEEDS = (0, 1)


def load_and_validate_h1_equalized_pcgrad_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("H1-A3 config schema_version must be 1")
    source = config.get("source", {})
    candidate = config.get("candidate", {})
    gate = config.get("gate", {})
    required_source = {
        "h1_required_decision": "innovation1_runtime_spn_rectangle_holdout_not_supported",
        "a1_required_decision": (
            "innovation1_runtime_spn_h1_source_gradient_imbalance_supported"
        ),
        "a2_required_decision": "innovation1_runtime_spn_h1_gradient_equalization_partial",
    }
    for key, expected in required_source.items():
        if source.get(key) != expected:
            raise ValueError(f"H1-A3 source field {key} drifted")
    required_candidate = {
        "gradient_combination": (
            "representation_l2_equalized_pcgrad_fixed_order"
        ),
        "representation_parameters": "all_except_shared_classifier",
        "classifier_gradient_combination": "raw_arithmetic_mean",
        "task_sampling": "unchanged_equal_one_batch_per_task",
        "conflict_projection": (
            "fixed_source_order_pcgrad_after_l2_equalization"
        ),
        "seeds": [0, 1],
        "expected_parameter_count": 442466,
    }
    for key, expected in required_candidate.items():
        if candidate.get(key) != expected:
            raise ValueError(f"H1-A3 candidate field {key} drifted")
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
        raise ValueError("H1-A3 target evaluations drifted")
    required_gate = {
        "target_auc_floor": 0.55,
        "target_topology_margin": 0.005,
        "a2_target_retention_tolerance": 0.02,
        "h1_source_macro_retention_tolerance": 0.01,
        "skinny_auc_improvement_over_a2": 0.01,
        "partial_skinny_improvement": 0.005,
        "minimum_conflict_projections_per_seed": 1,
        "required_seeds": [0, 1],
    }
    for key, expected in required_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"H1-A3 gate field {key} drifted")
    for path_key, hash_key in (
        ("h1_config_path", "h1_config_sha256"),
        ("a1_config_path", "a1_config_sha256"),
        ("a2_config_path", "a2_config_sha256"),
    ):
        if config_sha256(project_root / source[path_key]) != source.get(hash_key):
            raise ValueError(f"H1-A3 source hash drifted: {path_key}")
    a2_gate = _read_json(project_root / source["a2_output_root"] / "gate.json")
    if a2_gate.get("decision") != source["a2_required_decision"]:
        raise ValueError("H1-A3 A2 source gate drifted")
    return config


def run_h1_equalized_pcgrad(
    *,
    config: dict[str, Any],
    config_sha256_value: str,
    output_root: Path,
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    payload = run_h1_gradient_equalization(
        config=config,
        config_sha256_value=config_sha256_value,
        output_root=output_root,
        project_root=project_root,
        progress_callback=progress_callback,
    )
    a2_root = project_root / config["source"]["a2_output_root"]
    payload["a2_target_auc"] = _read_json(a2_root / "target-metrics.json")
    payload["a2_source_auc"] = _read_json(a2_root / "source-metrics.json")
    projections_by_seed = {
        str(seed): sum(
            int(row["conflict_projections"])
            for row in payload["gradient_scales"]
            if row["seed"] == seed
        )
        for seed in EXPECTED_SEEDS
    }
    payload["conflict_projections_by_seed"] = projections_by_seed
    payload["validation"]["checks"]["conflict_projections_observed"] = all(
        value >= config["gate"]["minimum_conflict_projections_per_seed"]
        for value in projections_by_seed.values()
    )
    payload["validation"]["status"] = (
        "pass"
        if all(payload["validation"]["checks"].values())
        else "fail"
    )
    return payload


def adjudicate_h1_equalized_pcgrad(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    per_seed: dict[str, Any] = {}
    full_pass = payload["validation"]["status"] == "pass"
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        target = payload["candidate_target_auc"][key]
        a2_target = payload["a2_target_auc"][key]
        source = payload["candidate_source_auc"][key]
        h1_source = payload["anchor_source_auc"][key]
        a2_source = payload["a2_source_auc"][key]
        correct = target["candidate_correct"]
        margins = {
            name: correct - target[name]
            for name in (
                "candidate_corrupted_target",
                "candidate_no_topology_target",
            )
        }
        source_macro = float(np.mean(list(source.values())))
        h1_macro = float(np.mean(list(h1_source.values())))
        skinny_delta = source["skinny64"] - a2_source["skinny64"]
        checks = {
            "target_auc_floor": correct >= gate_config["target_auc_floor"],
            "target_topology_margins": all(
                value >= gate_config["target_topology_margin"]
                for value in margins.values()
            ),
            "a2_target_retained": correct
            >= a2_target["candidate_correct"]
            - gate_config["a2_target_retention_tolerance"],
            "h1_source_macro_retained": source_macro
            >= h1_macro - gate_config["h1_source_macro_retention_tolerance"],
            "skinny_improved_over_a2": skinny_delta
            >= gate_config["skinny_auc_improvement_over_a2"],
            "conflict_projection_observed": payload[
                "conflict_projections_by_seed"
            ][key]
            >= gate_config["minimum_conflict_projections_per_seed"],
        }
        seed_pass = all(checks.values())
        full_pass = full_pass and seed_pass
        per_seed[key] = {
            "candidate_target_auc": correct,
            "a2_target_auc": a2_target["candidate_correct"],
            "target_delta_vs_a2": correct - a2_target["candidate_correct"],
            "target_margins": margins,
            "candidate_source_macro_auc": source_macro,
            "h1_source_macro_auc": h1_macro,
            "source_macro_delta_vs_h1": source_macro - h1_macro,
            "candidate_skinny_auc": source["skinny64"],
            "a2_skinny_auc": a2_source["skinny64"],
            "skinny_auc_delta_vs_a2": skinny_delta,
            "conflict_projections": payload["conflict_projections_by_seed"][key],
            "checks": checks,
            "pass": seed_pass,
        }
    partial = any(
        row["skinny_auc_delta_vs_a2"]
        >= gate_config["partial_skinny_improvement"]
        for row in per_seed.values()
    )
    if payload["validation"]["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_h1_equalized_pcgrad_protocol_invalid"
        next_action = "repair the exact projection, checkpoint, cache or leakage failure"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_h1_equalized_pcgrad_supported"
        next_action = (
            "preregister a second independent whole-cipher holdout with the "
            "same equalized fixed-order PCGrad Runtime-E4"
        )
    elif partial:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_equalized_pcgrad_partial"
        next_action = (
            "retain only the supported optimizer evidence and run a no-training "
            "representation accessibility audit before any further training"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_equalized_pcgrad_not_supported"
        next_action = (
            "close optimizer modifications and run a no-training per-cipher "
            "representation geometry and classifier accessibility audit"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": payload["validation"]["status"] == "pass",
        "full_pass": full_pass,
        "partial_skinny_improvement": partial,
        "per_seed": per_seed,
        "training_or_optimizer_steps_on_target": 0,
        "target_training_rows": 0,
        "claim_scope": (
            "local 2048/class/source optimizer-only RECTANGLE whole-cipher "
            "holdout diagnostic; not formal scale, universality, attack or SOTA"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "add architecture, cipher routing or another optimizer treatment",
            "train or select on RECTANGLE",
            "change samples, epochs, labels, negatives or remote scale",
            "claim universal adaptation from one holdout cipher",
        ],
    }


def write_h1_equalized_pcgrad_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    write_h1_gradient_equalization_artifacts(
        payload=payload,
        gate=gate,
        output_root=output_root,
    )
    _write_json(
        output_root / "conflict-projections.json",
        payload["conflict_projections_by_seed"],
    )
    render_h1_equalized_pcgrad_svg(payload, gate, output_root / "curves.svg")


def render_h1_equalized_pcgrad_svg(
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
        ("candidate_correct", "A3正确结构"),
        ("candidate_corrupted_target", "A3损坏结构"),
        ("candidate_no_topology_target", "A3无拓扑"),
        ("a2", "A2正确结构"),
        ("h1", "H1原始锚点"),
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F8C8D")
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
            key = str(seed)
            values = []
            for name, _ in target_labels:
                if name == "a2":
                    values.append(payload["a2_target_auc"][key]["candidate_correct"])
                elif name == "h1":
                    values.append(payload["anchor_target_auc"][key]["candidate_correct"])
                else:
                    values.append(payload["candidate_target_auc"][key][name])
            bars = axes[0, column].barh(
                range(len(target_labels)), values, color=colors
            )
            axes[0, column].axvline(0.5, color="#34495E")
            axes[0, column].axvline(0.55, color="#7B2CBF", linestyle="--")
            axes[0, column].set_xlim(0.48, 0.75)
            axes[0, column].set_yticks(
                range(len(target_labels)), [label for _, label in target_labels]
            )
            axes[0, column].bar_label(bars, fmt="%.4f", padding=3)
            axes[0, column].set_xlabel("未见 RECTANGLE 验证 AUC")
            axes[0, column].set_title(f"seed{seed}：零微调目标结果")

            y = np.arange(len(EXPECTED_SOURCES))
            series = (
                (payload["candidate_source_auc"][key], "A3投影", -0.22, "#0072B2"),
                (payload["a2_source_auc"][key], "A2归一化", 0.0, "#CC79A7"),
                (payload["anchor_source_auc"][key], "H1锚点", 0.22, "#AAB7B8"),
            )
            for source, label, offset, color in series:
                source_bars = axes[1, column].barh(
                    y + offset,
                    [source[name] for name in EXPECTED_SOURCES],
                    height=0.2,
                    color=color,
                    label=label,
                )
                axes[1, column].bar_label(
                    source_bars, fmt="%.3f", padding=2, fontsize=7
                )
            axes[1, column].axvline(0.5, color="#34495E")
            axes[1, column].set_xlim(0.4, 1.0)
            axes[1, column].set_yticks(
                y, [display[name] for name in EXPECTED_SOURCES]
            )
            axes[1, column].set_xlabel("四源验证 AUC")
            axes[1, column].set_title(f"seed{seed}：逐密码源验证对比")
            axes[1, column].legend(frameon=False, loc="lower right")
        fig.suptitle(
            "创新1 H1-A3：等化后固定顺序 PCGrad 的 RECTANGLE 整密码留出\n"
            "仅移除负表示梯度分量；RECTANGLE 不参与训练、选模或微调",
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


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_h1_equalized_pcgrad_supported": (
            "目标与源保持双seed全过，开放第二独立留出设计"
        ),
        "innovation1_runtime_spn_h1_equalized_pcgrad_partial": (
            "SKINNY部分改善但未全过，停止训练并转表示可达性审计"
        ),
        "innovation1_runtime_spn_h1_equalized_pcgrad_not_supported": (
            "冲突投影未修复源保持，关闭优化器改动并转表示审计"
        ),
        "innovation1_runtime_spn_h1_equalized_pcgrad_protocol_invalid": "协议无效",
    }.get(decision, decision)


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


__all__ = [
    "adjudicate_h1_equalized_pcgrad",
    "config_sha256",
    "load_and_validate_h1_equalized_pcgrad_config",
    "render_h1_equalized_pcgrad_svg",
    "run_h1_equalized_pcgrad",
    "write_h1_equalized_pcgrad_artifacts",
]
