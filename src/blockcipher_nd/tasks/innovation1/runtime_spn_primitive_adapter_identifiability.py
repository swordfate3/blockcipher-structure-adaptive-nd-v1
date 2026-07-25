from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
    config_sha256,
    load_and_validate_joint_config,
)
from blockcipher_nd.training.metrics import binary_auc


EXPECTED_PROBES = ("disabled", "source", "uniform", "shuffled", "amplified")
ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_and_validate_identifiability_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("adapter identifiability audit schema_version must be 1")
    source = payload.get("source", {})
    audit = payload.get("audit", {})
    gate = payload.get("gate", {})
    if tuple(audit.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("adapter identifiability audit must use seeds 0 and 1")
    expected_audit = {
        "split": "train",
        "rows_per_cipher": 4096,
        "batch_size": 256,
        "loss": "mse",
        "device": "cpu",
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"adapter identifiability audit field {key} drifted")
    expected_probes = {
        "disabled": {"mode": "correct", "scale": 0.0, "reference": "disabled"},
        "source": {"mode": "correct", "scale": 0.1, "reference": "disabled"},
        "uniform": {"mode": "uniform", "scale": 0.1, "reference": "source"},
        "shuffled": {"mode": "shuffled", "scale": 0.1, "reference": "source"},
        "amplified": {"mode": "correct", "scale": 0.5, "reference": "source"},
    }
    if audit.get("probes") != expected_probes:
        raise ValueError("adapter identifiability probe panel drifted")
    expected_source = {
        "required_decision": "innovation1_runtime_spn_primitive_adapter_joint_not_supported",
        "role": "correct",
        "mode": "correct",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"adapter identifiability source field {key} drifted")
    expected_gate = {
        "active_median_relative_rms": 0.1,
        "active_task_relative_rms": 0.05,
        "active_task_count": 4,
        "route_median_relative_rms": 0.05,
        "rank_collapse_effective_rank": 2.0,
        "scale_macro_auc_gain": 0.005,
        "scale_per_task_auc_floor": -0.005,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"adapter identifiability gate field {key} drifted")
    source_config_path = project_root / source["config_path"]
    source_config = load_and_validate_joint_config(source_config_path)
    if config_sha256(source_config_path) != source.get("config_sha256"):
        raise ValueError("adapter identifiability source config hash drifted")
    if source_config["run_id"] not in source["output_root"]:
        raise ValueError("adapter identifiability source output root drifted")
    return payload


def adapter_rank_profile(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for adapter in ("fan_in_1", "multi_source"):
        prefix = f"primitive_adapters.{adapter}"
        down = state_dict[f"{prefix}.down.weight"].detach().cpu().to(torch.float64)
        up = state_dict[f"{prefix}.up.weight"].detach().cpu().to(torch.float64)
        singular_values = torch.linalg.svdvals(up @ down)
        positive = singular_values[singular_values > 1e-12]
        normalized = positive / positive.sum() if positive.numel() else positive
        effective_rank = (
            float(torch.exp(-(normalized * torch.log(normalized)).sum()))
            if normalized.numel()
            else 0.0
        )
        profiles[adapter] = {
            "down_frobenius_norm": float(torch.linalg.vector_norm(down)),
            "up_frobenius_norm": float(torch.linalg.vector_norm(up)),
            "linearized_effective_rank": effective_rank,
            "linearized_numerical_rank": int(positive.numel()),
            "linearized_singular_values": [float(value) for value in singular_values],
            "nonlinearity": "GELU; linearized rank is a proxy only",
        }
    return profiles


def run_adapter_identifiability_audit(
    *,
    config: dict[str, Any],
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source = config["source"]
    source_config_path = project_root / source["config_path"]
    source_config = load_and_validate_joint_config(source_config_path)
    source_root = project_root / source["output_root"]
    source_gate = json.loads((source_root / "gate.json").read_text(encoding="utf-8"))
    structures = {
        item["name"]: load_runtime_spn_descriptor(
            item["runtime_structure_path"],
            rounds=int(source_config["model"]["runtime_rounds"]),
            round_start=int(item["runtime_round_start"]),
        ).structure
        for item in source_config["protocols"]
    }
    rows: list[dict[str, Any]] = []
    rank_profiles: dict[str, Any] = {}
    checkpoint_checks: list[dict[str, Any]] = []
    cache_checks: list[dict[str, Any]] = []
    for seed in config["audit"]["seeds"]:
        checkpoint_path = source_root / "checkpoints" / f"seed{seed}-correct.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_valid = all(
            (
                checkpoint.get("seed") == seed,
                checkpoint.get("role") == source["role"],
                checkpoint.get("mode") == source["mode"],
                checkpoint.get("config_sha256") == source["config_sha256"],
            )
        )
        checkpoint_checks.append(
            {"seed": seed, "path": str(checkpoint_path), "valid": checkpoint_valid}
        )
        rank_profiles[str(seed)] = adapter_rank_profile(checkpoint["state_dict"])
        models = {
            probe: _load_probe_model(
                source_config["model"],
                checkpoint["state_dict"],
                mode=probe_spec["mode"],
                scale=float(probe_spec["scale"]),
                device=config["audit"]["device"],
            )
            for probe, probe_spec in config["audit"]["probes"].items()
        }
        for task in EXPECTED_CIPHERS:
            _emit(progress_callback, "identifiability_task_start", seed=seed, task=task)
            cache_root = source_root / "cache" / f"seed{seed}" / task / "train"
            features, labels, cache_valid = _load_source_cache(
                cache_root,
                source_config=source_config,
                task=task,
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
            logits = {
                probe: _predict_logits(
                    model,
                    structures[task],
                    features,
                    batch_size=int(config["audit"]["batch_size"]),
                )
                for probe, model in models.items()
            }
            for probe in EXPECTED_PROBES:
                reference = config["audit"]["probes"][probe]["reference"]
                reference_logits = logits[reference]
                row = counterfactual_metrics(
                    labels=labels,
                    logits=logits[probe],
                    reference_logits=reference_logits,
                )
                rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "probe": probe,
                        "mode": config["audit"]["probes"][probe]["mode"],
                        "scale": config["audit"]["probes"][probe]["scale"],
                        "reference": reference,
                        **row,
                    }
                )
            _emit(progress_callback, "identifiability_task_done", seed=seed, task=task)
    expected_rows = len(EXPECTED_SEEDS) * len(EXPECTED_CIPHERS) * len(EXPECTED_PROBES)
    validation = {
        "status": "pass"
        if (
            source_gate.get("status") == "hold"
            and source_gate.get("decision") == source["required_decision"]
            and all(row["valid"] for row in checkpoint_checks)
            and all(row["valid"] for row in cache_checks)
            and len(rows) == expected_rows
            and all(_row_finite(row) for row in rows)
        )
        else "fail",
        "source_gate_status": source_gate.get("status"),
        "source_gate_decision": source_gate.get("decision"),
        "checkpoint_checks": checkpoint_checks,
        "cache_checks": cache_checks,
        "result_rows": len(rows),
        "expected_result_rows": expected_rows,
        "all_metrics_finite": all(_row_finite(row) for row in rows),
        "training_or_optimizer_steps": 0,
        "split": config["audit"]["split"],
    }
    return {
        "config": config,
        "rows": rows,
        "adapter_rank": rank_profiles,
        "validation": validation,
    }


def counterfactual_metrics(
    *,
    labels: np.ndarray,
    logits: np.ndarray,
    reference_logits: np.ndarray,
) -> dict[str, float]:
    labels_float = np.asarray(labels, dtype=np.float32)
    logits_float = np.asarray(logits, dtype=np.float64)
    reference = np.asarray(reference_logits, dtype=np.float64)
    delta = logits_float - reference
    probabilities = 1.0 / (1.0 + np.exp(-logits_float))
    reference_probabilities = 1.0 / (1.0 + np.exp(-reference))
    reference_std = max(float(np.std(reference)), 1e-12)
    return {
        "auc": binary_auc(labels_float, probabilities.astype(np.float32)),
        "reference_auc": binary_auc(
            labels_float,
            reference_probabilities.astype(np.float32),
        ),
        "auc_delta_vs_reference": float(
            binary_auc(labels_float, probabilities.astype(np.float32))
            - binary_auc(labels_float, reference_probabilities.astype(np.float32))
        ),
        "mean_abs_logit_delta_vs_reference": float(np.mean(np.abs(delta))),
        "rms_logit_delta_vs_reference": float(np.sqrt(np.mean(delta**2))),
        "relative_rms_logit_delta_vs_reference": float(
            np.sqrt(np.mean(delta**2)) / reference_std
        ),
        "threshold_flip_fraction_vs_reference": float(
            np.mean((logits_float >= 0.0) != (reference >= 0.0))
        ),
        "logit_std": float(np.std(logits_float)),
        "reference_logit_std": float(np.std(reference)),
    }


def adjudicate_adapter_identifiability(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    rows = payload["rows"]
    per_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        source_effects = [
            row["relative_rms_logit_delta_vs_reference"]
            for row in rows
            if row["seed"] == seed and row["probe"] == "source"
        ]
        route_effects = {
            probe: [
                row["relative_rms_logit_delta_vs_reference"]
                for row in rows
                if row["seed"] == seed and row["probe"] == probe
            ]
            for probe in ("uniform", "shuffled")
        }
        source_auc = {
            row["task"]: row["auc"]
            for row in rows
            if row["seed"] == seed and row["probe"] == "source"
        }
        amplified_auc = {
            row["task"]: row["auc"]
            for row in rows
            if row["seed"] == seed and row["probe"] == "amplified"
        }
        scale_deltas = {
            task: amplified_auc[task] - source_auc[task] for task in EXPECTED_CIPHERS
        }
        functionally_active = bool(
            np.median(source_effects) >= config["gate"]["active_median_relative_rms"]
            and sum(
                value >= config["gate"]["active_task_relative_rms"]
                for value in source_effects
            )
            >= config["gate"]["active_task_count"]
        )
        route_specialized = all(
            float(np.median(values)) >= config["gate"]["route_median_relative_rms"]
            for values in route_effects.values()
        )
        scale_useful = bool(
            float(np.mean(list(scale_deltas.values())))
            >= config["gate"]["scale_macro_auc_gain"]
            and min(scale_deltas.values()) >= config["gate"]["scale_per_task_auc_floor"]
        )
        per_seed[str(seed)] = {
            "source_relative_rms_by_task": dict(
                zip(EXPECTED_CIPHERS, source_effects, strict=True)
            ),
            "source_median_relative_rms": float(np.median(source_effects)),
            "source_active_task_count": sum(
                value >= config["gate"]["active_task_relative_rms"]
                for value in source_effects
            ),
            "route_median_relative_rms": {
                probe: float(np.median(values))
                for probe, values in route_effects.items()
            },
            "amplified_minus_source_auc_by_task": scale_deltas,
            "amplified_minus_source_macro_auc": float(
                np.mean(list(scale_deltas.values()))
            ),
            "functionally_active": functionally_active,
            "route_specialized": route_specialized,
            "useful_scale_sensitivity": scale_useful,
        }
    rank_collapsed = any(
        all(
            payload["adapter_rank"][str(seed)][adapter]["linearized_effective_rank"]
            < config["gate"]["rank_collapse_effective_rank"]
            for seed in EXPECTED_SEEDS
        )
        for adapter in ("fan_in_1", "multi_source")
    )
    protocol_valid = payload["validation"]["status"] == "pass"
    active = all(row["functionally_active"] for row in per_seed.values())
    specialized = all(row["route_specialized"] for row in per_seed.values())
    scale_useful = all(row["useful_scale_sensitivity"] for row in per_seed.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_runtime_spn_adapter_identifiability_invalid"
        next_action = "repair and rerun the frozen audit without interpreting metrics"
    elif not active:
        status = "pass"
        decision = "innovation1_runtime_spn_additive_adapter_functionally_weak"
        next_action = (
            "replace the weak additive residual with one parameter-matched local "
            "structure-conditioned FiLM/gated modulation; keep descriptors and data frozen"
        )
    elif rank_collapsed:
        status = "pass"
        decision = "innovation1_runtime_spn_adapter_rank_collapsed"
        next_action = "test one rank-regularized candidate before increasing rank or adding experts"
    elif not specialized:
        status = "pass"
        decision = "innovation1_runtime_spn_adapter_active_not_route_specialized"
        next_action = (
            "retain the shared backbone and refine one local descriptor or add one "
            "specialization objective; do not increase rank"
        )
    elif scale_useful:
        status = "pass"
        decision = "innovation1_runtime_spn_adapter_scale05_training_gate_open"
        next_action = (
            "train one parameter-matched scale-0.5 candidate against the frozen "
            "scale-0.1 source and required routing controls"
        )
    else:
        status = "pass"
        decision = "innovation1_runtime_spn_additive_adapter_replace_with_film"
        next_action = (
            "replace the active but non-improving additive Adapter with one matched "
            "structure-conditioned FiLM/gated modulation"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "per_seed": per_seed,
        "functionally_active_both_seeds": active,
        "route_specialized_both_seeds": specialized,
        "rank_collapsed": rank_collapsed,
        "useful_scale_sensitivity_both_seeds": scale_useful,
        "training_or_optimizer_steps": 0,
        "claim_scope": (
            "frozen-checkpoint training-split functional counterfactual audit; not a "
            "new trained model, validation gain, or unseen-cipher result"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "add experts, samples, epochs, rank, or remote compute mechanically",
            "select a scale from validation data",
            "route on cipher identity or global fingerprints",
        ],
    }


def write_adapter_identifiability_artifacts(
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
    _write_csv(output_root / "counterfactual_metrics.csv", payload["rows"])
    _write_json(output_root / "adapter_rank.json", payload["adapter_rank"])
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "functionally_active_both_seeds": gate["functionally_active_both_seeds"],
            "route_specialized_both_seeds": gate["route_specialized_both_seeds"],
            "rank_collapsed": gate["rank_collapsed"],
            "useful_scale_sensitivity_both_seeds": gate[
                "useful_scale_sensitivity_both_seeds"
            ],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    render_adapter_identifiability_svg(payload, gate, output_root / "curves.svg")


def render_adapter_identifiability_svg(
    payload: dict[str, Any],
    gate: dict[str, Any],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    task_labels = {
        "gift64": "GIFT",
        "skinny64": "SKINNY",
        "rectangle80": "RECTANGLE",
        "uknit64": "uKNIT",
        "dialga128": "Dialga",
    }
    probes = (
        ("source", "原比例 - 关闭 Adapter", "#0072B2", "o"),
        ("uniform", "均匀路由 - 正确路由", "#D55E00", "s"),
        ("shuffled", "打乱路由 - 正确路由", "#009E73", "^"),
        ("amplified", "放大比例 - 原比例", "#7A3E9D", "D"),
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 8.4),
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.78)},
    )
    y = np.arange(len(EXPECTED_CIPHERS))
    offsets = (-0.24, -0.08, 0.08, 0.24)
    values_all = [0.0, 0.05, 0.10]
    for seed, axis in zip(EXPECTED_SEEDS, axes[:2], strict=True):
        for offset, (probe, label, color, marker) in zip(offsets, probes, strict=True):
            values = [
                next(
                    row["relative_rms_logit_delta_vs_reference"]
                    for row in payload["rows"]
                    if row["seed"] == seed
                    and row["task"] == task
                    and row["probe"] == probe
                )
                for task in EXPECTED_CIPHERS
            ]
            values_all.extend(values)
            axis.scatter(
                values,
                y + offset,
                color=color,
                marker=marker,
                s=58,
                label=label,
                zorder=3,
            )
        axis.axvline(0.05, color="#777777", linestyle=":", linewidth=1.2)
        axis.axvline(0.10, color="#333333", linestyle="--", linewidth=1.2)
        axis.set_title(f"随机种子 seed{seed}", fontsize=13, pad=10)
        axis.set_xlabel("相对参考输出的 RMS logit 改变量", fontsize=11)
        axis.set_yticks(y, [task_labels[task] for task in EXPECTED_CIPHERS])
        axis.invert_yaxis()
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    high = max(values_all)
    for axis in axes[:2]:
        axis.set_xlim(-0.02 * max(1.0, high), high * 1.08)
    summary_axis = axes[2]
    summary_axis.axis("off")
    summary_axis.set_title("冻结审计裁决", fontsize=13, pad=10)
    rank_lines = []
    for seed in EXPECTED_SEEDS:
        rank_lines.append(f"seed{seed} 线性化有效秩")
        for adapter, label in (
            ("fan_in_1", "单来源"),
            ("multi_source", "多来源"),
        ):
            rank = payload["adapter_rank"][str(seed)][adapter][
                "linearized_effective_rank"
            ]
            rank_lines.append(f"  {label}: {rank:.2f} / 8")
    summary_lines = [
        *rank_lines,
        "",
        f"功能贡献充分：{'是' if gate['functionally_active_both_seeds'] else '否'}",
        f"路由已专门化：{'是' if gate['route_specialized_both_seeds'] else '否'}",
        f"低秩权重坍缩：{'是' if gate['rank_collapsed'] else '否'}",
        f"放大比例有用：{'是' if gate['useful_scale_sensitivity_both_seeds'] else '否'}",
        "",
        "虚线 0.10：功能贡献门槛",
        "点线 0.05：路由差异门槛",
        "所有比较使用同一 checkpoint，",
        "没有训练或更新参数。",
    ]
    summary_axis.text(
        0.02,
        0.96,
        "\n".join(summary_lines),
        transform=summary_axis.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.5,
    )
    fig.suptitle(
        "创新1：结构原语 Adapter 是否真正影响五密码共享模型\n"
        "数值越大，表示关闭、换路由或放大 Adapter 后，模型输出变化越明显",
        fontsize=16,
        y=0.985,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.46, 0.015),
    )
    fig.tight_layout(rect=(0.04, 0.08, 0.99, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _load_probe_model(
    model_config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    *,
    mode: str,
    scale: float,
    device: str,
) -> RuntimeE4EquivariantSpnDistinguisher:
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=int(model_config["hidden_dim"]),
        pair_embedding_dim=int(model_config["pair_embedding_dim"]),
        processor_steps=int(model_config["processor_steps"]),
        dropout=float(model_config["dropout"]),
        sbox_context_mode=model_config["sbox_context_mode"],
        cell_input_mode=model_config["cell_input_mode"],
        round_window_mode=model_config["round_window_mode"],
        primitive_adapter_mode=mode,
        primitive_adapter_rank=int(model_config["primitive_adapter_rank"]),
        primitive_adapter_scale=scale,
    )
    model = RuntimeE4EquivariantSpnDistinguisher(spec)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _load_source_cache(
    cache_root: Path,
    *,
    source_config: dict[str, Any],
    task: str,
    seed: int,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    features = np.load(cache_root / "features.npy", mmap_mode="r")
    labels = np.load(cache_root / "labels.npy", mmap_mode="r")
    metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    protocol = next(item for item in source_config["protocols"] if item["name"] == task)
    training = source_config["training"]
    valid = bool(
        features.shape[0] == labels.shape[0] == expected_rows
        and metadata.get("samples_total") == expected_rows
        and metadata.get("samples_per_class") == training["samples_per_class"]
        and metadata.get("pairs_per_sample") == training["pairs_per_sample"]
        and metadata.get("negative_mode") == training["negative_mode"]
        and metadata.get("sample_structure") == training["sample_structure"]
        and metadata.get("rounds") == protocol["rounds"]
        and metadata.get("input_difference") == int(protocol["input_difference"], 0)
        and metadata.get("seed") == seed
    )
    return features, labels, valid


def _predict_logits(
    model: RuntimeE4EquivariantSpnDistinguisher,
    structure: RuntimeSpnStructure,
    features: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), batch_size):
            stop = min(start + batch_size, int(features.shape[0]))
            batch = torch.as_tensor(
                np.asarray(features[start:stop]).copy(),
                dtype=torch.float32,
                device=device,
            )
            runtime = batch.reshape(batch.shape[0], -1, 2, structure.block_bits).flip(
                -1
            )
            values.append(model(runtime, structure).squeeze(1).cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def _row_finite(row: dict[str, Any]) -> bool:
    return all(
        np.isfinite(value)
        for key, value in row.items()
        if key
        in {
            "auc",
            "reference_auc",
            "auc_delta_vs_reference",
            "mean_abs_logit_delta_vs_reference",
            "rms_logit_delta_vs_reference",
            "relative_rms_logit_delta_vs_reference",
            "threshold_flip_fraction_vs_reference",
            "logit_std",
            "reference_logit_std",
        }
    )


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
    "EXPECTED_PROBES",
    "adapter_rank_profile",
    "adjudicate_adapter_identifiability",
    "counterfactual_metrics",
    "load_and_validate_identifiability_config",
    "render_adapter_identifiability_svg",
    "run_adapter_identifiability_audit",
    "write_adapter_identifiability_artifacts",
]
