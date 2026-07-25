from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
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
from blockcipher_nd.training.optim import compute_loss, make_loss


GRADIENT_VIEWS = (
    "shared_backbone",
    "all_adapters",
    "fan_in_1_adapter",
    "multi_source_adapter",
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_and_validate_audit_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("primitive descriptor audit schema_version must be 1")
    source = payload.get("source", {})
    audit = payload.get("audit", {})
    gate = payload.get("gate", {})
    if tuple(audit.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("primitive descriptor audit must use seeds 0 and 1")
    expected_audit = {
        "split": "train",
        "rows_per_cipher": 4096,
        "batch_size": 256,
        "loss": "mse",
        "device": "cpu",
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"primitive descriptor audit field {key} drifted")
    if tuple(audit.get("gradient_views", ())) != GRADIENT_VIEWS:
        raise ValueError("primitive descriptor audit gradient views drifted")
    expected_source = {
        "required_decision": "innovation1_runtime_spn_primitive_adapter_joint_not_supported",
        "role": "correct",
        "mode": "correct",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"primitive descriptor audit source field {key} drifted")
    expected_gate = {
        "minimum_collision_groups": 2,
        "weak_alignment_cosine": 0.1,
        "shared_negative_pair_fraction": 0.5,
        "shared_mean_cosine": 0.0,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"primitive descriptor audit gate field {key} drifted")

    source_config_path = project_root / source["config_path"]
    source_config = load_and_validate_joint_config(source_config_path)
    if config_sha256(source_config_path) != source.get("config_sha256"):
        raise ValueError("primitive descriptor audit source config hash drifted")
    if source_config["run_id"] not in source["output_root"]:
        raise ValueError("primitive descriptor audit source output root drifted")
    return payload


def build_descriptor_profiles(
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for name, structure in structures.items():
        round_profiles: list[dict[str, Any]] = []
        normalized_route_signature: list[tuple[int, int]] = []
        sbox_hashes: set[str] = set()
        for round_index in range(structure.rounds):
            inverse = structure.inverse_linear_matrices[round_index]
            row_fan_in = inverse.sum(dim=1).to(torch.long)
            fan_in_histogram = {
                str(int(value)): int((row_fan_in == value).sum())
                for value in torch.unique(row_fan_in, sorted=True)
            }
            cell_routes: list[str] = []
            cell_source_patterns: list[dict[str, Any]] = []
            for cell in range(structure.cells):
                target_rows = structure.cell_membership == cell
                target_fan_in = row_fan_in[target_rows]
                route = (
                    "fan_in_1"
                    if bool(torch.all(target_fan_in == 1))
                    else "multi_source"
                )
                cell_routes.append(route)
                source_bits = torch.nonzero(
                    inverse[target_rows].sum(dim=0), as_tuple=False
                ).flatten()
                source_cells = torch.unique(structure.cell_membership[source_bits])
                cell_source_patterns.append(
                    {
                        "route": route,
                        "row_fan_in": [int(value) for value in target_fan_in],
                        "source_cell_count": int(source_cells.numel()),
                    }
                )
                truth = structure.sbox_truth_bits[round_index, cell]
                sbox_hashes.add(hashlib.sha256(truth.numpy().tobytes()).hexdigest())
            fan_in_1_count = cell_routes.count("fan_in_1")
            multi_source_count = cell_routes.count("multi_source")
            divisor = math.gcd(fan_in_1_count, multi_source_count) or structure.cells
            normalized_route_signature.append(
                (fan_in_1_count // divisor, multi_source_count // divisor)
            )
            round_profiles.append(
                {
                    "round_index": round_index,
                    "fan_in_1_cells": fan_in_1_count,
                    "multi_source_cells": multi_source_count,
                    "fan_in_histogram": fan_in_histogram,
                    "cell_source_patterns": cell_source_patterns,
                }
            )
        signature = "|".join(
            f"{fan_in_1}:{multi_source}"
            for fan_in_1, multi_source in normalized_route_signature
        )
        profiles[name] = {
            "block_bits": structure.block_bits,
            "cells": structure.cells,
            "rounds": structure.rounds,
            "normalized_route_signature": signature,
            "unique_sbox_count": len(sbox_hashes),
            "unique_transition_count": structure.unique_transition_count,
            "transition_sha256s": list(structure.transition_sha256s()),
            "window_sha256": structure.window_sha256(),
            "round_profiles": round_profiles,
        }
    return profiles


def find_descriptor_collisions(
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_signature: dict[str, list[str]] = {}
    for name, profile in profiles.items():
        by_signature.setdefault(profile["normalized_route_signature"], []).append(name)
    collisions: list[dict[str, Any]] = []
    for signature, names in sorted(by_signature.items()):
        fingerprints = {profiles[name]["window_sha256"] for name in names}
        if len(names) > 1 and len(fingerprints) > 1:
            collisions.append(
                {
                    "normalized_route_signature": signature,
                    "tasks": sorted(names),
                    "distinct_window_fingerprints": len(fingerprints),
                }
            )
    return collisions


def run_primitive_descriptor_audit(
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
    profiles = build_descriptor_profiles(structures)
    collisions = find_descriptor_collisions(profiles)
    gradient_vectors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    gradient_norm_rows: list[dict[str, Any]] = []
    cache_checks: list[dict[str, Any]] = []
    checkpoint_checks: list[dict[str, Any]] = []

    for seed in config["audit"]["seeds"]:
        seed_key = str(seed)
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
            {
                "seed": seed,
                "path": str(checkpoint_path),
                "valid": checkpoint_valid,
            }
        )
        model = RuntimeE4EquivariantSpnDistinguisher(
            _source_model_spec(source_config["model"])
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(torch.device(config["audit"]["device"]))
        gradient_vectors[seed_key] = {}
        for name in EXPECTED_CIPHERS:
            _emit(progress_callback, "gradient_task_start", seed=seed, task=name)
            cache_root = source_root / "cache" / f"seed{seed}" / name / "train"
            features, labels, metadata, cache_valid = _load_and_validate_cache(
                cache_root,
                source_config=source_config,
                task_name=name,
                seed=seed,
                expected_rows=int(config["audit"]["rows_per_cipher"]),
            )
            cache_checks.append(
                {
                    "seed": seed,
                    "task": name,
                    "path": str(cache_root),
                    "valid": cache_valid,
                    "rows": int(labels.shape[0]),
                }
            )
            vectors = _mean_loss_gradients(
                model=model,
                structure=structures[name],
                features=features,
                labels=labels,
                batch_size=int(config["audit"]["batch_size"]),
                loss_name=config["audit"]["loss"],
            )
            gradient_vectors[seed_key][name] = vectors
            for view, vector in vectors.items():
                gradient_norm_rows.append(
                    {
                        "seed": seed,
                        "task": name,
                        "view": view,
                        "l2_norm": float(torch.linalg.vector_norm(vector)),
                        "finite": bool(torch.isfinite(vector).all()),
                    }
                )
            _emit(progress_callback, "gradient_task_done", seed=seed, task=name)

    cosine_rows = pairwise_gradient_cosines(gradient_vectors)
    expected_cosine_rows = len(EXPECTED_SEEDS) * len(GRADIENT_VIEWS) * 10
    validation = {
        "status": "pass"
        if (
            source_gate.get("status") == "hold"
            and source_gate.get("decision") == source["required_decision"]
            and all(row["valid"] for row in checkpoint_checks)
            and all(row["valid"] for row in cache_checks)
            and all(row["finite"] for row in gradient_norm_rows)
            and len(cosine_rows) == expected_cosine_rows
        )
        else "fail",
        "source_gate_status": source_gate.get("status"),
        "source_gate_decision": source_gate.get("decision"),
        "checkpoint_checks": checkpoint_checks,
        "cache_checks": cache_checks,
        "all_gradient_vectors_finite": all(row["finite"] for row in gradient_norm_rows),
        "cosine_rows": len(cosine_rows),
        "expected_cosine_rows": expected_cosine_rows,
        "training_or_optimizer_steps": 0,
        "split": config["audit"]["split"],
    }
    return {
        "config": config,
        "descriptor_profiles": profiles,
        "descriptor_collisions": collisions,
        "gradient_cosines": cosine_rows,
        "gradient_norms": gradient_norm_rows,
        "validation": validation,
    }


def pairwise_gradient_cosines(
    gradients: dict[str, dict[str, dict[str, torch.Tensor]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_key, by_task in gradients.items():
        for view in GRADIENT_VIEWS:
            for task_a, task_b in itertools.combinations(EXPECTED_CIPHERS, 2):
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


def adjudicate_primitive_descriptor_audit(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    cosine_rows = payload["gradient_cosines"]
    collisions = payload["descriptor_collisions"]
    same_route_pairs: list[dict[str, Any]] = []
    for collision in collisions:
        signature = collision["normalized_route_signature"]
        if all(part.startswith("1:0") for part in signature.split("|")):
            view = "fan_in_1_adapter"
        elif all(part.startswith("0:1") for part in signature.split("|")):
            view = "multi_source_adapter"
        else:
            view = "all_adapters"
        for task_a, task_b in itertools.combinations(collision["tasks"], 2):
            values = [
                row["cosine"]
                for row in cosine_rows
                if row["view"] == view
                and {row["task_a"], row["task_b"]} == {task_a, task_b}
            ]
            finite_values = [value for value in values if value is not None]
            mean = float(np.mean(finite_values)) if finite_values else None
            same_route_pairs.append(
                {
                    "tasks": [task_a, task_b],
                    "view": view,
                    "per_seed_cosines": finite_values,
                    "mean_cosine": mean,
                    "weak_or_conflicting": bool(
                        finite_values
                        and (
                            any(value < 0.0 for value in finite_values)
                            or mean < config["gate"]["weak_alignment_cosine"]
                        )
                    ),
                }
            )
    shared_by_seed: dict[str, dict[str, float]] = {}
    for seed in EXPECTED_SEEDS:
        values = [
            row["cosine"]
            for row in cosine_rows
            if row["seed"] == seed
            and row["view"] == "shared_backbone"
            and row["cosine"] is not None
        ]
        shared_by_seed[str(seed)] = {
            "mean_off_diagonal_cosine": float(np.mean(values)),
            "negative_pair_fraction": float(np.mean(np.asarray(values) < 0.0)),
        }
    descriptor_priority = bool(
        len(collisions) >= config["gate"]["minimum_collision_groups"]
        and any(row["weak_or_conflicting"] for row in same_route_pairs)
    )
    shared_conflict = all(
        metrics["mean_off_diagonal_cosine"] < config["gate"]["shared_mean_cosine"]
        or metrics["negative_pair_fraction"]
        >= config["gate"]["shared_negative_pair_fraction"]
        for metrics in shared_by_seed.values()
    )
    protocol_valid = payload["validation"]["status"] == "pass"
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_runtime_spn_primitive_descriptor_audit_invalid"
        next_action = "repair and rerun the frozen audit without interpreting metrics"
    elif descriptor_priority and shared_conflict:
        status = "pass"
        decision = "innovation1_runtime_spn_descriptor_and_shared_gradient_conflict"
        next_action = (
            "refine exactly one local primitive descriptor axis for the worst "
            "same-route collision; retain shared-gradient conflict as a required control"
        )
    elif descriptor_priority:
        status = "pass"
        decision = "innovation1_runtime_spn_primitive_descriptor_refinement_priority"
        next_action = (
            "refine exactly one local primitive descriptor axis while preserving the "
            "shared optimizer and all source protocols"
        )
    elif shared_conflict:
        status = "pass"
        decision = "innovation1_runtime_spn_shared_gradient_conflict_priority"
        next_action = (
            "keep the primitive descriptor frozen and test one parameter-matched "
            "multi-task gradient-conflict treatment"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_adapter_identifiability_audit_required"
        next_action = (
            "audit adapter residual scale and rank identifiability before adding "
            "experts, samples, epochs, or remote compute"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "descriptor_collision_groups": collisions,
        "same_route_gradient_pairs": same_route_pairs,
        "shared_backbone_gradient_summary": shared_by_seed,
        "descriptor_refinement_priority": descriptor_priority,
        "shared_gradient_conflict": shared_conflict,
        "training_or_optimizer_steps": 0,
        "claim_scope": (
            "frozen-checkpoint full-training-split gradient and descriptor audit; "
            "not a new distinguisher result and not unseen-cipher evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "mechanically add experts, samples, epochs, or remote compute",
            "route on cipher name, cipher ID, block width, or a global fingerprint",
            "use this audit as a universal or held-out-cipher claim",
        ],
    }


def write_primitive_descriptor_audit_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in payload["gradient_cosines"]
        ),
        encoding="utf-8",
    )
    _write_json(
        output_root / "descriptor_profiles.json",
        {
            "profiles": payload["descriptor_profiles"],
            "collisions": payload["descriptor_collisions"],
        },
    )
    _write_csv(output_root / "gradient_cosines.csv", payload["gradient_cosines"])
    _write_csv(output_root / "gradient_norms.csv", payload["gradient_norms"])
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "descriptor_refinement_priority": gate["descriptor_refinement_priority"],
            "shared_gradient_conflict": gate["shared_gradient_conflict"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    render_descriptor_gradient_audit_svg(payload, gate, output_root / "curves.svg")


def render_descriptor_gradient_audit_svg(
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
    display = {
        "gift64": "GIFT",
        "skinny64": "SKINNY",
        "rectangle80": "RECTANGLE",
        "uknit64": "uKNIT",
        "dialga128": "Dialga",
    }
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 1.0, 0.82), wspace=0.32, hspace=0.34
    )
    image_handle = None
    for row_index, view in enumerate(("shared_backbone", "all_adapters")):
        for column_index, seed in enumerate(EXPECTED_SEEDS):
            axis = fig.add_subplot(grid[row_index, column_index])
            matrix = _cosine_matrix(payload["gradient_cosines"], seed=seed, view=view)
            masked = np.ma.masked_invalid(matrix)
            image_handle = axis.imshow(masked, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
            for y in range(len(EXPECTED_CIPHERS)):
                for x in range(len(EXPECTED_CIPHERS)):
                    value = matrix[y, x]
                    label = "--" if np.isnan(value) else f"{value:.2f}"
                    axis.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white"
                        if not np.isnan(value) and abs(value) > 0.55
                        else "black",
                    )
            title = "共享主干" if view == "shared_backbone" else "两个原语 Adapter 合计"
            axis.set_title(f"seed{seed}：{title}", fontsize=12, pad=8)
            axis.set_xticks(
                range(len(EXPECTED_CIPHERS)),
                [display[name] for name in EXPECTED_CIPHERS],
                rotation=32,
                ha="right",
                fontsize=9,
            )
            axis.set_yticks(
                range(len(EXPECTED_CIPHERS)),
                [display[name] for name in EXPECTED_CIPHERS],
                fontsize=9,
            )
    collision_axis = fig.add_subplot(grid[:, 2])
    collision_axis.axis("off")
    collision_axis.set_title("当前两类路由的结构碰撞", fontsize=13, pad=12)
    lines = []
    for index, collision in enumerate(payload["descriptor_collisions"], start=1):
        signature = collision["normalized_route_signature"]
        route_text = (
            "全部单来源"
            if signature == "1:0|1:0"
            else "全部多来源"
            if signature == "0:1|0:1"
            else signature
        )
        task_text = " / ".join(display[name] for name in collision["tasks"])
        lines.extend(
            [
                f"碰撞 {index}：{route_text}",
                f"  {task_text}",
                f"  实际结构指纹数：{collision['distinct_window_fingerprints']}",
                "",
            ]
        )
    lines.extend(
        [
            "审计裁决",
            f"  描述符细化优先：{'是' if gate['descriptor_refinement_priority'] else '否'}",
            f"  共享梯度冲突：{'是' if gate['shared_gradient_conflict'] else '否'}",
            "",
            "说明",
            "  红色：梯度方向相近",
            "  蓝色：梯度方向相反",
            "  0.00 也可能表示两个任务激活了",
            "  互不重叠的 Adapter 参数块；归因时",
            "  只比较同一路由碰撞组。",
        ]
    )
    collision_axis.text(
        0.02,
        0.96,
        "\n".join(lines),
        transform=collision_axis.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.45,
    )
    fig.suptitle(
        "创新1：五密码共享 SPN 模型的结构描述符与梯度冲突审计\n"
        "同一格越接近 +1 表示两个密码希望参数向相似方向更新；越接近 -1 表示更新方向冲突",
        fontsize=16,
        y=0.985,
    )
    if image_handle is not None:
        colorbar_axis = fig.add_axes((0.14, 0.045, 0.50, 0.025))
        colorbar = fig.colorbar(image_handle, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("梯度余弦相似度", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.19)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _source_model_spec(model: dict[str, Any]) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=int(model["hidden_dim"]),
        pair_embedding_dim=int(model["pair_embedding_dim"]),
        processor_steps=int(model["processor_steps"]),
        dropout=float(model["dropout"]),
        sbox_context_mode=model["sbox_context_mode"],
        cell_input_mode=model["cell_input_mode"],
        round_window_mode=model["round_window_mode"],
        primitive_adapter_mode="correct",
        primitive_adapter_rank=int(model["primitive_adapter_rank"]),
        primitive_adapter_scale=float(model["primitive_adapter_scale"]),
    )


def _load_and_validate_cache(
    cache_root: Path,
    *,
    source_config: dict[str, Any],
    task_name: str,
    seed: int,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], bool]:
    features = np.load(cache_root / "features.npy", mmap_mode="r")
    labels = np.load(cache_root / "labels.npy", mmap_mode="r")
    metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
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
    return features, labels, metadata, valid


def _mean_loss_gradients(
    *,
    model: RuntimeE4EquivariantSpnDistinguisher,
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
        pair_bits = 2 * structure.block_bits
        runtime_features = batch_features.reshape(
            batch_features.shape[0], -1, 2, structure.block_bits
        ).flip(-1)
        if batch_features.shape[1] % pair_bits:
            raise ValueError("audit cache contains incomplete ciphertext pairs")
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
        is_adapter = name.startswith("primitive_adapters.")
        if view == "shared_backbone":
            return not is_adapter
        if view == "all_adapters":
            return is_adapter
        if view == "fan_in_1_adapter":
            return name.startswith("primitive_adapters.fan_in_1.")
        if view == "multi_source_adapter":
            return name.startswith("primitive_adapters.multi_source.")
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


def _cosine_matrix(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    view: str,
) -> np.ndarray:
    size = len(EXPECTED_CIPHERS)
    matrix = np.eye(size, dtype=np.float64)
    indices = {name: index for index, name in enumerate(EXPECTED_CIPHERS)}
    for row in rows:
        if row["seed"] != seed or row["view"] != view:
            continue
        a = indices[row["task_a"]]
        b = indices[row["task_b"]]
        value = np.nan if row["cosine"] is None else row["cosine"]
        matrix[a, b] = matrix[b, a] = value
    return matrix


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
    "adjudicate_primitive_descriptor_audit",
    "build_descriptor_profiles",
    "find_descriptor_collisions",
    "load_and_validate_audit_config",
    "pairwise_gradient_cosines",
    "render_descriptor_gradient_audit_svg",
    "run_primitive_descriptor_audit",
    "write_primitive_descriptor_audit_artifacts",
]
