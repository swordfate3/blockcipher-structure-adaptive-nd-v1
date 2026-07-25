from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential import (
    DifferentialDataset,
    DifferentialDatasetConfig,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTask,
    RuntimeSpnJointTrainingResult,
    evaluate_runtime_spn_joint,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig


EXPECTED_PROTOCOLS = ("gift64", "skinny64", "rectangle80", "uknit64", "dialga128")
EXPECTED_SOURCES = ("gift64", "skinny64", "uknit64", "dialga128")
EXPECTED_ROLES = ("correct", "corrupted", "no_topology")
EXPECTED_TARGET_EVALUATIONS = (
    "candidate_correct",
    "candidate_corrupted_target",
    "candidate_no_topology_target",
    "corrupted_source_control",
    "no_topology_source_control",
)
EXPECTED_SEEDS = (0, 1)
HOLDOUT_CIPHER = "rectangle80"


class RelationModeRuntimeE4(nn.Module):
    """Apply one shared dynamic-structure Runtime-E4 state with a fixed relation mode."""

    def __init__(self, spec: RuntimeParameterizedSpnSpec, relation_mode: str) -> None:
        super().__init__()
        if relation_mode not in {"true", "independent"}:
            raise ValueError("relation_mode must be true or independent")
        self.backbone = RuntimeE4EquivariantSpnDistinguisher(spec)
        self.relation_mode = relation_mode

    def forward(
        self,
        features: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        return self.backbone(features, structure, relation_mode=self.relation_mode)


def load_and_validate_holdout_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("whole-cipher holdout config schema_version must be 1")
    if config.get("holdout_cipher") != HOLDOUT_CIPHER:
        raise ValueError("H1 must hold out RECTANGLE-80")
    if tuple(config.get("source_ciphers", ())) != EXPECTED_SOURCES:
        raise ValueError("H1 source cipher panel drifted")
    if HOLDOUT_CIPHER in config["source_ciphers"]:
        raise ValueError("holdout cipher must not be a source task")
    protocols = config.get("protocols", [])
    if tuple(item.get("name") for item in protocols) != EXPECTED_PROTOCOLS:
        raise ValueError("H1 protocol panel drifted")
    training = config.get("training", {})
    if tuple(training.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("H1 must use seeds 0 and 1")
    if tuple(training.get("roles", {})) != EXPECTED_ROLES:
        raise ValueError("H1 source role panel drifted")
    required_roles = {
        "correct": {"structure": "correct", "relation_mode": "true"},
        "corrupted": {"structure": "corrupted", "relation_mode": "true"},
        "no_topology": {"structure": "correct", "relation_mode": "independent"},
    }
    if training["roles"] != required_roles:
        raise ValueError("H1 source interventions drifted")
    if tuple(config.get("target_evaluations", {})) != EXPECTED_TARGET_EVALUATIONS:
        raise ValueError("H1 target evaluation panel drifted")
    required_training = {
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "feature_encoding": "ciphertext_pair_bits",
        "sample_structure": "independent_pairs",
        "key_rotation_interval": 0,
        "epochs": 10,
        "batch_size": 256,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "lr_scheduler": "none",
        "checkpoint_metric": "val_macro_auc",
        "restore_best_checkpoint": True,
        "device": "cpu",
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(f"H1 training field {key} drifted")
    required_model = {
        "backbone": "RuntimeE4EquivariantSpnDistinguisher",
        "hidden_dim": 64,
        "pair_embedding_dim": 128,
        "processor_steps": 2,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet",
        "round_window_mode": "recurrent_window",
        "runtime_rounds": 2,
        "conditioner": "none",
        "expected_parameter_count": 442466,
    }
    for key, expected in required_model.items():
        if config.get("model", {}).get(key) != expected:
            raise ValueError(f"H1 model field {key} drifted")
    if set(config.get("gate", {})) != {
        "absolute_target_auc_floor",
        "margin",
        "seen_macro_margin",
        "required_seeds",
    }:
        raise ValueError("H1 gate fields drifted")
    return config


def verify_readiness(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    readiness = config["readiness"]
    gate = _read_json(project_root / readiness["gate_path"])
    if gate.get("status") != "pass":
        raise ValueError("H1 readiness did not pass")
    if gate.get("decision") != readiness["required_decision"]:
        raise ValueError("H1 readiness decision drifted")
    if not all(gate.get("checks", {}).values()):
        raise ValueError("H1 readiness contains a failed check")
    return gate


def build_holdout_readiness(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    structures = _load_structures(config)
    roles = {
        role: RelationModeRuntimeE4(_plain_spec(config["model"]), values["relation_mode"])
        for role, values in config["training"]["roles"].items()
    }
    state = roles["correct"].state_dict()
    for model in roles.values():
        model.load_state_dict(state, strict=True)
    parameter_counts = {
        role: sum(parameter.numel() for parameter in model.parameters())
        for role, model in roles.items()
    }
    target_probe = _target_control_probe(roles["correct"], structures[HOLDOUT_CIPHER])
    cache_probe = _cache_probe(config, project_root)
    smoke = _synthetic_holdout_smoke(config, structures)
    checks = {
        "holdout_absent_from_source_panel": HOLDOUT_CIPHER not in EXPECTED_SOURCES,
        "source_panel_exact": tuple(config["source_ciphers"]) == EXPECTED_SOURCES,
        "parameter_matched": set(parameter_counts.values()) == {442466},
        "strict_shared_initial_state": all(
            all(torch.equal(state[name], model.state_dict()[name]) for name in state)
            for model in roles.values()
        ),
        "one_state_handles_all_widths": _all_width_probe(state, config, structures),
        "target_controls_distinct": all(target_probe.values()),
        "cache_contract_ready": cache_probe["passed"],
        "target_train_cache_not_required": not cache_probe["target_train_referenced"],
        "synthetic_source_only_checkpoint": smoke["source_only_checkpoint"],
        "synthetic_target_after_training": smoke["target_evaluated_after_training"],
        "no_task_specific_trainable_state": smoke["task_specific_trainable_state"] is False,
    }
    passed = all(checks.values())
    gate = {
        "run_id": "i1_runtime_spn_rectangle_whole_cipher_holdout_h1_readiness_20260726",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_rectangle_holdout_readiness_passed"
            if passed
            else "innovation1_runtime_spn_rectangle_holdout_protocol_invalid"
        ),
        "checks": checks,
        "parameter_counts": parameter_counts,
        "target_control_probe": target_probe,
        "cache_probe": cache_probe,
        "smoke": smoke,
        "claim_scope": (
            "engineering and zero-leakage readiness only; no target AUC, transfer, "
            "universality, attack, SOTA, or breakthrough claim"
        ),
        "next_action": (
            "run the frozen 2048/class/source two-seed RECTANGLE whole-cipher holdout"
            if passed
            else "repair failed readiness checks before any H1 result-producing run"
        ),
    }
    manifest = {
        "holdout_cipher": HOLDOUT_CIPHER,
        "source_ciphers": list(EXPECTED_SOURCES),
        "parameter_counts": parameter_counts,
        "target_training_loaded": False,
        "checkpoint_selection_tasks": list(EXPECTED_SOURCES),
        "target_evaluation_timing": "after_all_source_roles_complete",
        "cache_probe": cache_probe,
    }
    return manifest, gate


def run_holdout_experiment(
    *,
    config: dict[str, Any],
    config_sha256: str,
    output_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    structures = _load_structures(config)
    checkpoint_root = output_root / "checkpoints"
    role_root = output_root / "role-results"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    role_root.mkdir(parents=True, exist_ok=True)
    roles: dict[int, dict[str, dict[str, Any]]] = {}
    target_loaded_after_sources: dict[int, bool] = {}
    for seed in EXPECTED_SEEDS:
        source_tasks = _load_source_tasks(
            config,
            seed=seed,
            structures=structures,
            progress_callback=progress_callback,
        )
        roles[seed] = {}
        for role in EXPECTED_ROLES:
            role_path = role_root / f"seed{seed}-{role}.json"
            checkpoint_path = checkpoint_root / f"seed{seed}-{role}.pt"
            resumed = _load_resumable_role(
                role_path,
                checkpoint_path,
                config_sha256=config_sha256,
            )
            if resumed is not None:
                roles[seed][role] = resumed
                _emit(progress_callback, "source_role_reused", seed=seed, role=role)
                continue
            intervention = config["training"]["roles"][role]
            tasks = _intervene_tasks(
                source_tasks,
                structure_mode=intervention["structure"],
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = RelationModeRuntimeE4(
                    _plain_spec(config["model"]),
                    intervention["relation_mode"],
                )
            _emit(progress_callback, "source_role_start", seed=seed, role=role)
            result = train_runtime_spn_joint(
                model,
                tasks,
                _training_config(config["training"], seed),
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda event, payload, seed=seed, role=role: progress_callback(
                        event,
                        {"seed": seed, "role": role, **payload},
                    )
                ),
            )
            checkpoint = {
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "seed": seed,
                "role": role,
                "config_sha256": config_sha256,
                "best_epoch": result.metadata["best_epoch"],
                "checkpoint_selection_tasks": list(EXPECTED_SOURCES),
                "holdout_cipher": HOLDOUT_CIPHER,
            }
            torch.save(checkpoint, checkpoint_path)
            payload = _source_role_payload(
                seed=seed,
                role=role,
                intervention=intervention,
                parameter_count=sum(p.numel() for p in model.parameters()),
                config_sha256=config_sha256,
                checkpoint_path=checkpoint_path,
                result=result,
            )
            _write_json(role_path, payload)
            roles[seed][role] = payload
            _emit(
                progress_callback,
                "source_role_done",
                seed=seed,
                role=role,
                best_epoch=result.metadata["best_epoch"],
            )
        all_sources_done = set(roles[seed]) == set(EXPECTED_ROLES)
        if not all_sources_done:
            raise RuntimeError("all source roles must complete before target loading")
        _emit(progress_callback, "target_validation_load_start", seed=seed)
        target_dataset = _load_target_validation(
            config,
            seed=seed,
            progress_callback=progress_callback,
        )
        target_loaded_after_sources[seed] = all_sources_done
        for name, evaluation in config["target_evaluations"].items():
            source_role = evaluation["source_role"]
            checkpoint_path = Path(roles[seed][source_role]["checkpoint_path"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = RelationModeRuntimeE4(
                _plain_spec(config["model"]),
                evaluation["relation_mode"],
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            structure = structures[HOLDOUT_CIPHER]
            if evaluation["structure"] == "corrupted":
                structure = structure.corrupted()
            metrics = _evaluate_target(
                model,
                target_dataset,
                structure,
                config["training"],
            )
            roles[seed].setdefault("target_evaluations", {})[name] = {
                "name": name,
                "source_role": source_role,
                "structure": evaluation["structure"],
                "relation_mode": evaluation["relation_mode"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "metrics": metrics,
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
    return _assemble_payload(
        config=config,
        config_sha256=config_sha256,
        roles=roles,
        target_loaded_after_sources=target_loaded_after_sources,
    )


def adjudicate_holdout_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    validation = payload["validation"]
    floor = float(config["gate"]["absolute_target_auc_floor"])
    margin = float(config["gate"]["margin"])
    seen_margin = float(config["gate"]["seen_macro_margin"])
    per_seed: dict[str, Any] = {}
    full_pass = validation["status"] == "pass"
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        target = payload["target_metrics"][key]
        seen = payload["source_macro_auc"][key]
        candidate = target["candidate_correct"]
        target_deltas = {
            name: candidate - target[name]
            for name in EXPECTED_TARGET_EVALUATIONS
            if name != "candidate_correct"
        }
        seen_deltas = {
            role: seen["correct"] - seen[role]
            for role in ("corrupted", "no_topology")
        }
        checks = {
            "target_auc_floor": candidate >= floor,
            "target_controls": all(value >= margin for value in target_deltas.values()),
            "seen_source_controls": all(
                value >= seen_margin for value in seen_deltas.values()
            ),
        }
        seed_pass = all(checks.values())
        full_pass = full_pass and seed_pass
        per_seed[key] = {
            "candidate_auc": candidate,
            "target_deltas": target_deltas,
            "seen_macro_auc": seen,
            "seen_deltas": seen_deltas,
            "checks": checks,
            "pass": seed_pass,
        }
    if validation["status"] != "pass":
        status = "fail"
        decision = "innovation1_runtime_spn_rectangle_holdout_protocol_invalid"
        next_action = "repair the exact leakage, cache, checkpoint, or geometry failure"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_rectangle_holdout_supported"
        next_action = "preregister a second independent whole-cipher holdout"
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_rectangle_holdout_not_supported"
        next_action = (
            "audit source-task calibration versus representation alignment; do not "
            "add target supervision, reopen residual modules, or scale remotely"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": validation["status"] == "pass",
        "full_pass": full_pass,
        "per_seed": per_seed,
        "claim_scope": (
            "local 2048/class/source RECTANGLE whole-cipher holdout only; not "
            "formal scale, universal SPN adaptation, attack, SOTA, or breakthrough"
        ),
        "blocked_actions": [
            "train or tune a RECTANGLE-specific head",
            "select checkpoints or thresholds on held-out RECTANGLE",
            "rescue a hold with more epochs, samples, adapters, MoE, or remote compute",
            "claim universality from one held-out cipher",
        ],
        "next_action": next_action,
    }


def write_holdout_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "source-metrics.json", payload["source_metrics"])
    _write_json(output_root / "target-metrics.json", payload["target_metrics"])
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": payload["config"]["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["rows"]),
        encoding="utf-8",
    )
    history = payload["history"]
    with (output_root / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        if history:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    render_holdout_svg(gate, output_root / "curves.svg")


def render_holdout_svg(gate: dict[str, Any], output: Path) -> None:
    labels = {
        "candidate_correct": "候选：正确结构",
        "candidate_corrupted_target": "同权重：损坏目标结构",
        "candidate_no_topology_target": "同权重：目标无拓扑",
        "corrupted_source_control": "损坏结构训练控制",
        "no_topology_source_control": "无拓扑训练控制",
    }
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F8C8D")
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.1), sharex=True, sharey=True)
        for axis, seed in zip(axes, EXPECTED_SEEDS):
            metrics = gate["per_seed"][str(seed)]
            candidate = metrics["candidate_auc"]
            values = [
                candidate,
                *[
                    candidate - metrics["target_deltas"][name]
                    for name in EXPECTED_TARGET_EVALUATIONS
                    if name != "candidate_correct"
                ],
            ]
            y = np.arange(len(EXPECTED_TARGET_EVALUATIONS))
            axis.barh(y, values, color=colors, height=0.62)
            axis.axvline(0.5, color="#475569", linewidth=1.2, label="随机基线 0.50")
            axis.axvline(
                0.55,
                color="#8E44AD",
                linewidth=1.2,
                linestyle="--",
                label="目标下限 0.55",
            )
            axis.set_title(f"随机种子 seed{seed}", fontsize=12)
            axis.set_xlabel("未见 RECTANGLE 验证集 AUC")
            axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
            axis.set_axisbelow(True)
            for index, value in enumerate(values):
                axis.text(
                    value + 0.003,
                    index,
                    f"{value:.4f}",
                    va="center",
                    fontsize=9,
                )
        axes[0].set_yticks(
            np.arange(len(EXPECTED_TARGET_EVALUATIONS)),
            [labels[name] for name in EXPECTED_TARGET_EVALUATIONS],
        )
        axes[0].invert_yaxis()
        all_values = [
            value
            for seed in EXPECTED_SEEDS
            for value in (
                [gate["per_seed"][str(seed)]["candidate_auc"]]
                + [
                    gate["per_seed"][str(seed)]["candidate_auc"] - delta
                    for delta in gate["per_seed"][str(seed)]["target_deltas"].values()
                ]
            )
        ]
        lower = min(0.47, min(all_values) - 0.03)
        upper = max(0.60, max(all_values) + 0.06)
        axes[0].set_xlim(lower, upper)
        title = "创新1：基础 Runtime-E4 的 RECTANGLE 整密码零微调留出"
        verdict = "通过，可设计第二个整密码留出" if gate["status"] == "pass" else "暂缓，零样本结构泛化门未通过"
        fig.suptitle(
            f"{title}\n训练仅含 GIFT / SKINNY / uKNIT / Dialga；RECTANGLE 不参与训练与选模\n本次裁决：{verdict}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )
        fig.tight_layout(rect=(0.05, 0.09, 0.98, 0.83))
        fig.savefig(output, format="svg", bbox_inches="tight")
        plt.close(fig)


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_structures(config: dict[str, Any]) -> dict[str, RuntimeSpnStructure]:
    return {
        item["name"]: load_runtime_spn_descriptor(
            item["runtime_structure_path"],
            rounds=int(config["model"]["runtime_rounds"]),
            round_start=int(item["runtime_round_start"]),
        ).structure
        for item in config["protocols"]
    }


def _plain_spec(model: dict[str, Any]) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=int(model["hidden_dim"]),
        pair_embedding_dim=int(model["pair_embedding_dim"]),
        processor_steps=int(model["processor_steps"]),
        dropout=float(model["dropout"]),
        sbox_context_mode=model["sbox_context_mode"],
        cell_input_mode=model["cell_input_mode"],
        round_window_mode=model["round_window_mode"],
    )


def _training_config(training: dict[str, Any], seed: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        seed=seed,
        device=training["device"],
        optimizer=training["optimizer"],
        weight_decay=float(training["weight_decay"]),
        lr_scheduler=training["lr_scheduler"],
        checkpoint_metric=training["checkpoint_metric"],
        restore_best_checkpoint=bool(training["restore_best_checkpoint"]),
        loss=training["loss"],
    )


def _load_source_tasks(
    config: dict[str, Any],
    *,
    seed: int,
    structures: dict[str, RuntimeSpnStructure],
    progress_callback: ProgressCallback | None,
) -> list[RuntimeSpnJointTask]:
    protocol_by_name = {item["name"]: item for item in config["protocols"]}
    cache_root = Path(config["training"]["cache_source_root"]) / f"seed{seed}"
    tasks: list[RuntimeSpnJointTask] = []
    for name in EXPECTED_SOURCES:
        protocol = protocol_by_name[name]
        train = _load_dataset(
            config,
            protocol,
            seed=seed,
            split="train",
            cache_root=cache_root,
            progress_callback=progress_callback,
        )
        validation = _load_dataset(
            config,
            protocol,
            seed=seed,
            split="validation",
            cache_root=cache_root,
            progress_callback=progress_callback,
        )
        tasks.append(
            RuntimeSpnJointTask(
                name=name,
                group="source",
                structure=structures[name],
                train_dataset=train,
                validation_dataset=validation,
            )
        )
    return tasks


def _load_target_validation(
    config: dict[str, Any],
    *,
    seed: int,
    progress_callback: ProgressCallback | None,
) -> DifferentialDataset:
    protocol = next(
        item for item in config["protocols"] if item["name"] == HOLDOUT_CIPHER
    )
    cache_root = Path(config["training"]["cache_source_root"]) / f"seed{seed}"
    return _load_dataset(
        config,
        protocol,
        seed=seed,
        split="validation",
        cache_root=cache_root,
        progress_callback=progress_callback,
    )


def _load_dataset(
    config: dict[str, Any],
    protocol: dict[str, Any],
    *,
    seed: int,
    split: str,
    cache_root: Path,
    progress_callback: ProgressCallback | None,
) -> DifferentialDataset:
    training = config["training"]
    validation = split == "validation"
    cipher = build_cipher(
        protocol["cipher_key"],
        int(protocol["rounds"]),
        key=int(protocol["validation_key" if validation else "train_key"], 0),
    )
    return make_chunked_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=int(protocol["input_difference"], 0),
            samples_per_class=int(
                training[
                    "validation_samples_per_class" if validation else "samples_per_class"
                ]
            ),
            seed=seed + (10_000 if validation else 0),
            feature_encoding=training["feature_encoding"],
            pairs_per_sample=int(training["pairs_per_sample"]),
            negative_mode=training["negative_mode"],
            key_rotation_interval=int(training["key_rotation_interval"]),
            sample_structure=training["sample_structure"],
        ),
        cache_dir=cache_root / protocol["name"] / split,
        chunk_size=512,
        workers=1,
        progress_callback=progress_callback,
        progress_context={
            "seed": seed,
            "cipher": protocol["name"],
            "split": split,
        },
    )


def _intervene_tasks(
    tasks: list[RuntimeSpnJointTask],
    *,
    structure_mode: str,
) -> list[RuntimeSpnJointTask]:
    if structure_mode not in {"correct", "corrupted"}:
        raise ValueError("source structure mode must be correct or corrupted")
    return [
        RuntimeSpnJointTask(
            name=task.name,
            group=task.group,
            structure=(
                task.structure if structure_mode == "correct" else task.structure.corrupted()
            ),
            train_dataset=task.train_dataset,
            validation_dataset=task.validation_dataset,
        )
        for task in tasks
    ]


def _evaluate_target(
    model: nn.Module,
    dataset: DifferentialDataset,
    structure: RuntimeSpnStructure,
    training: dict[str, Any],
) -> dict[str, float]:
    task = RuntimeSpnJointTask(
        name=HOLDOUT_CIPHER,
        group="holdout",
        structure=structure,
        train_dataset=dataset,
        validation_dataset=dataset,
    )
    return evaluate_runtime_spn_joint(
        model,
        [task],
        split="validation",
        batch_size=int(training["batch_size"]),
        device=torch.device("cpu"),
        loss=training["loss"],
    )[HOLDOUT_CIPHER]


def _source_role_payload(
    *,
    seed: int,
    role: str,
    intervention: dict[str, str],
    parameter_count: int,
    config_sha256: str,
    checkpoint_path: Path,
    result: RuntimeSpnJointTrainingResult,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "role": role,
        "intervention": intervention,
        "parameter_count": parameter_count,
        "config_sha256": config_sha256,
        "checkpoint_path": str(checkpoint_path),
        "history": result.history,
        "train_metrics": result.train_metrics,
        "validation_metrics": result.validation_metrics,
        "metadata": result.metadata,
    }


def _assemble_payload(
    *,
    config: dict[str, Any],
    config_sha256: str,
    roles: dict[int, dict[str, dict[str, Any]]],
    target_loaded_after_sources: dict[int, bool],
) -> dict[str, Any]:
    source_metrics: dict[str, Any] = {}
    source_macro: dict[str, Any] = {}
    target_metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    protocol_by_name = {item["name"]: item for item in config["protocols"]}
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        source_metrics[key] = {}
        source_macro[key] = {}
        for role in EXPECTED_ROLES:
            role_payload = roles[seed][role]
            source_metrics[key][role] = role_payload["validation_metrics"]
            source_macro[key][role] = float(
                np.mean(
                    [
                        role_payload["validation_metrics"][name]["auc"]
                        for name in EXPECTED_SOURCES
                    ]
                )
            )
            for name in EXPECTED_SOURCES:
                rows.append(
                    {
                        "run_id": config["run_id"],
                        "row_kind": "source_validation",
                        "seed": seed,
                        "role": role,
                        "cipher": name,
                        "cipher_display_name": protocol_by_name[name]["display_name"],
                        "rounds": protocol_by_name[name]["rounds"],
                        "parameter_count": role_payload["parameter_count"],
                        "samples_per_class": config["training"]["samples_per_class"],
                        "validation_samples_per_class": config["training"][
                            "validation_samples_per_class"
                        ],
                        "pairs_per_sample": config["training"]["pairs_per_sample"],
                        "negative_mode": config["training"]["negative_mode"],
                        "checkpoint": role_payload["checkpoint_path"],
                        "metrics": {
                            "train": role_payload["train_metrics"][name],
                            "validation": role_payload["validation_metrics"][name],
                        },
                        "config_sha256": config_sha256,
                    }
                )
            history.extend(
                {"seed": seed, "role": role, **row}
                for row in role_payload["history"]
            )
        evaluations = roles[seed]["target_evaluations"]
        target_metrics[key] = {
            name: evaluations[name]["metrics"]["auc"]
            for name in EXPECTED_TARGET_EVALUATIONS
        }
        for name in EXPECTED_TARGET_EVALUATIONS:
            evaluation = evaluations[name]
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "holdout_target",
                    "seed": seed,
                    "evaluation": name,
                    "source_role": evaluation["source_role"],
                    "cipher": HOLDOUT_CIPHER,
                    "cipher_display_name": protocol_by_name[HOLDOUT_CIPHER][
                        "display_name"
                    ],
                    "rounds": protocol_by_name[HOLDOUT_CIPHER]["rounds"],
                    "parameter_count": roles[seed][evaluation["source_role"]][
                        "parameter_count"
                    ],
                    "training_samples_per_class": 0,
                    "validation_samples_per_class": config["training"][
                        "validation_samples_per_class"
                    ],
                    "pairs_per_sample": config["training"]["pairs_per_sample"],
                    "negative_mode": config["training"]["negative_mode"],
                    "checkpoint": evaluation["checkpoint_path"],
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "metrics": {"validation": evaluation["metrics"]},
                    "optimizer_steps": 0,
                    "target_head_trained": False,
                    "config_sha256": config_sha256,
                }
            )
    checkpoints = [
        Path(roles[seed][role]["checkpoint_path"])
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    ]
    candidate_checkpoint_shared = all(
        len(
            {
                roles[seed]["target_evaluations"][name]["checkpoint_sha256"]
                for name in (
                    "candidate_correct",
                    "candidate_corrupted_target",
                    "candidate_no_topology_target",
                )
            }
        )
        == 1
        for seed in EXPECTED_SEEDS
    )
    parameter_counts = {
        roles[seed][role]["parameter_count"]
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    }
    validation_checks = {
        "result_rows": len(rows) == 34,
        "six_checkpoints_exist": len(checkpoints) == 6 and all(p.exists() for p in checkpoints),
        "parameter_matched": parameter_counts == {442466},
        "source_tasks_exclude_holdout": all(
            set(roles[seed][role]["metadata"]["task_names"]) == set(EXPECTED_SOURCES)
            for seed in EXPECTED_SEEDS
            for role in EXPECTED_ROLES
        ),
        "checkpoint_selection_source_only": all(
            HOLDOUT_CIPHER not in roles[seed][role]["metadata"]["task_names"]
            and roles[seed][role]["metadata"]["selected_checkpoint"] == "best"
            for seed in EXPECTED_SEEDS
            for role in EXPECTED_ROLES
        ),
        "target_loaded_after_all_source_roles": all(target_loaded_after_sources.values()),
        "candidate_same_checkpoint_counterfactuals": candidate_checkpoint_shared,
        "target_optimizer_steps_zero": all(
            roles[seed]["target_evaluations"][name]["optimizer_steps"] == 0
            for seed in EXPECTED_SEEDS
            for name in EXPECTED_TARGET_EVALUATIONS
        ),
        "target_head_never_trained": all(
            not roles[seed]["target_evaluations"][name]["target_head_trained"]
            for seed in EXPECTED_SEEDS
            for name in EXPECTED_TARGET_EVALUATIONS
        ),
        "strict_negative_mode": config["training"]["negative_mode"]
        == "encrypted_random_plaintexts",
    }
    validation = {
        "status": "pass" if all(validation_checks.values()) else "fail",
        "checks": validation_checks,
        "result_rows": len(rows),
        "parameter_counts": sorted(parameter_counts),
        "checkpoint_count": len(checkpoints),
        "holdout_cipher": HOLDOUT_CIPHER,
        "source_ciphers": list(EXPECTED_SOURCES),
        "target_training_rows": 0,
        "checkpoint_selection_tasks": list(EXPECTED_SOURCES),
        "cache_source_root": config["training"]["cache_source_root"],
    }
    return {
        "config": config,
        "source_metrics": source_metrics,
        "source_macro_auc": source_macro,
        "target_metrics": target_metrics,
        "rows": rows,
        "history": history,
        "validation": validation,
    }


def _cache_probe(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    root = project_root / config["training"]["cache_source_root"]
    required: list[Path] = []
    for seed in EXPECTED_SEEDS:
        for cipher in EXPECTED_SOURCES:
            for split in ("train", "validation"):
                required.extend(
                    root / f"seed{seed}" / cipher / split / name
                    for name in ("features.npy", "labels.npy", "metadata.json")
                )
        required.extend(
            root / f"seed{seed}" / HOLDOUT_CIPHER / "validation" / name
            for name in ("features.npy", "labels.npy", "metadata.json")
        )
    target_train = [
        root / f"seed{seed}" / HOLDOUT_CIPHER / "train" for seed in EXPECTED_SEEDS
    ]
    return {
        "passed": all(path.is_file() for path in required),
        "required_file_count": len(required),
        "missing": [str(path) for path in required if not path.is_file()],
        "target_train_referenced": False,
        "target_train_paths_present_but_unused": [
            str(path) for path in target_train if path.exists()
        ],
    }


def _target_control_probe(
    model: RelationModeRuntimeE4,
    structure: RuntimeSpnStructure,
) -> dict[str, bool]:
    generator = torch.Generator().manual_seed(260726)
    features = torch.randint(
        0,
        2,
        (3, 4, 2, structure.block_bits),
        generator=generator,
        dtype=torch.float32,
    )
    with torch.no_grad():
        correct = model.backbone(features, structure, relation_mode="true")
        corrupted = model.backbone(
            features,
            structure.corrupted(),
            relation_mode="true",
        )
        independent = model.backbone(features, structure, relation_mode="independent")
    return {
        "correct_differs_from_corrupted": not torch.equal(correct, corrupted),
        "correct_differs_from_no_topology": not torch.equal(correct, independent),
        "outputs_finite": bool(
            torch.isfinite(correct).all()
            and torch.isfinite(corrupted).all()
            and torch.isfinite(independent).all()
        ),
    }


def _all_width_probe(
    state: dict[str, torch.Tensor],
    config: dict[str, Any],
    structures: dict[str, RuntimeSpnStructure],
) -> bool:
    for structure in structures.values():
        model = RelationModeRuntimeE4(_plain_spec(config["model"]), "true")
        model.load_state_dict(state, strict=True)
        features = torch.zeros((1, 4, 2, structure.block_bits), dtype=torch.float32)
        with torch.no_grad():
            output = model(features, structure)
        if output.shape != (1, 1) or not torch.isfinite(output).all():
            return False
    return True


def _synthetic_holdout_smoke(
    config: dict[str, Any],
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, Any]:
    tasks = [
        RuntimeSpnJointTask(
            name=name,
            group="source",
            structure=structures[name],
            train_dataset=_synthetic_dataset(structures[name].block_bits, seed=index),
            validation_dataset=_synthetic_dataset(
                structures[name].block_bits,
                seed=100 + index,
            ),
        )
        for index, name in enumerate(EXPECTED_SOURCES)
    ]
    model = RelationModeRuntimeE4(_plain_spec(config["model"]), "true")
    result = train_runtime_spn_joint(
        model,
        tasks,
        TrainingConfig(
            epochs=1,
            batch_size=16,
            learning_rate=1e-4,
            seed=0,
            device="cpu",
            optimizer="adam",
            weight_decay=1e-5,
            lr_scheduler="none",
            checkpoint_metric="val_macro_auc",
            restore_best_checkpoint=True,
            loss="mse",
        ),
    )
    target = _synthetic_dataset(structures[HOLDOUT_CIPHER].block_bits, seed=999)
    target_metrics = _evaluate_target(
        model,
        target,
        structures[HOLDOUT_CIPHER],
        {"batch_size": 16, "loss": "mse"},
    )
    return {
        "source_only_checkpoint": result.metadata["task_names"]
        == list(EXPECTED_SOURCES),
        "target_evaluated_after_training": bool(np.isfinite(target_metrics["auc"])),
        "task_specific_trainable_state": result.metadata[
            "task_specific_trainable_state"
        ],
        "shared_state_dict_count": result.metadata["shared_state_dict_count"],
        "source_task_names": result.metadata["task_names"],
        "target_auc_smoke_only": target_metrics["auc"],
    }


def _synthetic_dataset(block_bits: int, *, seed: int) -> DifferentialDataset:
    rng = np.random.default_rng(seed)
    rows = 32
    return DifferentialDataset(
        features=rng.integers(
            0,
            2,
            size=(rows, 4 * 2 * block_bits),
            dtype=np.uint8,
        ),
        labels=np.asarray([index % 2 for index in range(rows)], dtype=np.uint8),
        metadata={"synthetic_readiness": True, "seed": seed},
    )


def _load_resumable_role(
    role_path: Path,
    checkpoint_path: Path,
    *,
    config_sha256: str,
) -> dict[str, Any] | None:
    if not role_path.exists() or not checkpoint_path.exists():
        return None
    payload = _read_json(role_path)
    if payload.get("config_sha256") != config_sha256:
        return None
    if payload.get("checkpoint_path") != str(checkpoint_path):
        return None
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "RelationModeRuntimeE4",
    "adjudicate_holdout_experiment",
    "build_holdout_readiness",
    "config_sha256",
    "load_and_validate_holdout_config",
    "render_holdout_svg",
    "run_holdout_experiment",
    "verify_readiness",
    "write_holdout_artifacts",
]
