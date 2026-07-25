from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_readiness import (
    FIVE_CIPHER_PROTOCOLS,
)
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTask,
    RuntimeSpnJointTrainingResult,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig


EXPECTED_CIPHERS = tuple(protocol.name for protocol in FIVE_CIPHER_PROTOCOLS)
EXPECTED_ROLES = ("dense", "correct", "uniform", "shuffled")
EXPECTED_SEEDS = (0, 1)


def load_and_validate_joint_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("joint primitive-adapter config schema_version must be 1")
    training = payload.get("training", {})
    model = payload.get("model", {})
    protocols = payload.get("protocols", [])
    if tuple(training.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("joint primitive-adapter config must use seeds 0 and 1")
    if tuple(training.get("roles", {}).keys()) != EXPECTED_ROLES:
        raise ValueError("joint primitive-adapter config has the wrong role panel")
    if tuple(item.get("name") for item in protocols) != EXPECTED_CIPHERS:
        raise ValueError("joint primitive-adapter config has the wrong cipher panel")
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
            raise ValueError(f"joint primitive-adapter training field {key} drifted")
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
        "primitive_adapter_rank": 8,
        "primitive_adapter_scale": 0.1,
    }
    for key, expected in required_model.items():
        if model.get(key) != expected:
            raise ValueError(f"joint primitive-adapter model field {key} drifted")
    if model.get("primitive_adapter_effect", "additive") not in {
        "additive",
        "multiplicative_gate",
    }:
        raise ValueError("joint primitive-adapter effect mode is unsupported")
    _validate_protocols(protocols)
    return payload


def verify_readiness(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    readiness = config["readiness"]
    path = project_root / readiness["gate_path"]
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise ValueError("primitive-adapter readiness gate did not pass")
    if gate.get("decision") != readiness["required_decision"]:
        raise ValueError("primitive-adapter readiness decision does not match config")
    if not all(gate.get("checks", {}).values()):
        raise ValueError("primitive-adapter readiness contains a failed check")
    return gate


def run_joint_experiment(
    *,
    config: dict[str, Any],
    config_sha256: str,
    output_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    structures = _load_structures(config)
    role_payloads: dict[int, dict[str, dict[str, Any]]] = {}
    checkpoints = output_root / "checkpoints"
    role_results = output_root / "role-results"
    checkpoints.mkdir(parents=True, exist_ok=True)
    role_results.mkdir(parents=True, exist_ok=True)
    for seed in config["training"]["seeds"]:
        configured_cache_root = config["training"].get("cache_source_root")
        seed_cache_root = (
            Path(configured_cache_root) / f"seed{seed}"
            if configured_cache_root
            else output_root / "cache" / f"seed{seed}"
        )
        tasks = _make_tasks(
            config=config,
            seed=int(seed),
            structures=structures,
            cache_root=seed_cache_root,
            progress_callback=progress_callback,
        )
        role_payloads[int(seed)] = {}
        for role, mode in config["training"]["roles"].items():
            role_path = role_results / f"seed{seed}-{role}.json"
            checkpoint_path = checkpoints / f"seed{seed}-{role}.pt"
            resumed = _load_resumable_role(
                role_path,
                checkpoint_path,
                config_sha256=config_sha256,
            )
            if resumed is not None:
                role_payloads[int(seed)][role] = resumed
                _emit(
                    progress_callback,
                    "joint_role_reused",
                    seed=seed,
                    role=role,
                )
                continue
            _emit(progress_callback, "joint_role_start", seed=seed, role=role)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                model = RuntimeE4EquivariantSpnDistinguisher(
                    _model_spec(config["model"], mode)
                )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            result = train_runtime_spn_joint(
                model,
                tasks,
                _training_config(config["training"], int(seed)),
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda event, payload, seed=seed, role=role: progress_callback(
                        event,
                        {"seed": seed, "role": role, **payload},
                    )
                ),
            )
            payload = _role_payload(
                seed=int(seed),
                role=role,
                mode=mode,
                parameter_count=parameter_count,
                config_sha256=config_sha256,
                checkpoint_path=checkpoint_path,
                result=result,
            )
            torch.save(
                {
                    "state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    },
                    "seed": int(seed),
                    "role": role,
                    "mode": mode,
                    "config_sha256": config_sha256,
                    "best_epoch": result.metadata["best_epoch"],
                },
                checkpoint_path,
            )
            _write_json(role_path, payload)
            role_payloads[int(seed)][role] = payload
            _emit(
                progress_callback,
                "joint_role_done",
                seed=seed,
                role=role,
                best_epoch=result.metadata["best_epoch"],
                best_val_macro_auc=result.metadata["best_val_macro_auc"],
            )
    return _assemble_experiment_payload(config, config_sha256, role_payloads)


def adjudicate_joint_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    margin = float(config["gate"]["margin"])
    floor = float(config["gate"]["per_cipher_floor"])
    metrics = payload["per_cipher_metrics"]
    aggregates = payload["aggregates"]
    router = payload["router_utilization"]
    gradients = payload["gradient_diagnostics"]
    per_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        control_deltas: dict[str, Any] = {}
        for control in ("dense", "uniform", "shuffled"):
            control_deltas[control] = {
                group: (
                    aggregates[seed_key]["correct"][group]
                    - aggregates[seed_key][control][group]
                )
                for group in ("core_macro_auc", "stress_macro_auc", "five_macro_auc")
            }
        per_cipher_dense = {
            cipher: (
                metrics[seed_key]["correct"][cipher]["validation"]["auc"]
                - metrics[seed_key]["dense"][cipher]["validation"]["auc"]
            )
            for cipher in EXPECTED_CIPHERS
        }
        adapter_traffic = {
            adapter: sum(
                router[seed_key]["correct"][cipher].get(adapter, 0.0)
                for cipher in EXPECTED_CIPHERS
            )
            for adapter in ("fan_in_1", "multi_source")
        }
        adapter_gradients = gradients[seed_key]["correct"][
            "adapter_gradient_mean_abs_sum"
        ]
        core_checks = {
            f"core_correct_minus_{control}_at_least_margin": (
                control_deltas[control]["core_macro_auc"] >= margin
            )
            for control in ("dense", "uniform", "shuffled")
        }
        core_checks["each_core_cipher_not_below_dense_floor"] = all(
            per_cipher_dense[cipher] >= floor
            for cipher in ("gift64", "skinny64", "rectangle80")
        )
        stress_checks = {
            f"stress_correct_minus_{control}_at_least_margin": (
                control_deltas[control]["stress_macro_auc"] >= margin
            )
            for control in ("dense", "uniform", "shuffled")
        }
        stress_checks["uknit_not_below_dense_floor"] = (
            per_cipher_dense["uknit64"] >= floor
        )
        stress_checks["dialga_not_below_dense_floor"] = (
            per_cipher_dense["dialga128"] >= floor
        )
        attribution_checks = {
            "both_adapters_have_traffic": all(
                value > 0.0 for value in adapter_traffic.values()
            ),
            "both_adapters_have_gradients": (
                adapter_gradients.get("primitive_adapters.fan_in_1", 0.0) > 0.0
                and adapter_gradients.get("primitive_adapters.multi_source", 0.0) > 0.0
            ),
            "all_gradients_finite": gradients[seed_key]["correct"][
                "all_gradients_finite"
            ],
        }
        per_seed[seed_key] = {
            "control_deltas": control_deltas,
            "per_cipher_correct_minus_dense": per_cipher_dense,
            "adapter_traffic": adapter_traffic,
            "core_checks": core_checks,
            "stress_checks": stress_checks,
            "attribution_checks": attribution_checks,
            "core_pass": all(core_checks.values()) and all(attribution_checks.values()),
            "stress_pass": all(stress_checks.values()),
        }
        per_seed[seed_key]["full_pass"] = bool(
            per_seed[seed_key]["core_pass"] and per_seed[seed_key]["stress_pass"]
        )
    protocol_valid = bool(payload["validation"]["status"] == "pass")
    full_pass = protocol_valid and all(
        per_seed[str(seed)]["full_pass"] for seed in EXPECTED_SEEDS
    )
    core_pass = protocol_valid and all(
        per_seed[str(seed)]["core_pass"] for seed in EXPECTED_SEEDS
    )
    if full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_primitive_adapter_five_cipher_supported"
        next_action = (
            "keep deterministic adapters and run preregistered whole-cipher holdouts "
            "for RECTANGLE, Dialga, and uKNIT; do not start learned MoE yet"
        )
    elif core_pass:
        status = "hold"
        decision = (
            "innovation1_runtime_spn_primitive_adapter_core_supported_new_cipher_hold"
        )
        next_action = (
            "preserve the core result and test one heterogeneous-S-box/round primitive "
            "locally; do not claim five-cipher support or start learned MoE"
        )
    else:
        status = "hold" if protocol_valid else "invalid"
        decision = (
            "innovation1_runtime_spn_primitive_adapter_joint_not_supported"
            if protocol_valid
            else "innovation1_runtime_spn_primitive_adapter_protocol_invalid"
        )
        next_action = (
            "audit task balancing, descriptor fan-in classification, parameter matching, "
            "and gradient flow; do not add experts, epochs, samples, or remote compute"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "full_pass": full_pass,
        "core_pass": core_pass,
        "margin": margin,
        "per_cipher_floor": floor,
        "per_seed": per_seed,
        "claim_scope": (
            "local 2048/class/cipher two-seed joint diagnostic only; not formal scale, "
            "not unseen-cipher transfer, and not a universal or breakthrough claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "mechanically increase samples, epochs, or remote compute after a hold",
            "use five-cipher macro AUC to hide a failed uKNIT or core cipher",
            "start learned soft or Top-2 MoE without a full joint and holdout pass",
        ],
    }


def write_joint_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    rows = payload["result_rows"]
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_history_csv(output_root / "history.csv", payload["history_rows"])
    _write_json(output_root / "per_cipher_metrics.json", payload["per_cipher_metrics"])
    _write_json(output_root / "router_utilization.json", payload["router_utilization"])
    _write_json(
        output_root / "gradient_diagnostics.json", payload["gradient_diagnostics"]
    )
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "aggregates": payload["aggregates"],
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    render_joint_margin_svg(payload, output_root / "curves.svg", gate=gate)


def render_joint_margin_svg(
    payload: dict[str, Any],
    output: Path,
    *,
    gate: dict[str, Any] | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    labels = {
        "gift64": "GIFT-64 r6",
        "skinny64": "SKINNY-64/64 r7",
        "rectangle80": "RECTANGLE-80 r6",
        "uknit64": "uKNIT-BC 前缀5轮",
        "dialga128": "Dialga-128 前缀4轮",
        "core_macro_auc": "核心组三密码平均",
        "stress_macro_auc": "新算法压力组平均",
        "five_macro_auc": "五密码总体平均（仅展示）",
    }
    categories = (
        *EXPECTED_CIPHERS,
        "core_macro_auc",
        "stress_macro_auc",
        "five_macro_auc",
    )
    has_gated_source = bool(
        gate
        and all(
            "gated_minus_additive_by_cipher"
            in gate.get("per_seed", {}).get(str(seed), {})
            for seed in EXPECTED_SEEDS
        )
    )
    has_film_source = bool(
        gate
        and all(
            "film_minus_additive" in gate.get("per_seed", {}).get(str(seed), {})
            for seed in EXPECTED_SEEDS
        )
    )
    has_typed_relation = bool(
        gate
        and all(
            "typed_minus_sources" in gate.get("per_seed", {}).get(str(seed), {})
            for seed in EXPECTED_SEEDS
        )
    )
    has_additive_source = has_gated_source or has_film_source
    axis_subject = (
        "正确GF(2)关系类型"
        if has_typed_relation
        else "正确局部结构描述"
        if has_film_source
        else "正确路由"
    )
    controls = (
        [
            ("dense", "固定完全图控制", "#0072B2", "o"),
            ("uniform", "关系类型无关控制", "#D55E00", "s"),
            ("shuffled", "关系类型打乱控制", "#009E73", "^"),
        ]
        if has_typed_relation
        else
        [
            ("dense", "固定条件稠密锚点", "#0072B2", "o"),
            ("uniform", "全cell均值描述控制", "#D55E00", "s"),
            ("shuffled", "特征打乱描述控制", "#009E73", "^"),
        ]
        if has_film_source
        else [
            ("dense", "普通稠密锚点", "#0072B2", "o"),
            ("uniform", "均匀混合控制", "#D55E00", "s"),
            ("shuffled", "打乱原语路由", "#009E73", "^"),
        ]
    )
    if has_additive_source:
        controls.append(("additive_source", "旧加法正确路由", "#CC79A7", "D"))
    margins_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    all_values = [0.0, 0.005]
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        margins_by_seed[seed] = {}
        for control, _label, _color, _marker in controls:
            if control == "additive_source":
                assert gate is not None
                seed_gate = gate["per_seed"][seed_key]
                if has_film_source:
                    source = seed_gate["film_minus_additive"]
                    values = dict(source["by_cipher"])
                    values["core_macro_auc"] = source["core_macro"]
                    values["stress_macro_auc"] = source["stress_macro"]
                    values["five_macro_auc"] = source["five_macro"]
                else:
                    values = dict(seed_gate["gated_minus_additive_by_cipher"])
                    values["core_macro_auc"] = seed_gate[
                        "gated_minus_additive_core_macro"
                    ]
                    values["stress_macro_auc"] = seed_gate[
                        "gated_minus_additive_stress_macro"
                    ]
                    values["five_macro_auc"] = float(
                        np.mean([values[cipher] for cipher in EXPECTED_CIPHERS])
                    )
            else:
                values = {}
                for cipher in EXPECTED_CIPHERS:
                    values[cipher] = (
                        payload["per_cipher_metrics"][seed_key]["correct"][cipher][
                            "validation"
                        ]["auc"]
                        - payload["per_cipher_metrics"][seed_key][control][cipher][
                            "validation"
                        ]["auc"]
                    )
                for aggregate in (
                    "core_macro_auc",
                    "stress_macro_auc",
                    "five_macro_auc",
                ):
                    values[aggregate] = (
                        payload["aggregates"][seed_key]["correct"][aggregate]
                        - payload["aggregates"][seed_key][control][aggregate]
                    )
            margins_by_seed[seed][control] = values
            all_values.extend(values.values())

    low = min(all_values)
    high = max(all_values)
    padding = max(0.004, 0.12 * max(0.01, high - low))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.5), sharey=True)
    y = np.arange(len(categories))
    offsets = np.linspace(-0.27, 0.27, len(controls))
    for seed, axis in zip(EXPECTED_SEEDS, axes, strict=True):
        axis.axvline(0.0, color="#333333", linewidth=1.2, label="无提升")
        axis.axvline(
            0.005,
            color="#7A3E9D",
            linewidth=1.2,
            linestyle="--",
            label="推进门槛 +0.005",
        )
        for offset, (control, label, color, marker) in zip(
            offsets, controls, strict=True
        ):
            values = [
                margins_by_seed[seed][control][category] for category in categories
            ]
            axis.scatter(
                values,
                y + offset,
                label=f"{axis_subject}减{label}",
                color=color,
                marker=marker,
                s=58,
                zorder=3,
            )
        axis.set_title(f"随机种子 seed{seed}", fontsize=13, pad=10)
        axis.set_xlabel(f"验证集 AUC 差值（{axis_subject} - 对照）", fontsize=11)
        axis.set_xlim(low - padding, high + padding)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.tick_params(axis="both", labelsize=10)
    axes[0].set_yticks(y, [labels[category] for category in categories])
    axes[0].invert_yaxis()
    if has_typed_relation:
        experiment_title = "创新1：五密码共享 Runtime-E4 Typed GF(2) 关系消息归因结果"
    elif has_film_source:
        experiment_title = "创新1：五密码共享 Runtime-E4 局部结构 True FiLM 归因结果"
    elif has_gated_source:
        experiment_title = "创新1：五密码共享 Runtime-E4 结构原语乘法门控归因结果"
    else:
        experiment_title = "创新1：五密码共享 Runtime-E4 结构原语适配器归因结果"
    decision_caption = ""
    if gate and (has_film_source or has_typed_relation):
        decision_caption = (
            "\n本次裁决：通过，可进入整密码留出"
            if gate.get("status") == "pass"
            else "\n本次裁决：暂缓，不进入整密码留出"
        )
    fig.suptitle(
        f"{experiment_title}\n"
        f"正值表示{axis_subject}优于同预算对照；紫色虚线是 +0.005 推进门槛"
        f"{decision_caption}",
        fontsize=16,
        y=0.985,
    )
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0.04, 0.08, 0.99, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _validate_protocols(protocols: list[dict[str, Any]]) -> None:
    expected = {protocol.name: protocol for protocol in FIVE_CIPHER_PROTOCOLS}
    for item in protocols:
        reference = expected[item["name"]]
        checks = {
            "group": reference.group,
            "cipher_key": reference.cipher_key,
            "rounds": reference.rounds,
            "input_difference": hex(reference.input_difference),
            "train_key": hex(reference.train_key),
            "validation_key": hex(reference.validation_key),
            "runtime_structure_path": reference.descriptor_path,
            "runtime_round_start": reference.descriptor_round_start,
        }
        for key, expected_value in checks.items():
            actual = item.get(key)
            if key in {"input_difference", "train_key", "validation_key"}:
                actual = hex(int(str(actual), 0))
            if actual != expected_value:
                raise ValueError(f"protocol {item['name']} field {key} drifted")


def _load_structures(config: dict[str, Any]) -> dict[str, RuntimeSpnStructure]:
    rounds = int(config["model"]["runtime_rounds"])
    return {
        item["name"]: load_runtime_spn_descriptor(
            item["runtime_structure_path"],
            rounds=rounds,
            round_start=int(item["runtime_round_start"]),
        ).structure
        for item in config["protocols"]
    }


def _make_tasks(
    *,
    config: dict[str, Any],
    seed: int,
    structures: dict[str, RuntimeSpnStructure],
    cache_root: Path,
    progress_callback: ProgressCallback | None,
) -> list[RuntimeSpnJointTask]:
    training = config["training"]
    tasks: list[RuntimeSpnJointTask] = []
    for protocol in config["protocols"]:
        datasets = {}
        for split, key_field, count_field, seed_offset in (
            ("train", "train_key", "samples_per_class", 0),
            (
                "validation",
                "validation_key",
                "validation_samples_per_class",
                10_000,
            ),
        ):
            cipher = build_cipher(
                protocol["cipher_key"],
                int(protocol["rounds"]),
                key=int(protocol[key_field], 0),
            )
            datasets[split] = make_chunked_differential_dataset(
                DifferentialDatasetConfig(
                    cipher=cipher,
                    input_difference=int(protocol["input_difference"], 0),
                    samples_per_class=int(training[count_field]),
                    seed=seed + seed_offset,
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
        tasks.append(
            RuntimeSpnJointTask(
                name=protocol["name"],
                group=protocol["group"],
                structure=structures[protocol["name"]],
                train_dataset=datasets["train"],
                validation_dataset=datasets["validation"],
            )
        )
    return tasks


def _model_spec(model: dict[str, Any], mode: str) -> RuntimeParameterizedSpnSpec:
    conditioning = model.get("primitive_conditioning")
    true_film = conditioning == "true_film"
    typed_relation = conditioning == "typed_relation_gnn_film"
    return RuntimeParameterizedSpnSpec(
        hidden_dim=int(model["hidden_dim"]),
        pair_embedding_dim=int(model["pair_embedding_dim"]),
        processor_steps=int(model["processor_steps"]),
        dropout=float(model["dropout"]),
        sbox_context_mode=model["sbox_context_mode"],
        cell_input_mode=model["cell_input_mode"],
        round_window_mode=model["round_window_mode"],
        primitive_adapter_mode="none" if true_film or typed_relation else mode,
        primitive_adapter_rank=int(model["primitive_adapter_rank"]),
        primitive_adapter_scale=float(model["primitive_adapter_scale"]),
        primitive_adapter_effect=model.get("primitive_adapter_effect", "additive"),
        primitive_film_mode=mode if true_film else "none",
        primitive_film_rank=int(model.get("primitive_film_rank", 10)),
        primitive_film_scale=float(model.get("primitive_film_scale", 0.1)),
        typed_relation_mode=mode if typed_relation else "none",
        typed_relation_scale=float(model.get("typed_relation_scale", 0.1)),
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


def _role_payload(
    *,
    seed: int,
    role: str,
    mode: str,
    parameter_count: int,
    config_sha256: str,
    checkpoint_path: Path,
    result: RuntimeSpnJointTrainingResult,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "role": role,
        "adapter_mode": mode,
        "parameter_count": parameter_count,
        "config_sha256": config_sha256,
        "checkpoint_path": str(checkpoint_path),
        "history": result.history,
        "train_metrics": result.train_metrics,
        "validation_metrics": result.validation_metrics,
        "metadata": result.metadata,
        "router_traffic": result.router_traffic,
        "gradient_diagnostics": result.gradient_diagnostics,
    }


def _load_resumable_role(
    role_path: Path,
    checkpoint_path: Path,
    *,
    config_sha256: str,
) -> dict[str, Any] | None:
    if not role_path.exists() or not checkpoint_path.exists():
        return None
    payload = json.loads(role_path.read_text(encoding="utf-8"))
    if payload.get("config_sha256") != config_sha256:
        return None
    if payload.get("checkpoint_path") != str(checkpoint_path):
        return None
    return payload


def _assemble_experiment_payload(
    config: dict[str, Any],
    config_sha256: str,
    roles: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    per_cipher: dict[str, Any] = {}
    aggregates: dict[str, Any] = {}
    router: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    protocol_by_name = {item["name"]: item for item in config["protocols"]}
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        per_cipher[seed_key] = {}
        aggregates[seed_key] = {}
        router[seed_key] = {}
        gradients[seed_key] = {}
        for role in EXPECTED_ROLES:
            payload = roles[seed][role]
            per_cipher[seed_key][role] = {}
            router[seed_key][role] = payload["router_traffic"]
            gradients[seed_key][role] = payload["gradient_diagnostics"]
            for cipher in EXPECTED_CIPHERS:
                per_cipher[seed_key][role][cipher] = {
                    "train": payload["train_metrics"][cipher],
                    "validation": payload["validation_metrics"][cipher],
                }
                protocol = protocol_by_name[cipher]
                result_rows.append(
                    {
                        "run_id": config["run_id"],
                        "seed": seed,
                        "role": role,
                        "adapter_mode": payload["adapter_mode"],
                        "parameter_count": payload["parameter_count"],
                        "cipher": cipher,
                        "cipher_display_name": protocol["display_name"],
                        "group": protocol["group"],
                        "rounds": protocol["rounds"],
                        "samples_per_class": config["training"]["samples_per_class"],
                        "validation_samples_per_class": config["training"][
                            "validation_samples_per_class"
                        ],
                        "pairs_per_sample": config["training"]["pairs_per_sample"],
                        "negative_mode": config["training"]["negative_mode"],
                        "epochs": config["training"]["epochs"],
                        "best_epoch": payload["metadata"]["best_epoch"],
                        "checkpoint": payload["checkpoint_path"],
                        "metrics": per_cipher[seed_key][role][cipher],
                        "config_sha256": config_sha256,
                    }
                )
            core = [
                payload["validation_metrics"][name]["auc"]
                for name in EXPECTED_CIPHERS
                if protocol_by_name[name]["group"] == "core"
            ]
            stress = [
                payload["validation_metrics"][name]["auc"]
                for name in EXPECTED_CIPHERS
                if protocol_by_name[name]["group"] == "stress"
            ]
            aggregates[seed_key][role] = {
                "core_macro_auc": float(np.mean(core)),
                "stress_macro_auc": float(np.mean(stress)),
                "five_macro_auc": float(
                    np.mean(
                        [
                            payload["validation_metrics"][name]["auc"]
                            for name in EXPECTED_CIPHERS
                        ]
                    )
                ),
            }
            for row in payload["history"]:
                history_rows.append({"seed": seed, "role": role, **row})
    checkpoints_exist = all(
        Path(roles[seed][role]["checkpoint_path"]).exists()
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    )
    parameter_counts = {
        roles[seed][role]["parameter_count"]
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    }
    validation = {
        "status": "pass"
        if len(result_rows) == 40 and checkpoints_exist and len(parameter_counts) == 1
        else "fail",
        "result_rows": len(result_rows),
        "expected_result_rows": 40,
        "all_checkpoints_exist": checkpoints_exist,
        "parameter_counts": sorted(parameter_counts),
        "parameter_matched": len(parameter_counts) == 1,
        "shared_checkpoints": 8,
        "task_specific_trainable_state": False,
        "strict_negative_mode": config["training"]["negative_mode"],
        "cache_source_root": config["training"].get(
            "cache_source_root",
            str(Path(config["run_id"]) / "cache"),
        ),
        "config_sha256": config_sha256,
    }
    return {
        "config": config,
        "config_sha256": config_sha256,
        "per_cipher_metrics": per_cipher,
        "aggregates": aggregates,
        "router_utilization": router,
        "gradient_diagnostics": gradients,
        "result_rows": result_rows,
        "history_rows": history_rows,
        "validation": validation,
    }


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    "EXPECTED_CIPHERS",
    "EXPECTED_ROLES",
    "EXPECTED_SEEDS",
    "adjudicate_joint_experiment",
    "config_sha256",
    "load_and_validate_joint_config",
    "render_joint_margin_svg",
    "run_joint_experiment",
    "verify_readiness",
    "write_joint_artifacts",
]
