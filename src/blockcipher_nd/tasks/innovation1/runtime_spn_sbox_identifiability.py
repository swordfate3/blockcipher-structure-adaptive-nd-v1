from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    runtime_spn_structure_from_truth_bits,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_holdout import (
    _file_sha256,
    _validate_checkpoint_payload,
    load_and_validate_dialga_holdout_config,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    RelationModeRuntimeE4,
    _load_structures,
    _load_target_validation,
    _plain_spec,
    config_sha256,
    load_and_validate_holdout_config,
)
from blockcipher_nd.training.metrics import (
    best_threshold_accuracy_and_threshold,
    binary_auc,
    predict_binary_probabilities,
)
from blockcipher_nd.training.types import ProgressCallback


CIPHERS = ("gift64", "skinny64", "rectangle80", "uknit64", "dialga128")
SOURCE_CIPHERS = CIPHERS[:4]
HOLDOUT_CIPHER = "dialga128"
EXPECTED_SEEDS = (0, 1)
CONTROL_MODES = (
    "exact",
    "broadcast_gift64",
    "broadcast_skinny64",
    "broadcast_rectangle80",
    "broadcast_uknit64_reference",
    "broadcast_dialga128",
    "identity",
    "input_permuted",
    "zero_descriptor",
)
DISPLAY_NAMES = {
    "gift64": "GIFT-64 r6",
    "skinny64": "SKINNY-64/64 r7",
    "rectangle80": "RECTANGLE-80 r6",
    "uknit64": "uKNIT-BC prefix-r5",
    "dialga128": "Dialga-128 prefix-r4",
}
CONTROL_LABELS = {
    "exact": "正确S盒",
    "broadcast_gift64": "GIFT S盒",
    "broadcast_skinny64": "SKINNY S盒",
    "broadcast_rectangle80": "RECTANGLE S盒",
    "broadcast_uknit64_reference": "uKNIT代表S盒",
    "broadcast_dialga128": "Dialga S盒",
    "identity": "恒等S盒",
    "input_permuted": "输入打乱S盒",
    "zero_descriptor": "零描述消融",
}


class _BoundRuntimeSpn(nn.Module):
    def __init__(
        self,
        model: RelationModeRuntimeE4,
        structure: RuntimeSpnStructure,
    ) -> None:
        super().__init__()
        self.model = model
        self.structure = structure

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_bits = 2 * self.structure.block_bits
        if features.ndim != 2 or features.shape[1] % pair_bits:
            raise ValueError("S1 features do not contain complete ciphertext pairs")
        runtime = features.reshape(
            features.shape[0], -1, 2, self.structure.block_bits
        ).flip(-1)
        return self.model(runtime, self.structure)


def load_and_validate_sbox_identifiability_config(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("S1 config schema_version must be 1")
    if config.get("experiment") != "innovation1_runtime_spn_sbox_identifiability_s1":
        raise ValueError("S1 experiment name drifted")
    if tuple(config.get("ciphers", ())) != CIPHERS:
        raise ValueError("S1 cipher panel drifted")
    if tuple(config.get("source_ciphers", ())) != SOURCE_CIPHERS:
        raise ValueError("S1 source panel drifted")
    if config.get("holdout_cipher") != HOLDOUT_CIPHER:
        raise ValueError("S1 holdout cipher drifted")
    if tuple(config.get("controls", ())) != CONTROL_MODES:
        raise ValueError("S1 control panel drifted")

    required_evaluation = {
        "seeds": [0, 1],
        "validation_samples_per_class": 1024,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "batch_size": 256,
        "device": "cpu",
        "relation_mode": "true",
        "parameter_count": 442466,
        "training_performed": False,
        "optimizer_steps": 0,
    }
    if config.get("evaluation") != required_evaluation:
        raise ValueError("S1 evaluation contract drifted")
    required_gate = {
        "probability_delta_floor": 0.000001,
        "auc_margin": 0.005,
        "required_seeds": [0, 1],
        "expected_rows": 90,
    }
    if config.get("gate") != required_gate:
        raise ValueError("S1 gate contract drifted")

    source = config.get("source", {})
    hashed_paths = (
        ("a8_config_path", "a8_config_sha256"),
        ("a8_gate_path", "a8_gate_sha256"),
        ("a8_validation_path", "a8_validation_sha256"),
        ("a8_results_path", "a8_results_sha256"),
        ("protocol_config_path", "protocol_config_sha256"),
    )
    for path_key, hash_key in hashed_paths:
        artifact = project_root / source[path_key]
        if _file_sha256(artifact) != source.get(hash_key):
            raise ValueError(f"S1 frozen source hash drifted: {path_key}")

    a8 = load_and_validate_dialga_holdout_config(
        project_root / source["a8_config_path"],
        project_root=project_root,
        require_readiness=True,
    )
    if config_sha256(project_root / source["a8_config_path"]) != source[
        "a8_config_sha256"
    ]:
        raise ValueError("S1 A8 config hash drifted")
    if a8["run_id"] != "i1_runtime_spn_dialga_holdout_a8_2048_seed0_seed1_20260726":
        raise ValueError("S1 A8 source run drifted")

    a8_gate = _read_json(project_root / source["a8_gate_path"])
    if a8_gate.get("status") != "hold" or a8_gate.get("decision") != source.get(
        "a8_required_decision"
    ):
        raise ValueError("S1 requires the frozen A8 hold decision")
    a8_validation = _read_json(project_root / source["a8_validation_path"])
    if a8_validation.get("status") != "pass" or not all(
        a8_validation.get("checks", {}).values()
    ):
        raise ValueError("S1 requires valid A8 evidence")

    checkpoints = source.get("checkpoints", {})
    if tuple(checkpoints) != ("0", "1"):
        raise ValueError("S1 checkpoint panel drifted")
    for seed in EXPECTED_SEEDS:
        item = checkpoints[str(seed)]
        for path_key, hash_key in (
            ("path", "sha256"),
            ("role_result_path", "role_result_sha256"),
        ):
            if _file_sha256(project_root / item[path_key]) != item.get(hash_key):
                raise ValueError(f"S1 seed{seed} frozen checkpoint evidence drifted")
        role = _read_json(project_root / item["role_result_path"])
        if (
            role.get("seed") != seed
            or role.get("role") != "correct_candidate"
            or role.get("relation_mode") != "true"
            or role.get("parameter_count") != 442466
            or role.get("config_sha256") != source["a8_config_sha256"]
            or role.get("checkpoint_sha256") != item["sha256"]
        ):
            raise ValueError(f"S1 seed{seed} role-result contract drifted")
    return config


def build_sbox_counterfactuals(
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, dict[str, dict[str, Any]]]:
    if tuple(structures) != CIPHERS:
        raise ValueError("S1 runtime structure panel drifted")
    templates = {
        "broadcast_gift64": structures["gift64"].sbox_truth_bits[-1, 0],
        "broadcast_skinny64": structures["skinny64"].sbox_truth_bits[-1, 0],
        "broadcast_rectangle80": structures["rectangle80"].sbox_truth_bits[-1, 0],
        "broadcast_uknit64_reference": structures["uknit64"].sbox_truth_bits[-1, 0],
        "broadcast_dialga128": structures["dialga128"].sbox_truth_bits[-1, 0],
    }
    identity = _truth_bits_from_table(torch.arange(16, dtype=torch.long))
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for cipher in CIPHERS:
        target = structures[cipher]
        truth_by_mode: dict[str, torch.Tensor] = {"exact": target.sbox_truth_bits}
        truth_by_mode.update(
            {
                mode: truth.reshape(1, 1, 64).repeat(
                    target.rounds, target.cells, 1
                )
                for mode, truth in templates.items()
            }
        )
        truth_by_mode["identity"] = identity.reshape(1, 1, 64).repeat(
            target.rounds, target.cells, 1
        )
        truth_by_mode["input_permuted"] = torch.roll(
            target.sbox_truth_bits.reshape(target.rounds, target.cells, 16, 4),
            shifts=1,
            dims=2,
        ).reshape(target.rounds, target.cells, 64)
        truth_by_mode["zero_descriptor"] = torch.zeros_like(
            target.sbox_truth_bits
        )
        controls: dict[str, dict[str, Any]] = {}
        for mode in CONTROL_MODES:
            truth = truth_by_mode[mode]
            structure = runtime_spn_structure_from_truth_bits(
                target.cell_membership,
                target.bit_role,
                truth,
                target.linear_matrices,
            )
            controls[mode] = {
                "structure": structure,
                "valid_sbox": _truth_is_permutation(truth),
                "equivalent_to_exact": torch.equal(
                    truth, target.sbox_truth_bits
                ),
                "sbox_truth_sha256": _tensor_sha256(truth),
                "control_source": _control_source(mode),
            }
        result[cipher] = controls
    return result


def run_sbox_identifiability(
    *,
    config: dict[str, Any],
    config_path: Path,
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source = config["source"]
    base = load_and_validate_holdout_config(
        project_root / source["protocol_config_path"]
    )
    structures = _load_structures(base)
    controls = build_sbox_counterfactuals(structures)
    a8_exact = _load_a8_exact_auc(project_root / source["a8_results_path"])
    cache_check = _validation_cache_check(base, project_root)
    rows: list[dict[str, Any]] = []
    checkpoint_checks: dict[str, bool] = {}

    for seed in EXPECTED_SEEDS:
        source_item = source["checkpoints"][str(seed)]
        checkpoint_path = project_root / source_item["path"]
        role_path = project_root / source_item["role_result_path"]
        role = _read_json(role_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        _validate_checkpoint_payload(
            checkpoint,
            seed=seed,
            role="correct_candidate",
            relation_mode="true",
            config_hash=source["a8_config_sha256"],
            initial_hash=role["initial_state_sha256"],
        )
        model = RelationModeRuntimeE4(_plain_spec(base["model"]), "true")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        checkpoint_checks[str(seed)] = (
            parameter_count == config["evaluation"]["parameter_count"]
            and _file_sha256(checkpoint_path) == source_item["sha256"]
        )
        _emit(progress_callback, "sbox_seed_start", seed=seed)

        for cipher in CIPHERS:
            dataset = _load_target_validation(
                base,
                seed=seed,
                progress_callback=progress_callback,
                holdout_cipher=cipher,
            )
            feature_hash = _array_sha256(dataset.features)
            label_hash = _array_sha256(dataset.labels)
            probabilities: dict[str, np.ndarray] = {}
            for mode in CONTROL_MODES:
                item = controls[cipher][mode]
                bound = _BoundRuntimeSpn(model, item["structure"])
                probabilities[mode] = predict_binary_probabilities(
                    bound,
                    dataset,
                    batch_size=config["evaluation"]["batch_size"],
                    device=config["evaluation"]["device"],
                )
            exact_probabilities = probabilities["exact"]
            for mode in CONTROL_MODES:
                item = controls[cipher][mode]
                probability = probabilities[mode]
                metrics = _metrics(dataset.labels, probability)
                delta = np.abs(probability - exact_probabilities)
                rows.append(
                    {
                        "run_id": config["run_id"],
                        "row_kind": "frozen_checkpoint_sbox_counterfactual",
                        "seed": seed,
                        "cipher": cipher,
                        "cipher_display_name": DISPLAY_NAMES[cipher],
                        "rounds": _protocol_rounds(base, cipher),
                        "role": "source" if cipher in SOURCE_CIPHERS else "holdout",
                        "control": mode,
                        "control_label": CONTROL_LABELS[mode],
                        "control_source": item["control_source"],
                        "valid_sbox": item["valid_sbox"],
                        "equivalent_to_exact": item["equivalent_to_exact"],
                        **metrics,
                        "max_abs_probability_delta_from_exact": float(delta.max()),
                        "mean_abs_probability_delta_from_exact": float(delta.mean()),
                        "probability_sha256": _array_sha256(probability),
                        "feature_sha256": feature_hash,
                        "label_sha256": label_hash,
                        "sbox_truth_sha256": item["sbox_truth_sha256"],
                        "linear_topology_sha256": _tensor_sha256(
                            item["structure"].linear_matrices
                        ),
                        "cell_membership_sha256": _tensor_sha256(
                            item["structure"].cell_membership
                        ),
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": source_item["sha256"],
                        "config_sha256": config_sha256(config_path),
                        "samples_per_class": config["evaluation"][
                            "validation_samples_per_class"
                        ],
                        "samples_total": int(len(dataset.labels)),
                        "pairs_per_sample": config["evaluation"]["pairs_per_sample"],
                        "negative_mode": config["evaluation"]["negative_mode"],
                        "parameter_count": parameter_count,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
            _emit(
                progress_callback,
                "sbox_cipher_done",
                seed=seed,
                cipher=cipher,
                exact_auc=next(
                    row["auc"]
                    for row in reversed(rows)
                    if row["seed"] == seed
                    and row["cipher"] == cipher
                    and row["control"] == "exact"
                ),
            )

    validation = _validate_rows(
        config=config,
        rows=rows,
        a8_exact=a8_exact,
        cache_check=cache_check,
        checkpoint_checks=checkpoint_checks,
    )
    return {
        "config": config,
        "rows": rows,
        "validation": validation,
        "a8_exact_auc": a8_exact,
        "cache_check": cache_check,
    }


def adjudicate_sbox_identifiability(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    grouped = {
        (seed, cipher, control): row
        for row in rows
        for seed, cipher, control in [
            (int(row["seed"]), str(row["cipher"]), str(row["control"]))
        ]
    }
    delta_floor = float(config["gate"]["probability_delta_floor"])
    auc_margin = float(config["gate"]["auc_margin"])
    responsiveness: dict[str, bool] = {}
    source_macro_margins: dict[str, dict[str, float]] = {}
    holdout_margins: dict[str, dict[str, float]] = {}

    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        for cipher in CIPHERS:
            candidates = [
                grouped[(seed, cipher, mode)]
                for mode in CONTROL_MODES
                if mode != "exact"
                and grouped[(seed, cipher, mode)]["valid_sbox"]
                and not grouped[(seed, cipher, mode)]["equivalent_to_exact"]
            ]
            responsiveness[f"seed{seed}_{cipher}"] = bool(candidates) and max(
                float(row["max_abs_probability_delta_from_exact"])
                for row in candidates
            ) > delta_floor

        source_macro_margins[seed_key] = {}
        for mode in CONTROL_MODES[1:]:
            eligible = [
                cipher
                for cipher in SOURCE_CIPHERS
                if not grouped[(seed, cipher, mode)]["equivalent_to_exact"]
            ]
            exact_mean = float(
                np.mean([grouped[(seed, cipher, "exact")]["auc"] for cipher in eligible])
            )
            control_mean = float(
                np.mean([grouped[(seed, cipher, mode)]["auc"] for cipher in eligible])
            )
            source_macro_margins[seed_key][mode] = exact_mean - control_mean

        exact_holdout = float(grouped[(seed, HOLDOUT_CIPHER, "exact")]["auc"])
        holdout_margins[seed_key] = {
            mode: exact_holdout - float(grouped[(seed, HOLDOUT_CIPHER, mode)]["auc"])
            for mode in CONTROL_MODES[1:]
            if not grouped[(seed, HOLDOUT_CIPHER, mode)]["equivalent_to_exact"]
        }

    protocol_valid = validation.get("status") == "pass" and all(
        validation.get("checks", {}).values()
    )
    responsive = all(responsiveness.values())
    source_identifiable = all(
        margin >= auc_margin
        for margins in source_macro_margins.values()
        for margin in margins.values()
    )
    holdout_identifiable = all(
        margin >= auc_margin
        for margins in holdout_margins.values()
        for margin in margins.values()
    )
    identifiable = responsive and source_identifiable and holdout_identifiable

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_runtime_spn_sbox_identifiability_protocol_invalid"
        next_action = "repair only the failed frozen-artifact or evaluation invariant"
    elif identifiable:
        status = "pass"
        decision = "innovation1_runtime_spn_sbox_identifiable"
        next_action = (
            "preregister one local cell-local S-box operator against unchanged "
            "Runtime-E4 and no-S-box controls"
        )
    elif responsive:
        status = "hold"
        decision = "innovation1_runtime_spn_sbox_responsive_but_not_identifiable"
        next_action = (
            "close the current truth-table conditioning path; use the observed "
            "aliasing to specify at most one local Boolean-operator hypothesis"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_sbox_descriptor_functionally_ignored"
        next_action = (
            "close unseen-S-box composition and consolidate the exact-GF(2) "
            "topology boundary"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "thresholds": {
            "probability_delta_floor": delta_floor,
            "auc_margin": auc_margin,
        },
        "responsiveness": responsiveness,
        "source_macro_margins": source_macro_margins,
        "holdout_margins": holdout_margins,
        "research_checks": {
            "descriptor_responsive_every_seed_cipher": responsive,
            "source_macro_sbox_identifiable": source_identifiable,
            "dialga_holdout_sbox_identifiable": holdout_identifiable,
        },
        "claim_scope": (
            "local A8 frozen-checkpoint five-cipher S-box identifiability audit; "
            "no training, formal scale, universality, attack, SOTA or breakthrough claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "increase A8 samples or epochs",
            "launch remote A8 scale-up",
            "add Dialga supervision or target-head fitting",
            "rescue A8 with Adapter, FiLM, MoE or typed relations",
            "revive dense DDT input",
        ],
    }


def write_sbox_identifiability_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "results.jsonl", payload["rows"])
    _write_csv(output_root / "sensitivity.csv", payload["rows"])
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
    render_sbox_identifiability_svg(payload["rows"], gate, output_root / "curves.svg")


def render_sbox_identifiability_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    grouped = {
        (int(row["seed"]), str(row["cipher"]), str(row["control"])): row
        for row in rows
    }
    controls = CONTROL_MODES[1:]
    labels = [CONTROL_LABELS[mode] for mode in controls]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(18, 10))
        all_margins = []
        matrices: dict[int, np.ndarray] = {}
        for seed in EXPECTED_SEEDS:
            matrix = np.asarray(
                [
                    [
                        grouped[(seed, cipher, "exact")]["auc"]
                        - grouped[(seed, cipher, mode)]["auc"]
                        for mode in controls
                    ]
                    for cipher in CIPHERS
                ],
                dtype=np.float64,
            )
            matrices[seed] = matrix
            all_margins.extend(matrix.reshape(-1).tolist())
        limit = max(0.01, max(abs(value) for value in all_margins))

        for column, seed in enumerate(EXPECTED_SEEDS):
            matrix = matrices[seed]
            image = axes[0, column].imshow(
                matrix,
                cmap="RdBu",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            for row_index in range(len(CIPHERS)):
                for control_index in range(len(controls)):
                    value = matrix[row_index, control_index]
                    axes[0, column].text(
                        control_index,
                        row_index,
                        f"{value:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > limit * 0.55 else "#111827",
                    )
            axes[0, column].set_xticks(
                range(len(controls)), labels, rotation=32, ha="right", fontsize=8
            )
            axes[0, column].set_yticks(
                range(len(CIPHERS)), [DISPLAY_NAMES[c] for c in CIPHERS]
            )
            axes[0, column].set_title(
                f"seed{seed}：正确S盒 AUC 减去控制AUC",
                loc="left",
                fontweight="bold",
            )
            figure.colorbar(image, ax=axes[0, column], fraction=0.025, pad=0.02)

            response = [
                max(
                    grouped[(seed, cipher, mode)][
                        "max_abs_probability_delta_from_exact"
                    ]
                    for mode in controls
                    if grouped[(seed, cipher, mode)]["valid_sbox"]
                    and not grouped[(seed, cipher, mode)]["equivalent_to_exact"]
                )
                for cipher in CIPHERS
            ]
            bars = axes[1, column].barh(
                range(len(CIPHERS)),
                response,
                color=["#2563EB", "#0F9D76", "#D97706", "#7C3AED", "#C2417B"],
            )
            axes[1, column].bar_label(
                bars,
                labels=[f"{value:.3e}" for value in response],
                padding=3,
                fontsize=8,
            )
            axes[1, column].axvline(
                gate["thresholds"]["probability_delta_floor"],
                color="#DC2626",
                linestyle="--",
                linewidth=1,
            )
            axes[1, column].set_yticks(
                range(len(CIPHERS)), [DISPLAY_NAMES[c] for c in CIPHERS]
            )
            axes[1, column].set_xlabel("相对正确S盒的最大预测概率变化")
            axes[1, column].set_title(
                f"seed{seed}：网络是否会响应S盒描述",
                loc="left",
                fontweight="bold",
            )
            axes[1, column].set_xlim(0.0, max(response) * 1.18 if max(response) else 1e-5)

        figure.suptitle(
            "创新1 S1：冻结检查点的S盒可识别性审计",
            x=0.055,
            y=0.985,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.945,
            "上图正值表示正确S盒更好，负值表示错误S盒反而更好；下图只表示模型会不会响应，不代表理解正确。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.055,
            0.912,
            f"裁决：{_decision_chinese(gate['decision'])}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )
        figure.tight_layout(rect=(0.04, 0.04, 0.99, 0.88), h_pad=2.8, w_pad=2.2)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches=None)
        plt.close(figure)


def _validate_rows(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    a8_exact: dict[tuple[int, str], float],
    cache_check: dict[str, Any],
    checkpoint_checks: dict[str, bool],
) -> dict[str, Any]:
    grouped = {
        (int(row["seed"]), str(row["cipher"]), str(row["control"])): row
        for row in rows
    }
    complete_keys = {
        (seed, cipher, mode)
        for seed in EXPECTED_SEEDS
        for cipher in CIPHERS
        for mode in CONTROL_MODES
    }
    topology_fields = (
        "linear_topology_sha256",
        "cell_membership_sha256",
        "feature_sha256",
        "label_sha256",
        "checkpoint_sha256",
    )
    checks = {
        "expected_result_rows": len(rows) == config["gate"]["expected_rows"],
        "complete_seed_cipher_control_panel": set(grouped) == complete_keys,
        "frozen_checkpoints_valid": all(checkpoint_checks.values()),
        "validation_cache_complete": cache_check["passed"],
        "same_topology_data_and_checkpoint_within_pair": all(
            len({grouped[(seed, cipher, mode)][field] for mode in CONTROL_MODES}) == 1
            for seed in EXPECTED_SEEDS
            for cipher in CIPHERS
            for field in topology_fields
        ),
        "only_zero_descriptor_is_not_a_valid_sbox": all(
            bool(row["valid_sbox"]) == (row["control"] != "zero_descriptor")
            for row in rows
        ),
        "equivalent_controls_are_probability_exact": all(
            row["probability_sha256"]
            == grouped[(row["seed"], row["cipher"], "exact")]["probability_sha256"]
            and row["max_abs_probability_delta_from_exact"] == 0.0
            for row in rows
            if row["equivalent_to_exact"]
        ),
        "a8_exact_auc_reproduced": all(
            abs(grouped[(seed, cipher, "exact")]["auc"] - a8_exact[(seed, cipher)])
            <= 1e-12
            for seed in EXPECTED_SEEDS
            for cipher in CIPHERS
        ),
        "no_training_or_optimizer_steps": all(
            row["training_performed"] is False and row["optimizer_steps"] == 0
            for row in rows
        ),
        "frozen_protocol": all(
            row["samples_per_class"] == 1024
            and row["samples_total"] == 2048
            and row["pairs_per_sample"] == 4
            and row["negative_mode"] == "encrypted_random_plaintexts"
            and row["parameter_count"] == 442466
            for row in rows
        ),
        "finite_metrics": all(
            all(
                isinstance(row[field], (int, float)) and math.isfinite(float(row[field]))
                for field in (
                    "auc",
                    "accuracy",
                    "best_accuracy",
                    "loss",
                    "max_abs_probability_delta_from_exact",
                    "mean_abs_probability_delta_from_exact",
                )
            )
            for row in rows
        ),
        "sha256_fields_present": all(
            _is_sha256(row[field])
            for row in rows
            for field in (
                "probability_sha256",
                "feature_sha256",
                "label_sha256",
                "sbox_truth_sha256",
                "linear_topology_sha256",
                "cell_membership_sha256",
                "checkpoint_sha256",
                "config_sha256",
            )
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "result_rows": len(rows),
        "expected_rows": config["gate"]["expected_rows"],
        "checkpoint_checks": checkpoint_checks,
        "cache_check": cache_check,
        "training_performed": False,
        "optimizer_steps": 0,
    }


def _load_a8_exact_auc(path: Path) -> dict[tuple[int, str], float]:
    result: dict[tuple[int, str], float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("row_kind") == "source_validation" and row.get("role") == (
            "correct_candidate"
        ):
            result[(int(row["seed"]), str(row["cipher"]))] = float(
                row["metrics"]["validation"]["auc"]
            )
        elif row.get("row_kind") == "holdout_target" and row.get("evaluation") == (
            "candidate_correct"
        ):
            result[(int(row["seed"]), str(row["cipher"]))] = float(
                row["metrics"]["validation"]["auc"]
            )
    expected = {(seed, cipher) for seed in EXPECTED_SEEDS for cipher in CIPHERS}
    if set(result) != expected:
        raise ValueError("S1 could not recover the complete A8 exact-AUC panel")
    return result


def _validation_cache_check(
    base: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    root = project_root / base["training"]["cache_source_root"]
    required = [
        root / f"seed{seed}" / cipher / "validation" / filename
        for seed in EXPECTED_SEEDS
        for cipher in CIPHERS
        for filename in ("features.npy", "labels.npy", "metadata.json")
    ]
    return {
        "passed": all(path.is_file() for path in required),
        "required_file_count": len(required),
        "missing": [str(path) for path in required if not path.is_file()],
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    label_array = np.asarray(labels, dtype=np.float32)
    probability_array = np.asarray(probabilities, dtype=np.float32)
    predictions = (probability_array >= 0.5).astype(np.float32)
    best_accuracy, threshold = best_threshold_accuracy_and_threshold(
        label_array, probability_array
    )
    return {
        "loss": float(np.mean(np.square(probability_array - label_array))),
        "accuracy": float(np.mean(predictions == label_array)),
        "auc": binary_auc(label_array, probability_array),
        "best_accuracy": best_accuracy,
        "calibrated_threshold": threshold,
        "mean_probability": float(probability_array.mean()),
    }


def _truth_bits_from_table(table: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(table, dtype=torch.long).reshape(16)
    shifts = torch.arange(4, dtype=torch.long)
    return (((values[:, None] >> shifts) & 1).reshape(64)).to(torch.uint8)


def _truth_is_permutation(truth: torch.Tensor) -> bool:
    bits = torch.as_tensor(truth, dtype=torch.long).reshape(-1, 16, 4)
    values = torch.sum(bits * (1 << torch.arange(4, dtype=torch.long)), dim=-1)
    expected = torch.arange(16, dtype=torch.long)
    return all(torch.equal(torch.sort(table).values, expected) for table in values)


def _control_source(mode: str) -> str | None:
    if mode.startswith("broadcast_"):
        return mode.removeprefix("broadcast_").removesuffix("_reference")
    return None


def _protocol_rounds(base: dict[str, Any], cipher: str) -> int:
    return int(next(item["rounds"] for item in base["protocols"] if item["name"] == cipher))


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _tensor_sha256(values: torch.Tensor) -> str:
    return _array_sha256(values.detach().cpu().contiguous().numpy())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_sbox_identifiable": "正确S盒在五密码双seed中可识别",
        "innovation1_runtime_spn_sbox_responsive_but_not_identifiable": (
            "模型会响应S盒，但不能稳定识别正确S盒"
        ),
        "innovation1_runtime_spn_sbox_descriptor_functionally_ignored": (
            "至少一个密码/seed未使用S盒描述"
        ),
        "innovation1_runtime_spn_sbox_identifiability_protocol_invalid": (
            "冻结证据或评估协议无效"
        ),
    }[decision]


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "CIPHERS",
    "CONTROL_MODES",
    "EXPECTED_SEEDS",
    "adjudicate_sbox_identifiability",
    "build_sbox_counterfactuals",
    "load_and_validate_sbox_identifiability_config",
    "render_sbox_identifiability_svg",
    "run_sbox_identifiability",
    "write_sbox_identifiability_artifacts",
]
