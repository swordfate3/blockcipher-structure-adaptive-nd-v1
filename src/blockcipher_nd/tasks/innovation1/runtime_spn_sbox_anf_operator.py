from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
    inverse_sbox_anf_contributions,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    runtime_spn_structure_from_truth_bits,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_holdout import (
    _cache_probe,
    _file_sha256,
    load_and_validate_dialga_holdout_config,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_identifiability import (
    _array_sha256,
    _metrics,
    _tensor_sha256,
    _truth_bits_from_table,
    _truth_is_permutation,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_uknit_heterogeneous_holdout import (
    _clone_state_dict,
    _state_dict_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    _load_source_tasks,
    _load_structures,
    _load_target_validation,
    _plain_spec,
    _training_config,
    config_sha256,
    load_and_validate_holdout_config,
)
from blockcipher_nd.training.metrics import predict_binary_probabilities
from blockcipher_nd.training.runtime_spn_joint import train_runtime_spn_joint
from blockcipher_nd.training.types import ProgressCallback


SOURCE_CIPHERS = ("gift64", "skinny64", "rectangle80", "uknit64")
HOLDOUT_CIPHER = "dialga128"
CIPHERS = SOURCE_CIPHERS + (HOLDOUT_CIPHER,)
EXPECTED_SEEDS = (0, 1)
OPERATOR_CONTROLS = ("exact", "input_permuted", "identity")
DISPLAY_NAMES = {
    "gift64": "GIFT-64 r6",
    "skinny64": "SKINNY-64/64 r7",
    "rectangle80": "RECTANGLE-80 r6",
    "uknit64": "uKNIT-BC prefix-r5",
    "dialga128": "Dialga-128 prefix-r4",
}
CONTROL_LABELS = {
    "exact": "正确逆S盒算子",
    "input_permuted": "输入打乱逆S盒算子",
    "identity": "恒等算子",
    "a8_anchor": "A8基础Runtime-E4",
}


class SboxAnfOperatorRuntimeE4(nn.Module):
    """Bind the exact runtime S-box operator during joint source training."""

    def __init__(self, spec: RuntimeParameterizedSpnSpec) -> None:
        super().__init__()
        self.backbone = RuntimeE4EquivariantSpnDistinguisher(spec)

    def forward(
        self,
        features: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        return self.backbone(
            features,
            structure,
            relation_mode="true",
            operator_structure=structure,
        )


class _BoundSboxAnfOperator(nn.Module):
    def __init__(
        self,
        model: SboxAnfOperatorRuntimeE4,
        structure: RuntimeSpnStructure,
        operator_structure: RuntimeSpnStructure,
    ) -> None:
        super().__init__()
        self.model = model
        self.structure = structure
        self.operator_structure = operator_structure

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_bits = 2 * self.structure.block_bits
        if features.ndim != 2 or features.shape[1] % pair_bits:
            raise ValueError("S2 features do not contain complete ciphertext pairs")
        runtime = features.reshape(
            features.shape[0], -1, 2, self.structure.block_bits
        ).flip(-1)
        return self.model.backbone(
            runtime,
            self.structure,
            relation_mode="true",
            operator_structure=self.operator_structure,
        )


def load_and_validate_sbox_anf_operator_config(
    path: Path,
    *,
    project_root: Path,
    require_readiness: bool,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("S2 config schema_version must be 1")
    if config.get("experiment") != "innovation1_runtime_spn_sbox_anf_operator_s2":
        raise ValueError("S2 experiment name drifted")
    if tuple(config.get("source_ciphers", ())) != SOURCE_CIPHERS:
        raise ValueError("S2 source cipher panel drifted")
    if config.get("holdout_cipher") != HOLDOUT_CIPHER:
        raise ValueError("S2 must hold out Dialga")
    if tuple(config.get("operator_controls", ())) != OPERATOR_CONTROLS:
        raise ValueError("S2 operator controls drifted")

    required_candidate = {
        "backbone": "RuntimeE4EquivariantSpnDistinguisher",
        "cell_input_mode": "state_triplet",
        "round_window_mode": "recurrent_window",
        "sbox_context_mode": "edge_gate",
        "sbox_context_scale": 0.0,
        "sbox_boolean_operator_mode": "inverse_anf_contribution_gate",
        "sbox_boolean_operator_scale": 0.25,
        "expected_parameter_count": 459234,
        "gradient_combination": (
            "representation_l2_equalized_pcgrad_fixed_order"
        ),
    }
    if config.get("candidate") != required_candidate:
        raise ValueError("S2 candidate contract drifted")
    required_training = {
        "seeds": [0, 1],
        "samples_per_class_per_source": 2048,
        "validation_samples_per_class_per_source": 1024,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "epochs": 10,
        "batch_size": 256,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "checkpoint_metric": "val_macro_auc",
        "restore_best_checkpoint": True,
        "device": "cpu",
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
    }
    if config.get("training") != required_training:
        raise ValueError("S2 training contract drifted")
    required_gate = {
        "auc_margin": 0.005,
        "anchor_retention_tolerance": 0.005,
        "target_auc_floor": 0.55,
        "probability_delta_floor": 0.000001,
        "required_seeds": [0, 1],
        "expected_result_rows": 40,
        "expected_history_rows": 20,
    }
    if config.get("gate") != required_gate:
        raise ValueError("S2 gate contract drifted")

    source = config.get("source", {})
    for path_key, hash_key in (
        ("protocol_config_path", "protocol_config_sha256"),
        ("a8_config_path", "a8_config_sha256"),
        ("a8_gate_path", "a8_gate_sha256"),
        ("a8_validation_path", "a8_validation_sha256"),
        ("a8_results_path", "a8_results_sha256"),
        ("s1_gate_path", "s1_gate_sha256"),
    ):
        artifact = project_root / source[path_key]
        if _file_sha256(artifact) != source.get(hash_key):
            raise ValueError(f"S2 frozen source hash drifted: {path_key}")

    base = load_and_validate_holdout_config(
        project_root / source["protocol_config_path"]
    )
    a8 = load_and_validate_dialga_holdout_config(
        project_root / source["a8_config_path"],
        project_root=project_root,
        require_readiness=True,
    )
    if tuple(a8["source_ciphers"]) != SOURCE_CIPHERS:
        raise ValueError("S2 no longer matches the A8 source panel")
    if base["training"]["samples_per_class"] != 2048:
        raise ValueError("S2 base samples_per_class drifted")
    if base["training"]["validation_samples_per_class"] != 1024:
        raise ValueError("S2 base validation scale drifted")
    if base["training"]["pairs_per_sample"] != 4:
        raise ValueError("S2 base pair count drifted")

    a8_gate = _read_json(project_root / source["a8_gate_path"])
    if a8_gate.get("status") != "hold" or a8_gate.get("decision") != source.get(
        "a8_required_decision"
    ):
        raise ValueError("S2 requires the frozen A8 hold decision")
    a8_validation = _read_json(project_root / source["a8_validation_path"])
    if a8_validation.get("status") != "pass" or not all(
        a8_validation.get("checks", {}).values()
    ):
        raise ValueError("S2 requires valid A8 evidence")
    s1_gate = _read_json(project_root / source["s1_gate_path"])
    if s1_gate.get("status") != "hold" or s1_gate.get("decision") != source.get(
        "s1_required_decision"
    ):
        raise ValueError("S2 requires the frozen S1 non-identifiability decision")

    if require_readiness:
        readiness = _read_json(project_root / _readiness_gate_path())
        if readiness.get("status") != "pass" or readiness.get("decision") != (
            "innovation1_runtime_spn_sbox_anf_operator_s2_readiness_passed"
        ):
            raise ValueError("S2 readiness did not pass")
        if not all(readiness.get("checks", {}).values()):
            raise ValueError("S2 readiness contains a failed check")
    return config


def build_sbox_operator_controls(
    structure: RuntimeSpnStructure,
) -> dict[str, RuntimeSpnStructure]:
    identity = _truth_bits_from_table(torch.arange(16, dtype=torch.long))
    truth = {
        "exact": structure.sbox_truth_bits,
        "input_permuted": torch.roll(
            structure.sbox_truth_bits.reshape(
                structure.rounds,
                structure.cells,
                16,
                4,
            ),
            shifts=1,
            dims=2,
        ).reshape(structure.rounds, structure.cells, 64),
        "identity": identity.reshape(1, 1, 64).repeat(
            structure.rounds,
            structure.cells,
            1,
        ),
    }
    return {
        mode: runtime_spn_structure_from_truth_bits(
            structure.cell_membership,
            structure.bit_role,
            truth[mode],
            structure.linear_matrices,
        )
        for mode in OPERATOR_CONTROLS
    }


def run_sbox_anf_operator_readiness(
    *,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    base = load_and_validate_holdout_config(
        project_root / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    candidate_spec = _candidate_spec(base, config)
    candidate = SboxAnfOperatorRuntimeE4(candidate_spec).eval()
    baseline = RuntimeE4EquivariantSpnDistinguisher(_plain_spec(base["model"])).eval()
    parameter_count = sum(parameter.numel() for parameter in candidate.parameters())
    controls = {
        cipher: build_sbox_operator_controls(structure)
        for cipher, structure in structures.items()
    }

    generator = torch.Generator().manual_seed(26_072_613)
    logit_deltas: dict[str, dict[str, float]] = {}
    pair_swap_errors: dict[str, float] = {}
    all_outputs_finite = True
    for cipher, structure in structures.items():
        features = torch.randint(
            0,
            2,
            (3, 4, 2, structure.block_bits),
            generator=generator,
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = {
                mode: candidate.backbone(
                    features,
                    structure,
                    operator_structure=operator,
                )
                for mode, operator in controls[cipher].items()
            }
            swapped = candidate.backbone(
                features.flip(2),
                structure,
                operator_structure=controls[cipher]["exact"],
            )
        logit_deltas[cipher] = {
            mode: float((logits["exact"] - logits[mode]).abs().max())
            for mode in OPERATOR_CONTROLS[1:]
        }
        pair_swap_errors[cipher] = float((logits["exact"] - swapped).abs().max())
        all_outputs_finite = all_outputs_finite and all(
            bool(torch.isfinite(value).all()) for value in logits.values()
        )

    relabel_errors = {
        cipher: _cell_relabel_error(
            candidate,
            structures[cipher],
            controls[cipher]["exact"],
        )
        for cipher in ("uknit64", "dialga128")
    }
    anf_reconstruction = {
        cipher: _anf_reconstructs_all_tables(structure)
        for cipher, structure in structures.items()
    }
    control_contracts = {
        cipher: _control_contract(structure, controls[cipher])
        for cipher, structure in structures.items()
    }
    gradients = _finite_gradient_probe(candidate, structures)
    cache = _cache_probe(base, project_root)
    a8_gate = _read_json(project_root / config["source"]["a8_gate_path"])
    s1_gate = _read_json(project_root / config["source"]["s1_gate_path"])
    checks = {
        "all_runtime_sboxes_are_permutations": all(
            _truth_is_permutation(structure.sbox_truth_bits)
            for structure in structures.values()
        ),
        "anf_reconstructs_every_inverse_sbox": all(anf_reconstruction.values()),
        "controls_change_only_operator_sboxes": all(control_contracts.values()),
        "candidate_parameter_geometry": parameter_count
        == config["candidate"]["expected_parameter_count"],
        "baseline_geometry_unchanged": sum(
            parameter.numel() for parameter in baseline.parameters()
        )
        == 442466
        and not any(
            name.startswith("sbox_boolean_operator_projection")
            for name in baseline.state_dict()
        ),
        "old_free_sbox_gate_disabled": candidate_spec.sbox_context_scale == 0.0,
        "operator_controls_change_logits": all(
            delta > 1e-6
            for cipher in CIPHERS
            for delta in logit_deltas[cipher].values()
        ),
        "pair_swap_invariant": all(error <= 1e-6 for error in pair_swap_errors.values()),
        "joint_cell_relabel_invariant": all(
            error <= 1e-6 for error in relabel_errors.values()
        ),
        "source_panel_exact_and_holdout_absent": tuple(config["source_ciphers"])
        == SOURCE_CIPHERS
        and HOLDOUT_CIPHER not in SOURCE_CIPHERS,
        "disk_cache_complete": cache["passed"],
        "target_train_cache_not_referenced": not cache["target_train_referenced"],
        "frozen_a8_and_s1_decisions_match": a8_gate.get("decision")
        == config["source"]["a8_required_decision"]
        and s1_gate.get("decision") == config["source"]["s1_required_decision"],
        "forward_backward_finite": all_outputs_finite
        and gradients["all_values_finite"]
        and gradients["operator_gradient_nonzero"],
        "result_contract_frozen": config["gate"]["expected_result_rows"] == 40
        and config["gate"]["expected_history_rows"] == 20,
    }
    passed = all(checks.values())
    return {
        "run_id": "i1_runtime_spn_sbox_anf_operator_s2_readiness_20260726",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_sbox_anf_operator_s2_readiness_passed"
            if passed
            else "innovation1_runtime_spn_sbox_anf_operator_s2_readiness_failed"
        ),
        "checks": checks,
        "parameter_count": parameter_count,
        "baseline_parameter_count": sum(
            parameter.numel() for parameter in baseline.parameters()
        ),
        "anf_reconstruction": anf_reconstruction,
        "control_contracts": control_contracts,
        "logit_deltas": logit_deltas,
        "pair_swap_errors": pair_swap_errors,
        "cell_relabel_errors": relabel_errors,
        "gradient_probe": gradients,
        "cache_probe": cache,
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
        "next_action": (
            "run the frozen local two-seed S2 diagnostic"
            if passed
            else "repair only the failed S2 readiness invariant before training"
        ),
    }


def run_sbox_anf_operator(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    base = load_and_validate_holdout_config(
        project_root / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    operator_controls = {
        cipher: build_sbox_operator_controls(structure)
        for cipher, structure in structures.items()
    }
    anchors = _load_a8_anchors(project_root / config["source"]["a8_results_path"])
    config_hash = config_sha256(config_path)
    checkpoint_root = output_root / "checkpoints"
    role_root = output_root / "role-results"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    role_root.mkdir(parents=True, exist_ok=True)
    roles: dict[int, dict[str, Any]] = {}
    datasets: dict[int, dict[str, Any]] = {}

    for seed in EXPECTED_SEEDS:
        _emit(progress_callback, "s2_seed_start", seed=seed)
        tasks = _load_source_tasks(
            base,
            seed=seed,
            structures=structures,
            progress_callback=progress_callback,
            source_ciphers=SOURCE_CIPHERS,
        )
        datasets[seed] = {task.name: task.validation_dataset for task in tasks}
        checkpoint_path = checkpoint_root / f"seed{seed}-candidate.pt"
        role_path = role_root / f"seed{seed}-candidate.json"
        role = _load_role(role_path, checkpoint_path, config_hash=config_hash)
        if role is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = SboxAnfOperatorRuntimeE4(_candidate_spec(base, config))
            initial_hash = _state_dict_sha256(_clone_state_dict(model.state_dict()))
            result = train_runtime_spn_joint(
                model,
                tasks,
                _training_config(base["training"], seed),
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda event, payload, seed=seed: progress_callback(
                        event,
                        {"seed": seed, "role": "candidate", **payload},
                    )
                ),
                gradient_combination=config["candidate"]["gradient_combination"],
            )
            checkpoint = {
                "state_dict": _clone_state_dict(model.state_dict()),
                "seed": seed,
                "role": "candidate",
                "config_sha256": config_hash,
                "initial_state_sha256": initial_hash,
                "best_epoch": result.metadata["best_epoch"],
                "checkpoint_selection_tasks": list(SOURCE_CIPHERS),
                "holdout_cipher": HOLDOUT_CIPHER,
                "operator_mode": config["candidate"][
                    "sbox_boolean_operator_mode"
                ],
            }
            torch.save(checkpoint, checkpoint_path)
            role = {
                "seed": seed,
                "role": "candidate",
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "config_sha256": config_hash,
                "initial_state_sha256": initial_hash,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "history": result.history,
                "train_metrics": result.train_metrics,
                "validation_metrics": result.validation_metrics,
                "metadata": result.metadata,
                "gradient_diagnostics": result.gradient_diagnostics,
            }
            _write_json(role_path, role)
            _emit(
                progress_callback,
                "s2_candidate_trained",
                seed=seed,
                best_epoch=result.metadata["best_epoch"],
            )
        else:
            _emit(progress_callback, "s2_candidate_reused", seed=seed)
        _validate_role(role, checkpoint_path, seed=seed, config_hash=config_hash)
        roles[seed] = role

        datasets[seed][HOLDOUT_CIPHER] = _load_target_validation(
            base,
            seed=seed,
            progress_callback=progress_callback,
            holdout_cipher=HOLDOUT_CIPHER,
        )

    readiness = _read_json(project_root / _readiness_gate_path())
    return _assemble_payload(
        config=config,
        config_hash=config_hash,
        base=base,
        structures=structures,
        operator_controls=operator_controls,
        anchors=anchors,
        roles=roles,
        datasets=datasets,
        readiness=readiness,
    )


def adjudicate_sbox_anf_operator(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    protocol_valid = payload["validation"].get("status") == "pass" and all(
        payload["validation"].get("checks", {}).values()
    )
    per_seed: dict[str, Any] = {}
    full_pass = protocol_valid
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        source = payload["source_macro_auc"][key]
        target = payload["target_auc"][key]
        probability = payload["probability_deltas"][key]
        source_margins = {
            "input_permuted": source["exact"] - source["input_permuted"],
            "identity": source["exact"] - source["identity"],
            "a8_anchor": source["exact"] - source["a8_anchor"],
        }
        target_margins = {
            "input_permuted": target["exact"] - target["input_permuted"],
            "identity": target["exact"] - target["identity"],
            "a8_anchor": target["exact"] - target["a8_anchor"],
        }
        checks = {
            "source_input_permuted_margin": source_margins["input_permuted"]
            >= gate_config["auc_margin"],
            "source_identity_margin": source_margins["identity"]
            >= gate_config["auc_margin"],
            "source_anchor_retained": source_margins["a8_anchor"]
            >= -gate_config["anchor_retention_tolerance"],
            "dialga_auc_floor": target["exact"] >= gate_config["target_auc_floor"],
            "dialga_input_permuted_margin": target_margins["input_permuted"]
            >= gate_config["auc_margin"],
            "dialga_identity_margin": target_margins["identity"]
            >= gate_config["auc_margin"],
            "dialga_anchor_retained": target_margins["a8_anchor"]
            >= -gate_config["anchor_retention_tolerance"],
            "operator_probability_responsive": all(
                probability[mode] > gate_config["probability_delta_floor"]
                for mode in OPERATOR_CONTROLS[1:]
            ),
        }
        seed_pass = all(checks.values())
        full_pass = full_pass and seed_pass
        per_seed[key] = {
            "source_macro_auc": source,
            "source_margins": source_margins,
            "dialga_auc": target,
            "dialga_margins": target_margins,
            "probability_deltas": probability,
            "checks": checks,
            "pass": seed_pass,
        }

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_runtime_spn_sbox_anf_operator_protocol_invalid"
        next_action = "repair only the failed S2 protocol invariant and rerun"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_sbox_anf_operator_supported"
        next_action = (
            "preregister a 65536/class/source remote confirmation against the same "
            "A8 anchor and same-checkpoint operator controls"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_sbox_anf_operator_not_supported"
        next_action = (
            "close S-box conditioning and retain only the supported exact-GF(2) "
            "topology contribution; do not scale or tune this operator"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "full_pass": full_pass,
        "per_seed": per_seed,
        "thresholds": gate_config,
        "claim_scope": (
            "local 2048/class/source two-seed S-box mechanism diagnostic; no formal "
            "scale, universality, attack, SOTA or breakthrough claim"
        ),
        "blocked_actions": [
            "tune the ANF residual scale inside S2",
            "increase samples, epochs, or pairs to rescue a hold",
            "launch remote training after a local hold",
            "add S-box-specific experts or target supervision",
            "revive dense DDT or closed uKNIT inverse-triplet/query routes",
        ],
        "next_action": next_action,
    }


def write_sbox_anf_operator_readiness_artifacts(
    readiness: dict[str, Any],
    *,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "validation.json", readiness)
    _write_json(output_root / "gate.json", readiness)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": readiness["run_id"],
            "status": readiness["status"],
            "decision": readiness["decision"],
            "next_action": readiness["next_action"],
        },
    )


def write_sbox_anf_operator_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "results.jsonl", payload["rows"])
    _write_csv(output_root / "history.csv", payload["history"])
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
    render_sbox_anf_operator_svg(gate, output_root / "curves.svg")


def render_sbox_anf_operator_svg(gate: dict[str, Any], output: Path) -> None:
    modes = ("exact", "input_permuted", "identity", "a8_anchor")
    colors = {
        "exact": "#0072B2",
        "input_permuted": "#D55E00",
        "identity": "#7F8C8D",
        "a8_anchor": "#009E73",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
        for column, seed in enumerate(EXPECTED_SEEDS):
            item = gate["per_seed"][str(seed)]
            panels = (
                (
                    axes[0, column],
                    item["source_macro_auc"],
                    f"seed{seed}：四个训练来源的平均 AUC",
                ),
                (
                    axes[1, column],
                    item["dialga_auc"],
                    f"seed{seed}：未参与训练的 Dialga AUC",
                ),
            )
            for axis, values, title in panels:
                y = np.arange(len(modes))
                bars = axis.barh(
                    y,
                    [values[mode] for mode in modes],
                    color=[colors[mode] for mode in modes],
                    height=0.62,
                )
                axis.axvline(0.5, color="#475569", linewidth=1.1, linestyle="--")
                axis.set_yticks(y, [CONTROL_LABELS[mode] for mode in modes])
                axis.invert_yaxis()
                axis.set_title(title, fontsize=12)
                axis.set_xlabel("AUC（越高越好）")
                axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
                axis.set_axisbelow(True)
                for bar, mode in zip(bars, modes, strict=True):
                    value = float(values[mode])
                    axis.text(
                        value + 0.002,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.4f}",
                        va="center",
                        fontsize=9,
                    )
                panel_values = [float(values[mode]) for mode in modes]
                axis.set_xlim(
                    min(0.47, min(panel_values) - 0.025),
                    max(0.60, max(panel_values) + 0.055),
                )
        verdict = (
            "通过：正确算子双 seed 同时胜过错误算子"
            if gate["status"] == "pass"
            else "暂缓：正确 S 盒算子未通过双 seed 语义门"
        )
        fig.suptitle(
            "创新1 S2：逐样本执行逆 S 盒 ANF，能否识别正确 S 盒？\n"
            "同一候选检查点仅替换算子；A8 为无 ANF 算子的同协议锚点\n"
            f"裁决：{verdict}",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )
        fig.tight_layout(rect=(0.03, 0.035, 0.99, 0.87), h_pad=2.4, w_pad=2.8)
        fig.savefig(output, format="svg", bbox_inches="tight")
        plt.close(fig)


def _candidate_spec(
    base: dict[str, Any],
    config: dict[str, Any],
) -> RuntimeParameterizedSpnSpec:
    plain = _plain_spec(base["model"])
    candidate = config["candidate"]
    return RuntimeParameterizedSpnSpec(
        hidden_dim=plain.hidden_dim,
        pair_embedding_dim=plain.pair_embedding_dim,
        processor_steps=plain.processor_steps,
        dropout=plain.dropout,
        sbox_context_scale=float(candidate["sbox_context_scale"]),
        sbox_context_mode=candidate["sbox_context_mode"],
        cell_input_mode=candidate["cell_input_mode"],
        round_window_mode=candidate["round_window_mode"],
        relation_activity_pooling_mode=plain.relation_activity_pooling_mode,
        sbox_boolean_operator_mode=candidate["sbox_boolean_operator_mode"],
        sbox_boolean_operator_scale=float(candidate["sbox_boolean_operator_scale"]),
    )


def _assemble_payload(
    *,
    config: dict[str, Any],
    config_hash: str,
    base: dict[str, Any],
    structures: dict[str, RuntimeSpnStructure],
    operator_controls: dict[str, dict[str, RuntimeSpnStructure]],
    anchors: dict[tuple[int, str], dict[str, Any]],
    roles: dict[int, dict[str, Any]],
    datasets: dict[int, dict[str, Any]],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    source_macro_auc: dict[str, dict[str, float]] = {}
    target_auc: dict[str, dict[str, float]] = {}
    probability_deltas: dict[str, dict[str, float]] = {}
    protocol_by_name = {item["name"]: item for item in base["protocols"]}

    for seed in EXPECTED_SEEDS:
        key = str(seed)
        role = roles[seed]
        checkpoint_path = Path(role["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = SboxAnfOperatorRuntimeE4(_candidate_spec(base, config)).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        history.extend(
            {"seed": seed, "role": "candidate", **row} for row in role["history"]
        )
        auc_by_cipher: dict[str, dict[str, float]] = {}
        target_probabilities: dict[str, np.ndarray] = {}

        for cipher in CIPHERS:
            structure = structures[cipher]
            dataset = datasets[seed][cipher]
            feature_hash = _array_sha256(dataset.features)
            label_hash = _array_sha256(dataset.labels)
            probabilities: dict[str, np.ndarray] = {}
            auc_by_cipher[cipher] = {}
            for mode in OPERATOR_CONTROLS:
                operator = operator_controls[cipher][mode]
                probabilities[mode] = predict_binary_probabilities(
                    _BoundSboxAnfOperator(model, structure, operator),
                    dataset,
                    batch_size=config["training"]["batch_size"],
                    device=config["training"]["device"],
                )
            exact_probability = probabilities["exact"]
            for mode in OPERATOR_CONTROLS:
                operator = operator_controls[cipher][mode]
                metrics = _metrics(dataset.labels, probabilities[mode])
                auc_by_cipher[cipher][mode] = float(metrics["auc"])
                delta = np.abs(probabilities[mode] - exact_probability)
                rows.append(
                    {
                        "run_id": config["run_id"],
                        "row_kind": (
                            "source_validation"
                            if cipher in SOURCE_CIPHERS
                            else "holdout_target"
                        ),
                        "seed": seed,
                        "role": "candidate",
                        "cipher": cipher,
                        "cipher_display_name": DISPLAY_NAMES[cipher],
                        "rounds": protocol_by_name[cipher]["rounds"],
                        "operator_control": mode,
                        "operator_control_label": CONTROL_LABELS[mode],
                        **metrics,
                        "max_abs_probability_delta_from_exact": float(delta.max()),
                        "mean_abs_probability_delta_from_exact": float(delta.mean()),
                        "probability_sha256": _array_sha256(probabilities[mode]),
                        "feature_sha256": feature_hash,
                        "label_sha256": label_hash,
                        "main_sbox_truth_sha256": _tensor_sha256(
                            structure.sbox_truth_bits
                        ),
                        "operator_sbox_truth_sha256": _tensor_sha256(
                            operator.sbox_truth_bits
                        ),
                        "linear_topology_sha256": _tensor_sha256(
                            structure.linear_matrices
                        ),
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": role["checkpoint_sha256"],
                        "config_sha256": config_hash,
                        "training_samples_per_class": (
                            config["training"]["samples_per_class_per_source"]
                            if cipher in SOURCE_CIPHERS
                            else 0
                        ),
                        "validation_samples_per_class": config["training"][
                            "validation_samples_per_class_per_source"
                        ],
                        "pairs_per_sample": config["training"]["pairs_per_sample"],
                        "negative_mode": config["training"]["negative_mode"],
                        "parameter_count": role["parameter_count"],
                        "optimizer_steps_on_target": 0,
                        "target_head_trained": False,
                    }
                )
            if cipher == HOLDOUT_CIPHER:
                target_probabilities = probabilities

            anchor = anchors[(seed, cipher)]
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": (
                        "source_validation"
                        if cipher in SOURCE_CIPHERS
                        else "holdout_target"
                    ),
                    "seed": seed,
                    "role": "a8_anchor",
                    "cipher": cipher,
                    "cipher_display_name": DISPLAY_NAMES[cipher],
                    "rounds": protocol_by_name[cipher]["rounds"],
                    "operator_control": "a8_anchor",
                    "operator_control_label": CONTROL_LABELS["a8_anchor"],
                    **anchor["metrics"],
                    "checkpoint": anchor["checkpoint"],
                    "checkpoint_sha256": anchor["checkpoint_sha256"],
                    "config_sha256": config_hash,
                    "training_samples_per_class": (
                        config["training"]["samples_per_class_per_source"]
                        if cipher in SOURCE_CIPHERS
                        else 0
                    ),
                    "validation_samples_per_class": config["training"][
                        "validation_samples_per_class_per_source"
                    ],
                    "pairs_per_sample": config["training"]["pairs_per_sample"],
                    "negative_mode": config["training"]["negative_mode"],
                    "parameter_count": 442466,
                    "optimizer_steps_on_target": 0,
                    "target_head_trained": False,
                }
            )

        source_macro_auc[key] = {
            mode: float(
                np.mean([auc_by_cipher[cipher][mode] for cipher in SOURCE_CIPHERS])
            )
            for mode in OPERATOR_CONTROLS
        }
        source_macro_auc[key]["a8_anchor"] = float(
            np.mean([anchors[(seed, cipher)]["metrics"]["auc"] for cipher in SOURCE_CIPHERS])
        )
        target_auc[key] = {
            mode: auc_by_cipher[HOLDOUT_CIPHER][mode] for mode in OPERATOR_CONTROLS
        }
        target_auc[key]["a8_anchor"] = float(
            anchors[(seed, HOLDOUT_CIPHER)]["metrics"]["auc"]
        )
        probability_deltas[key] = {
            mode: float(
                np.max(
                    np.abs(
                        target_probabilities["exact"] - target_probabilities[mode]
                    )
                )
            )
            for mode in OPERATOR_CONTROLS[1:]
        }

    validation = _validate_payload(
        config=config,
        rows=rows,
        history=history,
        roles=roles,
        readiness=readiness,
        anchors=anchors,
    )
    return {
        "config": config,
        "rows": rows,
        "history": history,
        "source_macro_auc": source_macro_auc,
        "target_auc": target_auc,
        "probability_deltas": probability_deltas,
        "validation": validation,
    }


def _validate_payload(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    history: list[dict[str, Any]],
    roles: dict[int, dict[str, Any]],
    readiness: dict[str, Any],
    anchors: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    candidate_rows = [row for row in rows if row["role"] == "candidate"]
    anchor_rows = [row for row in rows if row["role"] == "a8_anchor"]
    grouped = {
        (int(row["seed"]), str(row["cipher"]), str(row["operator_control"])): row
        for row in candidate_rows
    }
    expected = {
        (seed, cipher, mode)
        for seed in EXPECTED_SEEDS
        for cipher in CIPHERS
        for mode in OPERATOR_CONTROLS
    }
    same_fields = (
        "feature_sha256",
        "label_sha256",
        "linear_topology_sha256",
        "checkpoint_sha256",
    )
    checks = {
        "readiness_gate_matches": readiness.get("status") == "pass"
        and readiness.get("decision")
        == "innovation1_runtime_spn_sbox_anf_operator_s2_readiness_passed"
        and all(readiness.get("checks", {}).values()),
        "expected_result_rows": len(rows) == config["gate"]["expected_result_rows"],
        "complete_candidate_control_panel": set(grouped) == expected,
        "expected_anchor_rows": len(anchor_rows) == 10 and len(anchors) == 10,
        "expected_history_rows": len(history)
        == config["gate"]["expected_history_rows"],
        "candidate_checkpoints_valid": all(
            Path(roles[seed]["checkpoint_path"]).is_file()
            and _file_sha256(Path(roles[seed]["checkpoint_path"]))
            == roles[seed]["checkpoint_sha256"]
            for seed in EXPECTED_SEEDS
        ),
        "candidate_parameter_count": {
            int(row["parameter_count"]) for row in candidate_rows
        }
        == {config["candidate"]["expected_parameter_count"]},
        "same_data_topology_checkpoint_within_controls": all(
            len({grouped[(seed, cipher, mode)][field] for mode in OPERATOR_CONTROLS})
            == 1
            for seed in EXPECTED_SEEDS
            for cipher in CIPHERS
            for field in same_fields
        ),
        "operator_truth_changes_for_controls": all(
            grouped[(seed, cipher, "exact")]["operator_sbox_truth_sha256"]
            != grouped[(seed, cipher, mode)]["operator_sbox_truth_sha256"]
            for seed in EXPECTED_SEEDS
            for cipher in CIPHERS
            for mode in OPERATOR_CONTROLS[1:]
        ),
        "source_only_checkpoint_selection": all(
            tuple(roles[seed]["metadata"]["task_names"]) == SOURCE_CIPHERS
            and roles[seed]["metadata"]["selected_checkpoint"] == "best"
            for seed in EXPECTED_SEEDS
        ),
        "gradient_protocol_exact": all(
            roles[seed]["metadata"]["gradient_combination"]
            == config["candidate"]["gradient_combination"]
            and roles[seed]["gradient_diagnostics"]["all_gradients_finite"]
            for seed in EXPECTED_SEEDS
        ),
        "target_never_trained": all(
            row["training_samples_per_class"] == 0
            and row["optimizer_steps_on_target"] == 0
            and row["target_head_trained"] is False
            for row in rows
            if row["cipher"] == HOLDOUT_CIPHER
        ),
        "strict_negative_and_fixed_protocol": all(
            row["validation_samples_per_class"] == 1024
            and row["pairs_per_sample"] == 4
            and row["negative_mode"] == "encrypted_random_plaintexts"
            for row in rows
        ),
        "all_metrics_finite": all(
            all(
                isinstance(row[field], (int, float)) and math.isfinite(float(row[field]))
                for field in ("auc", "accuracy", "best_accuracy", "loss")
            )
            for row in rows
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "result_rows": len(rows),
        "expected_result_rows": config["gate"]["expected_result_rows"],
        "history_rows": len(history),
        "expected_history_rows": config["gate"]["expected_history_rows"],
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
    }


def _load_a8_anchors(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    anchors: dict[tuple[int, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        selected = row.get("row_kind") == "source_validation" and row.get("role") == (
            "correct_candidate"
        )
        selected = selected or (
            row.get("row_kind") == "holdout_target"
            and row.get("evaluation") == "candidate_correct"
        )
        if not selected:
            continue
        anchors[(int(row["seed"]), str(row["cipher"]))] = {
            "metrics": dict(row["metrics"]["validation"]),
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
    expected = {(seed, cipher) for seed in EXPECTED_SEEDS for cipher in CIPHERS}
    if set(anchors) != expected:
        raise ValueError("S2 could not recover the complete A8 anchor panel")
    return anchors


def _anf_reconstructs_all_tables(structure: RuntimeSpnStructure) -> bool:
    values = torch.arange(16, dtype=torch.long)
    bits = ((values[:, None] >> torch.arange(3, -1, -1)) & 1).to(torch.float32)
    states = bits[:, None, :].repeat(1, structure.cells, 1).reshape(
        16,
        structure.block_bits,
    )
    for round_index in range(structure.rounds):
        contributions = inverse_sbox_anf_contributions(
            states,
            structure,
            round_index=round_index,
        )
        reconstructed = torch.remainder(contributions.sum(dim=-1), 2.0)
        expected_state = structure.apply_inverse_sboxes(states, round_index)
        expected = RuntimeE4EquivariantSpnDistinguisher._ordered_cell_values(
            expected_state[:, None, :],
            structure,
        )[:, 0]
        if not torch.equal(reconstructed, expected):
            return False
    return True


def _control_contract(
    exact: RuntimeSpnStructure,
    controls: dict[str, RuntimeSpnStructure],
) -> bool:
    return bool(
        tuple(controls) == OPERATOR_CONTROLS
        and torch.equal(controls["exact"].sbox_truth_bits, exact.sbox_truth_bits)
        and not torch.equal(
            controls["input_permuted"].sbox_truth_bits,
            exact.sbox_truth_bits,
        )
        and not torch.equal(controls["identity"].sbox_truth_bits, exact.sbox_truth_bits)
        and all(_truth_is_permutation(item.sbox_truth_bits) for item in controls.values())
        and all(
            torch.equal(item.cell_membership, exact.cell_membership)
            and torch.equal(item.bit_role, exact.bit_role)
            and torch.equal(item.linear_matrices, exact.linear_matrices)
            and torch.equal(item.inverse_linear_matrices, exact.inverse_linear_matrices)
            for item in controls.values()
        )
    )


def _cell_relabel_error(
    model: SboxAnfOperatorRuntimeE4,
    structure: RuntimeSpnStructure,
    operator: RuntimeSpnStructure,
) -> float:
    permutation = tuple(reversed(range(structure.cells)))
    relabeled, bit_permutation = structure.relabel_cells(permutation)
    relabeled_operator, operator_bit_permutation = operator.relabel_cells(permutation)
    if not torch.equal(bit_permutation, operator_bit_permutation):
        return math.inf
    generator = torch.Generator().manual_seed(26_072_614 + structure.block_bits)
    features = torch.randint(
        0,
        2,
        (2, 4, 2, structure.block_bits),
        generator=generator,
        dtype=torch.float32,
    )
    relabeled_features = torch.empty_like(features)
    relabeled_features[..., bit_permutation] = features
    with torch.no_grad():
        original = model.backbone(
            features,
            structure,
            operator_structure=operator,
        )
        transformed = model.backbone(
            relabeled_features,
            relabeled,
            operator_structure=relabeled_operator,
        )
    return float((original - transformed).abs().max())


def _finite_gradient_probe(
    model: SboxAnfOperatorRuntimeE4,
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    generator = torch.Generator().manual_seed(26_072_615)
    for cipher in SOURCE_CIPHERS:
        structure = structures[cipher]
        features = torch.randint(
            0,
            2,
            (2, 4, 2, structure.block_bits),
            generator=generator,
            dtype=torch.float32,
        )
        losses.append(model(features, structure).square().mean())
    torch.stack(losses).mean().backward()
    operator_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "sbox_boolean_operator_projection" in name
    ]
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    result = {
        "all_values_finite": all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "operator_gradient_nonzero": bool(operator_gradients)
        and all(gradient is not None for gradient in operator_gradients)
        and sum(
            float(gradient.abs().sum())
            for gradient in operator_gradients
            if gradient is not None
        )
        > 0.0,
        "gradient_tensor_count": len(gradients),
    }
    model.zero_grad(set_to_none=True)
    return result


def _load_role(
    role_path: Path,
    checkpoint_path: Path,
    *,
    config_hash: str,
) -> dict[str, Any] | None:
    if not role_path.is_file() or not checkpoint_path.is_file():
        return None
    role = _read_json(role_path)
    if role.get("config_sha256") != config_hash:
        return None
    if role.get("checkpoint_path") != str(checkpoint_path):
        return None
    return role


def _validate_role(
    role: dict[str, Any],
    checkpoint_path: Path,
    *,
    seed: int,
    config_hash: str,
) -> None:
    if role.get("seed") != seed or role.get("role") != "candidate":
        raise ValueError("S2 resumed role identity drifted")
    if role.get("config_sha256") != config_hash:
        raise ValueError("S2 resumed role config hash drifted")
    if role.get("parameter_count") != 459234:
        raise ValueError("S2 resumed role parameter count drifted")
    if role.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("S2 resumed checkpoint hash drifted")
    if tuple(role.get("metadata", {}).get("task_names", ())) != SOURCE_CIPHERS:
        raise ValueError("S2 resumed role source panel drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {
        "seed": seed,
        "role": "candidate",
        "config_sha256": config_hash,
        "checkpoint_selection_tasks": list(SOURCE_CIPHERS),
        "holdout_cipher": HOLDOUT_CIPHER,
        "operator_mode": "inverse_anf_contribution_gate",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"S2 checkpoint metadata drifted: {key}")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("S2 checkpoint state_dict is missing")


def _readiness_gate_path() -> Path:
    return Path(
        "outputs/local_readiness/"
        "i1_runtime_spn_sbox_anf_operator_s2_readiness_20260726/gate.json"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
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
    "CIPHERS",
    "HOLDOUT_CIPHER",
    "OPERATOR_CONTROLS",
    "SOURCE_CIPHERS",
    "SboxAnfOperatorRuntimeE4",
    "adjudicate_sbox_anf_operator",
    "build_sbox_operator_controls",
    "load_and_validate_sbox_anf_operator_config",
    "render_sbox_anf_operator_svg",
    "run_sbox_anf_operator",
    "run_sbox_anf_operator_readiness",
    "write_sbox_anf_operator_artifacts",
    "write_sbox_anf_operator_readiness_artifacts",
]
