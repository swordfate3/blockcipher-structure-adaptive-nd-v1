from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.deterministic_token_edge_basis import (
    DeterministicTokenEdgeBasisK1AzProbe,
)
from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    PositionPreservingOperatorSpec,
    relabel_runtime_pairs,
    trainable_parameter_geometry,
    transported_position_ids,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1ay import (
    build_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_mandatory_token_gate_k1be import (
    build_candidate_probe as build_k1be_probe,
    load_and_validate_config as load_k1be_config,
    load_authority as load_k1be_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bc import (
    build_probe as build_k1bc_probe,
    load_and_validate_config as load_k1bc_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_deterministic_token_edge_basis_k1bg_readiness_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_deterministic_token_edge_basis_k1bg_readiness_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "1549994ed2d0fabf2ce2d4e20fa2b26f5ae27dc7d3dba264c85c04e2e40fe9d3"
)
REPLICAS = (0, 1)
PROBE_ROWS = 64
EXPECTED_PANEL_ROWS = 12
EXPECTED_GRADIENT_ROWS = 6
EXPECTED_TRAINABLE_PARAMETERS = 25_696
WRONG_CONDITIONS = (
    "same_summary_corrupted_operator",
    "cross_cipher_operator",
)
MODELS = ("candidate", "k1be", "k1bc")
BASIS_GRAM_TOLERANCE = 1e-6
WHOLE_PATH_K1BE_RATIO_MINIMUM = 0.5
TOPOLOGY_K1BC_MULTIPLIER = 4.0
RELABEL_TOLERANCE = 1e-5
SOURCE_REPLAY_TOLERANCE = 1e-12


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BG config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BG identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_deterministic_token_edge_basis_k1bg_readiness"
    ):
        raise ValueError("K1-BG experiment name drifted")
    if config.get("model") != {
        "hidden_dim": 32,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "modulation_scale": 0.05,
        "operator_token_dim": 18,
        "edge_basis": "sylvester_hadamard_18_to_32_orthonormal_then_rms_tanh",
        "basis_projection_trainable": False,
        "token_encoder_present": False,
        "sample_only_bypass": False,
        "structure_projection": False,
        "shared_across_widths": True,
        "cipher_identity": False,
        "per_cipher_modules": False,
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
    }:
        raise ValueError("K1-BG model contract drifted")
    if config.get("evaluation") != {
        "replicas": list(REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "conditions": [
            "correct_operator",
            "same_summary_corrupted_operator",
            "cross_cipher_operator",
            "disabled_k1az",
            "joint_relabel",
        ],
        "encoder_initialization_seeds": [40, 41],
        "probe_rows_per_panel": PROBE_ROWS,
        "gradient_rows_per_cipher": PROBE_ROWS,
        "expected_panel_rows": EXPECTED_PANEL_ROWS,
        "expected_gradient_rows": EXPECTED_GRADIENT_ROWS,
        "training_performed": False,
        "optimizer_steps": 0,
        "data_generation": False,
        "device": "cpu",
        "execution": "local_readiness",
    }:
        raise ValueError("K1-BG evaluation contract drifted")
    if config.get("gates") != {
        "fixed_basis_gram_max_abs_error": BASIS_GRAM_TOLERANCE,
        "whole_path_probability_rms_k1be_ratio_min": (
            WHOLE_PATH_K1BE_RATIO_MINIMUM
        ),
        "topology_probability_ratio_k1bc_multiplier_min": (
            TOPOLOGY_K1BC_MULTIPLIER
        ),
        "require_topology_share_strictly_above_k1be": True,
        "minimum_nonzero_modulation_delta": 0.0,
        "minimum_nonzero_logit_delta": 0.0,
        "minimum_nonzero_probability_delta": 0.0,
        "disabled_k1az_logit_replay_tolerance": 0.0,
        "joint_relabel_modulation_tolerance": RELABEL_TOLERANCE,
        "joint_relabel_logit_tolerance": RELABEL_TOLERANCE,
        "require_all_trainable_parameters_graph_connected": True,
        "require_fixed_parameter_geometry": True,
        "require_zero_updates_and_immutable_states": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BG gate contract drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Mapping[tuple[str, int, str], Any],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor | None]],
    Mapping[int, Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[tuple[int, str, str], Mapping[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {name: source_root / name for name in source["digests"]}
    source_config_path = project_root / str(source["config"])
    source_config = load_k1be_config(source_config_path)
    (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        corrupted_structures,
        cross_operators,
        inherited_checks,
    ) = load_k1be_authority(
        source_config,
        project_root=project_root,
        device=device,
    )
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    source_panels = _read_jsonl(paths["panel_results.jsonl"])
    panel_lookup = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["split"])): row
        for row in source_panels
    }
    checks = {
        "source_config_digest_exact": (
            file_sha256(source_config_path) == source["config_sha256"]
        ),
        "all_six_source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        )
        and len(paths) == 6,
        "source_gate_requires_deterministic_basis": (
            gate.get("status") == "hold"
            and gate.get("decision") == source["required_decision"]
            and gate.get("whole_path_retention_all") is True
            and gate.get("topology_share_lift_all") is False
            and not gate.get("failed_protocol_checks")
            and not gate.get("failed_compatibility_checks")
        ),
        "source_validation_passes": validation.get("status") == "pass"
        and not validation.get("errors"),
        "twelve_source_panels_rebound": len(source_panels) == EXPECTED_PANEL_ROWS
        and len(panel_lookup) == EXPECTED_PANEL_ROWS,
        "eighteen_datasets_rebound": len(datasets) == 18,
        **{f"k1be_{name}": bool(value) for name, value in inherited_checks.items()},
    }
    return (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        corrupted_structures,
        cross_operators,
        panel_lookup,
        checks,
    )


def build_candidate_probe(
    *,
    runtime_config: Mapping[str, Any],
    structures: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    initialization_seed: int,
    model_config: Mapping[str, Any],
    device: str,
) -> DeterministicTokenEdgeBasisK1AzProbe:
    cipher_configs = {
        str(row["cipher_key"]): row for row in runtime_config["ciphers"]
    }
    with torch.random.fork_rng():
        torch.manual_seed(initialization_seed)
        anchor = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            runtime_config["model"],
            {"gate_hidden_dim": 12},
        ).to(device)
        incompatible = anchor.load_state_dict(checkpoint["state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("K1-BG strict K1-AZ checkpoint load drifted")
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        probe = DeterministicTokenEdgeBasisK1AzProbe(
            anchor,
            PositionPreservingOperatorSpec(
                hidden_dim=int(model_config["hidden_dim"]),
                pair_embedding_dim=int(model_config["pair_embedding_dim"]),
                dropout=float(model_config["dropout"]),
                modulation_scale=float(model_config["modulation_scale"]),
            ),
        ).to(device)
    trainable = sum(
        parameter.numel()
        for parameter in probe.parameters()
        if parameter.requires_grad
    )
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError(f"K1-BG parameter count drifted: {trainable}")
    if any(parameter.requires_grad for parameter in probe.anchor.parameters()):
        raise ValueError("K1-BG anchor must remain frozen")
    if hasattr(probe.operator_encoder, "token_encoder"):
        raise ValueError("K1-BG learned token encoder must be absent")
    if hasattr(probe.operator_encoder, "structure_projection"):
        raise ValueError("K1-BG readiness-only projection must be absent")
    if structures["uknit64"].block_bits != 64 or structures["dialga128"].block_bits != 128:
        raise ValueError("K1-BG frozen widths drifted")
    return probe


def collect_readiness_rows(
    *,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, torch.Tensor | None]],
    source_checkpoints: Mapping[int, Mapping[str, Any]],
    corrupted_structures: Mapping[str, Any],
    cross_operators: Mapping[str, Any],
    source_panels: Mapping[tuple[int, str, str], Mapping[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    panels: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    k1be_config = load_k1be_config()
    k1bc_config = load_k1bc_config()
    replica_configs = {
        int(row["replica"]): row for row in k1bc_config["replicas"]
    }
    for replica, initialization_seed in zip(
        REPLICAS,
        config["evaluation"]["encoder_initialization_seeds"],
        strict=True,
    ):
        common = {
            "runtime_config": runtime_config,
            "structures": structures,
            "checkpoint": source_checkpoints[replica],
            "initialization_seed": int(initialization_seed),
            "device": device,
        }
        candidate = build_candidate_probe(
            **common,
            model_config=config["model"],
        )
        k1be = build_k1be_probe(
            **common,
            model_config=k1be_config["model"],
        )
        k1bc = build_k1bc_probe(
            **common,
            model_config=k1bc_config["model"],
        )
        candidate.eval()
        k1be.eval()
        k1bc.eval()
        states_before = {
            model: tensor_mapping_sha256(probe.state_dict())
            for model, probe in (("candidate", candidate), ("k1be", k1be), ("k1bc", k1bc))
        }
        projection = candidate.operator_encoder.basis_projection.detach().cpu()
        gram = projection @ projection.T
        gram_error = float(
            torch.max(torch.abs(gram - torch.eye(gram.shape[0]))).item()
        )
        projection_rank = int(torch.linalg.matrix_rank(projection).item())
        projection_sha256 = tensor_mapping_sha256({"basis_projection": projection})
        candidate_geometry = trainable_parameter_geometry(candidate.operator_encoder)
        for cipher in EXPECTED_CIPHERS:
            structure = structures[cipher]
            geometry.append(
                {
                    "run_id": RUN_ID,
                    "replica": replica,
                    "cipher_key": cipher,
                    "block_bits": structure.block_bits,
                    "trainable_parameter_count": sum(
                        parameter.numel()
                        for parameter in candidate.operator_encoder.parameters()
                    ),
                    "trainable_parameter_geometry": {
                        name: list(shape)
                        for name, shape in candidate_geometry.items()
                    },
                    "basis_projection_rank": projection_rank,
                    "basis_projection_gram_max_abs_error": gram_error,
                    "basis_projection_sha256": projection_sha256,
                    "basis_projection_trainable": candidate.basis_projection_trainable,
                    "token_encoder_present": candidate.token_encoder_present,
                    "sample_only_bypass": candidate.sample_only_bypass,
                    "readiness_only_projection_present": (
                        candidate.readiness_only_projection_present
                    ),
                    "uses_cipher_identity": candidate.uses_cipher_identity,
                    "uses_per_cipher_parameters": candidate.uses_per_cipher_parameters,
                }
            )
            seed = int(replica_configs[replica]["dataset_seeds"][cipher])
            summary = summaries[cipher]["correct_descriptor"]
            if summary is None:
                raise ValueError("K1-BG correct gate summary is missing")
            gradients.append(
                measure_gradient_coverage(
                    probe=candidate,
                    dataset=datasets[(cipher, seed, "train_seen")],
                    runtime_structure=structure,
                    operator_structure=structure,
                    summary=summary,
                    replica=replica,
                    cipher=cipher,
                    device=device,
                )
            )
            for split in FRESH_SPLITS:
                panels.append(
                    measure_panel(
                        candidate=candidate,
                        k1be=k1be,
                        k1bc=k1bc,
                        dataset=datasets[(cipher, seed, split)],
                        structure=structure,
                        corrupted=corrupted_structures[cipher],
                        cross_operator=cross_operators[cipher],
                        summary=summary,
                        source_panel=source_panels[(replica, cipher, split)],
                        replica=replica,
                        cipher=cipher,
                        seed=seed,
                        split=split,
                        device=device,
                    )
                )
        states_after = {
            model: tensor_mapping_sha256(probe.state_dict())
            for model, probe in (("candidate", candidate), ("k1be", k1be), ("k1bc", k1bc))
        }
        for row in (*panels, *gradients):
            if int(row["replica"]) == replica:
                for model in MODELS:
                    row[f"{model}_state_immutable"] = (
                        states_before[model] == states_after[model]
                    )
    return panels, gradients, geometry


def measure_panel(
    *,
    candidate: nn.Module,
    k1be: nn.Module,
    k1bc: nn.Module,
    dataset: Any,
    structure: Any,
    corrupted: Any,
    cross_operator: Any,
    summary: torch.Tensor,
    source_panel: Mapping[str, Any],
    replica: int,
    cipher: str,
    seed: int,
    split: str,
    device: str,
) -> dict[str, Any]:
    features = torch.as_tensor(
        np.array(dataset.features[:PROBE_ROWS], copy=True),
        dtype=torch.float32,
        device=device,
    )
    runtime_pairs = features.reshape(
        features.shape[0], -1, 2, structure.block_bits
    ).flip(-1)
    cell_permutation = torch.roll(torch.arange(structure.cells), shifts=-1)
    relabeled_structure, bit_permutation = structure.relabel_cells(
        cell_permutation.tolist()
    )
    position_ids = transported_position_ids(cell_permutation)
    relabeled_pairs = relabel_runtime_pairs(runtime_pairs, bit_permutation)
    relabeled_features = relabeled_pairs.flip(-1).reshape_as(features)
    operators = {
        "same_summary_corrupted_operator": corrupted,
        "cross_cipher_operator": cross_operator,
    }
    probes = {"candidate": candidate, "k1be": k1be, "k1bc": k1bc}
    with torch.inference_mode():
        correct = {
            model: _probe_correct(probe, features, runtime_pairs, structure, summary)
            for model, probe in probes.items()
        }
        row: dict[str, Any] = {
            "run_id": RUN_ID,
            "replica": replica,
            "cipher_key": cipher,
            "seed": seed,
            "split": split,
            "probe_rows": PROBE_ROWS,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for model in MODELS:
            row[f"{model}_whole_path_probability_rms"] = _rms(
                correct[model]["probability"] - correct[model]["disabled_probability"]
            )
        for condition, operator in operators.items():
            for model, probe in probes.items():
                wrong_modulation = probe.operator_encoder.sample_modulation(
                    runtime_pairs,
                    structure,
                    operator,
                )
                wrong_logits = probe.logits_with_operator(
                    features,
                    structure,
                    operator,
                    gate_summary=summary,
                )
                probability_rms = _rms(
                    correct[model]["probability"] - torch.sigmoid(wrong_logits)
                )
                row.update(
                    {
                        f"{model}_{condition}_modulation_rms": _rms(
                            correct[model]["modulation"] - wrong_modulation
                        ),
                        f"{model}_{condition}_logit_rms": _rms(
                            correct[model]["logits"] - wrong_logits
                        ),
                        f"{model}_{condition}_probability_rms": probability_rms,
                        f"{model}_{condition}_topology_share": probability_rms
                        / max(
                            float(row[f"{model}_whole_path_probability_rms"]),
                            1e-12,
                        ),
                    }
                )
        relabeled_modulation = candidate.operator_encoder.sample_modulation(
            relabeled_pairs,
            relabeled_structure,
            relabeled_structure,
            cell_position_ids=position_ids,
        )
        relabeled_logits = candidate.logits_with_operator(
            relabeled_features,
            relabeled_structure,
            relabeled_structure,
            gate_summary=summary,
            cell_position_ids=position_ids,
        )
        row.update(
            {
                "disabled_k1az_logit_replay_delta": _max_abs(
                    correct["candidate"]["disabled_logits"],
                    correct["k1be"]["disabled_logits"],
                ),
                "joint_relabel_modulation_delta": _max_abs(
                    correct["candidate"]["modulation"],
                    relabeled_modulation,
                ),
                "joint_relabel_logit_delta": _max_abs(
                    correct["candidate"]["logits"],
                    relabeled_logits,
                ),
            }
        )
    row["matched_k1be_source_replay_max_abs_delta"] = _source_replay_delta(
        row,
        source_panel,
    )
    return row


def _probe_correct(
    probe: nn.Module,
    features: torch.Tensor,
    runtime_pairs: torch.Tensor,
    structure: Any,
    summary: torch.Tensor,
) -> dict[str, torch.Tensor]:
    modulation = probe.operator_encoder.sample_modulation(
        runtime_pairs,
        structure,
        structure,
    )
    logits = probe.logits_with_operator(
        features,
        structure,
        structure,
        gate_summary=summary,
    )
    disabled_logits = probe.logits_with_operator(
        features,
        structure,
        structure,
        gate_summary=summary,
        enabled=False,
    )
    return {
        "modulation": modulation,
        "logits": logits,
        "probability": torch.sigmoid(logits),
        "disabled_logits": disabled_logits,
        "disabled_probability": torch.sigmoid(disabled_logits),
    }


def _source_replay_delta(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> float:
    pairs = [
        ("k1be_whole_path_probability_rms", "candidate_whole_path_probability_rms"),
        ("k1bc_whole_path_probability_rms", "anchor_whole_path_probability_rms"),
    ]
    for condition in WRONG_CONDITIONS:
        for metric in ("modulation_rms", "logit_rms", "probability_rms", "topology_share"):
            pairs.extend(
                (
                    (f"k1be_{condition}_{metric}", f"candidate_{condition}_{metric}"),
                    (f"k1bc_{condition}_{metric}", f"anchor_{condition}_{metric}"),
                )
            )
    return max(abs(float(row[left]) - float(source[right])) for left, right in pairs)


def measure_gradient_coverage(
    *,
    probe: nn.Module,
    dataset: Any,
    runtime_structure: Any,
    operator_structure: Any,
    summary: torch.Tensor,
    replica: int,
    cipher: str,
    device: str,
) -> dict[str, Any]:
    labels_array = np.asarray(dataset.labels).reshape(-1)
    positive = np.flatnonzero(labels_array == 1)[:32]
    negative = np.flatnonzero(labels_array == 0)[:32]
    indices = np.concatenate((positive, negative))
    features = torch.as_tensor(
        np.array(dataset.features[indices], copy=True),
        dtype=torch.float32,
        device=device,
    )
    labels = torch.as_tensor(
        np.array(dataset.labels[indices], copy=True),
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    named_parameters = list(probe.operator_encoder.named_parameters())
    logits = probe.logits_with_operator(
        features,
        runtime_structure,
        operator_structure,
        gate_summary=summary,
    )
    loss = nn.functional.mse_loss(torch.sigmoid(logits), labels)
    gradients = torch.autograd.grad(
        loss,
        [parameter for _name, parameter in named_parameters],
        allow_unused=True,
    )
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "rows": len(indices),
        "parameter_tensor_count": len(named_parameters),
        "graph_connected_tensor_count": sum(
            gradient is not None for gradient in gradients
        ),
        "nonzero_gradient_tensor_count": sum(
            gradient is not None and int(torch.count_nonzero(gradient)) > 0
            for gradient in gradients
        ),
        "trainable_parameter_count": sum(
            parameter.numel() for _name, parameter in named_parameters
        ),
        "loss": float(loss.detach().cpu()),
        "persistent_grads_none": all(
            parameter.grad is None for parameter in probe.parameters()
        ),
        "training_performed": False,
        "optimizer_steps": 0,
    }


def adjudicate_readiness(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    panels: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_panels = {
        (replica, cipher, split)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    actual_panels = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        for row in panels
    }
    geometries = {
        json.dumps(row["trainable_parameter_geometry"], sort_keys=True)
        for row in geometry
    }
    basis_digests = {str(row["basis_projection_sha256"]) for row in geometry}
    finite_fields = (
        [key for key in panels[0] if key.endswith(("_rms", "_share", "_delta"))]
        if panels
        else []
    )
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "twelve_panel_rows_complete": len(panels) == EXPECTED_PANEL_ROWS
        and actual_panels == expected_panels,
        "six_gradient_rows_complete": len(gradients) == EXPECTED_GRADIENT_ROWS,
        "six_fixed_geometry_rows_complete": len(geometry) == 6
        and len(geometries) == 1
        and len(basis_digests) == 1
        and all(
            int(row["trainable_parameter_count"])
            == EXPECTED_TRAINABLE_PARAMETERS
            for row in geometry
        ),
        "fixed_full_rank_basis_without_learned_token_encoder": all(
            int(row["basis_projection_rank"]) == 18
            and float(row["basis_projection_gram_max_abs_error"])
            <= BASIS_GRAM_TOLERANCE
            and row.get("basis_projection_trainable") is False
            and row.get("token_encoder_present") is False
            and row.get("sample_only_bypass") is False
            and row.get("readiness_only_projection_present") is False
            and row.get("uses_cipher_identity") is False
            and row.get("uses_per_cipher_parameters") is False
            for row in geometry
        ),
        "zero_updates_and_immutable_states": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and all(row.get(f"{model}_state_immutable") is True for model in MODELS)
            for row in (*panels, *gradients)
        ),
        "all_metrics_finite": all(
            all(math.isfinite(float(row[field])) for field in finite_fields)
            for row in panels
        )
        and all(math.isfinite(float(row["loss"])) for row in gradients),
    }
    compatibility_checks = {
        "matched_k1be_and_k1bc_source_replay_exact": all(
            float(row["matched_k1be_source_replay_max_abs_delta"])
            <= SOURCE_REPLAY_TOLERANCE
            for row in panels
        ),
        "disabled_path_exactly_replays_k1az": all(
            float(row["disabled_k1az_logit_replay_delta"]) == 0.0
            for row in panels
        ),
        "joint_relabel_is_equivariant": all(
            float(row["joint_relabel_modulation_delta"]) <= RELABEL_TOLERANCE
            and float(row["joint_relabel_logit_delta"]) <= RELABEL_TOLERANCE
            for row in panels
        ),
        "all_trainable_tensors_graph_connected": all(
            int(row["graph_connected_tensor_count"])
            == int(row["parameter_tensor_count"])
            and row.get("persistent_grads_none") is True
            for row in gradients
        ),
    }
    whole_path_checks = {
        _panel_key(row): float(row["candidate_whole_path_probability_rms"])
        >= WHOLE_PATH_K1BE_RATIO_MINIMUM
        * float(row["k1be_whole_path_probability_rms"])
        for row in panels
    }
    nonzero_topology_checks = {
        f"{_panel_key(row)}|{condition}": all(
            float(row[f"candidate_{condition}_{metric}_rms"]) > 0.0
            for metric in ("modulation", "logit", "probability")
        )
        for row in panels
        for condition in WRONG_CONDITIONS
    }
    topology_summaries: dict[str, dict[str, float | bool]] = {}
    for condition in WRONG_CONDITIONS:
        medians = {
            model: float(
                np.median(
                    [float(row[f"{model}_{condition}_topology_share"]) for row in panels]
                )
            )
            for model in MODELS
        }
        k1bc_multiplier = medians["candidate"] / max(medians["k1bc"], 1e-12)
        k1be_multiplier = medians["candidate"] / max(medians["k1be"], 1e-12)
        topology_summaries[condition] = {
            "candidate_median_topology_share": medians["candidate"],
            "k1be_median_topology_share": medians["k1be"],
            "k1bc_median_topology_share": medians["k1bc"],
            "candidate_to_k1bc_multiplier": k1bc_multiplier,
            "candidate_to_k1be_multiplier": k1be_multiplier,
            "passes_k1bc_four_times": k1bc_multiplier
            >= TOPOLOGY_K1BC_MULTIPLIER,
            "strictly_exceeds_k1be": medians["candidate"] > medians["k1be"],
        }
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_compatibility = [
        name for name, passed in compatibility_checks.items() if not passed
    ]
    whole_path_all = all(whole_path_checks.values())
    topology_nonzero_all = all(nonzero_topology_checks.values())
    topology_lift_all = all(
        bool(row["passes_k1bc_four_times"])
        and bool(row["strictly_exceeds_k1be"])
        for row in topology_summaries.values()
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1bg_protocol_invalid"
        next_action = "Repair only the failed source, basis, geometry, count, state or finite-metric binding."
    elif failed_compatibility:
        status = "hold"
        decision = "innovation1_uknit_family_k1bg_compatibility_incomplete"
        next_action = "Repair only source replay, graph connectivity, disabled replay or relabel equivariance; do not train."
    elif not whole_path_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bg_fixed_basis_path_too_weak"
        next_action = "Reject the fixed basis; do not rescue it by increasing the global scale."
    elif not topology_nonzero_all or not topology_lift_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bg_deterministic_basis_topology_lift_not_supported"
        next_action = "Reject the fixed basis and stop this learned edge-message family; audit a different structure-consumption primitive before any training."
    else:
        status = "pass"
        decision = "innovation1_uknit_family_k1bg_deterministic_basis_readiness_authorized"
        next_action = (
            "Preregister one local same-budget 2048/class/cipher, four-pair, "
            "two-replica, ten-epoch comparison against K1-BC and K1-BE."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "compatibility_checks": compatibility_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_compatibility_checks": failed_compatibility,
        "whole_path_checks": whole_path_checks,
        "whole_path_retention_all": whole_path_all,
        "nonzero_topology_checks": nonzero_topology_checks,
        "nonzero_topology_all": topology_nonzero_all,
        "topology_summaries": topology_summaries,
        "topology_share_lift_all": topology_lift_all,
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "remote_scale": "no",
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Zero-training local readiness on matched K1-BE/K1-BC 64-row probes "
            "and four-pair caches; not accuracy improvement, formal scale, an "
            "attack, unseen-cipher transfer, arbitrary-SPN proof or SOTA evidence."
        ),
    }


def run_readiness(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    _require_fresh_output_root(output_root)
    output_root.mkdir(parents=True)
    _append_progress(output_root / "progress.jsonl", "run_start")
    (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        corrupted_structures,
        cross_operators,
        source_panels,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-BG source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "evaluation": dict(config["evaluation"]),
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    panels, gradients, geometry = collect_readiness_rows(
        config=config,
        runtime_config=runtime_config,
        datasets=datasets,
        structures=structures,
        summaries=summaries,
        source_checkpoints=source_checkpoints,
        corrupted_structures=corrupted_structures,
        cross_operators=cross_operators,
        source_panels=source_panels,
        device=device,
    )
    gate = adjudicate_readiness(
        config=config,
        source_checks=source_checks,
        panels=panels,
        gradients=gradients,
        geometry=geometry,
    )
    results = [
        {
            "run_id": RUN_ID,
            "metric_type": "topology_share",
            "condition": condition,
            **summary,
        }
        for condition, summary in gate["topology_summaries"].items()
    ]
    results.append(
        {
            "run_id": RUN_ID,
            "metric_type": "readiness_summary",
            "whole_path_retention_all": gate["whole_path_retention_all"],
            "nonzero_topology_all": gate["nonzero_topology_all"],
            "topology_share_lift_all": gate["topology_share_lift_all"],
            "trainable_parameter_count": gate["trainable_parameter_count"],
            "optimizer_steps": 0,
        }
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "panel_rows": len(panels),
        "expected_panel_rows": EXPECTED_PANEL_ROWS,
        "gradient_rows": len(gradients),
        "expected_gradient_rows": EXPECTED_GRADIENT_ROWS,
        "geometry_rows": len(geometry),
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "topology_summaries": gate["topology_summaries"],
        "whole_path_retention_all": gate["whole_path_retention_all"],
        "compatibility_checks": gate["compatibility_checks"],
        "trainable_parameter_count": gate["trainable_parameter_count"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_json(output_root / "geometry.json", {"run_id": RUN_ID, "rows": geometry})
    _write_jsonl(output_root / "panel_results.jsonl", panels)
    _write_jsonl(output_root / "gradient_coverage.jsonl", gradients)
    _write_jsonl(output_root / "results.jsonl", results)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    return {
        "preflight": preflight,
        "panels": panels,
        "gradients": gradients,
        "geometry": geometry,
        "results": results,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _panel_key(row: Mapping[str, Any]) -> str:
    return f"replica{int(row['replica'])}|{row['cipher_key']}|{row['split']}"


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(values.to(torch.float64).square().mean()).cpu())


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).cpu())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    row = {"run_id": RUN_ID, "event": event, "time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _require_fresh_output_root(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"K1-BG output already exists: {path}")


__all__ = [
    "CONFIG_PATH",
    "RUN_ID",
    "adjudicate_readiness",
    "build_candidate_probe",
    "collect_readiness_rows",
    "load_and_validate_config",
    "load_authority",
    "measure_gradient_coverage",
    "measure_panel",
    "run_readiness",
]
