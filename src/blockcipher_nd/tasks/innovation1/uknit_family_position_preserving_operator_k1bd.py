from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_gradient_conflict_k1ap import (
    gradient_cosine,
    make_stratified_batches,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bc import (
    build_probe,
    load_and_validate_config as load_source_config,
    load_authority as load_source_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_position_preserving_operator_k1bd_"
    "gradient_coupling_audit_replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_position_preserving_operator_k1bd_"
    "gradient_coupling_audit_replica0_replica1_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "2bee09cd12de86c72b975238e1111dee9c13e45b72e0702752e0894101d30fed"
)
REPLICAS = (0, 1)
ENCODER_STATES = ("initial_encoder", "selected_encoder")
OPERATOR_CONDITIONS = (
    "correct_operator",
    "same_summary_corrupted_operator",
    "cross_cipher_operator",
)
WRONG_OPERATOR_CONDITIONS = OPERATOR_CONDITIONS[1:]
PARAMETER_GROUPS = (
    "connected_all",
    "bit_encoder",
    "token_encoder",
    "edge_message",
    "bit_update",
    "bit_update_norm",
    "pair_projection",
    "structure_projection",
)
BATCH_SIZE = 64
BATCH_TRIPLETS = 64
ROWS_PER_CLASS_PER_BATCH = 32
EXPECTED_NORM_ROWS = 18_432
EXPECTED_TOPOLOGY_PAIR_ROWS = 1_536
EXPECTED_CROSS_PAIR_ROWS = 768
EXPECTED_INTERVENTION_ROWS = 24
TOPOLOGY_COSINE_THRESHOLD = 0.99
TOPOLOGY_FREQUENCY_THRESHOLD = 0.90
TOPOLOGY_RELATIVE_NORM_THRESHOLD = 0.05
CONFLICT_COSINE_THRESHOLD = -0.05
CONFLICT_FREQUENCY_THRESHOLD = 0.50
NORM_RATIO_THRESHOLD = 4.0


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BD config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BD identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_position_preserving_operator_"
        "k1bd_gradient_coupling_audit"
    ):
        raise ValueError("K1-BD experiment name drifted")
    expected_audit = {
        "samples_per_class_per_cipher": 2048,
        "pairs_per_sample": 4,
        "batch_size": BATCH_SIZE,
        "positive_rows_per_batch": ROWS_PER_CLASS_PER_BATCH,
        "negative_rows_per_batch": ROWS_PER_CLASS_PER_BATCH,
        "batch_triplets_per_replica": BATCH_TRIPLETS,
        "encoder_states": list(ENCODER_STATES),
        "operator_conditions": list(OPERATOR_CONDITIONS),
        "parameter_groups": list(PARAMETER_GROUPS),
        "loss": "mse",
        "optimizer_steps": 0,
        "data_generation": False,
        "execution": "local_audit",
    }
    if config.get("audit") != expected_audit:
        raise ValueError("K1-BD audit protocol drifted")
    expected_gates = {
        "topology_gradient_median_cosine_min": TOPOLOGY_COSINE_THRESHOLD,
        "topology_gradient_high_cosine_frequency_min": (
            TOPOLOGY_FREQUENCY_THRESHOLD
        ),
        "topology_gradient_relative_norm_delta_max": (
            TOPOLOGY_RELATIVE_NORM_THRESHOLD
        ),
        "systematic_conflict_median_cosine_max": CONFLICT_COSINE_THRESHOLD,
        "systematic_conflict_negative_frequency_min": (
            CONFLICT_FREQUENCY_THRESHOLD
        ),
        "stable_gradient_norm_ratio_min": NORM_RATIO_THRESHOLD,
        "disconnected_group_gradient_max": 0.0,
        "require_same_signal_in_both_replicas": True,
        "optimizer_steps": 0,
        "remote_scale": "no",
    }
    if config.get("gates") != expected_gates:
        raise ValueError("K1-BD decision gates drifted")
    expected_replicas = [
        {
            "replica": 0,
            "encoder_initialization_seed": 40,
            "dataset_seeds": {"uknit64": 3, "midori64": 6, "dialga128": 0},
        },
        {
            "replica": 1,
            "encoder_initialization_seed": 41,
            "dataset_seeds": {"uknit64": 4, "midori64": 7, "dialga128": 1},
        },
    ]
    if config.get("replicas") != expected_replicas:
        raise ValueError("K1-BD replica contract drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Mapping[tuple[str, int, str], DiskDifferentialDataset],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor | None]],
    Mapping[int, Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    dict[int, dict[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {name: source_root / name for name in source["digests"]}
    artifact_checks = {
        name: path.is_file() and file_sha256(path) == digest
        for name, digest in source["digests"].items()
        for path in (paths[name],)
    }
    source_config_path = project_root / str(source["training_config"])
    source_config = load_source_config(source_config_path)
    (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        _anchors,
        corrupted_structures,
        cross_operators,
        inherited_checks,
    ) = load_source_authority(
        source_config,
        project_root=project_root,
        device=device,
    )
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    source_results = _read_jsonl(paths["results.jsonl"])
    source_controls = _read_jsonl(paths["controls.jsonl"])
    manifest = _read_json(paths["checkpoint_manifest.json"])
    encoder_checkpoints = _load_encoder_checkpoints(
        paths=paths,
        manifest=manifest,
        device=device,
    )
    checks = {
        "source_training_config_digest_exact": (
            file_sha256(source_config_path) == source["training_config_sha256"]
        ),
        "all_eight_source_artifact_digests_exact": all(artifact_checks.values())
        and len(artifact_checks) == 8,
        "source_gate_is_interpretable_hold": (
            gate.get("status") == "hold"
            and gate.get("decision")
            == (
                "innovation1_uknit_family_k1bc_position_preserving_"
                "operator_training_not_supported"
            )
            and gate.get("topology_results", {})
            .get("same_summary_corrupted_operator", {})
            .get("passing_panels")
            == 0
            and gate.get("topology_results", {})
            .get("cross_cipher_operator", {})
            .get("passing_panels")
            == 0
            and not gate.get("failed_protocol_checks")
        ),
        "source_validation_passes_exact_counts": (
            validation.get("status") == "pass"
            and validation.get("training_rows") == 2
            and validation.get("evaluation_rows") == 48
            and not validation.get("errors")
        ),
        "source_rows_complete": len(source_results) == 2
        and len(source_controls) == 48,
        "two_encoder_checkpoints_strictly_loaded": (
            set(encoder_checkpoints) == set(REPLICAS)
            and all(
                row["strict_encoder_state_load"]
                for row in encoder_checkpoints.values()
            )
        ),
        "eighteen_datasets_rebound": len(datasets) == 18,
        **{f"k1bc_{name}": bool(value) for name, value in inherited_checks.items()},
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
        encoder_checkpoints,
        checks,
    )


def audit_gradients(
    *,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    structures: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, torch.Tensor | None]],
    source_checkpoints: Mapping[int, Mapping[str, Any]],
    corrupted_structures: Mapping[str, Any],
    cross_operators: Mapping[str, Any],
    encoder_checkpoints: Mapping[int, Mapping[str, Any]],
    output_root: Path,
    device: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
]:
    norm_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    state_checks: dict[str, bool] = {}
    operators = {
        "correct_operator": structures,
        "same_summary_corrupted_operator": corrupted_structures,
        "cross_cipher_operator": cross_operators,
    }
    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        probe = build_probe(
            runtime_config=runtime_config,
            structures=structures,
            checkpoint=source_checkpoints[replica],
            initialization_seed=int(replica_config["encoder_initialization_seed"]),
            model_config=load_source_config()["model"],
            device=device,
        )
        initial_state = {
            name: value.detach().clone()
            for name, value in probe.operator_encoder.state_dict().items()
        }
        if (
            tensor_mapping_sha256(initial_state)
            != encoder_checkpoints[replica]["initial_encoder_state_sha256"]
        ):
            raise ValueError(f"K1-BD initial encoder hash drifted for replica {replica}")
        selected_state = encoder_checkpoints[replica]["encoder_state_dict"]
        state_dicts = {
            "initial_encoder": initial_state,
            "selected_encoder": selected_state,
        }
        batches = {
            cipher: make_stratified_batches(
                datasets[
                    (
                        cipher,
                        int(replica_config["dataset_seeds"][cipher]),
                        "train_seen",
                    )
                ],
                seed=81_000 + replica * 1_000 + list(EXPECTED_CIPHERS).index(cipher),
            )
            for cipher in EXPECTED_CIPHERS
        }
        for encoder_state, state_dict in state_dicts.items():
            incompatible = probe.operator_encoder.load_state_dict(
                state_dict, strict=True
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError("K1-BD strict encoder state load drifted")
            probe.eval()
            state_before = tensor_mapping_sha256(probe.state_dict())
            for batch_index in range(BATCH_TRIPLETS):
                connected_vectors: dict[tuple[str, str], torch.Tensor] = {}
                connected_norms: dict[tuple[str, str], float] = {}
                losses: dict[tuple[str, str], float] = {}
                for cipher in EXPECTED_CIPHERS:
                    dataset_seed = int(replica_config["dataset_seeds"][cipher])
                    dataset = datasets[(cipher, dataset_seed, "train_seen")]
                    indices = batches[cipher][batch_index]
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
                    summary = summaries[cipher]["correct_descriptor"]
                    if summary is None:
                        raise ValueError("K1-BD correct gate summary is missing")
                    for condition in OPERATOR_CONDITIONS:
                        metrics, connected, loss, output_sha256 = (
                            measure_gradient_groups(
                                probe=probe,
                                features=features,
                                labels=labels,
                                runtime_structure=structures[cipher],
                                operator_structure=operators[condition][cipher],
                                summary=summary,
                            )
                        )
                        connected_vectors[(cipher, condition)] = connected
                        connected_norms[(cipher, condition)] = float(
                            torch.linalg.vector_norm(connected)
                        )
                        losses[(cipher, condition)] = loss
                        for group, group_metrics in metrics.items():
                            norm_rows.append(
                                {
                                    "run_id": RUN_ID,
                                    "replica": replica,
                                    "encoder_state": encoder_state,
                                    "cipher_key": cipher,
                                    "condition": condition,
                                    "batch_index": batch_index,
                                    "parameter_group": group,
                                    "loss": loss,
                                    "gradient_norm": group_metrics["gradient_norm"],
                                    "nonzero_elements": group_metrics[
                                        "nonzero_elements"
                                    ],
                                    "parameter_elements": group_metrics[
                                        "parameter_elements"
                                    ],
                                    "nonzero_fraction": group_metrics[
                                        "nonzero_fraction"
                                    ],
                                    "output_sha256": output_sha256,
                                    "optimizer_steps": 0,
                                }
                            )
                for cipher in EXPECTED_CIPHERS:
                    correct = connected_vectors[(cipher, "correct_operator")]
                    correct_norm = connected_norms[(cipher, "correct_operator")]
                    for condition in WRONG_OPERATOR_CONDITIONS:
                        wrong = connected_vectors[(cipher, condition)]
                        wrong_norm = connected_norms[(cipher, condition)]
                        topology_rows.append(
                            {
                                "run_id": RUN_ID,
                                "replica": replica,
                                "encoder_state": encoder_state,
                                "cipher_key": cipher,
                                "wrong_condition": condition,
                                "batch_index": batch_index,
                                "correct_loss": losses[
                                    (cipher, "correct_operator")
                                ],
                                "wrong_loss": losses[(cipher, condition)],
                                "cosine": gradient_cosine(correct, wrong),
                                "relative_norm_difference": _relative_difference(
                                    correct_norm, wrong_norm
                                ),
                                "optimizer_steps": 0,
                            }
                        )
                for left_index, left in enumerate(EXPECTED_CIPHERS):
                    for right in EXPECTED_CIPHERS[left_index + 1 :]:
                        cross_rows.append(
                            {
                                "run_id": RUN_ID,
                                "replica": replica,
                                "encoder_state": encoder_state,
                                "batch_index": batch_index,
                                "cipher_pair": f"{left}__{right}",
                                "left_cipher": left,
                                "right_cipher": right,
                                "cosine": gradient_cosine(
                                    connected_vectors[(left, "correct_operator")],
                                    connected_vectors[(right, "correct_operator")],
                                ),
                                "optimizer_steps": 0,
                            }
                        )
            state_after = tensor_mapping_sha256(probe.state_dict())
            state_checks[f"replica{replica}_{encoder_state}_state_immutable"] = (
                state_before == state_after
            )
            state_checks[f"replica{replica}_{encoder_state}_grads_none"] = all(
                parameter.grad is None for parameter in probe.parameters()
            )
            _append_progress(
                output_root / "progress.jsonl",
                "gradient_state_done",
                replica=replica,
                encoder_state=encoder_state,
                batch_triplets=BATCH_TRIPLETS,
            )
    return norm_rows, topology_rows, cross_rows, state_checks


def measure_gradient_groups(
    *,
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    runtime_structure: Any,
    operator_structure: Any,
    summary: torch.Tensor,
) -> tuple[dict[str, dict[str, float | int]], torch.Tensor, float, str]:
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
    gradient_by_name = {
        name: gradient for (name, _parameter), gradient in zip(
            named_parameters, gradients, strict=True
        )
    }
    metrics: dict[str, dict[str, float | int]] = {}
    vectors: dict[str, torch.Tensor] = {}
    for group in PARAMETER_GROUPS:
        selected = [
            (name, parameter)
            for name, parameter in named_parameters
            if _parameter_in_group(name, group)
        ]
        if not selected:
            raise ValueError(f"K1-BD empty parameter group: {group}")
        chunks = []
        nonzero = 0
        elements = 0
        for name, parameter in selected:
            gradient = gradient_by_name[name]
            chunk = (
                torch.zeros_like(parameter)
                if gradient is None
                else gradient.detach()
            )
            chunks.append(chunk.reshape(-1))
            nonzero += int(torch.count_nonzero(chunk))
            elements += parameter.numel()
        vector = torch.cat(chunks)
        vectors[group] = vector
        metrics[group] = {
            "gradient_norm": float(torch.linalg.vector_norm(vector)),
            "nonzero_elements": nonzero,
            "parameter_elements": elements,
            "nonzero_fraction": nonzero / elements,
        }
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return (
        metrics,
        vectors["connected_all"],
        float(loss.detach().cpu()),
        _array_sha256(probabilities),
    )


def audit_interventions(
    *,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    structures: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, torch.Tensor | None]],
    source_checkpoints: Mapping[int, Mapping[str, Any]],
    corrupted_structures: Mapping[str, Any],
    cross_operators: Mapping[str, Any],
    encoder_checkpoints: Mapping[int, Mapping[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    state_checks: dict[str, bool] = {}
    wrong_operators = {
        "same_summary_corrupted_operator": corrupted_structures,
        "cross_cipher_operator": cross_operators,
    }
    source_model_config = load_source_config()["model"]
    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        probe = build_probe(
            runtime_config=runtime_config,
            structures=structures,
            checkpoint=source_checkpoints[replica],
            initialization_seed=int(replica_config["encoder_initialization_seed"]),
            model_config=source_model_config,
            device=device,
        )
        initial_state = {
            name: value.detach().clone()
            for name, value in probe.operator_encoder.state_dict().items()
        }
        state_dicts = {
            "initial_encoder": initial_state,
            "selected_encoder": encoder_checkpoints[replica]["encoder_state_dict"],
        }
        for encoder_state, state_dict in state_dicts.items():
            probe.operator_encoder.load_state_dict(state_dict, strict=True)
            probe.eval()
            state_before = tensor_mapping_sha256(probe.state_dict())
            for cipher in EXPECTED_CIPHERS:
                seed = int(replica_config["dataset_seeds"][cipher])
                summary = summaries[cipher]["correct_descriptor"]
                if summary is None:
                    raise ValueError("K1-BD correct gate summary is missing")
                for split in FRESH_SPLITS:
                    dataset = datasets[(cipher, seed, split)]
                    accumulators = {
                        "correct_contribution": _new_stats(),
                        "correct_vs_disabled_logit": _new_stats(),
                        "correct_vs_disabled_probability": _new_stats(),
                        **{
                            f"{condition}_{metric}": _new_stats()
                            for condition in WRONG_OPERATOR_CONDITIONS
                            for metric in ("contribution", "logit", "probability")
                        },
                    }
                    hashes = {
                        condition: hashlib.sha256()
                        for condition in (
                            "correct_operator",
                            *WRONG_OPERATOR_CONDITIONS,
                            "disabled_k1az",
                        )
                    }
                    with torch.no_grad():
                        for start in range(0, len(dataset.labels), BATCH_SIZE):
                            stop = min(start + BATCH_SIZE, len(dataset.labels))
                            features = torch.as_tensor(
                                np.array(dataset.features[start:stop], copy=True),
                                dtype=torch.float32,
                                device=device,
                            )
                            runtime_pairs = features.reshape(
                                features.shape[0],
                                -1,
                                2,
                                structures[cipher].block_bits,
                            ).flip(-1)
                            correct_modulation = (
                                probe.spec.modulation_scale
                                * torch.tanh(
                                    probe.operator_encoder.sample_modulation(
                                        runtime_pairs,
                                        structures[cipher],
                                        structures[cipher],
                                    )
                                )
                            )
                            correct_logits = probe.logits_with_operator(
                                features,
                                structures[cipher],
                                structures[cipher],
                                gate_summary=summary,
                            )
                            disabled_logits = probe.logits_with_operator(
                                features,
                                structures[cipher],
                                structures[cipher],
                                gate_summary=summary,
                                enabled=False,
                            )
                            correct_probabilities = torch.sigmoid(correct_logits)
                            disabled_probabilities = torch.sigmoid(disabled_logits)
                            _update_stats(
                                accumulators["correct_contribution"],
                                correct_modulation,
                            )
                            _update_stats(
                                accumulators["correct_vs_disabled_logit"],
                                correct_logits - disabled_logits,
                            )
                            _update_stats(
                                accumulators["correct_vs_disabled_probability"],
                                correct_probabilities - disabled_probabilities,
                            )
                            _update_hash(
                                hashes["correct_operator"], correct_probabilities
                            )
                            _update_hash(hashes["disabled_k1az"], disabled_probabilities)
                            for condition, operator_map in wrong_operators.items():
                                wrong_modulation = (
                                    probe.spec.modulation_scale
                                    * torch.tanh(
                                        probe.operator_encoder.sample_modulation(
                                            runtime_pairs,
                                            structures[cipher],
                                            operator_map[cipher],
                                        )
                                    )
                                )
                                wrong_logits = probe.logits_with_operator(
                                    features,
                                    structures[cipher],
                                    operator_map[cipher],
                                    gate_summary=summary,
                                )
                                wrong_probabilities = torch.sigmoid(wrong_logits)
                                _update_stats(
                                    accumulators[f"{condition}_contribution"],
                                    correct_modulation - wrong_modulation,
                                )
                                _update_stats(
                                    accumulators[f"{condition}_logit"],
                                    correct_logits - wrong_logits,
                                )
                                _update_stats(
                                    accumulators[f"{condition}_probability"],
                                    correct_probabilities - wrong_probabilities,
                                )
                                _update_hash(hashes[condition], wrong_probabilities)
                    row: dict[str, Any] = {
                        "run_id": RUN_ID,
                        "replica": replica,
                        "encoder_state": encoder_state,
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "rows": len(dataset.labels),
                        "optimizer_steps": 0,
                    }
                    for name, stats in accumulators.items():
                        finalized = _finalize_stats(stats)
                        row[f"{name}_rms"] = finalized["rms"]
                        row[f"{name}_max_abs"] = finalized["max_abs"]
                    row["probability_sha256"] = {
                        condition: digest.hexdigest()
                        for condition, digest in hashes.items()
                    }
                    rows.append(row)
            state_after = tensor_mapping_sha256(probe.state_dict())
            state_checks[
                f"replica{replica}_{encoder_state}_intervention_state_immutable"
            ] = state_before == state_after
    return rows, state_checks


def aggregate_results(
    norm_rows: Sequence[Mapping[str, Any]],
    topology_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    norm_groups: dict[tuple[str, int, str, str, str], list[Mapping[str, Any]]] = {}
    for row in norm_rows:
        key = (
            str(row["encoder_state"]),
            int(row["replica"]),
            str(row["condition"]),
            str(row["cipher_key"]),
            str(row["parameter_group"]),
        )
        norm_groups.setdefault(key, []).append(row)
    for key, rows in sorted(norm_groups.items()):
        encoder_state, replica, condition, cipher, group = key
        results.append(
            {
                "run_id": RUN_ID,
                "metric_type": "gradient_group",
                "encoder_state": encoder_state,
                "replica": replica,
                "condition": condition,
                "cipher_key": cipher,
                "parameter_group": group,
                "batch_count": len(rows),
                "median_gradient_norm": float(
                    np.median([float(row["gradient_norm"]) for row in rows])
                ),
                "maximum_gradient_norm": max(
                    float(row["gradient_norm"]) for row in rows
                ),
                "median_nonzero_fraction": float(
                    np.median([float(row["nonzero_fraction"]) for row in rows])
                ),
                "minimum_nonzero_fraction": min(
                    float(row["nonzero_fraction"]) for row in rows
                ),
                "optimizer_steps": 0,
            }
        )
    topology_groups: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = {}
    for row in topology_rows:
        key = (
            str(row["encoder_state"]),
            int(row["replica"]),
            str(row["cipher_key"]),
            str(row["wrong_condition"]),
        )
        topology_groups.setdefault(key, []).append(row)
    for key, rows in sorted(topology_groups.items()):
        encoder_state, replica, cipher, wrong_condition = key
        cosines = [float(row["cosine"]) for row in rows]
        norm_differences = [float(row["relative_norm_difference"]) for row in rows]
        results.append(
            {
                "run_id": RUN_ID,
                "metric_type": "topology_gradient_similarity",
                "encoder_state": encoder_state,
                "replica": replica,
                "cipher_key": cipher,
                "wrong_condition": wrong_condition,
                "batch_count": len(rows),
                "median_cosine": float(np.median(cosines)),
                "high_cosine_frequency": float(
                    np.mean(np.asarray(cosines) >= TOPOLOGY_COSINE_THRESHOLD)
                ),
                "median_relative_norm_difference": float(
                    np.median(norm_differences)
                ),
                "optimizer_steps": 0,
            }
        )
    cross_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in cross_rows:
        key = (
            str(row["encoder_state"]),
            int(row["replica"]),
            str(row["cipher_pair"]),
        )
        cross_groups.setdefault(key, []).append(row)
    for key, rows in sorted(cross_groups.items()):
        encoder_state, replica, cipher_pair = key
        cosines = [float(row["cosine"]) for row in rows]
        results.append(
            {
                "run_id": RUN_ID,
                "metric_type": "cross_cipher_gradient_similarity",
                "encoder_state": encoder_state,
                "replica": replica,
                "cipher_pair": cipher_pair,
                "batch_count": len(rows),
                "median_cosine": float(np.median(cosines)),
                "negative_cosine_frequency": float(
                    np.mean(np.asarray(cosines) < 0.0)
                ),
                "optimizer_steps": 0,
            }
        )
    results.extend(
        {"metric_type": "fresh_intervention", **dict(row)}
        for row in intervention_rows
    )
    return results


def adjudicate(
    *,
    source_checks: Mapping[str, bool],
    gradient_state_checks: Mapping[str, bool],
    intervention_state_checks: Mapping[str, bool],
    norm_rows: Sequence[Mapping[str, Any]],
    topology_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result_groups = [row for row in results if row.get("metric_type") == "gradient_group"]
    topology_summaries = [
        row
        for row in results
        if row.get("metric_type") == "topology_gradient_similarity"
    ]
    cross_summaries = [
        row
        for row in results
        if row.get("metric_type") == "cross_cipher_gradient_similarity"
    ]
    disconnected_rows = [
        row
        for row in norm_rows
        if row.get("parameter_group") == "structure_projection"
    ]
    connected_summaries = [
        row
        for row in result_groups
        if row.get("parameter_group") != "structure_projection"
    ]
    protocol_checks = {
        "all_source_bindings_exact": all(source_checks.values()),
        "all_states_immutable_and_grad_buffers_empty": all(
            (*gradient_state_checks.values(), *intervention_state_checks.values())
        )
        and len(gradient_state_checks) == 8
        and len(intervention_state_checks) == 4,
        "gradient_norm_rows_complete": len(norm_rows) == EXPECTED_NORM_ROWS,
        "topology_gradient_rows_complete": (
            len(topology_rows) == EXPECTED_TOPOLOGY_PAIR_ROWS
        ),
        "cross_cipher_gradient_rows_complete": (
            len(cross_rows) == EXPECTED_CROSS_PAIR_ROWS
        ),
        "fresh_intervention_rows_complete": (
            len(intervention_rows) == EXPECTED_INTERVENTION_ROWS
        ),
        "all_rows_zero_optimizer_steps": all(
            int(row.get("optimizer_steps", -1)) == 0
            for row in (*norm_rows, *topology_rows, *cross_rows, *intervention_rows)
        ),
        "all_connected_cosines_defined_and_finite": all(
            row.get("cosine") is not None and math.isfinite(float(row["cosine"]))
            for row in (*topology_rows, *cross_rows)
        ),
        "all_result_metrics_finite": _all_numeric_finite(results),
    }
    disconnected_supported = bool(disconnected_rows) and all(
        float(row["gradient_norm"]) == 0.0
        and int(row["nonzero_elements"]) == 0
        for row in disconnected_rows
    ) and all(
        float(row["maximum_gradient_norm"]) > 0.0
        and float(row["median_nonzero_fraction"]) > 0.0
        for row in connected_summaries
    )
    topology_passes = {
        _topology_key(row): (
            float(row["median_cosine"]) >= TOPOLOGY_COSINE_THRESHOLD
            and float(row["high_cosine_frequency"])
            >= TOPOLOGY_FREQUENCY_THRESHOLD
            and float(row["median_relative_norm_difference"])
            <= TOPOLOGY_RELATIVE_NORM_THRESHOLD
        )
        for row in topology_summaries
    }
    selected_topology_keys = {
        ("selected_encoder", replica, cipher, condition)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for condition in WRONG_OPERATOR_CONDITIONS
    }
    topology_indistinguishable = selected_topology_keys == {
        key for key, passed in topology_passes.items() if passed and key[0] == "selected_encoder"
    }
    initial_topology_pass_count = sum(
        passed for key, passed in topology_passes.items() if key[0] == "initial_encoder"
    )
    cross_by_key = {
        (str(row["encoder_state"]), int(row["replica"]), str(row["cipher_pair"])): row
        for row in cross_summaries
    }
    cipher_pairs = [
        f"{left}__{right}"
        for left_index, left in enumerate(EXPECTED_CIPHERS)
        for right in EXPECTED_CIPHERS[left_index + 1 :]
    ]
    conflict_by_replica: dict[str, dict[str, bool]] = {}
    for replica in REPLICAS:
        conflict_by_replica[str(replica)] = {}
        for pair in cipher_pairs:
            row = cross_by_key[("selected_encoder", replica, pair)]
            conflict_by_replica[str(replica)][pair] = (
                float(row["median_cosine"]) <= CONFLICT_COSINE_THRESHOLD
                and float(row["negative_cosine_frequency"])
                >= CONFLICT_FREQUENCY_THRESHOLD
            )
    stable_conflict_pairs = [
        pair
        for pair in cipher_pairs
        if all(conflict_by_replica[str(replica)][pair] for replica in REPLICAS)
    ]
    group_by_key = {
        (
            str(row["encoder_state"]),
            int(row["replica"]),
            str(row["condition"]),
            str(row["cipher_key"]),
            str(row["parameter_group"]),
        ): row
        for row in result_groups
    }
    norm_ratios: dict[str, dict[str, Any]] = {}
    for replica in REPLICAS:
        medians = {
            cipher: float(
                group_by_key[
                    (
                        "selected_encoder",
                        replica,
                        "correct_operator",
                        cipher,
                        "connected_all",
                    )
                ]["median_gradient_norm"]
            )
            for cipher in EXPECTED_CIPHERS
        }
        dominant = max(medians, key=medians.get)
        minimum = min(medians.values())
        norm_ratios[str(replica)] = {
            "dominant_cipher": dominant,
            "max_to_min_median_norm_ratio": (
                math.inf if minimum == 0.0 else medians[dominant] / minimum
            ),
            "median_norms": medians,
        }
    stable_norm_imbalance = (
        len({row["dominant_cipher"] for row in norm_ratios.values()}) == 1
        and all(
            float(row["max_to_min_median_norm_ratio"]) >= NORM_RATIO_THRESHOLD
            for row in norm_ratios.values()
        )
    )
    failed_protocol_checks = [
        name for name, passed in protocol_checks.items() if not passed
    ]
    if failed_protocol_checks:
        status = "invalid"
        decision = "innovation1_uknit_family_k1bd_protocol_invalid"
        next_action = "Repair only the failed binding, count, state or finite-metric invariant."
    elif topology_indistinguishable:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1bd_standard_classification_"
            "objective_topology_indistinguishable_supported"
        )
        next_action = (
            "Preregister one same-data correct-versus-wrong topology ranking "
            "auxiliary loss with the K1-BC classifier and benchmark frozen; "
            "exclude the disconnected readiness-only projection and do not scale."
        )
    elif stable_conflict_pairs:
        status = "pass"
        decision = "innovation1_uknit_family_k1bd_shared_gradient_conflict_supported"
        next_action = (
            "Compare one minimal shared-gradient combination rule against unchanged "
            "K1-BC; keep model, data, pairs, epochs, seeds and controls fixed."
        )
    elif stable_norm_imbalance:
        status = "pass"
        decision = "innovation1_uknit_family_k1bd_gradient_norm_imbalance_supported"
        next_action = (
            "Compare one fixed gradient-normalization rule against unchanged K1-BC; "
            "do not use PCGrad, MoE, experts or scale."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1bd_modulation_coupling_redesign_required"
        next_action = (
            "Redesign the topology-to-edge coupling to increase correct-versus-wrong "
            "downstream intervention before another training run; do not scale."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol_checks,
        "disconnected_structure_projection_supported": disconnected_supported,
        "disconnected_parameter_count": 12_672,
        "total_declared_trainable_parameter_count": 41_088,
        "disconnected_parameter_fraction": 12_672 / 41_088,
        "topology_gradient_indistinguishable": topology_indistinguishable,
        "topology_passes": {
            "|".join(map(str, key)): passed
            for key, passed in sorted(topology_passes.items())
        },
        "initial_topology_passing_panels": initial_topology_pass_count,
        "initial_topology_expected_panels": 12,
        "selected_topology_passing_panels": sum(
            topology_passes.get(key, False) for key in selected_topology_keys
        ),
        "selected_topology_expected_panels": 12,
        "systematic_conflict_by_replica": conflict_by_replica,
        "stable_conflict_pairs": stable_conflict_pairs,
        "stable_gradient_norm_imbalance": stable_norm_imbalance,
        "norm_ratios": norm_ratios,
        "remote_scale": "no",
        "blocked_actions": [
            "16-pair expansion",
            "larger samples, epochs, width, seeds, or remote execution",
            "cipher IDs, per-cipher heads, adapters, routers, MoE, or experts",
            "PCGrad or loss weighting inside this audit",
        ],
        "next_action": next_action,
        "claim_scope": (
            "Zero-update local audit of K1-BC initial and selected encoders using "
            "2048/class/cipher four-pair caches; not training, an attack, arbitrary-"
            "SPN proof, unseen-cipher transfer, formal scale or SOTA evidence."
        ),
    }


def run_audit(
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
        encoder_checkpoints,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-BD source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "audit": dict(config["audit"]),
    }
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {
                key: value
                for key, value in encoder_checkpoints[replica].items()
                if key != "encoder_state_dict"
            }
            for replica in REPLICAS
        ],
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    norm_rows, topology_rows, cross_rows, gradient_state_checks = audit_gradients(
        config=config,
        runtime_config=runtime_config,
        datasets=datasets,
        structures=structures,
        summaries=summaries,
        source_checkpoints=source_checkpoints,
        corrupted_structures=corrupted_structures,
        cross_operators=cross_operators,
        encoder_checkpoints=encoder_checkpoints,
        output_root=output_root,
        device=device,
    )
    intervention_rows, intervention_state_checks = audit_interventions(
        config=config,
        runtime_config=runtime_config,
        datasets=datasets,
        structures=structures,
        summaries=summaries,
        source_checkpoints=source_checkpoints,
        corrupted_structures=corrupted_structures,
        cross_operators=cross_operators,
        encoder_checkpoints=encoder_checkpoints,
        device=device,
    )
    results = aggregate_results(
        norm_rows, topology_rows, cross_rows, intervention_rows
    )
    gate = adjudicate(
        source_checks=source_checks,
        gradient_state_checks=gradient_state_checks,
        intervention_state_checks=intervention_state_checks,
        norm_rows=norm_rows,
        topology_rows=topology_rows,
        cross_rows=cross_rows,
        intervention_rows=intervention_rows,
        results=results,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "gradient_norm_rows": len(norm_rows),
        "expected_gradient_norm_rows": EXPECTED_NORM_ROWS,
        "topology_gradient_rows": len(topology_rows),
        "expected_topology_gradient_rows": EXPECTED_TOPOLOGY_PAIR_ROWS,
        "cross_cipher_gradient_rows": len(cross_rows),
        "expected_cross_cipher_gradient_rows": EXPECTED_CROSS_PAIR_ROWS,
        "intervention_rows": len(intervention_rows),
        "expected_intervention_rows": EXPECTED_INTERVENTION_ROWS,
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "topology_gradient_indistinguishable": gate[
            "topology_gradient_indistinguishable"
        ],
        "disconnected_structure_projection_supported": gate[
            "disconnected_structure_projection_supported"
        ],
        "stable_conflict_pairs": gate["stable_conflict_pairs"],
        "stable_gradient_norm_imbalance": gate[
            "stable_gradient_norm_imbalance"
        ],
        "norm_ratios": gate["norm_ratios"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "gradient_norms.jsonl", norm_rows)
    _write_jsonl(output_root / "topology_gradient_pairs.jsonl", topology_rows)
    _write_jsonl(output_root / "cross_cipher_gradient_pairs.jsonl", cross_rows)
    _write_jsonl(output_root / "interventions.jsonl", intervention_rows)
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
        "results": results,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _load_encoder_checkpoints(
    *,
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    device: str,
) -> dict[int, dict[str, Any]]:
    manifest_entries = {
        int(row["replica"]): row for row in manifest.get("entries", [])
    }
    checkpoints: dict[int, dict[str, Any]] = {}
    for replica in REPLICAS:
        name = f"checkpoints/replica{replica}_best.pt"
        path = paths[name]
        payload = torch.load(path, map_location=device, weights_only=False)
        state_dict = payload["encoder_state_dict"]
        state_sha256 = tensor_mapping_sha256(state_dict)
        entry = manifest_entries.get(replica, {})
        if (
            payload.get("run_id")
            != (
                "i1_uknit_family_position_preserving_operator_k1bc_"
                "2048_replica0_replica1_20260729"
            )
            or int(payload.get("replica", -1)) != replica
            or entry.get("sha256") != file_sha256(path)
            or entry.get("encoder_state_dict_sha256") != state_sha256
            or payload.get("encoder_state_dict_sha256") != state_sha256
        ):
            raise ValueError(f"K1-BD encoder checkpoint binding failed for replica {replica}")
        checkpoints[replica] = {
            "replica": replica,
            "path": str(path),
            "sha256": file_sha256(path),
            "best_epoch": int(payload["best_epoch"]),
            "initial_encoder_state_sha256": str(
                payload["initial_encoder_state_sha256"]
            ),
            "encoder_state_dict_sha256": state_sha256,
            "strict_encoder_state_load": True,
            "encoder_state_dict": state_dict,
        }
    return checkpoints


def _parameter_in_group(name: str, group: str) -> bool:
    prefix = name.split(".", 1)[0]
    if group == "connected_all":
        return prefix != "structure_projection"
    return prefix == group


def _relative_difference(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


def _topology_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["encoder_state"]),
        int(row["replica"]),
        str(row["cipher_key"]),
        str(row["wrong_condition"]),
    )


def _new_stats() -> dict[str, float | int]:
    return {"square_sum": 0.0, "count": 0, "max_abs": 0.0}


def _update_stats(stats: dict[str, float | int], values: torch.Tensor) -> None:
    detached = values.detach().to(torch.float64)
    stats["square_sum"] = float(stats["square_sum"]) + float(detached.square().sum())
    stats["count"] = int(stats["count"]) + detached.numel()
    stats["max_abs"] = max(
        float(stats["max_abs"]),
        float(detached.abs().max()) if detached.numel() else 0.0,
    )


def _finalize_stats(stats: Mapping[str, float | int]) -> dict[str, float]:
    count = int(stats["count"])
    if count <= 0:
        raise ValueError("K1-BD intervention accumulator is empty")
    return {
        "rms": math.sqrt(float(stats["square_sum"]) / count),
        "max_abs": float(stats["max_abs"]),
    }


def _update_hash(digest: Any, values: torch.Tensor) -> None:
    array = np.ascontiguousarray(values.detach().cpu().numpy())
    digest.update(array.tobytes(order="C"))


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _all_numeric_finite(rows: Sequence[Mapping[str, Any]]) -> bool:
    def finite(value: Any) -> bool:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, Mapping):
            return all(finite(item) for item in value.values())
        if isinstance(value, Sequence):
            return all(finite(item) for item in value)
        return True

    return all(finite(row) for row in rows)


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
        raise FileExistsError(f"K1-BD output root already exists: {path}")


__all__ = [
    "CONFIG_PATH",
    "RUN_ID",
    "adjudicate",
    "aggregate_results",
    "audit_gradients",
    "audit_interventions",
    "load_and_validate_config",
    "load_authority",
    "measure_gradient_groups",
    "run_audit",
]
