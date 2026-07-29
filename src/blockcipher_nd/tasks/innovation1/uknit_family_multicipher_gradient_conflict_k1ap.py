from __future__ import annotations

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
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    build_runtime_model,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao_training import (
    RUN_ID as SOURCE_RUN_ID,
    load_and_validate_config as load_source_config,
    load_sources,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_multicipher_gradient_conflict_k1ap_"
    "64batch_replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_multicipher_gradient_conflict_k1ap_"
    "64batch_replica0_replica1_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "0ac17ef02a5260e26fb85221ee1944a542bfcdba36b6d14a79caddb175bdbb16"
)
CONDITIONS = (
    "correct_runtime",
    "wrong_sbox_same_checkpoint",
    "transition_branch_off_same_checkpoint",
)
PARAMETER_GROUPS = ("all_trainable", "transition_semantic")
REPLICAS = (0, 1)
BATCH_SIZE = 64
ROWS_PER_CLASS_PER_BATCH = 32
BATCH_TRIPLETS = 64
EXPECTED_PAIR_ROWS = 2_304
EXPECTED_NORM_ROWS = 2_304
EXPECTED_SUMMARY_ROWS = 72
COSINE_THRESHOLD = -0.05
NEGATIVE_FREQUENCY_THRESHOLD = 0.50
NORM_RATIO_THRESHOLD = 4.0


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("K1-AP config must be a JSON object")
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AP config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-AP identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_multicipher_gradient_conflict_k1ap"
    ):
        raise ValueError("K1-AP experiment name drifted")
    if tuple(config.get("audit", {}).get("conditions", ())) != CONDITIONS:
        raise ValueError("K1-AP runtime conditions drifted")
    if tuple(config.get("audit", {}).get("parameter_groups", ())) != PARAMETER_GROUPS:
        raise ValueError("K1-AP parameter groups drifted")
    expected_audit = {
        "samples_per_class_per_cipher": 2048,
        "pairs_per_sample": 4,
        "batch_size": BATCH_SIZE,
        "positive_rows_per_batch": ROWS_PER_CLASS_PER_BATCH,
        "negative_rows_per_batch": ROWS_PER_CLASS_PER_BATCH,
        "batch_triplets_per_replica": BATCH_TRIPLETS,
        "conditions": list(CONDITIONS),
        "parameter_groups": list(PARAMETER_GROUPS),
        "loss": "mse",
        "optimizer_steps": 0,
        "data_generation": False,
        "execution": "local_audit",
    }
    if config.get("audit") != expected_audit:
        raise ValueError("K1-AP audit protocol drifted")
    expected_gates = {
        "systematic_conflict_median_cosine_max": COSINE_THRESHOLD,
        "systematic_conflict_negative_frequency_min": (
            NEGATIVE_FREQUENCY_THRESHOLD
        ),
        "stable_gradient_norm_ratio_min": NORM_RATIO_THRESHOLD,
        "require_same_signal_in_both_replicas": True,
        "optimizer_steps": 0,
        "remote_scale": "no",
    }
    if config.get("gates") != expected_gates:
        raise ValueError("K1-AP decision gates drifted")
    if [int(row["replica"]) for row in config.get("replicas", [])] != [0, 1]:
        raise ValueError("K1-AP replicas drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, int, str], DiskDifferentialDataset],
    dict[int, dict[str, Any]],
    dict[str, bool],
    list[dict[str, Any]],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {name: source_root / name for name in source["digests"]}
    artifact_digests_exact = {
        name: file_sha256(path) == digest
        for name, digest in source["digests"].items()
        for path in (paths[name],)
    }
    source_config_path = project_root / str(source["training_config"])
    source_config = load_source_config(source_config_path)
    readiness_config, dataset_rows, datasets, _anchors, source_checks = load_sources(
        source_config,
        project_root=project_root,
    )
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    checkpoint_manifest = _read_json(paths["checkpoint_manifest.json"])
    checkpoints = _load_checkpoints(
        paths=paths,
        checkpoint_manifest=checkpoint_manifest,
        device="cpu",
    )
    checks = {
        "source_training_config_digest_exact": (
            file_sha256(source_config_path) == source["training_config_sha256"]
        ),
        "all_six_source_artifact_digests_exact": all(
            artifact_digests_exact.values()
        ),
        "source_gate_is_interpretable_hold": (
            gate.get("run_id") == SOURCE_RUN_ID
            and gate.get("status") == "hold"
            and gate.get("wrong_sbox_margin_all_panels") is True
            and gate.get("retention_all_panels") is False
            and gate.get("branch_margin_all_panels") is False
            and not gate.get("failed_protocol_checks")
        ),
        "source_validation_passes_exact_counts": (
            validation.get("run_id") == SOURCE_RUN_ID
            and validation.get("status") == "pass"
            and validation.get("training_rows") == 2
            and validation.get("evaluation_rows") == 36
            and not validation.get("errors")
        ),
        "source_checkpoint_manifest_complete": (
            checkpoint_manifest.get("run_id") == SOURCE_RUN_ID
            and checkpoint_manifest.get("status") == "pass"
            and len(checkpoint_manifest.get("entries", [])) == 2
        ),
        "two_checkpoints_strictly_loaded": set(checkpoints) == set(REPLICAS)
        and all(checkpoint["strict_state_dict_load"] for checkpoint in checkpoints.values()),
        "eighteen_datasets_rebound": len(datasets) == 18,
        **{f"k1ao_{name}": value for name, value in source_checks.items()},
    }
    return readiness_config, datasets, checkpoints, checks, dataset_rows


def make_stratified_batches(
    dataset: DiskDifferentialDataset,
    *,
    seed: int,
) -> list[np.ndarray]:
    labels = np.asarray(dataset.labels).reshape(-1)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if len(positives) != 2048 or len(negatives) != 2048:
        raise ValueError("K1-AP requires exactly 2048 rows per class")
    rng = np.random.default_rng(seed)
    positives = rng.permutation(positives)
    negatives = rng.permutation(negatives)
    batches = []
    for batch_index in range(BATCH_TRIPLETS):
        start = batch_index * ROWS_PER_CLASS_PER_BATCH
        stop = (batch_index + 1) * ROWS_PER_CLASS_PER_BATCH
        indices = np.concatenate((positives[start:stop], negatives[start:stop]))
        batches.append(rng.permutation(indices))
    return batches


def audit_gradients(
    *,
    config: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
    checkpoints: Mapping[int, Mapping[str, Any]],
    output_root: Path,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness_config["ciphers"]
    }
    model_config = readiness_config["model"]
    pair_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    state_checks: dict[str, bool] = {}

    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        model = build_runtime_model(
            cipher_configs[EXPECTED_CIPHERS[0]], model_config
        ).to(device)
        model.load_state_dict(checkpoints[replica]["state_dict"], strict=True)
        model.eval()
        state_before = tensor_mapping_sha256(model.state_dict())
        correct_structures = {
            cipher_key: build_runtime_model(cipher_configs[cipher_key], model_config)
            .runtime_structure
            for cipher_key in EXPECTED_CIPHERS
        }
        wrong_structures = {
            cipher_key: build_runtime_model(
                cipher_configs[cipher_key], model_config, wrong_sbox=True
            ).runtime_structure
            for cipher_key in EXPECTED_CIPHERS
        }
        batches = {}
        for cipher_index, cipher_key in enumerate(EXPECTED_CIPHERS):
            dataset_seed = int(replica_config["dataset_seeds"][cipher_key])
            batches[cipher_key] = make_stratified_batches(
                datasets[(cipher_key, dataset_seed, "train_seen")],
                seed=70_000 + replica * 1_000 + cipher_index,
            )

        for condition in CONDITIONS:
            for batch_index in range(BATCH_TRIPLETS):
                gradients: dict[str, dict[str, torch.Tensor]] = {}
                losses: dict[str, float] = {}
                for cipher_key in EXPECTED_CIPHERS:
                    dataset_seed = int(replica_config["dataset_seeds"][cipher_key])
                    dataset = datasets[(cipher_key, dataset_seed, "train_seen")]
                    indices = batches[cipher_key][batch_index]
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
                    structure = (
                        wrong_structures[cipher_key]
                        if condition == "wrong_sbox_same_checkpoint"
                        else correct_structures[cipher_key]
                    )
                    group_vectors, loss = measure_gradient_vectors(
                        model=model,
                        features=features,
                        labels=labels,
                        structure=structure,
                        transition_branch_enabled=(
                            condition
                            != "transition_branch_off_same_checkpoint"
                        ),
                    )
                    gradients[cipher_key] = group_vectors
                    losses[cipher_key] = loss
                    for group, vector in group_vectors.items():
                        norm_rows.append(
                            {
                                "run_id": RUN_ID,
                                "replica": replica,
                                "condition": condition,
                                "parameter_group": group,
                                "batch_index": batch_index,
                                "cipher_key": cipher_key,
                                "loss": loss,
                                "gradient_norm": float(torch.linalg.vector_norm(vector)),
                                "optimizer_steps": 0,
                            }
                        )
                for left_index, left in enumerate(EXPECTED_CIPHERS):
                    for right in EXPECTED_CIPHERS[left_index + 1 :]:
                        for group in PARAMETER_GROUPS:
                            cosine = gradient_cosine(
                                gradients[left][group], gradients[right][group]
                            )
                            pair_rows.append(
                                {
                                    "run_id": RUN_ID,
                                    "replica": replica,
                                    "condition": condition,
                                    "parameter_group": group,
                                    "batch_index": batch_index,
                                    "cipher_pair": f"{left}__{right}",
                                    "left_cipher": left,
                                    "right_cipher": right,
                                    "left_loss": losses[left],
                                    "right_loss": losses[right],
                                    "cosine": cosine,
                                    "cosine_defined": cosine is not None,
                                    "optimizer_steps": 0,
                                }
                            )
            _append_progress(
                output_root / "progress.jsonl",
                "condition_done",
                replica=replica,
                condition=condition,
                batch_triplets=BATCH_TRIPLETS,
            )
        state_after = tensor_mapping_sha256(model.state_dict())
        state_checks[f"replica{replica}_state_immutable"] = (
            state_before == state_after == checkpoints[replica]["state_dict_sha256"]
        )
        state_checks[f"replica{replica}_all_parameter_grads_none"] = all(
            parameter.grad is None for parameter in model.parameters()
        )
    return pair_rows, norm_rows, state_checks


def measure_gradient_vectors(
    *,
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    structure: Any,
    transition_branch_enabled: bool,
) -> tuple[dict[str, torch.Tensor], float]:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    logits = model.logits_with_runtime(
        features,
        structure,
        apply_sboxes=True,
        transition_branch_enabled=transition_branch_enabled,
    )
    loss = nn.functional.mse_loss(torch.sigmoid(logits), labels)
    gradients = torch.autograd.grad(
        loss,
        [parameter for _name, parameter in named_parameters],
        allow_unused=True,
    )
    vectors = {
        "all_trainable": _flatten_gradients(named_parameters, gradients),
        "transition_semantic": _flatten_gradients(
            named_parameters,
            gradients,
            names={
                name
                for name, _parameter in named_parameters
                if _is_transition_semantic_parameter(name)
            },
        ),
    }
    return vectors, float(loss.detach().cpu())


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = left_norm * right_norm
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def aggregate_gradients(
    pair_rows: Sequence[Mapping[str, Any]],
    norm_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    pair_groups: dict[tuple[int, str, str, str], list[float]] = {}
    for row in pair_rows:
        key = (
            int(row["replica"]),
            str(row["condition"]),
            str(row["parameter_group"]),
            str(row["cipher_pair"]),
        )
        if row.get("cosine") is not None:
            pair_groups.setdefault(key, []).append(float(row["cosine"]))
        else:
            pair_groups.setdefault(key, [])
    for key, values in sorted(pair_groups.items()):
        replica, condition, parameter_group, cipher_pair = key
        summaries.append(
            {
                "run_id": RUN_ID,
                "metric_type": "pairwise_cosine",
                "replica": replica,
                "condition": condition,
                "parameter_group": parameter_group,
                "cipher_pair": cipher_pair,
                "batch_count": BATCH_TRIPLETS,
                "defined_count": len(values),
                "median_cosine": (
                    float(np.median(values)) if values else None
                ),
                "negative_cosine_frequency": (
                    float(np.mean(np.asarray(values) < 0.0)) if values else None
                ),
                "optimizer_steps": 0,
            }
        )
    norm_groups: dict[tuple[int, str, str, str], list[float]] = {}
    for row in norm_rows:
        key = (
            int(row["replica"]),
            str(row["condition"]),
            str(row["parameter_group"]),
            str(row["cipher_key"]),
        )
        norm_groups.setdefault(key, []).append(float(row["gradient_norm"]))
    for key, values in sorted(norm_groups.items()):
        replica, condition, parameter_group, cipher_key = key
        summaries.append(
            {
                "run_id": RUN_ID,
                "metric_type": "gradient_norm",
                "replica": replica,
                "condition": condition,
                "parameter_group": parameter_group,
                "cipher_key": cipher_key,
                "batch_count": len(values),
                "median_gradient_norm": float(np.median(values)),
                "optimizer_steps": 0,
            }
        )
    return summaries


def adjudicate(
    *,
    source_checks: Mapping[str, bool],
    state_checks: Mapping[str, bool],
    pair_rows: Sequence[Mapping[str, Any]],
    norm_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocol_checks = {
        "all_source_bindings_exact": all(source_checks.values()),
        "two_states_immutable_and_no_grad_buffers": all(state_checks.values())
        and len(state_checks) == 4,
        "pair_rows_complete": len(pair_rows) == EXPECTED_PAIR_ROWS,
        "norm_rows_complete": len(norm_rows) == EXPECTED_NORM_ROWS,
        "summary_rows_complete": len(summaries) == EXPECTED_SUMMARY_ROWS,
        "all_rows_have_zero_optimizer_steps": all(
            int(row.get("optimizer_steps", -1)) == 0
            for row in (*pair_rows, *norm_rows, *summaries)
        ),
        "all_correct_all_trainable_cosines_defined": all(
            row.get("cosine") is not None
            for row in pair_rows
            if row.get("condition") == "correct_runtime"
            and row.get("parameter_group") == "all_trainable"
        ),
    }
    pair_summary = {
        (
            int(row["replica"]),
            str(row["condition"]),
            str(row["parameter_group"]),
            str(row["cipher_pair"]),
        ): row
        for row in summaries
        if row.get("metric_type") == "pairwise_cosine"
    }
    norm_summary = {
        (
            int(row["replica"]),
            str(row["condition"]),
            str(row["parameter_group"]),
            str(row["cipher_key"]),
        ): row
        for row in summaries
        if row.get("metric_type") == "gradient_norm"
    }
    cipher_pairs = [
        f"{left}__{right}"
        for left_index, left in enumerate(EXPECTED_CIPHERS)
        for right in EXPECTED_CIPHERS[left_index + 1 :]
    ]
    systematic_by_replica: dict[str, dict[str, bool]] = {}
    for replica in REPLICAS:
        systematic_by_replica[str(replica)] = {}
        for cipher_pair in cipher_pairs:
            row = pair_summary[(replica, "correct_runtime", "all_trainable", cipher_pair)]
            systematic_by_replica[str(replica)][cipher_pair] = (
                float(row["median_cosine"]) <= COSINE_THRESHOLD
                and float(row["negative_cosine_frequency"])
                >= NEGATIVE_FREQUENCY_THRESHOLD
            )
    stable_conflict_pairs = [
        cipher_pair
        for cipher_pair in cipher_pairs
        if all(systematic_by_replica[str(replica)][cipher_pair] for replica in REPLICAS)
    ]
    norm_ratios: dict[str, dict[str, Any]] = {}
    for replica in REPLICAS:
        medians = {
            cipher_key: float(
                norm_summary[
                    (replica, "correct_runtime", "all_trainable", cipher_key)
                ]["median_gradient_norm"]
            )
            for cipher_key in EXPECTED_CIPHERS
        }
        dominant = max(medians, key=medians.get)
        minimum = min(medians.values())
        ratio = math.inf if minimum == 0.0 else medians[dominant] / minimum
        norm_ratios[str(replica)] = {
            "dominant_cipher": dominant,
            "max_to_min_median_norm_ratio": ratio,
            "median_norms": medians,
        }
    same_dominant = len(
        {row["dominant_cipher"] for row in norm_ratios.values()}
    ) == 1
    stable_norm_imbalance = same_dominant and all(
        float(row["max_to_min_median_norm_ratio"]) >= NORM_RATIO_THRESHOLD
        for row in norm_ratios.values()
    )
    failed_protocol_checks = [
        name for name, passed in protocol_checks.items() if not passed
    ]
    if failed_protocol_checks:
        status = "invalid"
        decision = "innovation1_uknit_family_k1ap_protocol_invalid"
        next_action = "Repair the exact failed binding or row/state invariant and rerun."
    elif stable_conflict_pairs:
        status = "pass"
        decision = "innovation1_uknit_family_k1ap_systematic_gradient_conflict_supported"
        next_action = (
            "Compare one minimal PCGrad shared-update rule against unchanged K1-AO; "
            "do not change data, pairs, epochs, model, seeds, controls, or scale."
        )
    elif stable_norm_imbalance:
        status = "pass"
        decision = "innovation1_uknit_family_k1ap_stable_gradient_norm_imbalance_supported"
        next_action = (
            "Compare one minimal gradient-normalization rule against unchanged K1-AO; "
            "do not use PCGrad, experts, larger data, or more pairs."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1ap_optimizer_conflict_not_supported"
        next_action = (
            "Return to transition representation design, prioritizing Midori branch "
            "identifiability; do not tune optimizer, pairs, samples, epochs, or width."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol_checks,
        "systematic_conflict_by_replica": systematic_by_replica,
        "stable_conflict_pairs": stable_conflict_pairs,
        "stable_gradient_norm_imbalance": stable_norm_imbalance,
        "norm_ratios": norm_ratios,
        "remote_scale": "no",
        "blocked_actions": [
            "16-pair expansion",
            "larger samples, epochs, width, or remote execution",
            "MoE, cipher IDs, per-cipher heads, adapters, or experts",
        ],
        "next_action": next_action,
        "claim_scope": (
            "Zero-update local gradient audit of two K1-AO checkpoints; not a "
            "training result, attack, arbitrary-SPN proof, or SOTA evidence."
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
    readiness_config, datasets, checkpoints, source_checks, dataset_rows = (
        load_authority(config, project_root=project_root)
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-AP source binding failed: {source_checks}")
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
                for key, value in checkpoints[replica].items()
                if key != "state_dict"
            }
            for replica in REPLICAS
        ],
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)

    pair_rows, norm_rows, state_checks = audit_gradients(
        config=config,
        readiness_config=readiness_config,
        datasets=datasets,
        checkpoints=checkpoints,
        output_root=output_root,
        device=device,
    )
    summaries = aggregate_gradients(pair_rows, norm_rows)
    gate = adjudicate(
        source_checks=source_checks,
        state_checks=state_checks,
        pair_rows=pair_rows,
        norm_rows=norm_rows,
        summaries=summaries,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "pair_rows": len(pair_rows),
        "expected_pair_rows": EXPECTED_PAIR_ROWS,
        "norm_rows": len(norm_rows),
        "expected_norm_rows": EXPECTED_NORM_ROWS,
        "summary_rows": len(summaries),
        "expected_summary_rows": EXPECTED_SUMMARY_ROWS,
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "stable_conflict_pairs": gate["stable_conflict_pairs"],
        "stable_gradient_norm_imbalance": gate[
            "stable_gradient_norm_imbalance"
        ],
        "norm_ratios": gate["norm_ratios"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "gradient_pairs.jsonl", pair_rows)
    _write_jsonl(output_root / "gradient_norms.jsonl", norm_rows)
    _write_jsonl(output_root / "results.jsonl", summaries)
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
        "results": summaries,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _load_checkpoints(
    *,
    paths: Mapping[str, Path],
    checkpoint_manifest: Mapping[str, Any],
    device: str,
) -> dict[int, dict[str, Any]]:
    manifest_entries = {
        int(row["replica"]): row for row in checkpoint_manifest.get("entries", [])
    }
    checkpoints = {}
    for replica in REPLICAS:
        name = f"checkpoints/replica{replica}_best.pt"
        path = paths[name]
        payload = torch.load(path, map_location=device, weights_only=False)
        state_dict = payload["state_dict"]
        state_dict_sha256 = tensor_mapping_sha256(state_dict)
        manifest = manifest_entries.get(replica, {})
        if (
            payload.get("run_id") != SOURCE_RUN_ID
            or int(payload.get("replica", -1)) != replica
            or int(payload.get("optimizer_steps", -1)) != 1_920
            or manifest.get("sha256") != file_sha256(path)
            or manifest.get("state_dict_sha256") != state_dict_sha256
        ):
            raise ValueError(f"K1-AP checkpoint binding failed for replica {replica}")
        checkpoints[replica] = {
            "replica": replica,
            "path": str(path),
            "sha256": file_sha256(path),
            "state_dict_sha256": state_dict_sha256,
            "best_epoch": int(payload["best_epoch"]),
            "optimizer_steps": int(payload["optimizer_steps"]),
            "strict_state_dict_load": True,
            "state_dict": state_dict,
        }
    return checkpoints


def _flatten_gradients(
    named_parameters: Sequence[tuple[str, torch.Tensor]],
    gradients: Sequence[torch.Tensor | None],
    *,
    names: set[str] | None = None,
) -> torch.Tensor:
    chunks = []
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        if names is not None and name not in names:
            continue
        chunks.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    if not chunks:
        raise ValueError("K1-AP gradient group is empty")
    return torch.cat(chunks)


def _is_transition_semantic_parameter(name: str) -> bool:
    return (
        name == "backbone.transition_gate"
        or name.startswith("backbone.sbox_transition_encoder.")
        or name.startswith("backbone.transition_projection.")
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    row = {"run_id": RUN_ID, "event": event, "time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _require_fresh_output_root(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"K1-AP output root already exists: {path}")


__all__ = [
    "CONFIG_PATH",
    "RUN_ID",
    "adjudicate",
    "aggregate_gradients",
    "audit_gradients",
    "gradient_cosine",
    "load_and_validate_config",
    "load_authority",
    "make_stratified_batches",
    "measure_gradient_vectors",
    "run_audit",
]
