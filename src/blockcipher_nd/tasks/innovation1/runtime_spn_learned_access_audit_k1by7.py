from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_neural_attribution_k1by6 as k1by6,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_learned_access_audit_k1by7_present_r7_"
    "seed2_seed3_retry1_20260801"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_learned_access_audit_k1by7_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "07c250050f175894074d80d33b3bec7c2574203dd2409afdea86ce4763ca188a"
)
TAPS = (
    "linear_histogram",
    "linear_primitive_expert",
    "cell_fusion",
    "pooled_stage_summary",
    "pre_classifier_representation",
)
CONDITIONS = ("correct", "affine_wrong_endpoint")
EXPECTED_SEEDS = (2, 3)
EXPECTED_RESULT_ROWS = len(TAPS) * len(CONDITIONS) * len(EXPECTED_SEEDS)
MARGIN_FLOOR = 0.005


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if _file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY7 config digest drifted")
    audit = config.get("audit", {})
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_learned_access_audit_k1by7"
        or audit.get("cipher") != "PRESENT-80"
        or audit.get("rounds") != 7
        or tuple(audit.get("seeds", ())) != EXPECTED_SEEDS
        or tuple(audit.get("conditions", ())) != CONDITIONS
        or tuple(audit.get("taps", ())) != TAPS
        or audit.get("validation_rows_per_seed") != 2048
        or audit.get("pairs_per_sample") != 16
        or audit.get("input_bits") != 2048
        or audit.get("batch_size") != 128
        or audit.get("device") != "cpu"
        or audit.get("neural_training_performed") is not False
        or audit.get("optimizer_steps") != 0
        or audit.get("discovery_rows") != "even_validation_indices"
        or audit.get("evaluation_rows") != "odd_validation_indices"
        or audit.get("discovery_rows_per_class") != 512
        or audit.get("evaluation_rows_per_class") != 512
        or audit.get("probe") != "variance_normalized_class_mean_difference"
        or float(audit.get("probe_epsilon", math.nan)) != 1e-6
        or float(
            audit.get("source_logit_replay_tolerance", math.nan)
        )
        != 1e-6
        or float(
            config.get("gates", {}).get(
                "correct_minus_affine_probe_auc_min",
                math.nan,
            )
        )
        != MARGIN_FLOOR
    ):
        raise ValueError("K1-BY7 frozen config contract drifted")
    return config


def source_binding_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    paths = source_artifact_paths(config)
    expected = source_expected_digests(config)
    checks = {
        f"{name}_digest_exact": path.is_file()
        and _file_sha256(path) == expected[name]
        for name, path in paths.items()
        if name in expected
    }
    try:
        k1by3_gate = _read_json(paths["k1by3_gate"])
        k1by3_validation = _read_json(paths["k1by3_validation"])
        k1by6_gate = _read_json(paths["k1by6_gate"])
        k1by6_validation = _read_json(paths["k1by6_validation"])
        datasets = validation_dataset_paths()
        balance_checks = {
            f"seed{seed}_even_odd_balance_exact": _split_balance_exact(
                np.load(labels_path, mmap_mode="r")
            )
            for seed, (_features_path, labels_path) in datasets.items()
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        k1by3_gate = {}
        k1by3_validation = {}
        k1by6_gate = {}
        k1by6_validation = {}
        balance_checks = {
            f"seed{seed}_even_odd_balance_exact": False for seed in EXPECTED_SEEDS
        }
    checks["k1by3_source_exact_hold"] = (
        k1by3_gate.get("status") == "hold"
        and k1by3_gate.get("decision")
        == "innovation1_runtime_spn_k1by3_permutation_attribution_not_supported"
        and k1by3_validation.get("status") == "pass"
    )
    checks["k1by6_source_exact_hold"] = (
        k1by6_gate.get("status") == "hold"
        and k1by6_gate.get("decision")
        == "innovation1_runtime_spn_k1by6_permutation_attribution_not_supported"
        and k1by6_validation.get("status") == "pass"
    )
    checks.update(balance_checks)
    checks["four_checkpoints_complete"] = all(
        paths[name].is_file()
        for name in (
            "correct_seed2_checkpoint",
            "correct_seed3_checkpoint",
            "affine_seed2_checkpoint",
            "affine_seed3_checkpoint",
        )
    )
    return checks


def build_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol_checks = {
        **source_binding_checks(config),
        "config_exact": load_and_validate_config() == config,
        "local_cpu_execution_frozen": config["audit"]["device"] == "cpu",
        "zero_optimizer_steps_frozen": config["audit"]["optimizer_steps"] == 0,
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            models, source_rows = load_models_and_source_rows(config, seed=2)
            fixture = torch.as_tensor(
                np.array(
                    np.load(validation_dataset_paths()[2][0], mmap_mode="r")[:4],
                    copy=True,
                ),
                dtype=torch.float32,
            )
            captures = {
                condition: capture_taps(model, fixture)
                for condition, model in models.items()
            }
            evidence_checks = {
                "all_taps_captured": all(
                    tuple(values) == TAPS for values in captures.values()
                ),
                "tap_shapes_match_between_conditions": all(
                    captures["correct"][tap].shape
                    == captures["affine_wrong_endpoint"][tap].shape
                    for tap in TAPS
                ),
                "tap_batches_finite": all(
                    torch.isfinite(values[tap]).all()
                    for values in captures.values()
                    for tap in TAPS
                ),
                "strict_checkpoint_rows_match_conditions": (
                    source_rows["correct"]["model"] == k1by6.CORRECT_MODEL
                    and source_rows["affine_wrong_endpoint"]["model"]
                    == k1by6.AFFINE_MODEL
                ),
                "models_retain_distinct_programs": (
                    models["correct"].compiled_program_semantic_sha256
                    != models[
                        "affine_wrong_endpoint"
                    ].compiled_program_semantic_sha256
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "tap_shapes": {
                    condition: {
                        tap: list(values[tap].shape) for tap in TAPS
                    }
                    for condition, values in captures.items()
                },
                "program_semantic_sha256": {
                    condition: model.compiled_program_semantic_sha256
                    for condition, model in models.items()
                },
            }
        except Exception as exc:  # pragma: no cover - fail-closed artifact path
            errors.append(f"{type(exc).__name__}: {exc}")
            evidence_checks["readiness_execution_succeeded"] = False
    status = (
        "pass"
        if protocol_checks
        and evidence_checks
        and all(protocol_checks.values())
        and all(evidence_checks.values())
        and not errors
        else "fail"
    )
    return {
        "run_id": RUN_ID,
        "status": status,
        "execution_authorized": status == "pass",
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "evidence_metrics": evidence_metrics,
        "errors": errors,
    }


def evaluate(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_size = int(config["audit"]["batch_size"])
    epsilon = float(config["audit"]["probe_epsilon"])
    result_rows: list[dict[str, Any]] = []
    replay: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        features_path, labels_path = validation_dataset_paths()[seed]
        features = np.load(features_path, mmap_mode="r")
        labels = np.asarray(np.load(labels_path, mmap_mode="r"), dtype=np.uint8)
        models, source_rows = load_models_and_source_rows(config, seed=seed)
        representations: dict[str, dict[str, list[np.ndarray]]] = {
            condition: {tap: [] for tap in TAPS} for condition in CONDITIONS
        }
        logits: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            batch = torch.as_tensor(
                np.array(features[start:stop], copy=True),
                dtype=torch.float32,
            )
            for condition, model in models.items():
                captured = capture_taps(model, batch)
                for tap in TAPS:
                    representations[condition][tap].append(
                        captured[tap].numpy(force=True)
                    )
                with torch.inference_mode():
                    probabilities = torch.sigmoid(model(batch).flatten())
                    logits[condition].append(probabilities.numpy(force=True))
        replay[str(seed)] = {}
        for condition in CONDITIONS:
            logit_array = np.concatenate(logits[condition])
            observed_auc = float(binary_auc(labels, logit_array))
            source_auc = float(source_rows[condition]["metrics"]["auc"])
            replay[str(seed)][condition] = {
                "source_auc": source_auc,
                "replayed_auc": observed_auc,
                "absolute_error": abs(observed_auc - source_auc),
            }
            for tap_index, tap in enumerate(TAPS):
                values = np.concatenate(representations[condition][tap])
                probe = mean_difference_probe(values, labels, epsilon=epsilon)
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "seed": seed,
                        "condition": condition,
                        "tap": tap,
                        "tap_index": tap_index,
                        "representation_shape": list(values.shape),
                        **probe,
                    }
                )
    return result_rows, replay


def mean_difference_probe(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, Any]:
    flattened = np.asarray(values, dtype=np.float32).reshape(len(labels), -1)
    discovery = np.arange(len(labels)) % 2 == 0
    evaluation = ~discovery
    train_x = flattened[discovery]
    train_y = labels[discovery]
    test_x = flattened[evaluation]
    test_y = labels[evaluation]
    center = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, epsilon)
    normalized = (train_x - center) / scale
    direction = normalized[train_y == 1].mean(axis=0) - normalized[
        train_y == 0
    ].mean(axis=0)
    if not np.any(direction):
        scores = np.zeros(len(test_y), dtype=np.float32)
    else:
        scores = ((test_x - center) / scale) @ direction
        scores = scores / math.sqrt(float(len(direction)))
    return {
        "probe_auc": float(binary_auc(test_y, scores)),
        "representation_width": int(flattened.shape[1]),
        "discovery_rows": int(discovery.sum()),
        "evaluation_rows": int(evaluation.sum()),
        "discovery_positive_rows": int(train_y.sum()),
        "discovery_negative_rows": int(len(train_y) - train_y.sum()),
        "evaluation_positive_rows": int(test_y.sum()),
        "evaluation_negative_rows": int(len(test_y) - test_y.sum()),
        "direction_l2": float(np.linalg.norm(direction)),
    }


def capture_taps(model: torch.nn.Module, features: torch.Tensor) -> dict[str, torch.Tensor]:
    histogram_inputs: list[torch.Tensor] = []
    primitive_outputs: list[torch.Tensor] = []
    fusion_outputs: list[torch.Tensor] = []
    pooled_inputs: list[torch.Tensor] = []
    classifier_inputs: list[torch.Tensor] = []

    handles = [
        model.conditioner.histogram_encoder.register_forward_hook(
            lambda _module, inputs, _output: histogram_inputs.append(inputs[0].detach())
        ),
        model.conditioner.permutation_expert.register_forward_hook(
            lambda _module, _inputs, output: primitive_outputs.append(output.detach())
        ),
        model.conditioner.cell_fusion.register_forward_hook(
            lambda _module, _inputs, output: fusion_outputs.append(output.detach())
        ),
        model.conditioner.stage_projection.register_forward_hook(
            lambda _module, inputs, _output: pooled_inputs.append(inputs[0].detach())
        ),
        model.backbone.classifier.register_forward_hook(
            lambda _module, inputs, _output: classifier_inputs.append(inputs[0].detach())
        ),
    ]
    try:
        with torch.inference_mode():
            model(features)
    finally:
        for handle in handles:
            handle.remove()
    if not (
        len(histogram_inputs) == 4
        and len(primitive_outputs) == 2
        and len(fusion_outputs) == 2
        and len(pooled_inputs) == 2
        and len(classifier_inputs) == 1
    ):
        raise ValueError("K1-BY7 hook call geometry drifted")
    return {
        "linear_histogram": torch.stack(histogram_inputs[::2], dim=1),
        "linear_primitive_expert": torch.stack(primitive_outputs, dim=1),
        "cell_fusion": torch.stack(fusion_outputs, dim=1),
        "pooled_stage_summary": torch.stack(pooled_inputs, dim=1),
        "pre_classifier_representation": classifier_inputs[0],
    }


def adjudicate(
    config: Mapping[str, Any],
    *,
    result_rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    readiness: Mapping[str, Any],
    sources_unchanged: bool,
) -> dict[str, Any]:
    mapped = {
        (int(row["seed"]), str(row["condition"]), str(row["tap"])): row
        for row in result_rows
    }
    protocol_checks = {
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("execution_authorized") is True
            and readiness.get("training_authorized") is False
            and readiness.get("optimizer_steps_authorized") == 0
        ),
        "source_bindings_still_exact": all(source_binding_checks(config).values()),
        "source_artifacts_unchanged": sources_unchanged,
        "twenty_internal_probe_rows_exact": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and set(mapped)
            == {
                (seed, condition, tap)
                for seed in EXPECTED_SEEDS
                for condition in CONDITIONS
                for tap in TAPS
            }
        ),
        "probe_rows_finite_and_balanced": all(
            math.isfinite(float(row.get("probe_auc", math.nan)))
            and int(row.get("discovery_positive_rows", -1)) == 512
            and int(row.get("discovery_negative_rows", -1)) == 512
            and int(row.get("evaluation_positive_rows", -1)) == 512
            and int(row.get("evaluation_negative_rows", -1)) == 512
            for row in result_rows
        ),
        "all_source_logits_replay": all(
            float(values[condition]["absolute_error"])
            <= float(config["audit"]["source_logit_replay_tolerance"])
            for values in replay.values()
            for condition in CONDITIONS
        ),
    }
    seed_results: dict[str, Any] = {}
    first_loss_by_seed: dict[str, str | None] = {}
    for seed in EXPECTED_SEEDS:
        taps: dict[str, Any] = {}
        first_loss: str | None = None
        for tap in TAPS:
            correct = float(mapped[(seed, "correct", tap)]["probe_auc"])
            affine = float(
                mapped[(seed, "affine_wrong_endpoint", tap)]["probe_auc"]
            )
            margin = correct - affine
            passed = margin >= MARGIN_FLOOR
            if first_loss is None and not passed:
                first_loss = tap
            taps[tap] = {
                "correct_probe_auc": correct,
                "affine_probe_auc": affine,
                "correct_minus_affine_probe_auc": margin,
                "margin_pass": passed,
            }
        source_margin = (
            float(replay[str(seed)]["correct"]["source_auc"])
            - float(replay[str(seed)]["affine_wrong_endpoint"]["source_auc"])
        )
        if first_loss is None and source_margin < MARGIN_FLOOR:
            first_loss = "final_classifier"
        first_loss_by_seed[str(seed)] = first_loss
        seed_results[str(seed)] = {
            "taps": taps,
            "source_logit_margin": source_margin,
            "first_margin_loss": first_loss,
        }
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    seed3_loss = first_loss_by_seed.get("3")
    localization_complete = seed3_loss is not None
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by7_protocol_invalid"
        next_action = "Repair only the failed frozen source, hook, probe or artifact invariant."
    elif not localization_complete:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by7_seed3_loss_not_localized"
        next_action = (
            "Add no new model or data. Audit only the source checkpoint selection and "
            "final classifier replay because the frozen tap sequence did not explain "
            "the observed seed3 reversal."
        )
    else:
        status = "pass"
        decision = f"innovation1_runtime_spn_k1by7_first_loss_{seed3_loss}_identified"
        next_action = _next_action(str(seed3_loss))
    return {
        "run_id": RUN_ID,
        "status": status,
        "method_status": "hold",
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol,
        "thresholds": {"correct_minus_affine_probe_auc_min": MARGIN_FLOOR},
        "source_logit_replay": replay,
        "seed_results": seed_results,
        "first_loss_by_seed": first_loss_by_seed,
        "seed3_localization_complete": localization_complete,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Zero-new-training internal representation audit on frozen K1-BY3/K1-BY6 "
            "PRESENT checkpoints and validation rows; the closed-form probe is mechanism "
            "diagnosis, not distinguisher, formal-scale, transfer or SOTA evidence."
        ),
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed, values in sorted(gate.get("seed_results", {}).items()):
        for tap_index, tap in enumerate(TAPS):
            row = values["taps"][tap]
            rows.append(
                {
                    "seed": int(seed),
                    "tap_index": tap_index,
                    "tap": tap,
                    **row,
                    "first_margin_loss": values["first_margin_loss"],
                }
            )
    return rows


def load_models_and_source_rows(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, torch.nn.Module], dict[str, Mapping[str, Any]]]:
    task = k1by6.task_map(k1by6.read_tasks())[seed]
    rows3 = _read_jsonl(ROOT / config["sources"]["k1by3_root"] / "results.jsonl")
    rows6 = _read_jsonl(ROOT / config["sources"]["k1by6_root"] / "results.jsonl")
    correct_row = _one_row(rows3, seed=seed, model=k1by6.CORRECT_MODEL)
    affine_row = _one_row(rows6, seed=seed, model=k1by6.AFFINE_MODEL)
    models = {
        "correct": k1by6.build_model_for_task(task, model_key=k1by6.CORRECT_MODEL),
        "affine_wrong_endpoint": k1by6.build_model_for_task(
            task,
            model_key=k1by6.AFFINE_MODEL,
        ),
    }
    source_rows = {
        "correct": correct_row,
        "affine_wrong_endpoint": affine_row,
    }
    for condition, model in models.items():
        checkpoint = Path(
            str(source_rows[condition]["training"]["checkpoint_output"])
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
    return models, source_rows


def validation_dataset_paths() -> dict[int, tuple[Path, Path]]:
    root = k1by6.K1BY3_CACHE_ROOT / "present80/r7/validation"
    result = {}
    for seed in EXPECTED_SEEDS:
        matches = list(root.glob(f"seed-{10000 + seed}_*"))
        if len(matches) != 1:
            raise ValueError(f"expected one K1-BY3 validation cache for seed{seed}")
        result[seed] = (matches[0] / "features.npy", matches[0] / "labels.npy")
    return result


def source_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    sources = config["sources"]
    root3 = ROOT / sources["k1by3_root"]
    root6 = ROOT / sources["k1by6_root"]
    rows3 = _read_jsonl(root3 / "results.jsonl")
    rows6 = _read_jsonl(root6 / "results.jsonl")
    return {
        "k1by3_plan": ROOT / sources["k1by3_plan"],
        "k1by3_results": root3 / "results.jsonl",
        "k1by3_gate": root3 / "gate.json",
        "k1by3_validation": root3 / "validation.json",
        "k1by6_plan": ROOT / sources["k1by6_plan"],
        "k1by6_results": root6 / "results.jsonl",
        "k1by6_gate": root6 / "gate.json",
        "k1by6_preflight": root6 / "preflight.json",
        "k1by6_validation": root6 / "validation.json",
        "correct_seed2_checkpoint": Path(
            str(_one_row(rows3, seed=2, model=k1by6.CORRECT_MODEL)["training"]["checkpoint_output"])
        ),
        "correct_seed3_checkpoint": Path(
            str(_one_row(rows3, seed=3, model=k1by6.CORRECT_MODEL)["training"]["checkpoint_output"])
        ),
        "affine_seed2_checkpoint": Path(
            str(_one_row(rows6, seed=2, model=k1by6.AFFINE_MODEL)["training"]["checkpoint_output"])
        ),
        "affine_seed3_checkpoint": Path(
            str(_one_row(rows6, seed=3, model=k1by6.AFFINE_MODEL)["training"]["checkpoint_output"])
        ),
    }


def source_expected_digests(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    d3 = sources["k1by3_digests"]
    d6 = sources["k1by6_digests"]
    return {
        "k1by3_plan": d3["plan"],
        "k1by3_results": d3["results.jsonl"],
        "k1by3_gate": d3["gate.json"],
        "k1by3_validation": d3["validation.json"],
        "correct_seed2_checkpoint": d3["correct_seed2_checkpoint"],
        "correct_seed3_checkpoint": d3["correct_seed3_checkpoint"],
        "k1by6_plan": d6["plan"],
        "k1by6_results": d6["results.jsonl"],
        "k1by6_gate": d6["gate.json"],
        "k1by6_preflight": d6["preflight.json"],
        "k1by6_validation": d6["validation.json"],
        "affine_seed2_checkpoint": d6["affine_seed2_checkpoint"],
        "affine_seed3_checkpoint": d6["affine_seed3_checkpoint"],
    }


def authority_digests(config: Mapping[str, Any]) -> dict[str, str]:
    paths = source_artifact_paths(config)
    paths.update(
        {
            f"validation_seed{seed}_features": feature_path
            for seed, (feature_path, _label_path) in validation_dataset_paths().items()
        }
    )
    paths.update(
        {
            f"validation_seed{seed}_labels": label_path
            for seed, (_feature_path, label_path) in validation_dataset_paths().items()
        }
    )
    return {name: _file_sha256(path) for name, path in paths.items()}


def _next_action(first_loss: str) -> str:
    if first_loss == "linear_histogram":
        return (
            "Keep the network frozen. Test one deterministic histogram association "
            "repair that preserves endpoint bundles and the K1-BY6 affine control; do "
            "not change pooling, capacity, data or ciphers."
        )
    if first_loss in {"linear_primitive_expert", "cell_fusion"}:
        return (
            "Change only primitive-expert normalization or semantic fusion at the "
            "localized tap, then rerun the exact K1-BY6 two-seed budget and controls."
        )
    if first_loss == "pooled_stage_summary":
        return (
            "Redesign only the cell-order-invariant aggregation contract while keeping "
            "the primitive expert, data and classifier frozen in the next diagnostic."
        )
    return (
        "Audit only primitive residual gating and final readout/checkpoint selection; "
        "retain all upstream representations, data and controls."
    )


def _split_balance_exact(labels: np.ndarray) -> bool:
    return all(
        len(split) == 1024
        and int(split.sum()) == 512
        and int(len(split) - split.sum()) == 512
        for split in (labels[::2], labels[1::2])
    )


def _one_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    model: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if int(row.get("seed", -1)) == seed and row.get("model") == model
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one source row for seed={seed}, model={model}")
    return matches[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CONFIG_PATH",
    "CONDITIONS",
    "RUN_ID",
    "TAPS",
    "adjudicate",
    "authority_digests",
    "build_readiness",
    "capture_taps",
    "comparison_rows",
    "evaluate",
    "load_and_validate_config",
    "mean_difference_probe",
    "source_binding_checks",
]
