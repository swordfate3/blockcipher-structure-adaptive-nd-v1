from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.position_histogram_residual import (
    deterministic_position_histogram,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import input_geometry
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    build_k1t_control,
)
from blockcipher_nd.training.data import make_loader
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_ctspn_compact_invariant_k1w_2048_seed_panel_20260728"
CONTROL_MODELS = {
    "compact_exact": "runtime_spn_ct_k1w_compact_histogram_true",
    "compact_wrong_sbox": "runtime_spn_ct_k1w_compact_histogram_wrong_sbox",
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONTROL_MODELS.items()}
EXPECTED_KEYS = {
    ("uknit64", 3, condition) for condition in CONTROL_MODELS
} | {
    ("uknit64", 4, condition) for condition in CONTROL_MODELS
} | {
    ("dialga128", 0, condition) for condition in CONTROL_MODELS
} | {
    ("dialga128", 1, condition) for condition in CONTROL_MODELS
}
EXPECTED_PARAMETER_COUNT = 137_516
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
FOLD_LOGIT_TOLERANCE = 1e-6
FOLD_METRIC_TOLERANCE = 1e-7
UKNIT_SEMANTIC_MARGIN = 0.010
UKNIT_AUC_FLOOR = 0.550
UKNIT_ANCHOR_TOLERANCE = 0.020
DIALGA_ANCHOR_TOLERANCE = 0.005
ANCHOR_AUCS = {
    ("uknit64", 3): 0.5654244422912598,
    ("uknit64", 4): 0.5940475463867188,
    ("dialga128", 0): 0.9597501754760742,
    ("dialga128", 1): 0.9547367095947266,
}

K1U_ROOT = ROOT / (
    "outputs/remote_results_incomplete/"
    "i1_uknit_family_ctspn_position_residual_k1u_medium_"
    "65536_seed3_seed4_20260728"
)
K1T_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_deterministic_position_residual_"
    "k1t_2048_seed3_seed4_20260728"
)
K1N_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_exact_operator_composition_"
    "k1n_2048_seed0_seed1_20260728"
)
D1_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725"
)
K1Q_CACHE_ROOT = ROOT / (
    "outputs/local_audit/"
    "i1_uknit_family_ctspn_difference_position_discovery_"
    "k1q_seed2_confirm_seed3_seed4_20260728/cache"
)
D1_CACHE_ROOT = D1_ROOT / "cache"

SOURCE_DIGESTS = {
    "k1u_gate": "79a5f3652b8a6125af8c987cb8b1df075fc8e992e73cdb5dc61dedbfbdb6c3ed",
    "k1t_gate": "f122f43f4d895a1b68fb696bd81df4e1d362880a3a12d9883933c932dd7f0dbf",
    "k1n_gate": "e2aed925c5d285f2856be791e1f6450b5e338f10e470572844539d86c1134a4f",
    "d1_gate": "e113227bbd541a3d5c11502793d5ebb5d75108c4f53e157326b5ac509cc10e67",
    "k1u_seed3_checkpoint": "ff43fb8a9787b60ae02dd79509d5702e0d1605455b795ca0aba7d9dcf017f750",
    "k1u_seed4_checkpoint": "c2709f21784a1e580caa0ae058be1e8b4cf6278cebd42bb053004683ca663c81",
    "k1u_seed3_features": "c97ee5b09bfc0c7c2863b2b2d4befcd536f1be3b90db1f1d05b36a5ce9d2fe44",
    "k1u_seed3_labels": "ef35e167c4a38cfbc8a50a324f4cf43bf4b998a66aa548d8718079939017e591",
    "k1u_seed4_features": "652e8ff219f00184dfcab9a4ad8deb7e9d513495ddc0746096f004feb7894526",
    "k1u_seed4_labels": "ef35e167c4a38cfbc8a50a324f4cf43bf4b998a66aa548d8718079939017e591",
}

K1U_CHECKPOINTS = {
    3: K1U_ROOT
    / "checkpoints/row0003_runtime_spn_ct_k1t_position_histogram_invariant_seed3.pt",
    4: K1U_ROOT
    / "checkpoints/row0006_runtime_spn_ct_k1t_position_histogram_invariant_seed4.pt",
}
K1U_VALIDATION_CACHES = {
    3: K1U_ROOT
    / "validation_cache/uknit64/r5/validation/seed-10003_14638234f18849a3",
    4: K1U_ROOT
    / "validation_cache/uknit64/r5/validation/seed-10004_624ea08c5ec79218",
}

SOURCE_CACHE_ROWS = (
    (
        "uknit64",
        3,
        "train",
        K1Q_CACHE_ROOT / "uknit64/r5/train/seed-3_010f3af7ff7c6b03",
        "7115a5f8616f79c2fa877be4e1a2b121868ae19dbe324c8835e7a65c219b04b8",
    ),
    (
        "uknit64",
        3,
        "validation",
        K1Q_CACHE_ROOT / "uknit64/r5/validation/seed-10003_222ac0f458b64b18",
        "e5b09d207d0fd0af5960d57cf56b6b7eed29c6ede3187bd1c95f0c4131831792",
    ),
    (
        "uknit64",
        4,
        "train",
        K1Q_CACHE_ROOT / "uknit64/r5/train/seed-4_e5ff32de2b652521",
        "e29ced8080972c32571f5fe39744093f80f47f8e83459e4b53be155e01d1edcd",
    ),
    (
        "uknit64",
        4,
        "validation",
        K1Q_CACHE_ROOT / "uknit64/r5/validation/seed-10004_f2b02ef8a58bdb97",
        "71c66435b22f59b75105f49465be76c778750571afe6fcd8a7941f854d626061",
    ),
    (
        "dialga128",
        0,
        "train",
        D1_CACHE_ROOT / "dialga128/r4/train/seed-0_3c38760cec3987ad",
        "fbd6e5cdea4658545224eb35f4150239349dfd189c8ecab367a898817acf7452",
    ),
    (
        "dialga128",
        0,
        "validation",
        D1_CACHE_ROOT / "dialga128/r4/validation/seed-10000_1a8a02f786407616",
        "7d8509e2a590c7e87d6b0b640ffa62baa5fd501b77352c0243d162f6b5ecd6e0",
    ),
    (
        "dialga128",
        1,
        "train",
        D1_CACHE_ROOT / "dialga128/r4/train/seed-1_7feb4fc6c2034702",
        "10c29a8a9bd010b7838c57dfaa48b090320e006460a53e1947a0e416e2062b73",
    ),
    (
        "dialga128",
        1,
        "validation",
        D1_CACHE_ROOT / "dialga128/r4/validation/seed-10001_6e40179b203c18cb",
        "cba2b6d4dea62a4c2bd6d707af2d3bc1d840f6c3297e7ab3ca8c29e3f32eee06",
    ),
)


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    mapped: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for task in tasks:
        condition = MODEL_TO_CONDITION.get(str(task.get("model_key")))
        if condition is None:
            continue
        key = (str(task["cipher_key"]), int(task["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-W task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-W task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == len(EXPECTED_KEYS)
        and set(mapped) == EXPECTED_KEYS
        and all(_task_protocol_frozen(key, task) for key, task in mapped.items())
    )


def build_k1w_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-W condition")
    _, pair_bits = input_geometry(str(task["cipher_key"]))
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=deepcopy(dict(task["model_options"])),
    )


def source_cache_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cipher, seed, split, cache_dir, expected_digest in SOURCE_CACHE_ROWS:
        dataset = _load_cache(cache_dir)
        observed_digest = differential_dataset_sha256(dataset)
        rows.append(
            {
                "cipher_key": cipher,
                "seed": seed,
                "split": split,
                "cache_dir": str(cache_dir),
                "rows": int(dataset.labels.shape[0]),
                "dataset_sha256": observed_digest,
                "expected_dataset_sha256": expected_digest,
                "digest_matches": observed_digest == expected_digest,
            }
        )
    return rows


def source_binding_checks(cache_rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    k1u_gate = _read_json(K1U_ROOT / "gate.json")
    k1t_gate = _read_json(K1T_ROOT / "gate.json")
    k1n_gate = _read_json(K1N_ROOT / "gate.json")
    d1_gate = _read_json(D1_ROOT / "gate.json")
    paths = {
        "k1u_gate": K1U_ROOT / "gate.json",
        "k1t_gate": K1T_ROOT / "gate.json",
        "k1n_gate": K1N_ROOT / "gate.json",
        "d1_gate": D1_ROOT / "gate.json",
        "k1u_seed3_checkpoint": K1U_CHECKPOINTS[3],
        "k1u_seed4_checkpoint": K1U_CHECKPOINTS[4],
        "k1u_seed3_features": K1U_VALIDATION_CACHES[3] / "features.npy",
        "k1u_seed3_labels": K1U_VALIDATION_CACHES[3] / "labels.npy",
        "k1u_seed4_features": K1U_VALIDATION_CACHES[4] / "features.npy",
        "k1u_seed4_labels": K1U_VALIDATION_CACHES[4] / "labels.npy",
    }
    return {
        "source_payload_digests_exact": all(
            path.is_file() and file_sha256(path) == SOURCE_DIGESTS[name]
            for name, path in paths.items()
        ),
        "eight_bound_source_caches_exact": len(cache_rows) == 8
        and all(bool(row.get("digest_matches")) for row in cache_rows)
        and {
            (str(row["cipher_key"]), int(row["seed"]), str(row["split"]))
            for row in cache_rows
        }
        == {
            (cipher, seed, split)
            for cipher, seed, split, _path, _digest in SOURCE_CACHE_ROWS
        },
        "k1u_compact_branch_activated": (
            k1u_gate.get("status") == "hold"
            and k1u_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1u_medium_signal_without_position_necessity"
            and not k1u_gate.get("failed_protocol_checks")
        ),
        "k1t_anchor_completed_pass": k1t_gate.get("status") == "pass"
        and not k1t_gate.get("failed_protocol_checks"),
        "k1n_dialga_anchor_retained": (
            k1n_gate.get("research_checks", {}).get(
                "dialga128_seed0_cross_key_validation_retains_anchor"
            )
            is True
            and k1n_gate.get("research_checks", {}).get(
                "dialga128_seed1_cross_key_validation_retains_anchor"
            )
            is True
        ),
        "d1_difference_signal_gate_pass": d1_gate.get("status") == "pass"
        and not d1_gate.get("failed_protocol_checks"),
    }


def fold_position_histogram_state(
    old_state: Mapping[str, torch.Tensor],
    compact_model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    target = compact_model.state_dict()
    folded: dict[str, torch.Tensor] = {}
    weight_name = "backbone.histogram_projection.0.weight"
    for name, target_value in target.items():
        if name not in old_state:
            raise ValueError(f"K1-W fold missing source tensor: {name}")
        source = (
            torch.as_tensor(old_state[name])
            .detach()
            .to(dtype=target_value.dtype)
            .clone()
        )
        if name == weight_name:
            if source.shape != (128, 640) or target_value.shape != (128, 40):
                raise ValueError("K1-W histogram projection fold geometry drifted")
            source = source.reshape(128, 5, 16, 8).sum(dim=2).reshape(128, 40)
        if source.shape != target_value.shape:
            raise ValueError(f"K1-W fold shape mismatch for {name}")
        folded[name] = source
    compact_model.load_state_dict(folded, strict=True)
    return folded


def replay_k1u_checkpoint_folds(
    tasks: Sequence[Mapping[str, Any]],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    mapped = task_map(tasks)
    seed_results: dict[str, Any] = {}
    for seed in (3, 4):
        task = mapped[("uknit64", seed, "compact_exact")]
        old_model = build_k1t_control(
            task=task,
            condition="invariant_histogram_residual",
            input_bits=512,
        )
        compact_model = build_k1w_control(
            task=task,
            condition="compact_exact",
            input_bits=512,
        )
        checkpoint = torch.load(
            K1U_CHECKPOINTS[seed],
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("K1-W K1-U checkpoint has no state_dict")
        old_model.load_state_dict(state, strict=True)
        fold_position_histogram_state(state, compact_model)
        dataset = _load_cache(K1U_VALIDATION_CACHES[seed])
        production = _compare_models(
            old_model,
            compact_model,
            dataset,
            batch_size=EXPECTED_BATCH_SIZE,
            device=device,
        )
        old_model = old_model.double()
        compact_model = compact_model.double()
        fold_position_histogram_state(state, compact_model)
        algebraic = _compare_models(
            old_model,
            compact_model,
            dataset,
            batch_size=EXPECTED_BATCH_SIZE,
            device=device,
        )
        production_max_logit_error = production.pop("max_logit_error")
        comparison = {
            **production,
            "production_float32_max_logit_error": production_max_logit_error,
            "algebraic_float64_max_logit_error": algebraic["max_logit_error"],
        }
        comparison["rows"] = int(dataset.labels.shape[0])
        comparison["checkpoint"] = str(K1U_CHECKPOINTS[seed])
        comparison["cache_dir"] = str(K1U_VALIDATION_CACHES[seed])
        comparison["old_anchor_auc"] = float(
            _read_json(K1U_ROOT / "gate.json")["seed_results"][str(seed)][
                "invariant_histogram_residual_auc"
            ]
        )
        comparison["anchor_auc_replayed"] = (
            abs(comparison["old_auc"] - comparison["old_anchor_auc"]) <= 1e-6
        )
        comparison["logits_equivalent"] = (
            comparison["algebraic_float64_max_logit_error"]
            <= FOLD_LOGIT_TOLERANCE
        )
        comparison["metrics_equivalent"] = (
            abs(comparison["old_auc"] - comparison["compact_auc"])
            <= FOLD_METRIC_TOLERANCE
            and abs(comparison["old_accuracy"] - comparison["compact_accuracy"])
            <= FOLD_METRIC_TOLERANCE
        )
        seed_results[str(seed)] = comparison
    return {
        "status": "pass"
        if all(
            result["anchor_auc_replayed"]
            and result["logits_equivalent"]
            and result["metrics_equivalent"]
            for result in seed_results.values()
        )
        else "fail",
        "seed_results": seed_results,
    }


def structural_readiness(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = task_map(tasks)
    rng = np.random.default_rng(20260728)
    models: dict[tuple[str, str], torch.nn.Module] = {}
    fixtures: dict[str, torch.Tensor] = {}
    for cipher, seed in (("uknit64", 3), ("dialga128", 0)):
        input_bits, _pair_bits = input_geometry(cipher)
        fixtures[cipher] = torch.as_tensor(
            rng.integers(0, 2, size=(7, input_bits), dtype=np.uint8),
            dtype=torch.float32,
        )
        for condition in CONTROL_MODELS:
            models[(cipher, condition)] = build_k1w_control(
                task=mapped[(cipher, seed, condition)],
                condition=condition,
                input_bits=input_bits,
            )
    geometries = {
        key: tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for key, model in models.items()
    }
    parameter_counts = {
        key: int(model_metadata(model)["trainable_parameter_count"])
        for key, model in models.items()
    }
    logits: dict[tuple[str, str], torch.Tensor] = {}
    histogram_changes: dict[str, float] = {}
    relabel_errors: dict[str, float] = {}
    gradient_checks: dict[str, bool] = {}
    for cipher in ("uknit64", "dialga128"):
        exact = models[(cipher, "compact_exact")]
        wrong = models[(cipher, "compact_wrong_sbox")]
        wrong.load_state_dict(deepcopy(exact.state_dict()), strict=True)
        fixture = fixtures[cipher]
        exact.eval()
        wrong.eval()
        with torch.no_grad():
            logits[(cipher, "compact_exact")] = exact(fixture)
            logits[(cipher, "compact_wrong_sbox")] = wrong(fixture)
            exact_histogram = _pooled_histogram(exact, fixture)
            wrong_histogram = _pooled_histogram(wrong, fixture)
            histogram_changes[cipher] = float(
                (exact_histogram - wrong_histogram).abs().max()
            )
            relabel_errors[cipher] = _joint_relabel_logit_delta(exact, fixture)
        exact.train()
        exact.zero_grad(set_to_none=True)
        labels = torch.arange(len(fixture), dtype=torch.float32).remainder(2)
        loss = torch.nn.functional.mse_loss(
            torch.sigmoid(exact(fixture)).flatten(),
            labels,
        )
        loss.backward()
        histogram_gradients = {
            name: parameter.grad
            for name, parameter in exact.named_parameters()
            if "histogram_" in name
        }
        gradient_checks[cipher] = bool(histogram_gradients) and all(
            gradient is not None
            and torch.isfinite(gradient).all()
            and float(gradient.detach().abs().sum()) > 0.0
            for gradient in histogram_gradients.values()
        )
    checks = {
        "identical_state_geometry_all_models": len(set(geometries.values())) == 1,
        "parameter_count_exact": set(parameter_counts.values())
        == {EXPECTED_PARAMETER_COUNT},
        "finite_shape_logits": all(
            values.shape == (7, 1) and torch.isfinite(values).all()
            for values in logits.values()
        ),
        "joint_cell_relabel_invariant": all(
            error <= FOLD_LOGIT_TOLERANCE for error in relabel_errors.values()
        ),
        "wrong_sbox_histograms_non_degenerate": all(
            change > 0.0 for change in histogram_changes.values()
        ),
        "wrong_sbox_logits_non_degenerate": all(
            not torch.equal(
                logits[(cipher, "compact_exact")],
                logits[(cipher, "compact_wrong_sbox")],
            )
            for cipher in ("uknit64", "dialga128")
        ),
        "compact_histogram_gradients_finite_nonzero": all(gradient_checks.values()),
        "no_identity_or_position_parameters": all(
            model.uses_cipher_identity is False
            and model.uses_absolute_cell_or_bit_identity is False
            and model.uses_runtime_native_cell_slots is False
            and not any(
                token in name
                for name, _parameter in model.named_parameters()
                for token in ("cipher_id", "cell_embedding", "cell_ordinal")
            )
            for model in models.values()
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "parameter_counts": {f"{key[0]}:{key[1]}": value for key, value in parameter_counts.items()},
        "relabel_max_logit_errors": relabel_errors,
        "wrong_sbox_histogram_max_changes": histogram_changes,
        "gradient_checks": gradient_checks,
    }


def build_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]] | None = None,
    bindings: Mapping[str, bool] | None = None,
    fold_replay: Mapping[str, Any] | None = None,
    structure: Mapping[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    rows = source_cache_manifest() if cache_rows is None else list(cache_rows)
    source_checks = (
        source_binding_checks(rows) if bindings is None else dict(bindings)
    )
    structural = structural_readiness(tasks) if structure is None else dict(structure)
    replay = (
        replay_k1u_checkpoint_folds(tasks, device=device)
        if fold_replay is None
        else dict(fold_replay)
    )
    protocol_checks = {
        "eight_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        **source_checks,
        "structural_readiness_pass": structural.get("status") == "pass"
        and all(structural.get("checks", {}).values()),
        "k1u_checkpoint_fold_replay_pass": replay.get("status") == "pass"
        and all(
            result.get("anchor_auc_replayed") is True
            and result.get("logits_equivalent") is True
            and result.get("metrics_equivalent") is True
            for result in replay.get("seed_results", {}).values()
        )
        and set(replay.get("seed_results", {})) == {"3", "4"},
    }
    status = "pass" if protocol_checks and all(protocol_checks.values()) else "fail"
    return {
        "run_id": RUN_ID,
        "status": status,
        "optimizer_step_authorized": status == "pass",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "source_cache_manifest": rows,
        "structural_readiness": structural,
        "k1u_checkpoint_fold_replay": replay,
    }


def adjudicate(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        "readiness_exact_pass": readiness.get("status") == "pass"
        and readiness.get("optimizer_step_authorized") is True
        and all(readiness.get("protocol_checks", {}).values()),
        "eight_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "eight_training_rows_complete": len(result_rows) == 8
        and set(rows) == EXPECTED_KEYS,
        "training_protocol_frozen": training_protocol_frozen(result_rows),
        "sixteen_source_cache_reuses_exact": _cache_reuse_protocol(progress_rows),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(_auc(row)) for row in rows.values()),
    }
    seed_results: dict[str, dict[str, Any]] = {"uknit64": {}, "dialga128": {}}
    research_checks: dict[str, bool] = {}
    for cipher, seed in (("uknit64", 3), ("uknit64", 4), ("dialga128", 0), ("dialga128", 1)):
        exact = _auc(rows[(cipher, seed, "compact_exact")]) if rows else math.nan
        wrong = _auc(rows[(cipher, seed, "compact_wrong_sbox")]) if rows else math.nan
        anchor = ANCHOR_AUCS[(cipher, seed)]
        result = {
            "compact_exact_auc": exact,
            "compact_wrong_sbox_auc": wrong,
            "anchor_auc": anchor,
            "exact_minus_anchor": exact - anchor,
            "exact_minus_wrong_sbox": exact - wrong,
        }
        seed_results[cipher][str(seed)] = result
        if cipher == "uknit64":
            threshold = max(UKNIT_AUC_FLOOR, anchor - UKNIT_ANCHOR_TOLERANCE)
            result["retention_threshold"] = threshold
            research_checks[f"uknit64_seed{seed}_retains_anchor"] = exact >= threshold
            research_checks[f"uknit64_seed{seed}_beats_wrong_sbox"] = (
                exact - wrong >= UKNIT_SEMANTIC_MARGIN
            )
        else:
            threshold = anchor - DIALGA_ANCHOR_TOLERANCE
            result["retention_threshold"] = threshold
            research_checks[f"dialga128_seed{seed}_retains_anchor"] = exact >= threshold
    protocol_valid = all(protocol_checks.values())
    all_research = bool(research_checks) and all(research_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1w_protocol_invalid"
        next_action = "repair only the failed protocol binding and rerun unchanged"
    elif all_research:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1w_compact_invariant_supported"
        next_action = (
            "freeze the compact architecture and compare 4 pairs versus 16 pairs "
            "inside it as a separate single-variable experiment"
        )
    elif any(
        not passed
        for name, passed in research_checks.items()
        if name.startswith("uknit64") and name.endswith("beats_wrong_sbox")
    ):
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1w_semantic_attribution_failed"
        next_action = "reject compact semantic scaling and audit histogram gate contribution"
    elif any(
        not passed
        for name, passed in research_checks.items()
        if name.startswith("uknit64")
    ):
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1w_uknit_retention_failed"
        next_action = "inspect compact histogram optimization without restoring redundant slots"
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1w_dialga_retention_failed"
        next_action = "audit compact residual interference on the frozen Dialga D1 backbone"
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "uknit_auc_floor": UKNIT_AUC_FLOOR,
            "uknit_anchor_tolerance": UKNIT_ANCHOR_TOLERANCE,
            "uknit_exact_minus_wrong_sbox": UKNIT_SEMANTIC_MARGIN,
            "dialga_anchor_tolerance": DIALGA_ANCHOR_TOLERANCE,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-cipher local 2048/class compact-invariant architecture diagnostic; "
            "not formal scale, attack, SOTA, shared-weight, zero-shot, transfer, or "
            "universal-SPN evidence"
        ),
        "blocked_actions": [
            "remote scale or 16 pairs from K1-W alone",
            "changing data, differences, epochs, seeds, labels, or negatives",
            "averaging across seeds to hide a failed retention or semantic gate",
        ],
    }


def result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    mapped: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        condition = MODEL_TO_CONDITION.get(str(row.get("model")))
        if condition is None:
            continue
        key = (str(row["cipher_key"]), int(row["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-W result: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-W result matrix is incomplete")
    return mapped


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cipher, seed_rows in gate.get("seed_results", {}).items():
        for seed, values in seed_rows.items():
            rows.append({"cipher_key": cipher, "seed": int(seed), **dict(values)})
    return sorted(rows, key=lambda row: (row["cipher_key"], row["seed"]))


def _task_protocol_frozen(
    key: tuple[str, int, str],
    task: Mapping[str, Any],
) -> bool:
    cipher, seed, _condition = key
    expected_rounds = 5 if cipher == "uknit64" else 4
    expected_difference = 0x0000400000000000 if cipher == "uknit64" else 0x40
    options = task.get("model_options", {})
    return (
        int(task.get("rounds", -1)) == expected_rounds
        and int(task.get("seed", -1)) == seed
        and int(task.get("samples_per_class", -1)) == 2048
        and int(task.get("validation_samples_total", -1)) == EXPECTED_VALIDATION_ROWS
        and int(task.get("pairs_per_sample", -1)) == 4
        and int(task.get("input_difference", -1)) == expected_difference
        and task.get("negative_mode") == "encrypted_random_plaintexts"
        and task.get("sample_structure") == "independent_pairs"
        and task.get("feature_encoding") == "ciphertext_pair_bits"
        and int(task.get("key_rotation_interval", -1)) == 0
        and task.get("loss") == "mse"
        and task.get("optimizer") == "adam"
        and float(task.get("learning_rate", math.nan)) == 1e-4
        and float(task.get("weight_decay", math.nan)) == 1e-5
        and task.get("checkpoint_metric") == "val_auc"
        and task.get("restore_best_checkpoint") is True
        and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
        and int(options.get("runtime_rounds", -1)) == 2
        and int(options.get("runtime_round_start", -1))
        == (3 if cipher == "uknit64" else 2)
        and int(options.get("pair_embedding_dim", -1)) == 128
        and int(options.get("histogram_value_dim", -1)) == 8
    )


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return len(rows) == 8 and all(
            int(row.get("samples_per_class", -1)) == 2048
            and int(row.get("pairs_per_sample", -1)) == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and int(row.get("trainable_parameter_count", -1))
            == EXPECTED_PARAMETER_COUNT
            and int(row.get("training", {}).get("input_bits", -1))
            == (512 if row.get("cipher_key") == "uknit64" else 1024)
            and int(row.get("training", {}).get("train_rows", -1))
            == EXPECTED_TRAIN_ROWS
            and int(row.get("training", {}).get("validation_rows", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("epochs_ran", -1))
            == EXPECTED_EPOCHS
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and Path(str(row.get("training", {}).get("checkpoint_output"))).is_file()
            for row in rows
        )
    except (TypeError, ValueError):
        return False


def _cache_reuse_protocol(rows: Sequence[Mapping[str, Any]]) -> bool:
    reuses = [row for row in rows if row.get("event") == "cache_reuse"]
    creates = [
        row
        for row in rows
        if row.get("event") in {"cache_start", "cache_done"}
    ]
    return len(reuses) == 16 and not creates


def _load_cache(cache_dir: Path) -> DiskDifferentialDataset:
    metadata_path = cache_dir / "metadata.json"
    features_path = cache_dir / "features.npy"
    labels_path = cache_dir / "labels.npy"
    if not all(path.is_file() for path in (metadata_path, features_path, labels_path)):
        raise ValueError(f"missing K1-W source cache payload: {cache_dir}")
    return DiskDifferentialDataset(
        features=np.load(features_path, mmap_mode="r"),
        labels=np.load(labels_path, mmap_mode="r"),
        metadata=_read_json(metadata_path),
        cache_dir=cache_dir,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_models(
    old_model: torch.nn.Module,
    compact_model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    *,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    selected = torch.device(device)
    old_model = old_model.to(selected).eval()
    compact_model = compact_model.to(selected).eval()
    old_probabilities: list[np.ndarray] = []
    compact_probabilities: list[np.ndarray] = []
    max_error = 0.0
    with torch.no_grad():
        for features, _labels in make_loader(dataset, batch_size, shuffle=False):
            dtype = next(old_model.parameters()).dtype
            features = features.to(selected, dtype=dtype)
            old_logits = old_model(features).squeeze(1)
            compact_logits = compact_model(features).squeeze(1)
            max_error = max(
                max_error,
                float((old_logits - compact_logits).abs().max().cpu()),
            )
            old_probabilities.append(torch.sigmoid(old_logits).cpu().numpy())
            compact_probabilities.append(torch.sigmoid(compact_logits).cpu().numpy())
    old_probs = np.concatenate(old_probabilities).astype(np.float32, copy=False)
    compact_probs = np.concatenate(compact_probabilities).astype(np.float32, copy=False)
    labels = np.asarray(dataset.labels, dtype=np.float32)
    return {
        "max_logit_error": max_error,
        "old_auc": binary_auc(labels, old_probs),
        "compact_auc": binary_auc(labels, compact_probs),
        "old_accuracy": float(((old_probs >= 0.5) == labels).mean()),
        "compact_accuracy": float(((compact_probs >= 0.5) == labels).mean()),
    }


def _pooled_histogram(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    runtime = features.reshape(
        features.shape[0],
        -1,
        2,
        model.runtime_structure.block_bits,
    ).flip(-1)
    return deterministic_position_histogram(
        runtime,
        model.runtime_structure,
        apply_sboxes=model.apply_sboxes,
    ).mean(dim=2)


def _forward_runtime(
    model: torch.nn.Module,
    runtime: torch.Tensor,
    structure: Any,
) -> torch.Tensor:
    base = model.backbone.base.encode(runtime, structure)
    edge = model.backbone.edge_residual_embedding(
        runtime,
        structure,
        apply_sboxes=model.apply_sboxes,
    )
    combined = base + torch.tanh(model.backbone.residual_gate) * torch.tanh(edge)
    histogram = model.backbone.histogram_embedding(
        runtime,
        structure,
        apply_sboxes=model.apply_sboxes,
    )
    combined = combined + torch.tanh(model.backbone.histogram_gate) * torch.tanh(
        histogram.repeat(1, 3)
    )
    return model.backbone.base.classifier(combined)


def _joint_relabel_logit_delta(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> float:
    structure = model.runtime_structure
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    runtime = features.reshape(features.shape[0], -1, 2, structure.block_bits).flip(-1)
    relabeled_runtime = torch.empty_like(runtime)
    relabeled_runtime[..., bit_permutation] = runtime
    original = _forward_runtime(model, runtime, structure)
    changed = _forward_runtime(model, relabeled_runtime, relabeled)
    return float((original - changed).abs().max())


def _auc(row: Mapping[str, Any]) -> float:
    return float(row["metrics"]["auc"])


__all__ = [
    "ANCHOR_AUCS",
    "CONTROL_MODELS",
    "EXPECTED_KEYS",
    "EXPECTED_PARAMETER_COUNT",
    "RUN_ID",
    "SOURCE_CACHE_ROWS",
    "adjudicate",
    "build_k1w_control",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "fold_position_histogram_state",
    "read_tasks",
    "replay_k1u_checkpoint_folds",
    "source_binding_checks",
    "source_cache_manifest",
    "structural_readiness",
    "task_map",
]
