from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.present_zhang_wang_keras import (
    PresentZhangWangKerasMCNDDistinguisher,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONFIRMATION_KEYS,
    DIFFERENCE_PROFILE,
    INPUT_DIFFERENCE,
)


RUN_ID = (
    "i1_uknit_r5_published_architecture_baselines_k1bz_16pair_"
    "2048_seed3_seed4_20260802"
)
K1BS_ROOT = Path(
    "outputs/local_diagnostic/"
    "i1_uknit_r5_neural_architecture_ablation_k1bs_16pair_"
    "2048_seed3_seed4_20260731"
)
EXPECTED_SEEDS = (3, 4)
EXPECTED_PAIRS = 16
PAIR_BITS = 128
EXPECTED_INPUT_BITS = EXPECTED_PAIRS * PAIR_BITS
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
ARCHITECTURES = {
    "zhang_wang_mcnd": "spn_zhang_wang_mcnd_adapter",
    "liu_case3_conv2d": "spn_liu_case3_conv2d_adapter",
}
MODEL_TO_ARCHITECTURE = {
    model: architecture for architecture, model in ARCHITECTURES.items()
}
K1BS_ANCHORS = {
    3: {"structure_expert": 0.902801514, "autond_dbitnet": 0.511321068},
    4: {"structure_expert": 0.932538986, "autond_dbitnet": 0.526423454},
}
SIGNAL_FLOOR = 0.550
AUTOND_MARGIN = 0.010


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=EXPECTED_PAIRS,
        difference_profile=None,
        difference_member=0,
    )


def task_map(
    tasks: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for task in tasks:
        architecture = MODEL_TO_ARCHITECTURE.get(str(task.get("model_key")))
        if architecture is None:
            continue
        key = (int(task["seed"]), architecture)
        if key in mapped:
            raise ValueError(f"duplicate K1-BZ task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BZ task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == 4
        and set(mapped) == expected_keys()
        and all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
            and int(task.get("samples_per_class", -1)) == 2048
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
            and task.get("difference_profile") == DIFFERENCE_PROFILE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            and int(task.get("train_key", -1)) == CONFIRMATION_KEYS[seed][0]
            and int(task.get("validation_key", -1)) == CONFIRMATION_KEYS[seed][1]
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            for (seed, _), task in mapped.items()
        )
    )


def build_architecture(
    task: Mapping[str, Any], architecture: str
) -> torch.nn.Module:
    return build_model(
        ARCHITECTURES[architecture],
        input_bits=EXPECTED_INPUT_BITS,
        hidden_bits=32,
        pair_bits=PAIR_BITS,
        structure="SPN",
        model_options=dict(task["model_options"]),
    )


def build_readiness(
    tasks: Sequence[Mapping[str, Any]], *, k1bs_root: Path = K1BS_ROOT
) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    protocol_checks = {
        "four_frozen_tasks_exact": len(tasks) == 4 and set(mapped) == expected_keys(),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
        "local_cuda_unavailable": not torch.cuda.is_available(),
        "k1bs_cache_available": (k1bs_root / "cache/uknit64").is_dir(),
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            torch.manual_seed(20260802)
            fixture = torch.randint(0, 2, (4, EXPECTED_INPUT_BITS)).float()
            models = {
                architecture: build_architecture(mapped[(3, architecture)], architecture)
                for architecture in ARCHITECTURES
            }
            parameter_counts = {
                architecture: int(model_metadata(model)["trainable_parameter_count"])
                for architecture, model in models.items()
            }
            output_shapes: dict[str, list[int]] = {}
            gradient_l1: dict[str, float] = {}
            for architecture, model in models.items():
                logits = model(fixture)
                output_shapes[architecture] = list(logits.shape)
                torch.nn.functional.mse_loss(
                    torch.sigmoid(logits).flatten(),
                    torch.arange(4, dtype=torch.float32).remainder(2),
                ).backward()
                gradient_l1[architecture] = sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )

            adapted = models["zhang_wang_mcnd"]
            legacy = PresentZhangWangKerasMCNDDistinguisher(
                input_bits=EXPECTED_INPUT_BITS, pair_bits=PAIR_BITS, base_channels=32
            )
            legacy.load_state_dict(adapted.state_dict(), strict=True)
            adapted.eval()
            legacy.eval()
            exact_legacy_match = torch.equal(adapted(fixture), legacy(fixture))

            liu = models["liu_case3_conv2d"]
            view = liu.case3_view(fixture)
            pairs = fixture.reshape(4, EXPECTED_PAIRS, 2, 64)
            expected_delta = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
            recovered_delta = view[:, :, 2].permute(0, 1, 3, 2).reshape(
                4, EXPECTED_PAIRS, 64
            )
            evidence_checks = {
                "both_models_accept_same_input": set(
                    tuple(shape) for shape in output_shapes.values()
                )
                == {(4, 1)},
                "finite_nonzero_backward_gradients": all(
                    math.isfinite(value) and value > 0 for value in gradient_l1.values()
                ),
                "zhang_adapter_exactly_matches_legacy_present_layout": exact_legacy_match,
                "liu_case3_view_shape_exact": tuple(view.shape)
                == (4, EXPECTED_PAIRS, 3, 4, 16),
                "liu_case3_delta_channel_exact": torch.equal(
                    recovered_delta, expected_delta
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "parameter_counts": parameter_counts,
                "output_shapes": output_shapes,
                "gradient_l1": gradient_l1,
                "liu_case3_view_shape": list(view.shape),
            }
        except Exception as exc:  # pragma: no cover - fail-closed artifact path
            errors.append(f"{type(exc).__name__}: {exc}")
            evidence_checks["readiness_execution_succeeded"] = False
    passed = (
        bool(protocol_checks)
        and bool(evidence_checks)
        and all(protocol_checks.values())
        and all(evidence_checks.values())
        and not errors
    )
    return {
        "run_id": RUN_ID,
        "status": "pass" if passed else "fail",
        "optimizer_step_authorized": passed,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "evidence_metrics": evidence_metrics,
        "errors": errors,
    }


def adjudicate(
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        "readiness_exact_pass": readiness.get("status") == "pass"
        and readiness.get("optimizer_step_authorized") is True,
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "four_training_rows_complete": len(result_rows) == 4
        and set(rows) == expected_keys(),
        "training_protocol_frozen": training_protocol_frozen(result_rows),
        "k1bs_cache_reused_without_creation": cache_protocol_frozen(progress_rows),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(_auc(row)) for row in rows.values()),
    }
    seed_results: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        if all((seed, architecture) in rows for architecture in ARCHITECTURES):
            aucs = {
                architecture: _auc(rows[(seed, architecture)])
                for architecture in ARCHITECTURES
            }
            seed_results[str(seed)] = {
                "auc_by_architecture": aucs,
                "k1bs_anchors": K1BS_ANCHORS[seed],
                "adapter_minus_autond": {
                    architecture: auc - K1BS_ANCHORS[seed]["autond_dbitnet"]
                    for architecture, auc in aucs.items()
                },
                "expert_minus_adapter": {
                    architecture: K1BS_ANCHORS[seed]["structure_expert"] - auc
                    for architecture, auc in aucs.items()
                },
            }
    eligible = [
        architecture
        for architecture in ARCHITECTURES
        if all(
            seed_results.get(str(seed), {})
            .get("auc_by_architecture", {})
            .get(architecture, -math.inf)
            >= SIGNAL_FLOOR
            and seed_results[str(seed)]["adapter_minus_autond"][architecture]
            >= AUTOND_MARGIN
            for seed in EXPECTED_SEEDS
        )
    ]
    strongest = (
        max(
            eligible,
            key=lambda architecture: sum(
                seed_results[str(seed)]["auc_by_architecture"][architecture]
                for seed in EXPECTED_SEEDS
            ),
        )
        if eligible
        else None
    )
    protocol_valid = all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_k1bz_published_baseline_protocol_invalid"
        remote_scale = "no"
        next_action = "repair only the failed frozen protocol invariant and rerun K1-BZ unchanged"
    elif strongest is None:
        status = "hold"
        decision = "innovation1_uknit_k1bz_no_published_adapter_local_promotion"
        remote_scale = "no"
        next_action = (
            "retain the completed rows as local diagnostics and do not remotely scale either "
            "published architecture adapter"
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_k1bz_published_adapter_remote_candidate"
        remote_scale = "candidate"
        next_action = (
            f"prepare a fresh uniquely named 65536/class remote matrix with the uKNIT "
            f"structure expert, AutoND and {strongest}; do not reuse K1-BT"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": remote_scale,
        "selected_remote_candidate": strongest,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "adapter_auc": SIGNAL_FLOOR,
            "adapter_minus_autond": AUTOND_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class uKNIT r5 published-architecture adaptation "
            "diagnostic; not an original-paper protocol reproduction or formal evidence"
        ),
        "blocked_actions": [
            "calling K1-BZ paper-scale or formal evidence",
            "reporting original-paper accuracy as a same-protocol comparison",
            "attributing the raw Liu adapter to inverse-round feature engineering",
            "modifying or reusing the protocol-invalid K1-BT run id",
        ],
    }


def result_map(
    rows: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        architecture = MODEL_TO_ARCHITECTURE.get(str(row.get("model")))
        if architecture is None:
            continue
        key = (int(row["seed"]), architecture)
        if key in mapped:
            raise ValueError(f"duplicate K1-BZ result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BZ result matrix is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == 4 and all(
        row.get("model") in MODEL_TO_ARCHITECTURE
        and int(row.get("rounds", -1)) == 5
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("training", {}).get("input_bits", -1))
        == EXPECTED_INPUT_BITS
        and int(row.get("training", {}).get("train_rows", -1)) == EXPECTED_TRAIN_ROWS
        and int(row.get("training", {}).get("validation_rows", -1))
        == EXPECTED_VALIDATION_ROWS
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and row.get("training", {}).get("selected_checkpoint") == "best"
        and Path(str(row.get("training", {}).get("checkpoint_output", ""))).is_file()
        for row in rows
    )


def cache_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    events = [
        row
        for row in rows
        if row.get("event") in {"cache_start", "cache_reuse"}
        and row.get("split") in {"train", "validation"}
    ]
    return (
        sum(row.get("event") == "cache_start" for row in events) == 0
        and sum(row.get("event") == "cache_reuse" for row in events) == 8
    )


def expected_keys() -> set[tuple[int, str]]:
    return {
        (seed, architecture)
        for seed in EXPECTED_SEEDS
        for architecture in ARCHITECTURES
    }


def _auc(row: Mapping[str, Any]) -> float:
    return float(row.get("metrics", {}).get("auc", math.nan))


__all__ = [
    "ARCHITECTURES",
    "K1BS_ANCHORS",
    "K1BS_ROOT",
    "RUN_ID",
    "adjudicate",
    "build_readiness",
    "candidate_protocol_frozen",
    "read_tasks",
    "task_map",
]
