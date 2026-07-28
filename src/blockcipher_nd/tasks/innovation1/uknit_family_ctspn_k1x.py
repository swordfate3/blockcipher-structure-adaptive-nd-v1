from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    build_k1t_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1w import (
    K1Q_CACHE_ROOT,
    build_k1w_control,
    fold_position_histogram_state,
    read_tasks,
    task_map,
)
from blockcipher_nd.training.data import make_loader
from blockcipher_nd.training.metrics import binary_auc
from blockcipher_nd.training.optim import compute_loss


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_ctspn_compact_optimization_geometry_k1x_20260728"
K1W_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_compact_invariant_k1w_2048_seed_panel_20260728"
)
K1T_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_deterministic_position_residual_"
    "k1t_2048_seed3_seed4_20260728"
)
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_compact_invariant_k1w_2048_seed_panel.csv"
)
EXPECTED_SEEDS = (3, 4)
EXPECTED_BATCH_SIZE = 64
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_INFERENCE_ROWS = len(EXPECTED_SEEDS) * 2 * 3
EXPECTED_GRADIENT_ROWS = len(EXPECTED_SEEDS)
REPLAY_TOLERANCE = 1e-6
GRADIENT_RELATIVE_TOLERANCE = 1e-9
GRADIENT_ABSOLUTE_FLOOR = 1e-12
UPDATE_RATIO_MIN = 15.999
UPDATE_RATIO_MAX = 16.001
WEAK_AUC_DELTA = 0.010

SOURCE_PATHS = {
    "k1w_gate": K1W_ROOT / "gate.json",
    "k1w_results": K1W_ROOT / "results.jsonl",
    "k1t_gate": K1T_ROOT / "gate.json",
    "k1t_results": K1T_ROOT / "results.jsonl",
    "k1t_checkpoint_manifest": K1T_ROOT / "checkpoint_manifest.json",
}
SOURCE_DIGESTS = {
    "k1w_gate": "8f94cd31798638313d21c632445004ceb9d3fee545b5d3813b1ed6e4b998e338",
    "k1w_results": "75a7bdad3fb64b562c92545f4734e14dfad6c2d002b0099c5c02c0a1495a37e7",
    "k1t_gate": "f122f43f4d895a1b68fb696bd81df4e1d362880a3a12d9883933c932dd7f0dbf",
    "k1t_results": "adafb1217298ade5ad7bda4aff5a53742e951e5c737babb30a449362e948563a",
    "k1t_checkpoint_manifest": (
        "b971fa9e25b70c4a7a76caff608a886e758b356091881fd40ab42ff2e4289bc9"
    ),
}
K1W_CHECKPOINTS = {
    3: K1W_ROOT
    / "checkpoints/row0001_runtime_spn_ct_k1w_compact_histogram_true_seed3.pt",
    4: K1W_ROOT
    / "checkpoints/row0003_runtime_spn_ct_k1w_compact_histogram_true_seed4.pt",
}
K1W_CHECKPOINT_DIGESTS = {
    3: "3a53e7e1c648a5e2998f014285ed61218bae86fd1e5fa6a2216fc32dbbb76821",
    4: "9f5e6f001e732ae4c2c74ceb300e928e63a0042426d7afa9e1a546258accc5df",
}
K1T_CHECKPOINTS = {
    3: K1T_ROOT
    / "checkpoints/row0003_runtime_spn_ct_k1t_position_histogram_invariant_seed3.pt",
    4: K1T_ROOT
    / "checkpoints/row0006_runtime_spn_ct_k1t_position_histogram_invariant_seed4.pt",
}
K1T_CHECKPOINT_DIGESTS = {
    3: "aea483d1438617472216d1f7a70574ae99b0c2bf32ede601fa4a58bd1eed55ae",
    4: "21a91a173b3929759971f1ae198f26d3d1e5e0dc802ffa3686183600d635620b",
}
TRAIN_CACHES = {
    3: K1Q_CACHE_ROOT / "uknit64/r5/train/seed-3_010f3af7ff7c6b03",
    4: K1Q_CACHE_ROOT / "uknit64/r5/train/seed-4_e5ff32de2b652521",
}
VALIDATION_CACHES = {
    3: K1Q_CACHE_ROOT
    / "uknit64/r5/validation/seed-10003_222ac0f458b64b18",
    4: K1Q_CACHE_ROOT
    / "uknit64/r5/validation/seed-10004_f2b02ef8a58bdb97",
}


def source_binding_checks() -> dict[str, bool]:
    checks = {
        f"{name}_sha256_exact": path.is_file()
        and file_sha256(path) == SOURCE_DIGESTS[name]
        for name, path in SOURCE_PATHS.items()
    }
    checks.update(
        {
            f"k1w_seed{seed}_checkpoint_sha256_exact": path.is_file()
            and file_sha256(path) == K1W_CHECKPOINT_DIGESTS[seed]
            for seed, path in K1W_CHECKPOINTS.items()
        }
    )
    checks.update(
        {
            f"k1t_seed{seed}_checkpoint_sha256_exact": path.is_file()
            and file_sha256(path) == K1T_CHECKPOINT_DIGESTS[seed]
            for seed, path in K1T_CHECKPOINTS.items()
        }
    )
    checks.update(
        {
            f"seed{seed}_source_caches_exist": all(
                (cache / name).is_file()
                for cache in (TRAIN_CACHES[seed], VALIDATION_CACHES[seed])
                for name in ("features.npy", "labels.npy", "metadata.json")
            )
            for seed in EXPECTED_SEEDS
        }
    )
    if checks and all(checks.values()):
        k1w_gate = read_json(SOURCE_PATHS["k1w_gate"])
        k1t_gate = read_json(SOURCE_PATHS["k1t_gate"])
        checks["k1w_hold_decision_exact"] = (
            k1w_gate.get("status") == "hold"
            and k1w_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1w_semantic_attribution_failed"
        )
        checks["k1t_supported_decision_exact"] = (
            k1t_gate.get("status") == "pass"
            and k1t_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1t_deterministic_position_residual_supported"
        )
    return checks


def audit_inference_panel(
    *,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    tasks = task_map(read_tasks(PLAN))
    k1w_gate = read_json(SOURCE_PATHS["k1w_gate"])
    k1t_gate = read_json(SOURCE_PATHS["k1t_gate"])
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        task = tasks[("uknit64", seed, "compact_exact")]
        validation = load_cache(VALIDATION_CACHES[seed])
        k1w_state = load_state(K1W_CHECKPOINTS[seed])
        k1t_state = load_state(K1T_CHECKPOINTS[seed])

        k1w_exact = build_k1w_control(
            task=task,
            condition="compact_exact",
            input_bits=512,
        )
        k1w_wrong = build_k1w_control(
            task=task,
            condition="compact_wrong_sbox",
            input_bits=512,
        )
        k1w_exact.load_state_dict(k1w_state, strict=True)
        k1w_wrong.load_state_dict(k1w_state, strict=True)

        folded_exact = build_k1w_control(
            task=task,
            condition="compact_exact",
            input_bits=512,
        )
        folded_wrong = build_k1w_control(
            task=task,
            condition="compact_wrong_sbox",
            input_bits=512,
        )
        fold_position_histogram_state(k1t_state, folded_exact)
        fold_position_histogram_state(k1t_state, folded_wrong)

        source_aucs = {
            "k1w_compact": float(
                k1w_gate["seed_results"]["uknit64"][str(seed)][
                    "compact_exact_auc"
                ]
            ),
            "k1t_folded": float(
                k1t_gate["seed_results"][str(seed)]["cross_key_validation"][
                    "invariant_histogram_residual_auc"
                ]
            ),
        }
        for checkpoint_kind, exact_model, wrong_model in (
            ("k1w_compact", k1w_exact, k1w_wrong),
            ("k1t_folded", folded_exact, folded_wrong),
        ):
            exact = evaluate_model(
                exact_model,
                validation,
                device=device,
                zero_histogram_gate=False,
            )
            zero = evaluate_model(
                exact_model,
                validation,
                device=device,
                zero_histogram_gate=True,
            )
            wrong = evaluate_model(
                wrong_model,
                validation,
                device=device,
                zero_histogram_gate=False,
            )
            common = {
                "run_id": RUN_ID,
                "cipher_key": "uknit64",
                "rounds": 5,
                "seed": seed,
                "checkpoint_kind": checkpoint_kind,
                "rows": EXPECTED_VALIDATION_ROWS,
                "split": "cross_key_validation",
                "dataset_path": str(VALIDATION_CACHES[seed]),
                "checkpoint_path": str(
                    K1W_CHECKPOINTS[seed]
                    if checkpoint_kind == "k1w_compact"
                    else K1T_CHECKPOINTS[seed]
                ),
                "source_auc": source_aucs[checkpoint_kind],
                "histogram_effective_gate": float(
                    torch.tanh(exact_model.backbone.histogram_gate.detach())
                ),
                "strict_state_dict_load": True,
                "training_performed": False,
                "optimizer_steps": 0,
            }
            rows.extend(
                (
                    {
                        **common,
                        "condition": "exact",
                        **exact,
                        "full_minus_zero_auc": exact["auc"] - zero["auc"],
                        "exact_minus_wrong_sbox_auc": exact["auc"] - wrong["auc"],
                    },
                    {
                        **common,
                        "condition": "zero_histogram_gate",
                        **zero,
                    },
                    {
                        **common,
                        "condition": "wrong_sbox_same_checkpoint",
                        **wrong,
                    },
                )
            )
    return rows


def audit_gradient_panel() -> list[dict[str, Any]]:
    tasks = task_map(read_tasks(PLAN))
    return [
        audit_gradient_seed(
            seed=seed,
            task=tasks[("uknit64", seed, "compact_exact")],
            dataset=load_cache(TRAIN_CACHES[seed]),
        )
        for seed in EXPECTED_SEEDS
    ]


def audit_gradient_seed(
    *,
    seed: int,
    task: Mapping[str, Any],
    dataset: DiskDifferentialDataset,
) -> dict[str, Any]:
    old = build_k1t_control(
        task=task,
        condition="invariant_histogram_residual",
        input_bits=512,
    ).double()
    compact = build_k1w_control(
        task=task,
        condition="compact_exact",
        input_bits=512,
    ).double()
    source_state = load_state(K1T_CHECKPOINTS[seed])
    old.load_state_dict(source_state, strict=True)
    fold_position_histogram_state(source_state, compact)
    old.eval()
    compact.eval()
    old_state_before = tensor_mapping_sha256(old.state_dict())
    compact_state_before = tensor_mapping_sha256(compact.state_dict())

    features = torch.as_tensor(
        np.asarray(dataset.features[:EXPECTED_BATCH_SIZE]).copy(),
        dtype=torch.float64,
    )
    labels = torch.as_tensor(
        np.asarray(dataset.labels[:EXPECTED_BATCH_SIZE]).copy(),
        dtype=torch.float64,
    )
    old.zero_grad(set_to_none=True)
    compact.zero_grad(set_to_none=True)
    old_logits = old(features).squeeze(1)
    compact_logits = compact(features).squeeze(1)
    old_loss = compute_loss(nn.MSELoss(), old_logits, labels, "mse")
    compact_loss = compute_loss(nn.MSELoss(), compact_logits, labels, "mse")
    old_loss.backward()
    compact_loss.backward()

    old_weight = old.backbone.histogram_projection[0].weight
    compact_weight = compact.backbone.histogram_projection[0].weight
    if old_weight.grad is None or compact_weight.grad is None:
        raise ValueError("K1-X histogram projection gradient is missing")
    old_gradient = old_weight.grad.detach().reshape(128, 5, 16, 8)
    compact_gradient = compact_weight.grad.detach().reshape(128, 5, 8)
    expanded = compact_gradient.unsqueeze(2).expand_as(old_gradient)
    slot_max_abs_error = float((old_gradient - expanded).abs().max())
    gradient_scale = max(float(compact_gradient.abs().max()), GRADIENT_ABSOLUTE_FLOOR)
    slot_relative_error = slot_max_abs_error / gradient_scale
    folded_gradient = old_gradient.sum(dim=2)
    expected_folded = 16.0 * compact_gradient
    folded_max_abs_error = float((folded_gradient - expected_folded).abs().max())
    folded_relative_error = folded_max_abs_error / max(
        float(expected_folded.abs().max()),
        GRADIENT_ABSOLUTE_FLOOR,
    )
    compact_norm = float(torch.linalg.vector_norm(compact_gradient))
    folded_norm = float(torch.linalg.vector_norm(folded_gradient))
    update_ratio = folded_norm / compact_norm if compact_norm else math.nan

    old.zero_grad(set_to_none=True)
    compact.zero_grad(set_to_none=True)
    old_state_after = tensor_mapping_sha256(old.state_dict())
    compact_state_after = tensor_mapping_sha256(compact.state_dict())
    return {
        "run_id": RUN_ID,
        "cipher_key": "uknit64",
        "rounds": 5,
        "seed": seed,
        "batch_rows": EXPECTED_BATCH_SIZE,
        "dataset_path": str(TRAIN_CACHES[seed]),
        "checkpoint_path": str(K1T_CHECKPOINTS[seed]),
        "old_compact_max_abs_logit_error": float(
            (old_logits.detach() - compact_logits.detach()).abs().max()
        ),
        "old_compact_abs_loss_error": abs(
            float(old_loss.detach()) - float(compact_loss.detach())
        ),
        "slot_gradient_max_abs_error": slot_max_abs_error,
        "slot_gradient_relative_error": slot_relative_error,
        "folded_gradient_max_abs_error": folded_max_abs_error,
        "folded_gradient_relative_error": folded_relative_error,
        "compact_gradient_norm": compact_norm,
        "folded_old_gradient_norm": folded_norm,
        "folded_effective_update_ratio": update_ratio,
        "old_state_sha256_before": old_state_before,
        "old_state_sha256_after": old_state_after,
        "compact_state_sha256_before": compact_state_before,
        "compact_state_sha256_after": compact_state_after,
        "state_restored_exact": old_state_before == old_state_after
        and compact_state_before == compact_state_after,
        "training_performed": False,
        "optimizer_steps": 0,
    }


def adjudicate(
    *,
    inference_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    inference = inference_map(inference_rows)
    gradients = {int(row["seed"]): row for row in gradient_rows}
    expected_inference = {
        (seed, checkpoint, condition)
        for seed in EXPECTED_SEEDS
        for checkpoint in ("k1w_compact", "k1t_folded")
        for condition in (
            "exact",
            "zero_histogram_gate",
            "wrong_sbox_same_checkpoint",
        )
    }
    replay_errors = {
        f"{checkpoint}_seed{seed}": abs(
            float(inference[(seed, checkpoint, "exact")]["auc"])
            - float(inference[(seed, checkpoint, "exact")]["source_auc"])
        )
        for seed in EXPECTED_SEEDS
        for checkpoint in ("k1w_compact", "k1t_folded")
        if (seed, checkpoint, "exact") in inference
    }
    protocol_checks = {
        **dict(source_checks),
        "twelve_inference_rows_exact": len(inference_rows)
        == EXPECTED_INFERENCE_ROWS
        and set(inference) == expected_inference,
        "two_gradient_rows_exact": len(gradient_rows) == EXPECTED_GRADIENT_ROWS
        and set(gradients) == set(EXPECTED_SEEDS),
        "source_auc_replayed": len(replay_errors) == 4
        and max(replay_errors.values(), default=math.inf) <= REPLAY_TOLERANCE,
        "finite_inference_metrics": bool(inference_rows)
        and all(
            math.isfinite(float(row.get(metric, math.nan)))
            for row in inference_rows
            for metric in ("auc", "accuracy")
        ),
        "strict_load_zero_training": all(
            row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in inference_rows
        ),
        "gradient_state_unchanged_zero_steps": len(gradient_rows)
        == EXPECTED_GRADIENT_ROWS
        and all(
            row.get("state_restored_exact") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in gradient_rows
        ),
        "folded_forward_equivalence": len(gradient_rows)
        == EXPECTED_GRADIENT_ROWS
        and all(
            float(row.get("old_compact_max_abs_logit_error", math.inf)) <= 1e-10
            and float(row.get("old_compact_abs_loss_error", math.inf)) <= 1e-12
            for row in gradient_rows
        ),
    }
    research_checks: dict[str, bool] = {}
    seed_results: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        exact = inference.get((seed, "k1w_compact", "exact"), {})
        zero = inference.get((seed, "k1w_compact", "zero_histogram_gate"), {})
        wrong = inference.get(
            (seed, "k1w_compact", "wrong_sbox_same_checkpoint"),
            {},
        )
        gradient = gradients.get(seed, {})
        full_minus_zero = float(exact.get("auc", math.nan)) - float(
            zero.get("auc", math.nan)
        )
        exact_minus_wrong = float(exact.get("auc", math.nan)) - float(
            wrong.get("auc", math.nan)
        )
        ratio = float(gradient.get("folded_effective_update_ratio", math.nan))
        slot_error = float(gradient.get("slot_gradient_relative_error", math.inf))
        folded_error = float(
            gradient.get("folded_gradient_relative_error", math.inf)
        )
        research_checks[f"seed{seed}_slot_gradients_equal"] = (
            slot_error <= GRADIENT_RELATIVE_TOLERANCE
        )
        research_checks[f"seed{seed}_folded_gradient_exact"] = (
            folded_error <= GRADIENT_RELATIVE_TOLERANCE
        )
        research_checks[f"seed{seed}_effective_update_ratio_16x"] = (
            UPDATE_RATIO_MIN <= ratio <= UPDATE_RATIO_MAX
        )
        research_checks[f"seed{seed}_k1w_histogram_contribution_weak"] = (
            abs(full_minus_zero) <= WEAK_AUC_DELTA
        )
        research_checks[f"seed{seed}_k1w_same_state_semantics_weak"] = (
            abs(exact_minus_wrong) <= WEAK_AUC_DELTA
        )
        seed_results[str(seed)] = {
            "k1w_exact_auc": exact.get("auc"),
            "k1w_zero_histogram_auc": zero.get("auc"),
            "k1w_wrong_sbox_same_checkpoint_auc": wrong.get("auc"),
            "k1w_full_minus_zero_auc": full_minus_zero,
            "k1w_exact_minus_wrong_sbox_auc": exact_minus_wrong,
            "k1w_histogram_effective_gate": exact.get(
                "histogram_effective_gate"
            ),
            "folded_effective_update_ratio": ratio,
            "slot_gradient_relative_error": slot_error,
            "folded_gradient_relative_error": folded_error,
            "k1t_folded_exact_auc": inference.get(
                (seed, "k1t_folded", "exact"), {}
            ).get("auc"),
            "k1t_folded_zero_histogram_auc": inference.get(
                (seed, "k1t_folded", "zero_histogram_gate"), {}
            ).get("auc"),
            "k1t_folded_wrong_sbox_same_checkpoint_auc": inference.get(
                (seed, "k1t_folded", "wrong_sbox_same_checkpoint"), {}
            ).get("auc"),
        }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_valid = bool(research_checks) and all(research_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1x_protocol_invalid"
        next_action = "repair only the failed source, replay, intervention, or zero-step audit binding"
    elif research_valid:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1x_16x_optimization_geometry_supported"
        )
        next_action = (
            "run K1-Y changing only the compact first projection weight learning-rate multiplier to 16x"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1x_optimization_geometry_not_sufficient"
        )
        next_action = (
            "do not tune compact learning rates; select a different invariant aggregator or audit branch interference"
        )
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
        "source_replay_errors": replay_errors,
        "seed_results": seed_results,
        "thresholds": {
            "gradient_relative_tolerance": GRADIENT_RELATIVE_TOLERANCE,
            "update_ratio_min": UPDATE_RATIO_MIN,
            "update_ratio_max": UPDATE_RATIO_MAX,
            "weak_auc_delta": WEAK_AUC_DELTA,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local zero-training post-result optimization-geometry audit; "
            "not formal scale, attack, SOTA, transfer, family ceiling, or optimizer proof beyond the frozen projection"
        ),
        "blocked_actions": [
            "remote scale or sixteen pairs",
            "changing data, epochs, seeds, architecture, labels, or negatives",
            "training K1-Y unless every K1-X protocol and research check passes",
        ],
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: DiskDifferentialDataset,
    *,
    device: str,
    zero_histogram_gate: bool,
) -> dict[str, float]:
    selected = torch.device(device)
    model = model.to(selected).eval()
    probability_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for features, _labels in make_loader(
            dataset,
            batch_size=EXPECTED_BATCH_SIZE,
            shuffle=False,
        ):
            features = features.to(selected)
            logits = forward_with_histogram_gate(
                model,
                features,
                zero_histogram_gate=zero_histogram_gate,
            ).squeeze(1)
            probability_chunks.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(probability_chunks).astype(np.float32, copy=False)
    labels = np.asarray(dataset.labels, dtype=np.float32)
    return {
        "auc": binary_auc(labels, probabilities),
        "accuracy": float(((probabilities >= 0.5) == labels).mean()),
    }


def forward_with_histogram_gate(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    zero_histogram_gate: bool,
) -> torch.Tensor:
    runtime = features.reshape(
        features.shape[0],
        -1,
        2,
        model.runtime_structure.block_bits,
    ).flip(-1)
    base = model.backbone.base.encode(runtime, model.runtime_structure)
    edge = model.backbone.edge_residual_embedding(
        runtime,
        model.runtime_structure,
        apply_sboxes=model.apply_sboxes,
    )
    combined = base + torch.tanh(model.backbone.residual_gate) * torch.tanh(edge)
    histogram = model.backbone.histogram_embedding(
        runtime,
        model.runtime_structure,
        apply_sboxes=model.apply_sboxes,
    )
    histogram_gate = (
        torch.zeros_like(model.backbone.histogram_gate)
        if zero_histogram_gate
        else torch.tanh(model.backbone.histogram_gate)
    )
    combined = combined + histogram_gate * torch.tanh(histogram.repeat(1, 3))
    return model.backbone.base.classifier(combined)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"K1-X checkpoint has no state_dict: {path}")
    return {
        str(name): torch.as_tensor(value).detach().clone()
        for name, value in state.items()
    }


def load_cache(path: Path) -> DiskDifferentialDataset:
    metadata_path = path / "metadata.json"
    features_path = path / "features.npy"
    labels_path = path / "labels.npy"
    if not all(item.is_file() for item in (metadata_path, features_path, labels_path)):
        raise ValueError(f"missing K1-X source cache: {path}")
    return DiskDifferentialDataset(
        features=np.load(features_path, mmap_mode="r"),
        labels=np.load(labels_path, mmap_mode="r"),
        metadata=read_json(metadata_path),
        cache_dir=path,
    )


def inference_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["checkpoint_kind"]),
            str(row["condition"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-X inference row: {key}")
        mapped[key] = row
    return mapped


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), **dict(result)}
        for seed, result in sorted(gate.get("seed_results", {}).items())
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "EXPECTED_GRADIENT_ROWS",
    "EXPECTED_INFERENCE_ROWS",
    "RUN_ID",
    "adjudicate",
    "audit_gradient_panel",
    "audit_gradient_seed",
    "audit_inference_panel",
    "comparison_rows",
    "forward_with_histogram_gate",
    "inference_map",
    "source_binding_checks",
]
