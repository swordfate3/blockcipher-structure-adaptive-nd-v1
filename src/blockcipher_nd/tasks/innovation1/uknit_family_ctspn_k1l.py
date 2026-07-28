from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import (
    EXPECTED_SPLITS,
)
from blockcipher_nd.training.data import make_loader
from blockcipher_nd.training.metrics import binary_auc
from blockcipher_nd.training.optim import compute_loss


RUN_ID = "i1_uknit_family_ctspn_residual_attribution_k1l_20260728"
EXPECTED_SOURCE_DECISION = (
    "innovation1_uknit_family_ctspn_k1k_dialga_retained_"
    "operator_attribution_not_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "gate": "8922bd1d03de41547f33329b869204d2d05d664514674699f6661a7eaf758055",
    "checkpoint_manifest": (
        "1c826e182c3762d389a6d575ddbc755331a6a0123fcba87dde7f856006b8473f"
    ),
    "dataset_manifest": (
        "ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0"
    ),
    "controls": "b08832d8f01fe0091a1a1f07e507dc830833662204c2fbc618f9702eca06d3a0",
    "validation": (
        "02583f9045e8d4fd54900e3f54a40a56939d1516356beaa93a6e584e1d06fe8c"
    ),
}
EXPECTED_CHECKPOINT_DIGESTS = {
    ("uknit64", 0): (
        "f50a17dd09e5c7084081c1233f9b80bccc89aff342baa3d00a47c9f8ac320626"
    ),
    ("uknit64", 1): (
        "fa58547e023c7a620b666d5e19a89790677f98d17f119abd7ad29f28542359c0"
    ),
    ("dialga128", 0): (
        "d5db221f288b18940744af13f38d7ba543715e45f46c326fb0511c2fbb35b7a4"
    ),
    ("dialga128", 1): (
        "b2515e623507ff1374cb9debb743b4ec1c030bed7759998d10f5c8b5d6403846"
    ),
}
AUDIT_CONDITIONS = (
    "native_full",
    "gate_zero",
    "slot0_only",
    "slot1_only",
    "residual_row_shuffle",
    "reversed_full",
    "corrupted_full",
    "no_topology_full",
)
FULL_TOPOLOGY_CONDITIONS = {
    "native_full": "exact_ordered",
    "reversed_full": "operator_reversed",
    "corrupted_full": "operator_corrupted",
    "no_topology_full": "no_topology",
}
EXPECTED_RESULT_ROWS = (
    len(EXPECTED_CIPHERS)
    * len(EXPECTED_SEEDS)
    * len(EXPECTED_SPLITS)
    * len(AUDIT_CONDITIONS)
)
EXPECTED_GRADIENT_ROWS = len(EXPECTED_CIPHERS) * len(EXPECTED_SEEDS) * 2
REPLAY_TOLERANCE = 1e-7
CLOSED_GATE = 0.001
ACTIVE_GATE = 0.010
OPERATOR_MARGIN = 0.005
ROW_SHUFFLE_EXPLAINED = 0.80
ZERO_GRADIENT_TOLERANCE = 1e-12
OPEN_GRADIENT_FLOOR = 1e-8


@dataclass(frozen=True)
class ResidualPathOutputs:
    base_embeddings: torch.Tensor
    bounded_residuals: torch.Tensor
    full_logits: np.ndarray
    zero_logits: np.ndarray
    effective_gate: float


def label_blind_row_permutation(
    row_count: int,
    *,
    cipher: str,
    seed: int,
    split: str,
) -> torch.Tensor:
    if cipher not in EXPECTED_CIPHERS or split not in EXPECTED_SPLITS:
        raise ValueError("K1-L row permutation requires a frozen cipher and split")
    if seed not in EXPECTED_SEEDS or row_count < 2:
        raise ValueError("K1-L row permutation requires a frozen seed and rows")
    cipher_index = EXPECTED_CIPHERS.index(cipher)
    split_index = EXPECTED_SPLITS.index(split)
    generator = torch.Generator().manual_seed(
        20260728 + 1009 * cipher_index + 101 * seed + split_index
    )
    permutation = torch.randperm(row_count, generator=generator)
    if torch.equal(permutation, torch.arange(row_count)):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def collect_residual_path_outputs(
    model: torch.nn.Module,
    dataset: DifferentialDataset,
    *,
    batch_size: int,
    slot_mask: tuple[bool, bool] = (True, True),
) -> ResidualPathOutputs:
    model.eval()
    base_chunks: list[torch.Tensor] = []
    residual_chunks: list[torch.Tensor] = []
    full_chunks: list[torch.Tensor] = []
    zero_chunks: list[torch.Tensor] = []
    gate = float(torch.tanh(model.backbone.residual_gate.detach()))
    with torch.inference_mode():
        for features, _labels in make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        ):
            runtime = project_features(model, features)
            base = model.backbone.base.encode(runtime, model.runtime_structure)
            residual = model.backbone.edge_residual_embedding(
                runtime,
                model.runtime_structure,
                slot_mask=slot_mask,
            )
            bounded = torch.tanh(residual)
            zero_logits = model.backbone.base.classifier(base).squeeze(1)
            full_logits = model.backbone.base.classifier(
                base + gate * bounded
            ).squeeze(1)
            base_chunks.append(base.cpu())
            residual_chunks.append(bounded.cpu())
            full_chunks.append(full_logits.cpu())
            zero_chunks.append(zero_logits.cpu())
    return ResidualPathOutputs(
        base_embeddings=torch.cat(base_chunks),
        bounded_residuals=torch.cat(residual_chunks),
        full_logits=torch.cat(full_chunks).numpy().astype(np.float32, copy=False),
        zero_logits=torch.cat(zero_chunks).numpy().astype(np.float32, copy=False),
        effective_gate=gate,
    )


def shuffled_residual_logits(
    model: torch.nn.Module,
    outputs: ResidualPathOutputs,
    permutation: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    if sorted(permutation.tolist()) != list(range(len(outputs.base_embeddings))):
        raise ValueError("K1-L residual row permutation must be bijective")
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(permutation), batch_size):
            stop = min(start + batch_size, len(permutation))
            base = outputs.base_embeddings[start:stop]
            residual = outputs.bounded_residuals[permutation[start:stop]]
            logits = model.backbone.base.classifier(
                base + outputs.effective_gate * residual
            ).squeeze(1)
            chunks.append(logits.cpu())
    return torch.cat(chunks).numpy().astype(np.float32, copy=False)


def residual_metrics(
    labels: np.ndarray,
    outputs: ResidualPathOutputs,
    *,
    logits: np.ndarray | None = None,
) -> dict[str, float]:
    selected_logits = outputs.full_logits if logits is None else logits
    probabilities = sigmoid_numpy(selected_logits)
    zero_probabilities = sigmoid_numpy(outputs.zero_logits)
    contribution = selected_logits - outputs.zero_logits
    residual_norms = torch.linalg.vector_norm(outputs.bounded_residuals, dim=1)
    return {
        "auc": binary_auc(labels, probabilities),
        "zero_gate_auc": binary_auc(labels, zero_probabilities),
        "full_minus_zero_auc": (
            binary_auc(labels, probabilities)
            - binary_auc(labels, zero_probabilities)
        ),
        "residual_contribution_auc": binary_auc(labels, contribution),
        "mean_abs_residual_logit_contribution": float(np.abs(contribution).mean()),
        "rms_residual_logit_contribution": float(
            np.sqrt(np.square(contribution).mean())
        ),
        "max_abs_residual_logit_contribution": float(np.abs(contribution).max()),
        "mean_bounded_residual_embedding_norm": float(residual_norms.mean()),
    }


def audit_gradient_path(
    model: torch.nn.Module,
    dataset: DifferentialDataset,
    *,
    effective_gate: float,
    batch_size: int = 64,
) -> dict[str, Any]:
    if effective_gate not in {0.0, 0.05}:
        raise ValueError("K1-L gradient proof permits only gates 0.0 and 0.05")
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_sha = tensor_mapping_sha256(state)
    features = torch.as_tensor(
        np.asarray(dataset.features[:batch_size]),
        dtype=torch.float32,
    )
    labels = torch.as_tensor(
        np.asarray(dataset.labels[:batch_size]),
        dtype=torch.float32,
    )
    model.eval()
    model.zero_grad(set_to_none=True)
    raw_gate = 0.0 if effective_gate == 0.0 else math.atanh(effective_gate)
    with torch.no_grad():
        model.backbone.residual_gate.fill_(raw_gate)
    logits = model(features).squeeze(1)
    loss = compute_loss(nn.MSELoss(), logits, labels, "mse")
    loss.backward()
    groups = {
        "gate": ("backbone.residual_gate",),
        "cell_encoder": ("backbone.cell_encoder.",),
        "edge_encoder": ("backbone.edge_encoder.",),
        "cell_update": (
            "backbone.cell_update.",
            "backbone.cell_update_norm.",
        ),
        "residual_projection": ("backbone.residual_pair_projection.",),
    }
    gradient_norms = {
        group: grouped_gradient_norm(model, prefixes)
        for group, prefixes in groups.items()
    }
    model.zero_grad(set_to_none=True)
    model.load_state_dict(state, strict=True)
    restored_sha = tensor_mapping_sha256(model.state_dict())
    return {
        "gate_condition": (
            "exact_zero" if effective_gate == 0.0 else "effective_0p05"
        ),
        "effective_gate": effective_gate,
        "raw_gate": raw_gate,
        "loss": float(loss.detach()),
        "gradient_norms": gradient_norms,
        "state_sha256_before": initial_sha,
        "state_sha256_after": restored_sha,
        "state_restored_exact": initial_sha == restored_sha,
        "training_performed": False,
        "optimizer_steps": 0,
    }


def grouped_gradient_norm(
    model: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> float:
    squared = 0.0
    found = False
    for name, parameter in model.named_parameters():
        if not any(name == prefix or name.startswith(prefix) for prefix in prefixes):
            continue
        found = True
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum())
    if not found:
        raise ValueError(f"K1-L gradient group not found: {prefixes}")
    return math.sqrt(squared)


def adjudicate_k1l(
    *,
    result_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    results = result_map(result_rows)
    gradients = gradient_map(gradient_rows)
    expected_results = {
        (cipher, seed, split, condition)
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in AUDIT_CONDITIONS
    }
    expected_gradients = {
        (cipher, seed, condition)
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for condition in ("exact_zero", "effective_0p05")
    }
    replay_deltas = [
        abs(float(row.get("auc", math.nan)) - float(row.get("source_auc", math.nan)))
        for row in result_rows
        if row.get("condition") in FULL_TOPOLOGY_CONDITIONS
    ]
    protocol_checks = {
        **dict(source_checks),
        "ninety_six_result_rows_exact": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and set(results) == expected_results
        ),
        "eight_gradient_rows_exact": (
            len(gradient_rows) == EXPECTED_GRADIENT_ROWS
            and set(gradients) == expected_gradients
        ),
        "full_topology_auc_replay_exact": (
            len(replay_deltas)
            == len(EXPECTED_CIPHERS)
            * len(EXPECTED_SEEDS)
            * len(EXPECTED_SPLITS)
            * len(FULL_TOPOLOGY_CONDITIONS)
            and max(replay_deltas, default=math.inf) <= REPLAY_TOLERANCE
        ),
        "finite_result_metrics": all(
            math.isfinite(float(row.get("auc", math.nan)))
            and math.isfinite(float(row.get("residual_contribution_auc", math.nan)))
            for row in result_rows
        ),
        "deterministic_nonidentity_row_shuffle": all(
            row.get("row_permutation_bijective") is True
            and row.get("row_permutation_nonidentity") is True
            for row in result_rows
            if row.get("condition") == "residual_row_shuffle"
        ),
        "gradient_state_restored": all(
            row.get("state_restored_exact") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in gradient_rows
        ),
    }
    zero_starved = all(
        max(
            float(gradients[(cipher, seed, "exact_zero")]["gradient_norms"][group])
            for group in (
                "cell_encoder",
                "edge_encoder",
                "cell_update",
                "residual_projection",
            )
        )
        <= ZERO_GRADIENT_TOLERANCE
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
    )
    opened_gradients = all(
        max(
            float(
                gradients[(cipher, seed, "effective_0p05")]["gradient_norms"][group]
            )
            for group in (
                "cell_encoder",
                "edge_encoder",
                "cell_update",
                "residual_projection",
            )
        )
        > OPEN_GRADIENT_FLOOR
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
    )
    gate_by_checkpoint = {
        cipher: {
            str(seed): abs(
                float(
                    results[(cipher, seed, "train_seen", "native_full")][
                        "effective_gate"
                    ]
                )
            )
            for seed in EXPECTED_SEEDS
        }
        for cipher in EXPECTED_CIPHERS
    }
    uknit_closed = all(
        gate_by_checkpoint["uknit64"][str(seed)] <= CLOSED_GATE
        for seed in EXPECTED_SEEDS
    )
    dialga_active = all(
        gate_by_checkpoint["dialga128"][str(seed)] >= ACTIVE_GATE
        for seed in EXPECTED_SEEDS
    )
    operator_specific = all(
        contribution_operator_margin(results, cipher, seed, split) >= OPERATOR_MARGIN
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    dialga_operator_specific = all(
        contribution_operator_margin(results, "dialga128", seed, split)
        >= OPERATOR_MARGIN
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    row_shuffle_specific = all(
        float(
            results[(cipher, seed, split, "residual_row_shuffle")].get(
                "explained_fraction", 0.0
            )
        )
        >= ROW_SHUFFLE_EXPLAINED
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    research_checks = {
        "uknit_both_learned_gates_effectively_closed": uknit_closed,
        "dialga_both_learned_gates_active": dialga_active,
        "exact_zero_gate_starves_residual_path_gradients": zero_starved,
        "effective_0p05_gate_opens_residual_path_gradients": opened_gradients,
        "all_fresh_residual_contributions_operator_specific": operator_specific,
        "dialga_fresh_residual_contributions_operator_specific": (
            dialga_operator_specific
        ),
        "all_fresh_residual_contributions_sample_specific": row_shuffle_specific,
    }
    protocol_valid = all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1l_protocol_invalid"
        next_action = (
            "repair only the failed K1-L source binding or intervention and rerun unchanged"
        )
    elif uknit_closed and zero_starved and opened_gradients:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1l_uknit_zero_gate_gradient_"
            "starvation_supported"
        )
        next_action = (
            "freeze K1-M to change only the bounded gate-opening schedule on the "
            "same K1-K architecture, data, controls, scale, and seeds"
        )
    elif not uknit_closed:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1l_active_residual_"
            "uknit_signal_not_supported"
        )
        next_action = (
            "stop gate scheduling and test exact heterogeneous S-box/operator "
            "composition as the next single variable"
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1l_mechanism_inconclusive"
        next_action = (
            "hold architecture changes and repair the missing gradient or residual attribution proof"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "gate_by_checkpoint": gate_by_checkpoint,
        "max_full_topology_auc_replay_delta": max(replay_deltas, default=math.inf),
        "thresholds": {
            "closed_gate": CLOSED_GATE,
            "active_gate": ACTIVE_GATE,
            "operator_margin": OPERATOR_MARGIN,
            "row_shuffle_explained": ROW_SHUFFLE_EXPLAINED,
            "zero_gradient_tolerance": ZERO_GRADIENT_TOLERANCE,
            "open_gradient_floor": OPEN_GRADIENT_FLOOR,
        },
        "next_action": next_action,
        "claim_scope": (
            "zero-training post-result mechanism attribution over four frozen K1-K "
            "checkpoints and twelve frozen caches; not a blind confirmatory gate, "
            "formal scale, attack, SOTA, arbitrary-SPN transfer, or uKNIT ceiling"
        ),
        "blocked_actions": [
            "remote scale or more samples, epochs, pairs, seeds, width, or experts",
            "S-box, DDT, trail, key, cipher identity, partial decryption, or raw bypass before the K1-L mechanism decision",
            "treating post-result gate thresholds as blind preregistration",
        ],
    }


def project_features(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    return features.reshape(
        features.shape[0],
        -1,
        2,
        model.runtime_structure.block_bits,
    ).flip(-1)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(logits)
    return torch.sigmoid(tensor).numpy().astype(np.float32, copy=False)


def result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["cipher_key"]),
            int(row["seed"]),
            str(row["split"]),
            str(row["condition"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-L result row: {key}")
        mapped[key] = row
    return mapped


def gradient_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    mapped: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["cipher_key"]),
            int(row["seed"]),
            str(row["gate_condition"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-L gradient row: {key}")
        mapped[key] = row
    return mapped


def contribution_operator_margin(
    rows: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cipher: str,
    seed: int,
    split: str,
) -> float:
    exact = float(
        rows[(cipher, seed, split, "native_full")]["residual_contribution_auc"]
    )
    controls = (
        float(rows[(cipher, seed, split, condition)]["residual_contribution_auc"])
        for condition in ("reversed_full", "corrupted_full", "no_topology_full")
    )
    return min(exact - control for control in controls)


__all__ = [
    "ACTIVE_GATE",
    "AUDIT_CONDITIONS",
    "CLOSED_GATE",
    "EXPECTED_CHECKPOINT_DIGESTS",
    "EXPECTED_GRADIENT_ROWS",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SOURCE_DECISION",
    "EXPECTED_SOURCE_DIGESTS",
    "FULL_TOPOLOGY_CONDITIONS",
    "RUN_ID",
    "adjudicate_k1l",
    "audit_gradient_path",
    "collect_residual_path_outputs",
    "label_blind_row_permutation",
    "residual_metrics",
    "shuffled_residual_logits",
]
