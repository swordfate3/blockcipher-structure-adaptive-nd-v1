from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.present_balance_profile_operator import (
    PresentBalanceProfileOperator,
    PresentBalanceProfileOperatorSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    anf_prefix_features,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)
from blockcipher_nd.training.metrics import binary_auc


SOURCE_RUN_ID = "i2_present_r4_unit_balance_profile_readiness_20260718"
SOURCE_DECISION = "innovation2_present_unit_balance_profile_prefix_ready"
AUDIT_EPOCHS = 2
AUDIT_BATCH_SIZE = 8
AUDIT_HIDDEN_DIM = 32
AUDIT_STEPS = 2
AUDIT_SEED = 0


@dataclass(frozen=True)
class ProfileOperatorReadinessConfig:
    run_id: str
    mode: str = "readiness"
    epochs: int = AUDIT_EPOCHS
    batch_size: int = AUDIT_BATCH_SIZE
    hidden_dim: int = AUDIT_HIDDEN_DIM
    steps: int = AUDIT_STEPS
    seed: int = AUDIT_SEED
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "readiness"}:
            raise ValueError("mode must be smoke or readiness")
        if min(self.epochs, self.batch_size, self.hidden_dim, self.steps) <= 0:
            raise ValueError("training dimensions must be positive")
        if self.mode == "readiness" and (
            self.epochs != AUDIT_EPOCHS
            or self.batch_size != AUDIT_BATCH_SIZE
            or self.hidden_dim != AUDIT_HIDDEN_DIM
            or self.steps != AUDIT_STEPS
            or self.seed != AUDIT_SEED
            or self.dropout != 0.10
            or self.device != "cpu"
        ):
            raise ValueError("E66 readiness protocol is frozen")


def load_profile_operator_sources(
    profile_root: Path, atlas_root: Path
) -> dict[str, Any]:
    profile_gate = json.loads((profile_root / "gate.json").read_text(encoding="utf-8"))
    profile_metadata = json.loads(
        (profile_root / "metadata.json").read_text(encoding="utf-8")
    )
    profile_targets = np.load(profile_root / "profile_targets.npy")
    profile_observed = np.load(profile_root / "profile_observed.npy")
    with (profile_root / "matched_unit_contrast.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        matched_rows = list(csv.DictReader(handle))
    with (profile_root / "features.csv").open(encoding="utf-8", newline="") as handle:
        observed_features = list(csv.DictReader(handle))
    atlas_gate = json.loads((atlas_root / "gate.json").read_text(encoding="utf-8"))
    structures_payload = json.loads(
        (atlas_root / "structures.json").read_text(encoding="utf-8")
    )
    structures = structures_payload["structures"]
    prefix_features = build_full_prefix_features(structures)
    parsed_rows = [
        {
            "split": row["split"],
            "structure_index": int(row["structure_index"]),
            "output_bit": int(row["mask_index"]),
            "label": int(row["label"]),
        }
        for row in matched_rows
    ]
    return {
        "profile_gate": profile_gate,
        "profile_metadata": profile_metadata,
        "profile_targets": np.asarray(profile_targets, dtype=np.int8),
        "profile_observed": np.asarray(profile_observed, dtype=np.bool_),
        "matched_rows": parsed_rows,
        "observed_features": observed_features,
        "atlas_gate": atlas_gate,
        "structures": structures,
        "prefix_features": prefix_features,
        "source_hashes": {
            "profile": {
                name: _sha256(profile_root / name)
                for name in (
                    "gate.json",
                    "metadata.json",
                    "profile_targets.npy",
                    "profile_observed.npy",
                    "matched_unit_contrast.csv",
                    "features.csv",
                )
            },
            "atlas": {
                name: _sha256(atlas_root / name)
                for name in ("gate.json", "structures.json")
            },
        },
    }


def build_full_prefix_features(structures: list[dict[str, Any]]) -> np.ndarray:
    features = np.empty((len(structures), 64, 39), dtype=np.float64)
    for structure in structures:
        index = int(structure["index"])
        active_bits = tuple(int(bit) for bit in structure["active_bits"])
        supports = {
            rounds: possible_active_monomials(active_bits, rounds)
            for rounds in (1, 2, 3)
        }
        for output_bit in range(64):
            features[index, output_bit] = anf_prefix_features(
                np.asarray([output_bit], dtype=np.int64), supports
            )
    return features


def validate_profile_operator_sources(
    sources: dict[str, Any], *, strict: bool
) -> dict[str, bool | float]:
    rows = sources["matched_rows"]
    train_structures = {
        row["structure_index"] for row in rows if row["split"] == "train"
    }
    validation_structures = {
        row["structure_index"] for row in rows if row["split"] == "validation"
    }
    replay_errors = []
    for row in sources["observed_features"]:
        structure = int(row["structure_index"])
        output_bit = int(row["output_bit"])
        expected = np.asarray(
            [float(row[f"anf_prefix_{column:02d}"]) for column in range(39)],
            dtype=np.float64,
        )
        replay_errors.append(
            float(
                np.max(
                    np.abs(sources["prefix_features"][structure, output_bit] - expected)
                )
            )
        )
    replay_error = max(replay_errors, default=math.inf)
    checks: dict[str, bool | float] = {
        "profile_run_id_matches": sources["profile_gate"].get("run_id")
        == SOURCE_RUN_ID,
        "profile_decision_matches": sources["profile_gate"].get("decision")
        == SOURCE_DECISION,
        "profile_status_pass": sources["profile_gate"].get("status") == "pass",
        "atlas_decision_ready": sources["atlas_gate"].get("decision")
        == "innovation2_present_universal_balance_atlas_ready",
        "profile_shape_is_96x64": sources["profile_targets"].shape == (96, 64),
        "observed_shape_is_96x64": sources["profile_observed"].shape == (96, 64),
        "prefix_shape_is_96x64x39": sources["prefix_features"].shape
        == (96, 64, 39),
        "prefix_features_finite": bool(np.isfinite(sources["prefix_features"]).all()),
        "observed_edges_are_476": int(np.sum(sources["profile_observed"])) == 476,
        "matched_rows_are_476": len(rows) == 476,
        "train_validation_structures_disjoint": train_structures.isdisjoint(
            validation_structures
        ),
        "observed_prefix_replay_at_most_1e12": replay_error <= 1e-12,
        "source_hashes_present": all(
            len(value) == 64
            for group in sources["source_hashes"].values()
            for value in group.values()
        ),
        "observed_prefix_replay_max_abs_error": replay_error,
    }
    if strict:
        checks.update(
            {
                "train_structures_are_50": len(train_structures) == 50,
                "validation_structures_are_18": len(validation_structures) == 18,
                "structure_count_is_96": len(sources["structures"]) == 96,
            }
        )
    return checks


def measure_profile_operator_contract(
    config: ProfileOperatorReadinessConfig, sources: dict[str, Any]
) -> dict[str, Any]:
    true_player = _present_player()
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    models = {
        mode: _make_model(config, mode, true_player, corrupted_player, dropout=0.0)
        for mode in ("independent", "true", "corrupted")
    }
    _copy_parameters(models["independent"], models["true"])
    _copy_parameters(models["independent"], models["corrupted"])
    contract_indices = sorted(
        {int(row["structure_index"]) for row in sources["matched_rows"]}
    )[:4]
    features = torch.from_numpy(
        sources["prefix_features"][contract_indices].astype(np.float32)
    )
    targets = torch.from_numpy(
        sources["profile_targets"][contract_indices].astype(np.float32)
    )
    observed = torch.from_numpy(sources["profile_observed"][contract_indices])
    logits = models["true"](features)
    loss = masked_binary_cross_entropy(logits, targets, observed)
    explicit_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[observed], targets[observed]
    )
    loss.backward()
    with torch.no_grad():
        true_logits = models["true"](features)
        corrupted_logits = models["corrupted"](features)
    permutation = _cell_permutation()
    relabeled_player = _conjugate_player(true_player, permutation)
    relabeled = _make_model(config, "true", relabeled_player, relabeled_player, dropout=0.0)
    _copy_parameters(models["true"], relabeled)
    permuted_features = torch.empty_like(features)
    permuted_features[:, permutation] = features
    with torch.no_grad():
        original = models["true"](features)
        permuted = relabeled(permuted_features)
    expected = torch.empty_like(original)
    expected[:, permutation] = original
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    forbidden = ("certificate", "witness", "parity", "full_cube", "label")
    return {
        "output_shape": list(logits.shape),
        "masked_loss_explicit_max_abs_error": float(
            torch.abs(loss - explicit_loss).detach()
        ),
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
        "true_corrupted_logit_max_abs_difference": float(
            torch.max(torch.abs(true_logits - corrupted_logits))
        ),
        "cell_relabel_max_abs_error": float(torch.max(torch.abs(permuted - expected))),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in models["true"].parameters()
        ),
        "forbidden_named_state_absent": not any(
            token in name
            for name in models["true"].state_dict()
            for token in forbidden
        ),
        "shared_steps": config.steps,
        "prefix_input_dim": 39,
    }


def masked_binary_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    if logits.shape != targets.shape or observed.shape != targets.shape:
        raise ValueError("logits, targets, and observed must share shape")
    if not bool(observed.any()):
        raise ValueError("masked loss requires at least one observed target")
    return nn.functional.binary_cross_entropy_with_logits(
        logits[observed], targets[observed]
    )


def train_profile_operator_matrix(
    config: ProfileOperatorReadinessConfig,
    sources: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    rows = sources["matched_rows"]
    train_indices = sorted(
        {row["structure_index"] for row in rows if row["split"] == "train"}
    )
    validation_indices = sorted(
        {row["structure_index"] for row in rows if row["split"] == "validation"}
    )
    true_player = _present_player()
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    trained = []
    histories = []
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for mode in ("independent", "true", "corrupted"):
        _seed_everything(config.seed)
        model = _make_model(config, mode, true_player, corrupted_player).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best: dict[str, Any] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(1, config.epochs + 1):
            model.train()
            generator = torch.Generator().manual_seed(config.seed + epoch)
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            for start in range(0, len(order), config.batch_size):
                batch = [train_indices[order[index]] for index in range(start, min(start + config.batch_size, len(order)))]
                features, targets, observed = _batch_tensors(sources, batch, config.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = masked_binary_cross_entropy(logits, targets, observed)
                loss.backward()
                optimizer.step()
            train_metrics = _evaluate(model, sources, train_indices, config)
            validation_metrics = _evaluate(model, sources, validation_indices, config)
            history = {
                "row_id": f"profile_{mode}_seed{config.seed}",
                "relation_mode": mode,
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            histories.append(history)
            if best is None or validation_metrics["auc"] > best["validation_auc"]:
                best = {
                    "run_id": config.run_id,
                    "task": "innovation2_present_balance_profile_operator_readiness",
                    "row_id": history["row_id"],
                    "relation_mode": mode,
                    "seed": config.seed,
                    "best_epoch": epoch,
                    "epochs_completed": epoch,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        if best is None or best_state is None:
            raise RuntimeError("profile operator training produced no checkpoint")
        best["epochs_completed"] = config.epochs
        torch.save(best_state, checkpoints / f"{best['row_id']}.pt")
        trained.append(best)
    return {"trained_rows": trained, "history_rows": histories}


def adjudicate_profile_operator_readiness(
    config: ProfileOperatorReadinessConfig,
    source_checks: dict[str, bool | float],
    contract: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    boolean_source_checks = {
        key: value for key, value in source_checks.items() if isinstance(value, bool)
    }
    protocol_checks = {
        **boolean_source_checks,
        "output_shape_is_4x64": contract["output_shape"] == [4, 64],
        "masked_loss_matches_explicit": contract[
            "masked_loss_explicit_max_abs_error"
        ]
        <= 1e-7,
        "parameter_counts_match": contract["parameter_counts_match"],
        "true_corrupted_logits_differ": contract[
            "true_corrupted_logit_max_abs_difference"
        ]
        >= 1e-6,
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "forbidden_named_state_absent": contract["forbidden_named_state_absent"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "all_three_rows_present": set(by_mode) == {"independent", "true", "corrupted"},
        "all_rows_completed_two_epochs": len(rows) == 3
        and all(row["epochs_completed"] == config.epochs for row in rows),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "train_auc",
                "train_accuracy",
                "train_loss",
                "validation_auc",
                "validation_accuracy",
                "validation_loss",
            )
        ),
    }
    optimization_checks = {
        "independent_validation_auc_in_range": 0.55
        <= float(by_mode.get("independent", {}).get("validation_auc", 0.0))
        <= 0.95,
        "true_validation_auc_in_range": 0.55
        <= float(by_mode.get("true", {}).get("validation_auc", 0.0))
        <= 0.95,
        "corrupted_validation_auc_in_range": 0.35
        <= float(by_mode.get("corrupted", {}).get("validation_auc", 0.0))
        <= 0.95,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_profile_operator_protocol_invalid"
        action = "repair source replay, model contract, masked loss, or training protocol"
    elif not all(optimization_checks.values()):
        status = "hold"
        decision = "innovation2_present_profile_operator_optimization_not_ready"
        action = "stop formal training; two-epoch safe-prefix learning is not ready"
    else:
        status = "pass"
        decision = "innovation2_present_profile_operator_readiness_passed"
        action = "prepare E67 30-epoch seed0 independent/true/corrupted attribution"
    metrics = {
        "e65_prefix_ridge_validation_auc": 0.7936111111111112,
        "rows": rows,
        "true_minus_independent": float(by_mode.get("true", {}).get("validation_auc", 0.0))
        - float(by_mode.get("independent", {}).get("validation_auc", 0.0)),
        "true_minus_corrupted": float(by_mode.get("true", {}).get("validation_auc", 0.0))
        - float(by_mode.get("corrupted", {}).get("validation_auc", 0.0)),
        "contract": contract,
    }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "optimization_checks": optimization_checks,
        "metrics": metrics,
        "claim_scope": (
            "two-epoch local readiness of a prefix-guided 64-node profile operator "
            "on PRESENT-80 r4 strict unit-balance labels; no performance, high-round, "
            "new-attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "remote_scale": False,
        },
    }


def _make_model(
    config: ProfileOperatorReadinessConfig,
    mode: str,
    true_player: np.ndarray,
    corrupted_player: np.ndarray,
    *,
    dropout: float | None = None,
) -> PresentBalanceProfileOperator:
    player = corrupted_player if mode == "corrupted" else true_player
    return PresentBalanceProfileOperator(
        PresentBalanceProfileOperatorSpec(
            input_dim=39,
            hidden_dim=config.hidden_dim,
            steps=config.steps,
            dropout=config.dropout if dropout is None else dropout,
            relation_mode=mode,
        ),
        torch.from_numpy(np.asarray(player, dtype=np.int64)),
    )


def _evaluate(
    model: PresentBalanceProfileOperator,
    sources: dict[str, Any],
    indices: list[int],
    config: ProfileOperatorReadinessConfig,
) -> dict[str, float]:
    model.eval()
    all_logits = []
    all_targets = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start : start + config.batch_size]
            features, targets, observed = _batch_tensors(sources, batch, config.device)
            logits = model(features)
            losses.append(float(masked_binary_cross_entropy(logits, targets, observed)))
            all_logits.append(logits[observed].cpu().numpy())
            all_targets.append(targets[observed].cpu().numpy())
    scores = np.concatenate(all_logits)
    labels = np.concatenate(all_targets)
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    return {
        "auc": float(binary_auc(labels.astype(np.float32), scores.astype(np.float64))),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "loss": float(np.mean(losses)),
    }


def _batch_tensors(
    sources: dict[str, Any], indices: list[int], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(
        sources["prefix_features"][indices].astype(np.float32)
    ).to(device)
    targets = torch.from_numpy(
        sources["profile_targets"][indices].astype(np.float32)
    ).to(device)
    observed = torch.from_numpy(sources["profile_observed"][indices]).to(device)
    return features, targets, observed


def _copy_parameters(source: nn.Module, target: nn.Module) -> None:
    target_parameters = dict(target.named_parameters())
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            target_parameters[name].copy_(parameter)


def _present_player() -> np.ndarray:
    return np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)], dtype=np.int64
    )


def _cell_permutation() -> np.ndarray:
    cell_order = np.asarray([5, 2, 13, 0, 8, 15, 1, 10, 6, 12, 3, 14, 9, 4, 11, 7])
    permutation = np.empty(64, dtype=np.int64)
    for source_cell, target_cell in enumerate(cell_order):
        for lane in range(4):
            permutation[4 * source_cell + lane] = 4 * int(target_cell) + lane
    return permutation


def _conjugate_player(player: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    relabeled = np.empty_like(player)
    for source in range(64):
        relabeled[permutation[source]] = permutation[player[source]]
    return relabeled


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def serializable_config(config: ProfileOperatorReadinessConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ProfileOperatorReadinessConfig",
    "adjudicate_profile_operator_readiness",
    "build_full_prefix_features",
    "load_profile_operator_sources",
    "masked_binary_cross_entropy",
    "measure_profile_operator_contract",
    "serializable_config",
    "train_profile_operator_matrix",
    "validate_profile_operator_sources",
]
