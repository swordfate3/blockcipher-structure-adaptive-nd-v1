from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


E85_RUN_ID = (
    "i2_present_gift_r4_topology_parameterized_shared_profile_operator_"
    "readiness_seed0_20260719"
)
E85_DECISION = "innovation2_shared_profile_operator_readiness_passed"
PRESENT_ANCHOR_RUN_ID = (
    "i2_present_r4_r3_only_profile_operator_attribution_seed0_20260718"
)
PRESENT_ANCHOR_DECISION = "innovation2_present_r3_only_neural_gain_attributed"
PRESENT_ANCHOR_TRUE_AUC = 0.9455555555555556
GIFT_ANCHOR_RUN_ID = (
    "i2_gift64_r4_r3_only_profile_operator_attribution_seed0_20260719"
)
GIFT_ANCHOR_DECISION = "innovation2_gift64_r3_only_neural_gain_attributed"
GIFT_ANCHOR_TRUE_AUC = 0.913111342351717
EXPECTED_PARAMETER_COUNT = 4_795


@dataclass(frozen=True)
class SharedProfileAttributionConfig:
    run_id: str
    epochs: int = 30
    batch_size: int = 8
    hidden_dim: int = 32
    steps: int = 2
    seed: int = 0
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.epochs != 30
            or self.batch_size != 8
            or self.hidden_dim != 32
            or self.steps != 2
            or self.seed != 0
            or self.dropout != 0.10
            or self.learning_rate != 1e-3
            or self.weight_decay != 1e-4
            or self.device != "cpu"
        ):
            raise ValueError("E86 seed0 attribution protocol is frozen")

    @property
    def task_name(self) -> str:
        return "innovation2_present_gift_shared_profile_operator_attribution"


def load_e86_sources(
    e85_root: Path,
    present_anchor_root: Path,
    gift_anchor_root: Path,
) -> dict[str, Any]:
    roots = {
        "e85": e85_root,
        "present_anchor": present_anchor_root,
        "gift_anchor": gift_anchor_root,
    }
    payload: dict[str, Any] = {}
    hashes: dict[str, dict[str, str]] = {}
    for name, root in roots.items():
        gate_path = root / "gate.json"
        results_path = root / "results.jsonl"
        payload[name] = {
            "gate": json.loads(gate_path.read_text(encoding="utf-8")),
            "rows": [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
        }
        hashes[name] = {
            "gate.json": _sha256(gate_path),
            "results.jsonl": _sha256(results_path),
        }
    return {"sources": payload, "hashes": hashes}


def validate_e86_sources(bundle: dict[str, Any]) -> dict[str, bool]:
    e85 = bundle["sources"]["e85"]
    present = bundle["sources"]["present_anchor"]
    gift = bundle["sources"]["gift_anchor"]
    e85_gate = e85["gate"]
    present_gate = present["gate"]
    gift_gate = gift["gate"]
    present_true = _row_for_mode(present["rows"], "true")
    gift_true = _row_for_mode(gift["rows"], "true")
    return {
        "e85_run_id_matches": e85_gate.get("run_id") == E85_RUN_ID,
        "e85_status_pass": e85_gate.get("status") == "pass",
        "e85_decision_matches": e85_gate.get("decision") == E85_DECISION,
        "e85_protocol_checks_pass": bool(e85_gate.get("protocol_checks"))
        and all(e85_gate["protocol_checks"].values()),
        "e85_readiness_checks_pass": bool(e85_gate.get("readiness_checks"))
        and all(e85_gate["readiness_checks"].values()),
        "e85_three_rows_present": {
            row.get("relation_mode") for row in e85["rows"]
        }
        == {"independent", "true", "corrupted"},
        "e85_rows_completed_two_epochs": len(e85["rows"]) == 3
        and all(row.get("epochs_completed") == 2 for row in e85["rows"]),
        "present_anchor_run_id_matches": present_gate.get("run_id")
        == PRESENT_ANCHOR_RUN_ID,
        "present_anchor_status_pass": present_gate.get("status") == "pass",
        "present_anchor_decision_matches": present_gate.get("decision")
        == PRESENT_ANCHOR_DECISION,
        "present_anchor_protocol_pass": bool(present_gate.get("protocol_checks"))
        and all(present_gate["protocol_checks"].values()),
        "present_anchor_true_auc_matches": math.isclose(
            float(present_true.get("validation_auc", float("nan"))),
            PRESENT_ANCHOR_TRUE_AUC,
            abs_tol=1e-12,
        ),
        "present_anchor_true_is_30_epoch_seed0": present_true.get(
            "epochs_completed"
        )
        == 30
        and present_true.get("seed") == 0,
        "gift_anchor_run_id_matches": gift_gate.get("run_id")
        == GIFT_ANCHOR_RUN_ID,
        "gift_anchor_status_pass": gift_gate.get("status") == "pass",
        "gift_anchor_decision_matches": gift_gate.get("decision")
        == GIFT_ANCHOR_DECISION,
        "gift_anchor_protocol_pass": bool(gift_gate.get("protocol_checks"))
        and all(gift_gate["protocol_checks"].values()),
        "gift_anchor_true_auc_matches": math.isclose(
            float(gift_true.get("validation_auc", float("nan"))),
            GIFT_ANCHOR_TRUE_AUC,
            abs_tol=1e-12,
        ),
        "gift_anchor_true_is_30_epoch_seed0": gift_true.get("epochs_completed")
        == 30
        and gift_true.get("seed") == 0,
        "source_hashes_present": all(
            len(value) == 64
            for group in bundle["hashes"].values()
            for value in group.values()
        ),
    }


def adjudicate_shared_profile_attribution(
    config: SharedProfileAttributionConfig,
    source_checks: dict[str, bool],
    profile_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    rows = matrix["rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    protocol_checks = {
        **source_checks,
        **profile_checks,
        "output_shapes_are_4x64": set(
            tuple(value) for value in contract["output_shapes"].values()
        )
        == {(4, 64)},
        "masked_losses_match_explicit": max(
            contract["masked_loss_explicit_max_abs_errors"].values()
        )
        <= 1e-7,
        "parameter_counts_are_4795": set(contract["parameter_counts"].values())
        == {EXPECTED_PARAMETER_COUNT},
        "parameter_counts_match": contract["parameter_counts_match"],
        "initial_parameters_match": contract[
            "initial_parameter_max_abs_difference"
        ]
        == 0.0,
        "runtime_topology_state_absent": contract[
            "runtime_topology_state_absent"
        ],
        "cipher_specific_named_state_absent": contract[
            "cipher_specific_named_state_absent"
        ],
        "runtime_topology_changes_logits": min(
            contract["topology_logit_max_abs_differences"].values()
        )
        >= 1e-6,
        "cell_relabel_equivariant": max(
            contract["cell_relabel_max_abs_errors"].values()
        )
        <= 1e-6,
        "all_players_are_valid": contract["true_players_are_distinct"]
        and contract["corrupted_players_are_distinct_from_true"]
        and contract["all_players_are_permutations"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "three_rows_present": set(by_mode)
        == {"independent", "true", "corrupted"},
        "all_rows_completed_30_epochs": len(rows) == 3
        and all(row.get("epochs_completed") == 30 for row in rows),
        "schedule_is_7_plus_14_each_epoch": all(
            len(audit["epoch_batch_counts"]) == 30
            and all(
                counts == {"present": 7, "gift": 14}
                for counts in audit["epoch_batch_counts"]
            )
            for audit in matrix["schedule_audits"].values()
        ),
        "each_row_has_630_updates": all(
            audit["total_updates"] == 630
            for audit in matrix["schedule_audits"].values()
        ),
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key.endswith(("_auc", "_accuracy", "_loss"))
        ),
    }
    candidate_checks: dict[str, bool] = {}
    relation_checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {"rows": rows, "contract": contract}
    anchors = {
        "present": PRESENT_ANCHOR_TRUE_AUC,
        "gift": GIFT_ANCHOR_TRUE_AUC,
    }
    for cipher, anchor in anchors.items():
        true_row = by_mode.get("true", {})
        true_auc = float(true_row.get(f"{cipher}_validation_auc", 0.0))
        train_auc = float(true_row.get(f"{cipher}_train_auc", 0.0))
        independent_auc = float(
            by_mode.get("independent", {}).get(f"{cipher}_validation_auc", 0.0)
        )
        corrupted_auc = float(
            by_mode.get("corrupted", {}).get(f"{cipher}_validation_auc", 0.0)
        )
        candidate_checks[f"{cipher}_true_within_0p03_of_anchor"] = (
            true_auc >= anchor - 0.03
        )
        candidate_checks[f"{cipher}_train_validation_gap_at_most_0p15"] = (
            train_auc - true_auc <= 0.15
        )
        relation_checks[f"{cipher}_true_minus_independent_at_least_0p03"] = (
            true_auc - independent_auc >= 0.03
        )
        relation_checks[f"{cipher}_true_minus_corrupted_at_least_0p03"] = (
            true_auc - corrupted_auc >= 0.03
        )
        metrics[f"{cipher}_true_minus_anchor"] = true_auc - anchor
        metrics[f"{cipher}_true_minus_independent"] = true_auc - independent_auc
        metrics[f"{cipher}_true_minus_corrupted"] = true_auc - corrupted_auc
        metrics[f"{cipher}_true_train_validation_gap"] = train_auc - true_auc
    macro_anchor = (PRESENT_ANCHOR_TRUE_AUC + GIFT_ANCHOR_TRUE_AUC) / 2.0
    macro_true = float(by_mode.get("true", {}).get("macro_validation_auc", 0.0))
    candidate_checks["macro_true_within_0p03_of_separate_anchors"] = (
        macro_true >= macro_anchor - 0.03
    )
    metrics.update(
        {
            "macro_separate_anchor_auc": macro_anchor,
            "macro_true_auc": macro_true,
            "macro_true_minus_separate_anchor": macro_true - macro_anchor,
            "shared_parameter_count": EXPECTED_PARAMETER_COUNT,
            "separate_parameter_count": 2 * EXPECTED_PARAMETER_COUNT,
        }
    )

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_shared_profile_operator_attribution_protocol_invalid"
        action = "repair E85/anchor sources, 30-epoch schedule, topology, or artifacts"
    elif not all(candidate_checks.values()):
        status = "hold"
        decision = "innovation2_shared_profile_operator_quality_not_retained"
        action = "retain separate E73/E79 models and close shared-parameter branch"
    elif not all(relation_checks.values()):
        status = "hold"
        decision = "innovation2_shared_profile_operator_topology_not_attributed"
        action = "retain separate E73/E79 models and close shared-parameter branch"
    else:
        status = "pass"
        decision = "innovation2_shared_profile_operator_seed0_attributed"
        action = "run E87 identical 30-epoch shared matrix with seed1"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "candidate_checks": candidate_checks,
        "relation_checks": relation_checks,
        "metrics": metrics,
        "claim_scope": (
            "30-epoch local seed0 attribution of one 4,795-parameter topology-"
            "parameterized profile operator jointly trained on PRESENT-80 r4 and "
            "GIFT-64 r4 strict unit-balance profiles; no two-seed, zero-shot, "
            "unseen-cipher, high-round, attack, remote-scale, novelty, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "seed1": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: SharedProfileAttributionConfig) -> dict[str, Any]:
    return asdict(config)


def _row_for_mode(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("relation_mode") == mode), {})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "GIFT_ANCHOR_TRUE_AUC",
    "PRESENT_ANCHOR_TRUE_AUC",
    "SharedProfileAttributionConfig",
    "adjudicate_shared_profile_attribution",
    "load_e86_sources",
    "serializable_config",
    "validate_e86_sources",
]
