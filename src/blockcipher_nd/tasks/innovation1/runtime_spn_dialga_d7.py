from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    MODELS as D1_MODELS,
    adjudicate_runtime_spn_dialga_d1,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d6 import (
    D6_MODELS,
    EXPECTED_PARAMETER_COUNT,
    SEEDS,
    TOPOLOGY_RESIDUAL_MODE,
)


D1_RUN_ID = "i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725"
D1_PASS_DECISION = "innovation1_dialga_runtime_e4_d1_two_seed_supported"
CORRECT_AUC_FLOOR = 0.520
CONTROL_MARGIN = 0.005
ANCHOR_RETENTION_TOLERANCE = 0.010


def adjudicate_runtime_spn_dialga_d7(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    persisted_d1_gate: dict[str, Any],
    replayed_d1_gate: dict[str, Any],
    d1_validation: dict[str, Any],
    expected_cache_root: str | Path,
    cache_audit: dict[str, Any],
) -> dict[str, Any]:
    by_key = {
        (int(row.get("seed", -1)), str(row.get("model", ""))): row
        for row in rows
    }
    expected_keys = {
        (seed, model) for seed in SEEDS for model in D6_MODELS.values()
    }
    groups = {
        seed: {
            role: by_key.get((seed, model), {})
            for role, model in D6_MODELS.items()
        }
        for seed in SEEDS
    }
    flat_rows = [groups[seed][role] for seed in SEEDS for role in D6_MODELS]
    projected_gate = adjudicate_runtime_spn_dialga_d1(
        run_id=run_id,
        rows=_project_to_d1_contract(groups),
    )
    d1_anchors = _d1_correct_anchors(d1_rows)

    protocol_checks = {
        f"projected_{key}": value
        for key, value in projected_gate["protocol_checks"].items()
    }
    protocol_checks.update(
        {
            "six_exact_e5_rows": len(rows) == 6 and set(by_key) == expected_keys,
            "exact_d7_model_options": _model_options_exact(groups),
            "equal_expected_e5_parameter_geometry": all(
                row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
                and row.get("trainable_parameter_count")
                == EXPECTED_PARAMETER_COUNT
                for row in flat_rows
            ),
            "gated_residual_metadata_complete": all(
                _valid_gate_metadata(row, role=role)
                for seed in SEEDS
                for role, row in groups[seed].items()
            ),
            "exact_d1_cache_root_recorded": all(
                _same_path(
                    row.get("training", {}).get("dataset_cache_root"),
                    expected_cache_root,
                )
                for row in flat_rows
            )
            and all(
                _same_path(
                    row.get("training", {}).get("dataset_cache_root"),
                    expected_cache_root,
                )
                for row in d1_rows
            ),
            "d1_cache_reused_without_generation": cache_audit.get("status")
            == "pass",
            "d1_source_rows_complete": len(d1_rows) == 6
            and set(d1_anchors) == set(SEEDS),
            "d1_source_gate_recomputed_exactly": persisted_d1_gate
            == replayed_d1_gate,
            "d1_source_is_exact_completed_pass": (
                persisted_d1_gate.get("run_id") == D1_RUN_ID
                and persisted_d1_gate.get("status") == "pass"
                and persisted_d1_gate.get("decision") == D1_PASS_DECISION
                and all(persisted_d1_gate.get("protocol_checks", {}).values())
                and all(persisted_d1_gate.get("research_checks", {}).values())
            ),
            "d1_source_validation_passed": (
                d1_validation.get("run_id") == D1_RUN_ID
                and d1_validation.get("status") == "pass"
                and d1_validation.get("checks")
                == persisted_d1_gate.get("protocol_checks")
            ),
        }
    )

    aucs = {
        f"seed{seed}": {
            role: float(groups[seed][role].get("metrics", {}).get("auc", math.nan))
            for role in D6_MODELS
        }
        for seed in SEEDS
    }
    margins = {
        f"seed{seed}": {
            "correct_minus_corrupted": aucs[f"seed{seed}"]["correct"]
            - aucs[f"seed{seed}"]["corrupted"],
            "correct_minus_no_topology": aucs[f"seed{seed}"]["correct"]
            - aucs[f"seed{seed}"]["no_topology"],
            "correct_minus_d1": aucs[f"seed{seed}"]["correct"]
            - d1_anchors.get(seed, math.nan),
        }
        for seed in SEEDS
    }
    research_checks: dict[str, bool] = {}
    for seed in SEEDS:
        seed_key = f"seed{seed}"
        research_checks[f"{seed_key}_correct_auc_at_least_0p520"] = (
            aucs[seed_key]["correct"] >= CORRECT_AUC_FLOOR
        )
        research_checks[f"{seed_key}_correct_exceeds_corrupted_by_0p005"] = (
            margins[seed_key]["correct_minus_corrupted"] >= CONTROL_MARGIN
        )
        research_checks[f"{seed_key}_correct_exceeds_no_topology_by_0p005"] = (
            margins[seed_key]["correct_minus_no_topology"] >= CONTROL_MARGIN
        )
        research_checks[f"{seed_key}_retains_d1_within_0p010"] = (
            margins[seed_key]["correct_minus_d1"]
            >= -ANCHOR_RETENTION_TOLERANCE
        )

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_dialga_runtime_e5_d7_protocol_invalid"
        next_action = "repair only the failed frozen D7 check and rerun unchanged"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_dialga_runtime_e5_d7_r4_regression_supported"
        next_action = (
            "retain Runtime-E5 as a mechanically sound optional architecture, keep "
            "Dialga r5 closed, and prioritize supported Runtime-E4 cross-cipher work"
        )
    else:
        status = "hold"
        decision = "innovation1_dialga_runtime_e5_d7_r4_regression_not_supported"
        next_action = (
            "discard Runtime-E5 gated residuals, keep Runtime-E4 as the supported "
            "runtime backbone, and do not run another Dialga E5 experiment"
        )

    learned_gates = {
        f"seed{seed}": {
            role: {
                "raw": float(groups[seed][role].get("topology_gate_final_raw", math.nan)),
                "bounded": float(
                    groups[seed][role].get("topology_gate_final_bounded", math.nan)
                ),
            }
            for role in D6_MODELS
        }
        for seed in SEEDS
    }
    return {
        "run_id": run_id,
        "task": "innovation1_dialga128_runtime_e5_d7_r4_regression",
        "cipher": "Dialga-128",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "d1_correct_anchors": {
            f"seed{seed}": d1_anchors.get(seed, math.nan) for seed in SEEDS
        },
        "learned_topology_gates": learned_gates,
        "cache_audit": cache_audit,
        "thresholds": {
            "correct_auc": CORRECT_AUC_FLOOR,
            "control_margin": CONTROL_MARGIN,
            "d1_retention_tolerance": ANCHOR_RETENTION_TOLERANCE,
        },
        "claim_scope": (
            "Dialga-128 prefix-r4 two-seed local 2048/class Runtime-E5 "
            "architecture regression only; not evidence that E5 repairs r5, formal "
            "scale, an attack, paper reproduction, SOTA, or universal-SPN support"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "reopen Dialga prefix-r5 Runtime-E5",
            "remote GPU or sample scale-up",
            "change the D1 difference, pairs, keys, epochs, negatives, or metric",
            "claim E5 improves Runtime-E4 from a retention regression",
        ],
    }


def _project_to_d1_contract(
    groups: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for seed in SEEDS:
        for role in D6_MODELS:
            row = deepcopy(groups[seed][role])
            row["model"] = D1_MODELS[role]
            projected.append(row)
    return projected


def _d1_correct_anchors(rows: list[dict[str, Any]]) -> dict[int, float]:
    anchors: dict[int, float] = {}
    for row in rows:
        if row.get("model") != D1_MODELS["correct"]:
            continue
        seed = int(row.get("seed", -1))
        if seed not in SEEDS or seed in anchors:
            return {}
        try:
            auc = float(row["metrics"]["auc"])
        except (KeyError, TypeError, ValueError):
            return {}
        if not math.isfinite(auc):
            return {}
        anchors[seed] = auc
    return anchors


def _model_options_exact(
    groups: dict[int, dict[str, dict[str, Any]]],
) -> bool:
    common = {
        "runtime_structure_path": "configs/runtime/spn/dialga128.json",
        "runtime_round_start": 2,
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet",
        "round_window_mode": "recurrent_window",
        "runtime_structure_window_control": "full",
    }
    for seed in SEEDS:
        for role in D6_MODELS:
            expected = dict(common)
            if role == "corrupted":
                expected["topology_corruption_seed"] = 20260725
            if groups[seed][role].get("training", {}).get("model_options") != expected:
                return False
    return True


def _valid_gate_metadata(row: dict[str, Any], *, role: str) -> bool:
    try:
        initial = float(row.get("topology_gate_initial", math.nan))
        raw = float(row.get("topology_gate_final_raw", math.nan))
        bounded = float(row.get("topology_gate_final_bounded", math.nan))
    except (TypeError, ValueError):
        return False
    metadata_valid = (
        row.get("topology_residual_mode") == TOPOLOGY_RESIDUAL_MODE
        and initial == 0.0
        and math.isfinite(raw)
        and math.isfinite(bounded)
        and abs(math.tanh(raw) - bounded) <= 1e-7
        and abs(bounded) <= 1.0
    )
    return metadata_valid and (
        role != "no_topology" or (raw == 0.0 and bounded == 0.0)
    )


def _same_path(left: Any, right: str | Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


__all__ = [
    "ANCHOR_RETENTION_TOLERANCE",
    "CONTROL_MARGIN",
    "CORRECT_AUC_FLOOR",
    "D1_PASS_DECISION",
    "D1_RUN_ID",
    "adjudicate_runtime_spn_dialga_d7",
]
