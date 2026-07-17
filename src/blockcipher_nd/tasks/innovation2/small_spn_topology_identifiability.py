from __future__ import annotations

from typing import Any

import numpy as np


MINIMUM_COUNTS = {
    "selected_cells": 512,
    "train_p_sensitive_any_s": 192,
    "train_p_sensitive_all_s": 64,
    "heldout_p3_novel_any_s": 192,
    "heldout_p3_novel_all_s": 64,
    "dual_p_effect_cells": 128,
    "dual_p_effect_positive": 32,
    "dual_p_effect_negative": 32,
    "train_interaction_cells": 128,
    "full_interaction_cells": 192,
}


def topology_identifiability_metrics(
    labels: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    matrix = np.asarray(labels, dtype=np.bool_)
    selected_mask = np.asarray(selected, dtype=np.bool_)
    if matrix.shape[0] != 16:
        raise ValueError("labels must contain 16 ordered S-box/P-layer variants")
    if matrix.shape[1:] != selected_mask.shape:
        raise ValueError("selected mask must match the label cell shape")
    cube = matrix.reshape(4, 4, *matrix.shape[1:])[:, :, selected_mask]
    selected_cells = int(selected_mask.sum())

    train = cube[:3, :3]
    train_p_var_by_s = np.any(train != train[:, :1], axis=1)
    train_s_var_by_p = np.any(train != train[:1], axis=0)
    heldout_p3_novel_by_s = np.any(
        cube[:3, 3][:, None, :] != cube[:3, :3], axis=1
    )
    dual_p_effect = np.any(cube[3, 3][None, :] != cube[3, :3], axis=0)

    train_interaction = _interaction_mask(train)
    full_interaction = _interaction_mask(cube)
    dual_target = cube[3, 3]
    dual_effect_target = dual_target[dual_p_effect]
    counts = {
        "selected_cells": selected_cells,
        "train_p_sensitive_any_s": int(np.any(train_p_var_by_s, axis=0).sum()),
        "train_p_sensitive_all_s": int(np.all(train_p_var_by_s, axis=0).sum()),
        "train_s_sensitive_any_p": int(np.any(train_s_var_by_p, axis=0).sum()),
        "train_s_sensitive_all_p": int(np.all(train_s_var_by_p, axis=0).sum()),
        "heldout_p3_novel_any_s": int(
            np.any(heldout_p3_novel_by_s, axis=0).sum()
        ),
        "heldout_p3_novel_all_s": int(
            np.all(heldout_p3_novel_by_s, axis=0).sum()
        ),
        "dual_p_effect_cells": int(dual_p_effect.sum()),
        "dual_p_effect_positive": int(dual_effect_target.sum()),
        "dual_p_effect_negative": int(len(dual_effect_target) - dual_effect_target.sum()),
        "train_interaction_cells": int(train_interaction.sum()),
        "full_interaction_cells": int(full_interaction.sum()),
    }
    fractions = {
        key: float(value / selected_cells) if selected_cells else 0.0
        for key, value in counts.items()
        if key != "selected_cells" and not key.startswith("dual_p_effect_")
    }
    fractions["dual_p_effect_cells"] = (
        float(counts["dual_p_effect_cells"] / selected_cells)
        if selected_cells
        else 0.0
    )
    per_round: list[dict[str, Any]] = []
    selected_positions = np.argwhere(selected_mask)
    for round_index in range(matrix.shape[1]):
        cell_positions = selected_positions[:, 0] == round_index
        total = int(cell_positions.sum())
        per_round.append(
            {
                "round_index": round_index,
                "selected_cells": total,
                "train_p_sensitive_any_s": int(
                    np.any(train_p_var_by_s, axis=0)[cell_positions].sum()
                ),
                "dual_p_effect_cells": int(dual_p_effect[cell_positions].sum()),
                "full_interaction_cells": int(full_interaction[cell_positions].sum()),
            }
        )
    return {"counts": counts, "fractions": fractions, "per_round": per_round}


def adjudicate_topology_identifiability(
    *,
    run_id: str,
    metrics: dict[str, Any],
    readiness: dict[str, bool],
) -> dict[str, Any]:
    counts = metrics["counts"]
    width_checks = {
        f"{key}_at_least_{minimum}": int(counts[key]) >= minimum
        for key, minimum in MINIMUM_COUNTS.items()
    }
    if not all(readiness.values()):
        status = "fail"
        decision = "innovation2_small_spn_topology_label_audit_protocol_invalid"
        action = "repair source ownership, shape, train-only selection, or variant ordering"
    elif all(width_checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_topology_labels_identifiable"
        action = "design a train-only P-sensitive interaction benchmark before any new model"
    else:
        status = "hold"
        decision = "innovation2_small_spn_topology_labels_not_identifiable"
        action = "stop this synthetic label route and return to provider or target design"
    return {
        "run_id": run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "width_checks": width_checks,
        "thresholds": MINIMUM_COUNTS,
        "metrics": metrics,
        "claim_scope": (
            "deterministic identifiability audit of frozen exact labels on a 16-bit "
            "synthetic SPN family; no neural training or real-cipher claim"
        ),
        "next_action": {"action": action, "training": False, "remote_scale": False},
    }


def _interaction_mask(cube: np.ndarray) -> np.ndarray:
    base = cube[0, 0]
    mixed = cube ^ cube[:, :1] ^ cube[:1, :] ^ base
    return np.any(mixed, axis=(0, 1))
