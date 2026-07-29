from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.models.structure.spn.cell_joint_gf2_operator_response import (
    cell_joint_response_feature_dim,
    extract_cell_joint_gf2_operator_features,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_exact_gf2_operator_response_k1bh import (
    EXPECTED_FEATURE_ROWS,
    EXPECTED_FRESH_ROWS,
    EXPECTED_PAIRS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
    EXPECTED_TRAIN_ROWS,
    FEATURE_BATCH_SIZE,
    FRESH_SPLITS,
    REPLICAS,
    RESULT_CONDITIONS,
    adjudicate_k1bh,
    evaluate_k1bh,
    load_and_validate_config as load_k1bh_config,
    load_authority as load_k1bh_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_cell_joint_gf2_operator_response_k1bi_audit_"
    "replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_cell_joint_gf2_operator_response_"
    "k1bi_audit_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "65d1576a25091a14beee6be394b89c8007250e75d30271a0c0c9d967050534f0"
)


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BI config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BI identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_cell_joint_gf2_operator_response_k1bi_audit"
    ):
        raise ValueError("K1-BI experiment name drifted")
    if config.get("feature") != {
        "views": [
            "raw",
            "inverse_linear_0",
            "inverse_linear_1",
            "composed_1_then_0",
        ],
        "raw_channels": ["left", "right", "left_xor_right"],
        "cell_value_categories": 16,
        "pair_reduction": "categorical_histogram_mean",
        "runtime_cell_membership_used": True,
        "runtime_bit_role_used": True,
        "cell_position_and_view_order_preserved": True,
        "sbox_semantics_used": False,
        "expected_feature_dims": {
            "uknit64": 3072,
            "midori64": 3072,
            "dialga128": 6144,
        },
    }:
        raise ValueError("K1-BI feature contract drifted")
    probe = config.get("probe", {})
    if (
        probe.get("family") != "diagonal_fisher"
        or probe.get("variance_floor") != 1e-6
        or probe.get("fit_condition") != "correct_operator"
        or probe.get("counterfactuals_reuse_correct_fit") is not True
        or probe.get("fit_rows") != EXPECTED_TRAIN_ROWS
        or probe.get("fresh_rows") != EXPECTED_FRESH_ROWS
    ):
        raise ValueError("K1-BI probe contract drifted")
    if config.get("evaluation") != {
        "replicas": list(REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "conditions": list(RESULT_CONDITIONS),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "expected_feature_rows": EXPECTED_FEATURE_ROWS,
        "expected_scorer_rows": EXPECTED_SCORER_ROWS,
        "pairs_per_sample": EXPECTED_PAIRS,
        "neural_training_performed": False,
        "optimizer_steps": 0,
        "data_generation": False,
        "device": "cpu",
        "execution": "local_audit",
    }:
        raise ValueError("K1-BI evaluation contract drifted")
    if config.get("gates") != {
        "correct_operator_auc_min": 0.55,
        "correct_minus_identity_auc_min": 0.01,
        "correct_minus_each_wrong_operator_auc_min": 0.01,
        "correct_minus_label_shuffle_auc_min": 0.03,
        "label_shuffle_auc_min": 0.47,
        "label_shuffle_auc_max": 0.53,
        "midori_dialga_anchor_max_auc_drop": 0.02,
        "minimum_nonzero_response_rms": 0.0,
        "require_every_replica_cipher_fresh_split": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BI gate contract drifted")
    expected_shuffle_keys = {f"replica{replica}" for replica in REPLICAS}
    shuffle_seeds = probe.get("label_shuffle_seeds", {})
    if set(shuffle_seeds) != expected_shuffle_keys or any(
        set(shuffle_seeds[f"replica{replica}"]) != set(EXPECTED_CIPHERS)
        for replica in REPLICAS
    ):
        raise ValueError("K1-BI label-shuffle seed contract drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    list[dict[str, Any]],
    Mapping[tuple[str, int, str], Any],
    Mapping[str, RuntimeSpnStructure],
    Mapping[str, RuntimeSpnStructure],
    Mapping[str, RuntimeSpnStructure],
    dict[str, Any],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    source_config_path = project_root / str(source["config"])
    source_config = load_k1bh_config(source_config_path)
    (
        dataset_rows,
        datasets,
        structures,
        corrupted_structures,
        cross_operators,
        inherited_checks,
    ) = load_k1bh_authority(
        source_config,
        project_root=project_root,
        device=device,
    )
    paths = {name: source_root / name for name in source["digests"]}
    source_gate = _read_json(paths["gate.json"])
    source_validation = _read_json(paths["validation.json"])
    source_results = _read_jsonl(paths["results.jsonl"])
    source_features = _read_jsonl(paths["feature_manifest.jsonl"])
    source_scorers = _read_jsonl(paths["scorers.jsonl"])
    source_summary = _read_json(paths["summary.json"])
    source_datasets = _read_jsonl(paths["dataset_manifest.jsonl"])
    checks = {
        "source_config_digest_exact": (
            file_sha256(source_config_path) == source["config_sha256"]
        ),
        "all_seven_k1bh_artifact_digests_exact": len(paths) == 7
        and all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        ),
        "k1bh_clean_hold_requires_cell_joint_representation": (
            source_gate.get("status") == source["required_status"]
            and source_gate.get("decision") == source["required_decision"]
            and not source_gate.get("failed_protocol_checks")
            and source_gate.get("diagnostic_checks", {}).get(
                "label_shuffle_auc_within_symmetric_chance_band"
            )
            is False
        ),
        "k1bh_validation_passed": (
            source_validation.get("status") == "pass"
            and not source_validation.get("errors")
        ),
        "k1bh_artifact_rows_complete": (
            len(source_results) == EXPECTED_RESULT_ROWS
            and len(source_features) == EXPECTED_FEATURE_ROWS
            and len(source_scorers) == EXPECTED_SCORER_ROWS
            and len(source_datasets) == 18
            and source_summary.get("decision") == source["required_decision"]
        ),
        **{f"k1bh_{name}": bool(value) for name, value in inherited_checks.items()},
    }
    return (
        dataset_rows,
        datasets,
        structures,
        corrupted_structures,
        cross_operators,
        source_gate,
        checks,
    )


def evaluate_k1bi(
    *,
    config: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, RuntimeSpnStructure],
    corrupted_structures: Mapping[str, RuntimeSpnStructure],
    cross_operators: Mapping[str, RuntimeSpnStructure],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows, scorer_rows, result_rows = evaluate_k1bh(
        config=config,
        dataset_rows=dataset_rows,
        datasets=datasets,
        structures=structures,
        corrupted_structures=corrupted_structures,
        cross_operators=cross_operators,
        batch_size=FEATURE_BATCH_SIZE,
        feature_extractor=extract_cell_joint_gf2_operator_features,
        feature_dim=lambda structure: cell_joint_response_feature_dim(
            structure.cells
        ),
    )
    for row in (*feature_rows, *scorer_rows, *result_rows):
        row["run_id"] = RUN_ID
        row["representation"] = "runtime_cell_joint_16_value_histogram"
    return feature_rows, scorer_rows, result_rows


def adjudicate_k1bi(
    *,
    config: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_gate: Mapping[str, Any],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    compatibility_config = {
        **config,
        "gates": {
            **config["gates"],
            "label_shuffle_auc_max": config["gates"]["label_shuffle_auc_max"],
        },
    }
    base = adjudicate_k1bh(
        config=compatibility_config,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
        source_checks=source_checks,
    )
    protocol_checks = dict(base["protocol_checks"])
    protocol_checks["config_digest_exact"] = (
        file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256
    )
    panels = list(base["panels"])
    research_checks = {
        name: bool(value)
        for name, value in base["research_checks"].items()
        if not name.endswith("_label_shuffle_near_chance")
    }
    panel_map = {
        (int(panel["replica"]), str(panel["cipher_key"]), str(panel["split"])): panel
        for panel in panels
    }
    source_panel_map = {
        (int(panel["replica"]), str(panel["cipher_key"]), str(panel["split"])): panel
        for panel in source_gate.get("panels", [])
    }
    for key, panel in panel_map.items():
        replica, cipher, split = key
        prefix = f"replica{replica}_{cipher}_{split}"
        research_checks[f"{prefix}_label_shuffle_in_symmetric_chance_band"] = (
            float(config["gates"]["label_shuffle_auc_min"])
            <= float(panel["label_shuffle_auc"])
            <= float(config["gates"]["label_shuffle_auc_max"])
        )
        if cipher in {"midori64", "dialga128"}:
            anchor = source_panel_map.get(key)
            research_checks[f"{prefix}_retains_k1bh_anchor"] = (
                anchor is not None
                and float(panel["correct_auc"])
                >= float(anchor["correct_auc"])
                - float(config["gates"]["midori_dialga_anchor_max_auc_drop"])
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    correct_signal_all = bool(panels) and all(
        float(panel["correct_auc"])
        >= float(config["gates"]["correct_operator_auc_min"])
        for panel in panels
    )
    shuffle_symmetric_all = bool(panels) and all(
        float(config["gates"]["label_shuffle_auc_min"])
        <= float(panel["label_shuffle_auc"])
        <= float(config["gates"]["label_shuffle_auc_max"])
        for panel in panels
    )
    anchor_retained_all = all(
        value
        for name, value in research_checks.items()
        if name.endswith("_retains_k1bh_anchor")
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_k1bi_protocol_invalid"
        next_action = (
            "Repair only the failed K1-BH binding, cell reconstruction, scorer reuse "
            "or artifact invariant and rerun the frozen audit."
        )
    elif not shuffle_symmetric_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bi_shuffle_attribution_not_supported"
        next_action = (
            "Hold architecture work. Preregister an orientation-invariant shuffled-label "
            "null with multiple frozen permutations on the identical cell-joint features; "
            "do not tune the representation, difference, pair count or data budget."
        )
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1bi_cell_joint_topology_signal_supported"
        )
        next_action = (
            "Preregister a shared position-preserving neural residual that consumes exact "
            "transported native-cell categorical tokens, with K1-BI and K1-BH as frozen "
            "mechanism anchors and the same operator controls."
        )
    elif not correct_signal_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bi_cell_joint_signal_unstable"
        next_action = (
            "Stop the linear-only response route. Bind the already-supported runtime "
            "S-box-aware five-stage cell statistic as the next family primitive; do not "
            "return to edge-message pooling or mechanical scale-up."
        )
    elif not anchor_retained_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bi_anchor_regression"
        next_action = (
            "Audit the categorical reconstruction and Fisher variance handling before "
            "architecture work; the new representation lost a required K1-BH anchor."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1bi_not_topology_identifying"
        next_action = (
            "Audit cell-joint wrong-operator equivalence before neural work; correct "
            "responses are predictive but do not uniquely identify topology everywhere."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "thresholds": dict(config["gates"]),
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "panels": panels,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Two-replica, three-cipher local deterministic runtime-cell categorical "
            "GF(2) response audit on 4096 train and 2048 fresh total rows per panel "
            "with four pairs; not neural training, formal scale, an attack, SOTA, "
            "arbitrary-SPN generalization or a model improvement."
        ),
    }


def run_audit(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("K1-BI is a frozen local CPU audit")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"K1-BI output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _append_progress(output_root / "progress.jsonl", "run_start")
    (
        dataset_rows,
        datasets,
        structures,
        corrupted_structures,
        cross_operators,
        source_gate,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-BI source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_checks": source_checks,
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "device": device,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    rebound_rows = [
        {**row, "source_run_id": row.get("run_id"), "run_id": RUN_ID}
        for row in dataset_rows
    ]
    _write_jsonl(output_root / "dataset_manifest.jsonl", rebound_rows)
    _append_progress(
        output_root / "progress.jsonl",
        "cell_joint_response_start",
        expected_feature_rows=EXPECTED_FEATURE_ROWS,
        expected_scorer_rows=EXPECTED_SCORER_ROWS,
        expected_result_rows=EXPECTED_RESULT_ROWS,
    )
    feature_rows, scorer_rows, result_rows = evaluate_k1bi(
        config=config,
        dataset_rows=dataset_rows,
        datasets=datasets,
        structures=structures,
        corrupted_structures=corrupted_structures,
        cross_operators=cross_operators,
    )
    gate = adjudicate_k1bi(
        config=config,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
        source_gate=source_gate,
        source_checks=source_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "feature_rows": len(feature_rows),
        "expected_feature_rows": EXPECTED_FEATURE_ROWS,
        "scorer_rows": len(scorer_rows),
        "expected_scorer_rows": EXPECTED_SCORER_ROWS,
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "remote_scale": gate["remote_scale"],
        "panels": gate["panels"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
        "feature_rows": len(feature_rows),
        "scorer_rows": len(scorer_rows),
        "result_rows": len(result_rows),
        "optimizer_steps": 0,
    }
    _write_jsonl(output_root / "feature_manifest.jsonl", feature_rows)
    _write_jsonl(output_root / "scorers.jsonl", scorer_rows)
    _write_jsonl(output_root / "results.jsonl", result_rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(result_rows),
    )
    return {
        "preflight": preflight,
        "features": feature_rows,
        "scorers": scorer_rows,
        "results": result_rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    row = {"run_id": RUN_ID, "event": event, "time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = [
    "CONFIG_PATH",
    "RUN_ID",
    "adjudicate_k1bi",
    "evaluate_k1bi",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
