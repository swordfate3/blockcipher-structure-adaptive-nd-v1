from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    RUN_ID as K1_RUN_ID,
    _build_control_model,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    CANDIDATE_MODEL,
)


RUN_ID = "i1_uknit_family_ctspn_endpoint_alignment_k1a_20260728"
K1_DECISION = "innovation1_uknit_family_ctspn_k1_linear_schedule_not_supported"
EXPECTED_CIPHERS = ("uknit64", "dialga128")
EXPECTED_SEEDS = (0, 1)
CONTROL_CONDITIONS = ("repeat_last", "rotated")
EXPECTED_RESULT_ROWS = 8
PROBE_ROWS = 32
PROBE_SEED = 20260728


def run_endpoint_alignment_audit(
    *,
    task_rows: Sequence[Mapping[str, Any]],
    k1_gate: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    probe_rows: int = PROBE_ROWS,
) -> dict[str, Any]:
    if probe_rows <= 0:
        raise ValueError("K1-A probe_rows must be positive")
    tasks = _candidate_task_map(task_rows)
    checkpoints = _candidate_checkpoint_map(checkpoint_manifest)
    results: list[dict[str, Any]] = []
    strict_loads: list[bool] = []
    state_hashes_match: list[bool] = []
    input_widths: list[int] = []

    for cipher_index, cipher in enumerate(EXPECTED_CIPHERS):
        for seed in EXPECTED_SEEDS:
            task = tasks[(cipher, seed)]
            checkpoint = checkpoints[(cipher, seed)]
            checkpoint_path = Path(str(checkpoint["path"]))
            if file_sha256(checkpoint_path) != checkpoint.get("sha256"):
                raise ValueError(f"K1-A checkpoint hash mismatch: {checkpoint_path}")
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = payload["state_dict"]
            source_state_sha256 = tensor_mapping_sha256(state_dict)
            input_bits = 512 if cipher == "uknit64" else 1024
            generator = torch.Generator().manual_seed(
                PROBE_SEED + 100 * cipher_index + seed
            )
            features = torch.randint(
                0,
                2,
                (probe_rows, input_bits),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.float32)
            probe_sha256 = hashlib.sha256(
                features.numpy().tobytes(order="C")
            ).hexdigest()

            correct = _build_control_model(
                task=task,
                source_role="candidate",
                condition="correct_ordered",
                input_bits=input_bits,
            )
            correct.load_state_dict(state_dict, strict=True)
            correct.eval()
            strict_loads.append(True)
            state_hashes_match.append(
                tensor_mapping_sha256(correct.state_dict()) == source_state_sha256
            )
            input_widths.append(int(correct.backbone.edge_encoder[0].in_features))
            with torch.inference_mode():
                correct_views, correct_summary, correct_logits = frozen_k1_stages(
                    correct, features
                )
            correct_endpoints = native_endpoint_signature(correct)

            for condition in CONTROL_CONDITIONS:
                control = _build_control_model(
                    task=task,
                    source_role="candidate",
                    condition=condition,
                    input_bits=input_bits,
                )
                control.load_state_dict(state_dict, strict=True)
                control.eval()
                strict_loads.append(True)
                control_state_sha256 = tensor_mapping_sha256(control.state_dict())
                state_hashes_match.append(control_state_sha256 == source_state_sha256)
                with torch.inference_mode():
                    views, summary, logits = frozen_k1_stages(control, features)
                endpoints = native_endpoint_signature(control)
                results.append(
                    {
                        "run_id": RUN_ID,
                        "source_run_id": K1_RUN_ID,
                        "cipher_key": cipher,
                        "seed": seed,
                        "condition": condition,
                        "probe_rows": probe_rows,
                        "probe_sha256": probe_sha256,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": str(checkpoint["sha256"]),
                        "state_dict_sha256": source_state_sha256,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "edge_input_values": input_widths[-1],
                        "edges": int(correct_views.shape[-2]),
                        "transitions": int(correct_views.shape[2]),
                        "native_endpoint_fraction_changed": float(
                            (endpoints != correct_endpoints)
                            .any(dim=-1)
                            .to(torch.float32)
                            .mean()
                        ),
                        "edge_value_max_abs_delta": _max_abs_delta(
                            views, correct_views
                        ),
                        "edge_value_mean_abs_delta": _mean_abs_delta(
                            views, correct_views
                        ),
                        "transition_summary_max_abs_delta": _max_abs_delta(
                            summary, correct_summary
                        ),
                        "transition_summary_mean_abs_delta": _mean_abs_delta(
                            summary, correct_summary
                        ),
                        "logit_max_abs_delta": _max_abs_delta(
                            logits, correct_logits
                        ),
                        "logit_mean_abs_delta": _mean_abs_delta(
                            logits, correct_logits
                        ),
                        "source_candidate_auc": float(
                            k1_gate["seed_results"][cipher][str(seed)][
                                "candidate_auc"
                            ]
                        ),
                        "source_candidate_minus_control_auc": float(
                            k1_gate["seed_results"][cipher][str(seed)][
                                f"candidate_minus_{condition}"
                            ]
                        ),
                    }
                )

    gate = adjudicate_endpoint_alignment(
        k1_gate=k1_gate,
        results=results,
        task_keys=set(tasks),
        checkpoint_keys=set(checkpoints),
        strict_loads=strict_loads,
        state_hashes_match=state_hashes_match,
        edge_input_widths=input_widths,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(results),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_rows": 0,
        "optimizer_steps": 0,
        "result_rows": len(results),
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    return {
        "results": results,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def adjudicate_endpoint_alignment(
    *,
    k1_gate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    task_keys: set[tuple[str, int]],
    checkpoint_keys: set[tuple[str, int]],
    strict_loads: Sequence[bool],
    state_hashes_match: Sequence[bool],
    edge_input_widths: Sequence[int],
) -> dict[str, Any]:
    expected_keys = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    grouped = {
        (str(row.get("cipher_key")), int(row.get("seed", -1)), str(row.get("condition"))): row
        for row in results
    }
    expected_result_keys = {
        (cipher, seed, condition)
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for condition in CONTROL_CONDITIONS
    }
    protocol_checks = {
        "k1_hold_gate_protocol_clean": (
            k1_gate.get("run_id") == K1_RUN_ID
            and k1_gate.get("status") == "hold"
            and k1_gate.get("decision") == K1_DECISION
            and bool(k1_gate.get("protocol_checks"))
            and all(k1_gate.get("protocol_checks", {}).values())
        ),
        "four_candidate_tasks_complete": task_keys == expected_keys,
        "four_selected_candidate_checkpoints_complete": checkpoint_keys
        == expected_keys,
        "eight_control_rows_complete": (
            len(results) == EXPECTED_RESULT_ROWS
            and set(grouped) == expected_result_keys
        ),
        "strict_state_dict_load": bool(strict_loads) and all(strict_loads),
        "same_learned_state_per_control": bool(state_hashes_match)
        and all(state_hashes_match),
        "edge_tokens_have_no_endpoint_identity": bool(edge_input_widths)
        and set(edge_input_widths) == {12},
        "zero_training_and_optimizer_steps": all(
            row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in results
        ),
        "finite_probe_metrics": all(_row_is_finite(row) for row in results),
    }
    research_checks: dict[str, bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            for condition, endpoint_floor in (
                ("repeat_last", 0.45),
                ("rotated", 0.95),
            ):
                row = grouped.get((cipher, seed, condition), {})
                prefix = f"{cipher}_seed{seed}_{condition}"
                research_checks[f"{prefix}_native_endpoints_change"] = (
                    float(row.get("native_endpoint_fraction_changed", -1.0))
                    >= endpoint_floor
                )
                research_checks[f"{prefix}_raw_edge_values_change"] = (
                    float(row.get("edge_value_max_abs_delta", 0.0)) > 0.0
                )
                if cipher == "dialga128":
                    research_checks[f"{prefix}_pooled_summary_collapses"] = (
                        float(
                            row.get("transition_summary_max_abs_delta", math.inf)
                        )
                        <= 1e-5
                    )
                    research_checks[f"{prefix}_final_logit_collapses"] = (
                        float(row.get("logit_max_abs_delta", math.inf)) <= 1e-4
                    )
                    research_checks[f"{prefix}_observed_auc_ties"] = (
                        abs(
                            float(
                                row.get(
                                    "source_candidate_minus_control_auc", math.inf
                                )
                            )
                        )
                        <= 1e-5
                    )
            if cipher == "uknit64":
                row = grouped.get((cipher, seed, "repeat_last"), {})
                research_checks[f"{cipher}_seed{seed}_source_auc_below_floor"] = (
                    float(row.get("source_candidate_auc", math.inf)) < 0.520
                )

    protocol_valid = all(protocol_checks.values())
    alignment_supported = protocol_valid and all(research_checks.values())
    status = "pass" if alignment_supported else ("hold" if protocol_valid else "invalid")
    if alignment_supported:
        decision = "innovation1_uknit_family_ctspn_endpoint_alignment_loss_confirmed"
        next_action = (
            "Implement K1-B at the identical local budget: retain exact canonical "
            "state views and add only fixed-width native cell-position and directed "
            "bit-role endpoint channels; require uKNIT improvement, correct-order "
            "control dominance, and Dialga retention before any scale or K2 work."
        )
    elif protocol_valid:
        decision = "innovation1_uknit_family_ctspn_endpoint_alignment_loss_not_confirmed"
        next_action = (
            "Do not train an endpoint-aware candidate. Inspect temporal aggregation "
            "and K1 control construction at zero training until the failed alignment "
            "check is explained."
        )
    else:
        decision = "innovation1_uknit_family_ctspn_endpoint_alignment_audit_invalid"
        next_action = (
            "Repair only the failed K1 checkpoint replay, task alignment, or metric "
            "validation and rerun K1-A without training."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
        "thresholds": {
            "repeat_last_native_endpoint_fraction": 0.45,
            "rotated_native_endpoint_fraction": 0.95,
            "dialga_transition_summary_max_abs_delta": 1e-5,
            "dialga_logit_max_abs_delta": 1e-4,
            "dialga_auc_tie_abs_delta": 1e-5,
            "uknit_source_auc_floor": 0.520,
        },
        "next_action": next_action,
        "claim_scope": (
            "zero-training K1 checkpoint representation audit for uKNIT-BC "
            "prefix-r5 and Dialga-128 prefix-r4 only; confirms a native endpoint "
            "identity loss mechanism, not a better neural distinguisher, attack, "
            "transfer result, SOTA result, formal family taxonomy or scale result"
        ),
        "blocked_actions": [
            "increase samples, pairs, epochs or model width from K1-A",
            "remote launch from a zero-training audit",
            "start K2 before a position-preserving K1 candidate passes",
            "add MoE, DDT, trail, partial decryption, guessed keys or cipher-id routing",
            "use Dialga absolute AUC to hide uKNIT failure",
            "include generalized-Feistel MSX in the CT-SPN claim",
        ],
    }


def frozen_k1_stages(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    runtime = features.reshape(
        features.shape[0],
        -1,
        2,
        model.runtime_structure.block_bits,
    ).flip(-1)
    backbone = model.backbone
    views = backbone.canonical_transition_edge_views(
        runtime,
        model.runtime_structure,
        model.canonical_schedule,
        relation_mode=model.relation_mode,
        endpoint_identity_mode=backbone.spec.endpoint_identity_mode,
    )
    batch, pair_count, transitions, edges, _ = views.shape
    hidden = backbone.edge_encoder(views).reshape(
        batch * pair_count * transitions,
        edges,
        backbone.token_dim,
    )
    for block in backbone.mixer_blocks:
        hidden = block(hidden)
    hidden = backbone.sequence_norm(hidden)
    summary = (
        hidden.mean(dim=1)
        + hidden.max(dim=1).values
        + torch.sqrt(hidden.square().mean(dim=1).clamp_min(1e-8))
    ) / 3.0
    summary = summary.reshape(batch, pair_count, transitions, -1)
    return views, summary, model(features)


def native_endpoint_signature(model: torch.nn.Module) -> torch.Tensor:
    schedule = model.canonical_schedule
    structure = model.runtime_structure
    targets = schedule.canonical_edge_index[0]
    sources = schedule.canonical_edge_index[1]
    rows = []
    for input_native, output_native in schedule.factors:
        native_targets = torch.tensor(output_native, dtype=torch.long)[targets]
        native_sources = torch.tensor(input_native, dtype=torch.long)[sources]
        rows.append(
            torch.stack(
                (
                    structure.cell_membership[native_targets],
                    structure.bit_role[native_targets],
                    structure.cell_membership[native_sources],
                    structure.bit_role[native_sources],
                ),
                dim=-1,
            )
        )
    return torch.stack(rows)


def write_endpoint_alignment_artifacts(
    payload: Mapping[str, Any], output_root: Path
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["results"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    for name in ("validation", "gate", "summary"):
        (output_root / f"{name}.json").write_text(
            json.dumps(payload[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _candidate_task_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result = {
        (str(row["cipher_key"]), int(row["seed"])): row
        for row in rows
        if row.get("model_key") == CANDIDATE_MODEL
    }
    expected = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    if set(result) != expected:
        raise ValueError("K1-A requires all four frozen K1 candidate tasks")
    return result


def _candidate_checkpoint_map(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    if manifest.get("run_id") != K1_RUN_ID or manifest.get("status") != "pass":
        raise ValueError("K1-A requires the completed K1 checkpoint manifest")
    entries = manifest.get("entries", ())
    result = {
        (str(row["cipher_key"]), int(row["seed"])): row
        for row in entries
        if row.get("model") == CANDIDATE_MODEL
        and row.get("selected_checkpoint") == "best"
    }
    expected = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    if set(result) != expected:
        raise ValueError("K1-A requires four selected K1 candidate checkpoints")
    return result


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def _mean_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().mean())


def _row_is_finite(row: Mapping[str, Any]) -> bool:
    fields = (
        "native_endpoint_fraction_changed",
        "edge_value_max_abs_delta",
        "edge_value_mean_abs_delta",
        "transition_summary_max_abs_delta",
        "transition_summary_mean_abs_delta",
        "logit_max_abs_delta",
        "logit_mean_abs_delta",
        "source_candidate_auc",
        "source_candidate_minus_control_auc",
    )
    return all(
        isinstance(row.get(name), (int, float))
        and math.isfinite(float(row[name]))
        for name in fields
    )


__all__ = [
    "CONTROL_CONDITIONS",
    "EXPECTED_RESULT_ROWS",
    "PROBE_ROWS",
    "RUN_ID",
    "adjudicate_endpoint_alignment",
    "frozen_k1_stages",
    "native_endpoint_signature",
    "run_endpoint_alignment_audit",
    "write_endpoint_alignment_artifacts",
]
