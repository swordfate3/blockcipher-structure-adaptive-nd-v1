from __future__ import annotations

import gc
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.models.structure.spn.small_spn_pair_relation_models import (
    PairLocalBlock,
    PairTriangleBlock,
    SmallSpnPairRelationReasoner,
    SmallSpnPairRelationSpec,
)


@dataclass(frozen=True)
class RealSpnTransferAuditConfig:
    run_id: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu or cuda")


@dataclass(frozen=True)
class LabelSourceSpec:
    source_id: str
    relative_root: str
    expected_run_id: str
    expected_decision: str


LABEL_SOURCE_SPECS = (
    LabelSourceSpec(
        "present_e11b",
        "local_audits/i2_present_r7_hwang_kernel_convergence_high16_128keys_seed0_20260717",
        "i2_present_r7_hwang_kernel_convergence_high16_128keys_seed0_20260717",
        "innovation2_present_r7_hwang_kernel_reproduced",
    ),
    LabelSourceSpec(
        "present_e12",
        "local_audits/i2_present_r7_active_block_kernel_diversity_128keys_seed0_20260717",
        "i2_present_r7_active_block_kernel_diversity_128keys_seed0_20260717",
        "innovation2_present_r7_active_block_kernel_diversity_ready",
    ),
    LabelSourceSpec(
        "present_e16",
        "local_audits/i2_present_r7_inactive_context_kernel_diversity_128keys_seed0_20260717",
        "i2_present_r7_inactive_context_kernel_diversity_128keys_seed0_20260717",
        "innovation2_inactive_context_kernel_diversity_ready",
    ),
    LabelSourceSpec(
        "present_e17b",
        "local_audits/i2_present_r7_equal_prevalence_context_mask_readiness_seed0_20260717",
        "i2_present_r7_equal_prevalence_context_mask_readiness_seed0_20260717",
        "innovation2_equal_prevalence_context_label_shortcut_dominated",
    ),
    LabelSourceSpec(
        "present_e17c",
        "local_audits/i2_present_r7_context_mask_group_disjoint_readiness_seed0_20260717",
        "i2_present_r7_context_mask_group_disjoint_readiness_seed0_20260717",
        "innovation2_group_disjoint_shortcut_generalizes",
    ),
    LabelSourceSpec(
        "present_e18",
        "local_audits/i2_present_r7_fresh_expanded_context_kernel_128keys_seed0_20260717",
        "i2_present_r7_fresh_expanded_context_kernel_128keys_seed0_20260717",
        "innovation2_context_kernel_fresh_key_unstable",
    ),
    LabelSourceSpec(
        "present_e19",
        "local_audits/i2_present_r7_context_mask_balance_rate_128keys_seed0_20260717",
        "i2_present_r7_context_mask_balance_rate_128keys_seed0_20260717",
        "innovation2_balance_rate_interaction_not_reproducible",
    ),
    LabelSourceSpec(
        "skinny_e20",
        "local_audits/i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717",
        "i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717",
        "innovation2_skinny_r7_hwang_kernel_reproduced",
    ),
    LabelSourceSpec(
        "skinny_e21",
        "local_audits/i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717",
        "i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717",
        "innovation2_skinny_r8_hwang_kernel_reproduced",
    ),
    LabelSourceSpec(
        "skinny_e22",
        "local_audits/i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717",
        "i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717",
        "innovation2_skinny_r8_geometry_kernel_not_diverse",
    ),
    LabelSourceSpec(
        "skinny_e23",
        "local_audits/i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717",
        "i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717",
        "innovation2_skinny_r8_bottom_row_pair_family_not_closed",
    ),
    LabelSourceSpec(
        "skinny_e24",
        "local_audits/i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717",
        "i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717",
        "innovation2_skinny_r7_single_cell_kernel_not_diverse",
    ),
)


def load_label_sources(outputs_root: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for spec in LABEL_SOURCE_SPECS:
        root = outputs_root / spec.relative_root
        gate_path = root / "gate.json"
        metadata_path = root / "metadata.json"
        results_path = root / "results.jsonl"
        if not all(path.is_file() for path in (gate_path, metadata_path, results_path)):
            raise FileNotFoundError(f"missing E42 source artifact under {root}")
        gate = _read_json(gate_path)
        metadata = _read_json(metadata_path)
        results = _read_jsonl(results_path)
        readiness = gate.get("readiness_checks", {})
        record = {
            "source_id": spec.source_id,
            "root": str(root),
            "run_id": gate.get("run_id"),
            "decision": gate.get("decision"),
            "status": gate.get("status"),
            "expected_run_id": spec.expected_run_id,
            "expected_decision": spec.expected_decision,
            "run_id_matches": gate.get("run_id") == spec.expected_run_id,
            "decision_matches": gate.get("decision") == spec.expected_decision,
            "protocol_checks_present": bool(readiness),
            "protocol_checks_pass": bool(readiness) and all(readiness.values()),
            "result_rows": len(results),
            "gate_sha256": _sha256(gate_path),
            "metadata_sha256": _sha256(metadata_path),
            "results_sha256": _sha256(results_path),
        }
        sources[spec.source_id] = {
            "gate": gate,
            "metadata": metadata,
            "results": results,
            "record": record,
        }
    return sources


def build_label_readiness(
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    present_e17b = sources["present_e17b"]["gate"]
    present_e17c = sources["present_e17c"]["gate"]
    present_e18 = sources["present_e18"]
    anchor_rows = [
        row
        for row in present_e18["results"]
        if row.get("context_origin") == "e16_anchor"
    ]
    present_fresh_rate = _safe_ratio(
        sum(row.get("source_signature_reproduced") is True for row in anchor_rows),
        len(anchor_rows),
    )
    present_shortcut_auc = max(
        float(present_e17c["metrics"][key]["directional_auc"])
        for key in (
            "context_disjoint_bitwise",
            "mask_disjoint_bitwise",
            "dual_disjoint_bitwise",
        )
    )
    present_results = present_e18["results"]
    rows = [
        _label_row(
            family_id="present_r7_context",
            cipher="PRESENT-80",
            rounds=7,
            independent_structure_groups=len(present_results),
            nontrivial_joint_kernel_structures=sum(
                int(row.get("joint_kernel_dimension", 0)) > 0
                for row in present_results
            ),
            distinct_fresh_key_stable_signatures=len(
                {
                    row["joint_basis_signature"]
                    for row in present_results
                    if row.get("joint_basis_signature")
                }
            ),
            positive_label_prevalence=float(present_e17b["positive_rate"]),
            fresh_key_reproduction_rate=present_fresh_rate,
            group_disjoint_both_classes=bool(
                present_e17c["readiness_checks"][
                    "all_group_train_test_splits_have_both_classes"
                ]
            ),
            strongest_simple_marginal_auc=present_shortcut_auc,
            protocol_checks_pass=_sources_protocol_pass(
                sources,
                (
                    "present_e11b",
                    "present_e12",
                    "present_e16",
                    "present_e17b",
                    "present_e17c",
                    "present_e18",
                    "present_e19",
                ),
            ),
            evidence="E18 64 contexts + E17b labels + E17c group shortcuts",
        ),
        _label_row_from_kernel_family(
            family_id="skinny_r7_single_cell",
            cipher="SKINNY-64/64",
            rounds=7,
            independent_groups=16,
            gate=sources["skinny_e24"]["gate"],
            protocol_checks_pass=_sources_protocol_pass(
                sources, ("skinny_e20", "skinny_e24")
            ),
            evidence="E20 exact anchor + E24 16 single-cell geometries",
        ),
        _label_row_from_kernel_family(
            family_id="skinny_r8_adjacent_pair",
            cipher="SKINNY-64/64",
            rounds=8,
            independent_groups=16,
            gate=sources["skinny_e22"]["gate"],
            protocol_checks_pass=_sources_protocol_pass(
                sources, ("skinny_e21", "skinny_e22")
            ),
            evidence="E21 exact anchor + E22 16 adjacent-pair geometries",
        ),
        _label_row_from_kernel_family(
            family_id="skinny_r8_bottom_row_pair",
            cipher="SKINNY-64/64",
            rounds=8,
            independent_groups=6,
            gate=sources["skinny_e23"]["gate"],
            protocol_checks_pass=_sources_protocol_pass(
                sources, ("skinny_e21", "skinny_e22", "skinny_e23")
            ),
            evidence="E21/E22 anchors + E23 six bottom-row pairs",
        ),
    ]
    return rows


def make_present64_fixture() -> dict[str, np.ndarray]:
    player = np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)],
        dtype=np.int64,
    )
    active = np.zeros((8, 64), dtype=np.float32)
    active[0, 48:64] = 1.0
    active[1, 0:16] = 1.0
    active[2, 0:4] = 1.0
    active[3, 60:64] = 1.0
    active[4, 12:20] = 1.0
    active[5, 28:36] = 1.0
    active[6, ::4] = 1.0
    active[7, 1::4] = 1.0
    masks = (
        1 << 0,
        (1 << 4) | (1 << 12),
        (1 << 16) | (1 << 48),
        (1 << 20) | (1 << 28) | (1 << 52) | (1 << 60),
        0xF,
        0xF << 16,
        0xF << 32,
        0xF << 48,
    )
    mask_bits = np.asarray(
        [[float((mask >> bit) & 1) for bit in range(64)] for mask in masks],
        dtype=np.float32,
    )
    return {
        "sboxes": np.asarray([PRESENT_SBOX], dtype=np.uint8),
        "players": player[None, :],
        "structure_active": active,
        "output_mask_bits": mask_bits,
    }


def measure_present64_contract(
    fixture: dict[str, np.ndarray],
) -> dict[str, Any]:
    local_spec = _model_spec("local", hidden_dim=16)
    corrupted_spec = SmallSpnPairRelationSpec(
        **{**local_spec.__dict__, "topology_mode": "corrupted"}
    )
    triangle_spec = _model_spec("triangle", hidden_dim=16)
    torch.manual_seed(42001)
    local = _make_model(local_spec, fixture).eval()
    corrupted = _make_model(corrupted_spec, fixture).eval()
    triangle = _make_model(triangle_spec, fixture).eval()
    _copy_parameters(local, corrupted)

    cell_permutation = np.roll(np.arange(16, dtype=np.int64), 5)
    node_permutation = np.asarray(
        [4 * cell_permutation[node // 4] + node % 4 for node in range(64)],
        dtype=np.int64,
    )
    inverse = np.argsort(node_permutation)
    relabeled_fixture = {
        **fixture,
        "players": node_permutation[fixture["players"][:, inverse]],
        "structure_active": fixture["structure_active"][:, inverse],
        "output_mask_bits": fixture["output_mask_bits"][:, inverse],
    }
    relabeled = _make_model(local_spec, relabeled_fixture).eval()
    _copy_parameters(local, relabeled)

    variants = torch.zeros(8, dtype=torch.long)
    rounds = torch.arange(8, dtype=torch.long) % 2
    structures = torch.arange(8, dtype=torch.long)
    masks = torch.arange(8, dtype=torch.long)
    with torch.no_grad():
        initial, _ = local.build_initial_relation(variants, rounds, structures, masks)
        expected = local(variants, rounds, structures, masks)
        relabeled_output = relabeled(variants, rounds, structures, masks)
        corrupted_output = corrupted(variants, rounds, structures, masks)

        relation = torch.randn(2, 64, 64, 16)
        changed = relation.clone()
        changed[0, 3, 47] += torch.randn(16)
        baseline_pairs = local.local_block(relation)
        changed_pairs = local.local_block(changed)
        pair_delta = torch.abs(baseline_pairs - changed_pairs)
        pair_delta[0, 3, 47] = 0.0

    local_parameters = sum(parameter.numel() for parameter in local.parameters())
    triangle_parameters = sum(
        parameter.numel() for parameter in triangle.parameters()
    )
    return {
        "initial_pair_shape": list(initial.shape),
        "initial_pair_shape_matches": list(initial.shape) == [8, 64, 64, 16],
        "pair_count": int(initial.shape[1] * initial.shape[2]),
        "step_schedule": local.step_counts(torch.arange(2)).tolist(),
        "shared_local_block_count": sum(
            isinstance(module, PairLocalBlock) for module in local.modules()
        ),
        "shared_triangle_block_count": sum(
            isinstance(module, PairTriangleBlock) for module in local.modules()
        ),
        "local_parameter_count": local_parameters,
        "triangle_parameter_count": triangle_parameters,
        "parameter_counts_match": local_parameters == triangle_parameters,
        "cell_relabeling_max_abs_logit_error": float(
            torch.max(torch.abs(expected - relabeled_output))
        ),
        "true_corrupted_max_abs_logit_difference": float(
            torch.max(torch.abs(expected - corrupted_output))
        ),
        "local_off_pair_influence_max_abs": float(pair_delta.max()),
        "all_outputs_finite": bool(
            torch.isfinite(expected).all()
            and torch.isfinite(relabeled_output).all()
            and torch.isfinite(corrupted_output).all()
        ),
    }


def measure_model_grid(
    fixture: dict[str, np.ndarray], *, device: str = "cpu"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    torch_device = torch.device(device)
    for processor_mode in ("local", "triangle"):
        for hidden_dim in (16, 32, 64):
            for batch_size in (1, 2, 4, 8):
                rows.append(
                    _measure_model_case(
                        fixture,
                        processor_mode=processor_mode,
                        hidden_dim=hidden_dim,
                        batch_size=batch_size,
                        device=torch_device,
                    )
                )
    return rows


def adjudicate_real_spn_transfer_readiness(
    config: RealSpnTransferAuditConfig,
    source_records: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    model_contract: dict[str, Any],
    model_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_rows = [
        row
        for row in model_rows
        if int(row["hidden_dim"]) == 32 and int(row["batch_size"]) == 4
    ]
    protocol = {
        "expected_label_sources_present": len(source_records)
        == len(LABEL_SOURCE_SPECS),
        "all_source_run_ids_match": all(
            record["run_id_matches"] for record in source_records
        ),
        "all_source_decisions_match": all(
            record["decision_matches"] for record in source_records
        ),
        "all_source_protocol_checks_pass": all(
            record["protocol_checks_pass"] for record in source_records
        ),
        "four_label_families_present": len(label_rows) == 4,
        "twenty_four_model_cases_present": len(model_rows) == 24,
        "all_model_metrics_finite": all(
            math.isfinite(float(row["elapsed_seconds"]))
            and math.isfinite(float(row["peak_process_rss_bytes"]))
            for row in model_rows
            if row["success"]
        ),
        "initial_pair_shape_is_8x64x64x16": bool(
            model_contract["initial_pair_shape_matches"]
        ),
        "pair_count_is_4096": int(model_contract["pair_count"]) == 4096,
        "step_schedule_is_7_8": model_contract["step_schedule"] == [7, 8],
        "one_local_and_zero_triangle_blocks": int(
            model_contract["shared_local_block_count"]
        )
        == 1
        and int(model_contract["shared_triangle_block_count"]) == 0,
        "local_triangle_parameter_counts_match": bool(
            model_contract["parameter_counts_match"]
        ),
        "cell_relabeling_error_at_most_1e_6": float(
            model_contract["cell_relabeling_max_abs_logit_error"]
        )
        <= 1e-6,
        "true_corrupted_logit_difference_at_least_1e_5": float(
            model_contract["true_corrupted_max_abs_logit_difference"]
        )
        >= 1e-5,
        "local_off_pair_influence_is_zero": float(
            model_contract["local_off_pair_influence_max_abs"]
        )
        == 0.0,
        "contract_outputs_finite": bool(model_contract["all_outputs_finite"]),
        "hidden32_batch4_both_processors_succeed": len(target_rows) == 2
        and all(row["success"] for row in target_rows),
    }
    if not all(protocol.values()):
        return _gate(
            config,
            "fail",
            "innovation2_real_spn_pair_state_transfer_protocol_invalid",
            protocol,
            label_rows,
            model_contract,
            model_rows,
            "repair source ownership, 64-bit pair-state, invariance, memory, or metric protocol",
        )

    ready_families = [row["family_id"] for row in label_rows if row["train_ready"]]
    model_ready = all(
        protocol[key]
        for key in (
            "initial_pair_shape_is_8x64x64x16",
            "pair_count_is_4096",
            "step_schedule_is_7_8",
            "cell_relabeling_error_at_most_1e_6",
            "true_corrupted_logit_difference_at_least_1e_5",
            "local_off_pair_influence_is_zero",
            "hidden32_batch4_both_processors_succeed",
        )
    )
    if ready_families and model_ready:
        status = "pass"
        decision = "innovation2_real_spn_pair_state_transfer_ready"
        action = "freeze a local seed0 matrix on the strongest ready real-SPN label family"
    elif model_ready:
        status = "hold"
        decision = "innovation2_real_spn_pair_state_label_bank_not_ready"
        action = "build a wider verified real-SPN structure-to-linear-mask label atlas before neural training"
    else:
        status = "hold"
        decision = "innovation2_real_spn_pair_state_model_not_ready"
        action = "repair or reduce the 64-bit pair-state model before any label generation scale-up"
    gate = _gate(
        config,
        status,
        decision,
        protocol,
        label_rows,
        model_contract,
        model_rows,
        action,
    )
    gate["metrics"] = {
        "ready_label_families": ready_families,
        "ready_label_family_count": len(ready_families),
        "model_ready": model_ready,
        "successful_model_cases": sum(row["success"] for row in model_rows),
        "model_cases": len(model_rows),
        "maximum_peak_process_rss_bytes": max(
            int(row["peak_process_rss_bytes"])
            for row in model_rows
            if row["success"]
        ),
    }
    return gate


def _label_row_from_kernel_family(
    *,
    family_id: str,
    cipher: str,
    rounds: int,
    independent_groups: int,
    gate: dict[str, Any],
    protocol_checks_pass: bool,
    evidence: str,
) -> dict[str, Any]:
    metrics = gate["metrics"]
    nontrivial_key = next(
        key for key in metrics if "nontrivial" in key and "structures" in key
    )
    signature_key = next(key for key in metrics if "distinct" in key)
    survival_key = next(key for key in metrics if "survival" in key)
    return _label_row(
        family_id=family_id,
        cipher=cipher,
        rounds=rounds,
        independent_structure_groups=independent_groups,
        nontrivial_joint_kernel_structures=int(metrics[nontrivial_key]),
        distinct_fresh_key_stable_signatures=int(metrics[signature_key]),
        positive_label_prevalence=None,
        fresh_key_reproduction_rate=float(metrics[survival_key]),
        group_disjoint_both_classes=False,
        strongest_simple_marginal_auc=None,
        protocol_checks_pass=protocol_checks_pass,
        evidence=evidence,
    )


def _label_row(
    *,
    family_id: str,
    cipher: str,
    rounds: int,
    independent_structure_groups: int,
    nontrivial_joint_kernel_structures: int,
    distinct_fresh_key_stable_signatures: int,
    positive_label_prevalence: float | None,
    fresh_key_reproduction_rate: float,
    group_disjoint_both_classes: bool,
    strongest_simple_marginal_auc: float | None,
    protocol_checks_pass: bool,
    evidence: str,
) -> dict[str, Any]:
    checks = {
        "independent_structure_groups_at_least_32": independent_structure_groups
        >= 32,
        "nontrivial_joint_kernel_structures_at_least_8": nontrivial_joint_kernel_structures
        >= 8,
        "distinct_fresh_key_stable_signatures_at_least_4": distinct_fresh_key_stable_signatures
        >= 4,
        "positive_label_prevalence_in_0p10_0p90": positive_label_prevalence
        is not None
        and 0.10 <= positive_label_prevalence <= 0.90,
        "fresh_key_reproduction_rate_at_least_0p90": fresh_key_reproduction_rate
        >= 0.90,
        "group_disjoint_split_has_both_classes": group_disjoint_both_classes,
        "strongest_simple_marginal_auc_at_most_0p65": strongest_simple_marginal_auc
        is not None
        and strongest_simple_marginal_auc <= 0.65,
        "cryptographic_protocol_checks_pass": protocol_checks_pass,
    }
    return {
        "family_id": family_id,
        "cipher": cipher,
        "rounds": rounds,
        "independent_structure_groups": independent_structure_groups,
        "nontrivial_joint_kernel_structures": nontrivial_joint_kernel_structures,
        "distinct_fresh_key_stable_signatures": distinct_fresh_key_stable_signatures,
        "positive_label_prevalence": positive_label_prevalence,
        "fresh_key_reproduction_rate": fresh_key_reproduction_rate,
        "group_disjoint_both_classes": group_disjoint_both_classes,
        "strongest_simple_marginal_auc": strongest_simple_marginal_auc,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "required_checks": len(checks),
        "train_ready": all(checks.values()),
        "evidence": evidence,
    }


def _measure_model_case(
    fixture: dict[str, np.ndarray],
    *,
    processor_mode: str,
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "processor_mode": processor_mode,
        "hidden_dim": hidden_dim,
        "path_rank": max(2, hidden_dim // 8),
        "batch_size": batch_size,
        "state_bits": 64,
        "pair_count": 4096,
        "device": str(device),
        "estimated_relation_bytes": batch_size * 64 * 64 * hidden_dim * 4,
    }
    started = time.perf_counter()
    try:
        torch.manual_seed(42002 + hidden_dim + batch_size)
        model = _make_model(
            _model_spec(processor_mode, hidden_dim=hidden_dim), fixture
        ).to(device)
        model.train()
        variants = torch.zeros(batch_size, dtype=torch.long, device=device)
        rounds = torch.arange(batch_size, dtype=torch.long, device=device) % 2
        structures = torch.arange(batch_size, dtype=torch.long, device=device) % 8
        masks = (torch.arange(batch_size, dtype=torch.long, device=device) * 3) % 8
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        logits = model(variants, rounds, structures, masks)
        loss = logits.square().mean()
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        row.update(
            {
                "success": True,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "logits_finite": bool(torch.isfinite(logits).all()),
                "gradients_finite": bool(gradients_finite),
                "peak_process_rss_bytes": peak_memory,
                "error": "",
            }
        )
    except (MemoryError, RuntimeError, ValueError) as error:
        row.update(
            {
                "success": False,
                "parameter_count": 0,
                "logits_finite": False,
                "gradients_finite": False,
                "peak_process_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                )
                * 1024,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    row["elapsed_seconds"] = time.perf_counter() - started
    for name in ("model", "variants", "rounds", "structures", "masks", "logits", "loss"):
        if name in locals():
            del locals()[name]
    gc.collect()
    return row


def _model_spec(processor_mode: str, *, hidden_dim: int) -> SmallSpnPairRelationSpec:
    return SmallSpnPairRelationSpec(
        topology_mode="true",
        processor_mode=processor_mode,
        state_bits=64,
        round_categories=2,
        round_step_offset=7,
        hidden_dim=hidden_dim,
        path_rank=max(2, hidden_dim // 8),
        dropout=0.0,
    )


def _make_model(
    spec: SmallSpnPairRelationSpec, fixture: dict[str, np.ndarray]
) -> SmallSpnPairRelationReasoner:
    return SmallSpnPairRelationReasoner(
        spec,
        sboxes=fixture["sboxes"],
        players=fixture["players"],
        structure_active_bits=fixture["structure_active"],
        output_mask_bits=fixture["output_mask_bits"],
    )


def _copy_parameters(source: nn.Module, target: nn.Module) -> None:
    source_parameters = dict(source.named_parameters())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.copy_(source_parameters[name])


def _sources_protocol_pass(
    sources: dict[str, dict[str, Any]], source_ids: tuple[str, ...]
) -> bool:
    return all(
        sources[source_id]["record"]["run_id_matches"]
        and sources[source_id]["record"]["decision_matches"]
        and sources[source_id]["record"]["protocol_checks_pass"]
        for source_id in source_ids
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _gate(
    config: RealSpnTransferAuditConfig,
    status: str,
    decision: str,
    protocol: dict[str, bool],
    label_rows: list[dict[str, Any]],
    model_contract: dict[str, Any],
    model_rows: list[dict[str, Any]],
    action: str,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": protocol,
        "label_family_checks": {
            row["family_id"]: row["checks"] for row in label_rows
        },
        "model_contract": model_contract,
        "model_case_success": {
            f"{row['processor_mode']}_h{row['hidden_dim']}_b{row['batch_size']}": row[
                "success"
            ]
            for row in model_rows
        },
        "claim_scope": (
            "local label-bank and 64-bit pair-state transfer readiness over existing "
            "PRESENT/SKINNY empirical-kernel artifacts; not neural training or a new "
            "cryptanalytic property"
        ),
        "next_action": {"action": action, "remote_scale": False, "training": False},
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
