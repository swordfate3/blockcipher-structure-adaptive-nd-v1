from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.present_query_cone_sparse_anf_growth import (
    required_state_cone,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
)


RUN_ID = (
    "i2_output_prediction_opm1_present_r3_selected_output_"
    "structural_baseline_audit_20260722"
)


@dataclass(frozen=True)
class PresentSelectedOutputStructuralBaselineConfig:
    run_id: str = RUN_ID
    rounds: int = 3
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES

    def __post_init__(self) -> None:
        if self.rounds != 3:
            raise ValueError("OPM1 is frozen to PRESENT round three")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPM1 positions must match OP10 through OPC1")


def sbox_coordinate_metrics(output_bit_lsb: int) -> dict[str, int]:
    if output_bit_lsb not in range(4):
        raise ValueError("S-box output bit must be in [0, 3]")
    truth = [(PRESENT_SBOX[value] >> output_bit_lsb) & 1 for value in range(16)]
    anf = list(truth)
    for variable in range(4):
        for mask in range(16):
            if mask & (1 << variable):
                anf[mask] ^= anf[mask ^ (1 << variable)]
    degree = max(
        (mask.bit_count() for mask, coefficient in enumerate(anf) if coefficient),
        default=0,
    )
    maximum_walsh = max(
        abs(
            sum(
                (-1) ** (truth[value] ^ ((value & mask).bit_count() & 1))
                for value in range(16)
            )
        )
        for mask in range(16)
    )
    return {
        "output_bit_lsb": output_bit_lsb,
        "weight": sum(truth),
        "anf_degree": degree,
        "nonlinearity": 8 - maximum_walsh // 2,
        "anf_terms": sum(anf),
    }


def audit_present_selected_output_structural_baseline(
    config: PresentSelectedOutputStructuralBaselineConfig,
) -> dict[str, Any]:
    sbox_metrics = {
        bit: sbox_coordinate_metrics(bit) for bit in range(4)
    }
    selected = set(config.selected_msb_indices)
    rows: list[dict[str, Any]] = []
    for msb_index in range(64):
        internal_bit = 63 - msb_index
        source_mask = Present80.inverse_permutation_layer(1 << internal_bit)
        source_bit = source_mask.bit_length() - 1
        coordinate = source_bit % 4
        cone_widths = {
            rounds: len(
                required_state_cone(
                    rounds=rounds,
                    output_bits=(internal_bit,),
                )[0]
            )
            for rounds in range(1, config.rounds + 1)
        }
        rows.append(
            {
                "run_id": config.run_id,
                "msb_index": msb_index,
                "internal_bit": internal_bit,
                "last_round_sbox_source_bit": source_bit,
                "selected": msb_index in selected,
                "sbox_output_bit_lsb": coordinate,
                "round1_input_cone_bits": cone_widths[1],
                "round2_input_cone_bits": cone_widths[2],
                "round3_input_cone_bits": cone_widths[3],
                "sbox_output_weight": sbox_metrics[coordinate]["weight"],
                "sbox_anf_degree": sbox_metrics[coordinate]["anf_degree"],
                "sbox_nonlinearity": sbox_metrics[coordinate]["nonlinearity"],
                "sbox_anf_terms": sbox_metrics[coordinate]["anf_terms"],
                "sample_classification": False,
            }
        )

    selected_rows = [row for row in rows if row["selected"]]
    unselected_rows = [row for row in rows if not row["selected"]]
    all_round3_widths = {int(row["round3_input_cone_bits"]) for row in rows}
    selected_round3_widths = {
        int(row["round3_input_cone_bits"]) for row in selected_rows
    }
    unselected_round3_widths = {
        int(row["round3_input_cone_bits"]) for row in unselected_rows
    }
    selected_coordinates = {
        int(row["sbox_output_bit_lsb"]) for row in selected_rows
    }
    selected_degrees = {
        int(sbox_metrics[coordinate]["anf_degree"])
        for coordinate in selected_coordinates
    }
    all_degrees = {
        int(metrics["anf_degree"]) for metrics in sbox_metrics.values()
    }
    selected_nonlinearities = {
        int(sbox_metrics[coordinate]["nonlinearity"])
        for coordinate in selected_coordinates
    }
    all_nonlinearities = {
        int(metrics["nonlinearity"]) for metrics in sbox_metrics.values()
    }
    protocol_checks = {
        "official_present_zero_key_vector_matches": Present80(
            rounds=31, key=0
        ).encrypt(0)
        == 0x5579C1387B228445,
        "present_sbox_is_a_permutation": sorted(PRESENT_SBOX) == list(range(16)),
        "selected_positions_match_op10_through_opc1": tuple(
            row["msb_index"] for row in selected_rows
        )
        == config.selected_msb_indices,
        "msb_to_internal_conversion_is_bijective": {
            int(row["internal_bit"]) for row in rows
        }
        == set(range(64)),
        "inverse_player_sources_are_single_bits": all(
            Present80.inverse_permutation_layer(1 << int(row["internal_bit"])).bit_count()
            == 1
            for row in rows
        ),
        "inverse_player_sources_round_trip": all(
            Present80.permutation_layer(1 << int(row["last_round_sbox_source_bit"]))
            == 1 << int(row["internal_bit"])
            for row in rows
        ),
        "all_output_positions_are_audited": len(rows) == 64,
        "labels_are_output_positions_not_sample_classes": True,
    }
    execution_checks = {
        "sixty_four_result_rows_complete": len(rows) == 64,
        "eight_selected_rows_complete": len(selected_rows) == 8,
        "four_sbox_coordinates_complete": set(sbox_metrics) == set(range(4)),
        "all_metrics_are_finite": all(
            math.isfinite(float(row[field]))
            for row in rows
            for field in (
                "round1_input_cone_bits",
                "round2_input_cone_bits",
                "round3_input_cone_bits",
                "sbox_output_weight",
                "sbox_anf_degree",
                "sbox_nonlinearity",
                "sbox_anf_terms",
            )
        ),
    }
    coarse_baseline_checks = {
        "all_round3_output_bits_have_equal_input_cone_width": len(
            all_round3_widths
        )
        == 1,
        "selected_and_unselected_round3_cone_widths_match": selected_round3_widths
        == unselected_round3_widths,
        "selected_sbox_coordinates_are_not_uniquely_minimum_degree": min(
            selected_degrees
        )
        > min(all_degrees),
        "selected_and_all_sbox_coordinates_have_same_nonlinearity": selected_nonlinearities
        == all_nonlinearities,
        "all_sbox_output_coordinates_are_balanced": all(
            int(metrics["weight"]) == 8 for metrics in sbox_metrics.values()
        ),
    }
    valid = all(protocol_checks.values()) and all(execution_checks.values())
    if not valid:
        status = "fail"
        decision = (
            "innovation2_present_r3_selected_output_structural_baseline_"
            "protocol_invalid"
        )
        action = "repair only the bit order, dependency cone, S-box metrics, or artifacts"
    elif all(coarse_baseline_checks.values()):
        status = "pass"
        decision = (
            "innovation2_present_r3_selected_output_not_explained_by_"
            "coarse_structure_baselines"
        )
        action = "retain the frozen OPC1 matrix and report the easy-bit mechanism as unresolved beyond coarse cone and S-box-coordinate baselines"
    else:
        status = "hold"
        decision = (
            "innovation2_present_r3_selected_output_coarse_structure_"
            "candidate_requires_controlled_test"
        )
        action = "preregister one matched control for the separated structural quantity without changing output positions"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "coarse_baseline_checks": coarse_baseline_checks,
        "metrics": {
            "selected_msb_indices": list(config.selected_msb_indices),
            "selected_internal_bits": [
                int(row["internal_bit"]) for row in selected_rows
            ],
            "selected_last_round_sbox_source_bits": [
                int(row["last_round_sbox_source_bit"]) for row in selected_rows
            ],
            "selected_sbox_output_bits_lsb": sorted(selected_coordinates),
            "round1_input_cone_widths": sorted(
                {int(row["round1_input_cone_bits"]) for row in rows}
            ),
            "round2_input_cone_widths": sorted(
                {int(row["round2_input_cone_bits"]) for row in rows}
            ),
            "round3_input_cone_widths": sorted(all_round3_widths),
            "sbox_coordinate_metrics": [sbox_metrics[bit] for bit in range(4)],
        },
        "claim_scope": (
            "deterministic PRESENT r3 all-64-output dependency-cone and S-box-coordinate baseline; "
            "not a causal explanation, neural attribution, r4 evidence, sample classification, or SOTA"
        ),
        "next_action": {
            "action": action,
            "opc1_unchanged": True,
            "new_output_position_search_authorized": False,
            "remote_training_authorized": False,
        },
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_selected_output_structural_baseline_audit",
        "cipher": "PRESENT-80",
        "config": {
            **asdict(config),
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "neural_training": False,
        "sample_classification": False,
        "claim_scope": gate["claim_scope"],
    }
    return {
        "rows": rows,
        "metadata": metadata,
        "summary": {
            "run_id": config.run_id,
            "metadata": metadata,
            "rows": rows,
            "gate": gate,
        },
        "gate": gate,
    }
