from __future__ import annotations

from copy import deepcopy

from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d5 import (
    EXPECTED_BIT_INDICES,
    EXPECTED_KEY_ROLES,
    PANEL_SPECS,
    adjudicate_difference_screen,
    evaluate_difference_candidate,
    prepare_difference_screen_panel,
)


def _row(bit: int, key_role: str, auc: float) -> dict[str, object]:
    key, seed = PANEL_SPECS[key_role]
    return {
        "key_role": key_role,
        "key": key,
        "panel_seed": seed,
        "bit_index_lsb": bit,
        "input_difference": 1 << bit,
        "input_hamming_weight": 1,
        "input_cell_id": bit // 4,
        "input_cell_bit_role": bit % 4,
        "cipher": "Dialga-128",
        "rounds": 5,
        "tweak": 0,
        "calibration_rows_per_class": 512,
        "evaluation_rows_per_class": 512,
        "pairs_per_row": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "screen_model": "bit_marginal_bernoulli_naive_bayes",
        "screen_auc": auc,
        "max_abs_output_bit_bias": 0.1,
        "mean_abs_output_bit_bias": 0.01,
        "panel_sha256": ("1" if key_role == "train_key" else "2") * 64,
        "calibration_negative_sha256": ("3" if key_role == "train_key" else "4") * 64,
        "evaluation_negative_sha256": ("5" if key_role == "train_key" else "6") * 64,
        "calibration_positive_xor_sha256": f"{(bit % 10):x}" * 64,
        "evaluation_positive_xor_sha256": f"{((bit + 1) % 10):x}" * 64,
        "evaluation_score_sha256": f"{((bit + 2) % 10):x}" * 64,
        "runtime_structure_window_sha256": "a" * 64,
        "runtime_structure_cell_membership_sha256": "b" * 64,
        "runtime_structure_bit_role_sha256": "c" * 64,
        "training_performed": False,
        "data_generation_performed": True,
    }


def _rows(*, candidate_bit: int | None = None) -> list[dict[str, object]]:
    rows = []
    for bit in EXPECTED_BIT_INDICES:
        for key_role in EXPECTED_KEY_ROLES:
            auc = 0.51
            if bit == 6:
                auc = 0.53
            if bit == candidate_bit:
                auc = 0.56
            rows.append(_row(bit, key_role, auc))
    return rows


def test_d5_gate_shortlists_candidate_that_beats_both_keys_and_anchor() -> None:
    gate = adjudicate_difference_screen(run_id="d5-pass", rows=_rows(candidate_bit=9))

    assert gate["status"] == "pass"
    assert (
        gate["decision"]
        == "innovation1_dialga_runtime_e4_d5_difference_candidate_supported"
    )
    assert gate["shortlist"][0]["bit_index_lsb"] == 9
    assert all(gate["protocol_checks"].values())


def test_d5_gate_holds_when_no_candidate_beats_anchor() -> None:
    gate = adjudicate_difference_screen(run_id="d5-hold", rows=_rows())

    assert gate["status"] == "hold"
    assert gate["shortlist"] == []


def test_d5_gate_fails_closed_on_plaintext_panel_drift() -> None:
    rows = _rows(candidate_bit=9)
    rows[2] = deepcopy(rows[2])
    rows[2]["panel_sha256"] = "f" * 64

    gate = adjudicate_difference_screen(run_id="d5-invalid", rows=rows)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["shared_plaintext_panels_within_key"] is False


def test_d5_candidate_evaluation_is_deterministic_and_strict() -> None:
    panel = prepare_difference_screen_panel(
        key_role="train_key",
        key=0,
        seed=41000,
        calibration_rows_per_class=8,
        evaluation_rows_per_class=8,
        pairs_per_row=2,
    )
    structure = dialga128_runtime_structure(2, round_start=2)
    first = evaluate_difference_candidate(
        bit_index=6, panel=panel, runtime_structure=structure
    )
    second = evaluate_difference_candidate(
        bit_index=6, panel=panel, runtime_structure=structure
    )

    assert first == second
    assert first["input_difference"] == 0x40
    assert first["negative_mode"] == "encrypted_random_plaintexts"
    assert first["training_performed"] is False
    assert first["data_generation_performed"] is True
    assert 0.0 <= first["screen_auc"] <= 1.0
