from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_real_spn_pair_state_transfer import (
    render_real_spn_transfer_svg,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    destination_cell_rotation,
)
from blockcipher_nd.tasks.innovation2.real_spn_pair_state_transfer_readiness import (
    LABEL_SOURCE_SPECS,
    RealSpnTransferAuditConfig,
    adjudicate_real_spn_transfer_readiness,
    make_present64_fixture,
    measure_present64_contract,
)


def test_destination_cell_rotation_generalizes_to_64_bits() -> None:
    rotation = destination_cell_rotation(64)
    assert sorted(rotation.tolist()) == list(range(64))
    assert rotation[:8].tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert rotation[-4:].tolist() == [0, 1, 2, 3]


def test_present64_pair_state_contract_is_equivariant_and_topology_sensitive() -> None:
    contract = measure_present64_contract(make_present64_fixture())
    assert contract["initial_pair_shape"] == [8, 64, 64, 16]
    assert contract["pair_count"] == 4096
    assert contract["step_schedule"] == [7, 8]
    assert contract["shared_local_block_count"] == 1
    assert contract["shared_triangle_block_count"] == 0
    assert contract["parameter_counts_match"] is True
    assert contract["cell_relabeling_max_abs_logit_error"] <= 1e-6
    assert contract["true_corrupted_max_abs_logit_difference"] >= 1e-5
    assert contract["local_off_pair_influence_max_abs"] == 0.0
    assert contract["all_outputs_finite"] is True


def test_real_spn_transfer_gate_holds_only_for_missing_label_bank() -> None:
    source_records = [
        {
            "run_id_matches": True,
            "decision_matches": True,
            "protocol_checks_pass": True,
        }
        for _ in LABEL_SOURCE_SPECS
    ]
    label_rows = [
        {
            "family_id": f"family_{index}",
            "train_ready": False,
            "checks": {"labels": False},
        }
        for index in range(4)
    ]
    contract = {
        "initial_pair_shape_matches": True,
        "pair_count": 4096,
        "step_schedule": [7, 8],
        "shared_local_block_count": 1,
        "shared_triangle_block_count": 0,
        "parameter_counts_match": True,
        "cell_relabeling_max_abs_logit_error": 1e-8,
        "true_corrupted_max_abs_logit_difference": 0.02,
        "local_off_pair_influence_max_abs": 0.0,
        "all_outputs_finite": True,
    }
    model_rows = [
        {
            "processor_mode": processor,
            "hidden_dim": hidden,
            "batch_size": batch,
            "success": True,
            "elapsed_seconds": 0.1,
            "peak_process_rss_bytes": 1024,
        }
        for processor in ("local", "triangle")
        for hidden in (16, 32, 64)
        for batch in (1, 2, 4, 8)
    ]
    config = RealSpnTransferAuditConfig(run_id="e42")
    hold = adjudicate_real_spn_transfer_readiness(
        config, source_records, label_rows, contract, model_rows
    )
    assert hold["status"] == "hold"
    assert hold["decision"] == (
        "innovation2_real_spn_pair_state_label_bank_not_ready"
    )
    assert hold["metrics"]["model_ready"] is True

    ready_rows = [{**row, "train_ready": index == 0} for index, row in enumerate(label_rows)]
    ready = adjudicate_real_spn_transfer_readiness(
        config, source_records, ready_rows, contract, model_rows
    )
    assert ready["decision"] == "innovation2_real_spn_pair_state_transfer_ready"


def test_real_spn_transfer_plot_explains_label_and_model_gates(tmp_path: Path) -> None:
    label_rows = [
        {
            "family_id": family_id,
            "passed_checks": passed,
            "required_checks": 8,
            "train_ready": False,
        }
        for family_id, passed in (
            ("present_r7_context", 6),
            ("skinny_r7_single_cell", 4),
            ("skinny_r8_adjacent_pair", 3),
            ("skinny_r8_bottom_row_pair", 3),
        )
    ]
    model_rows = [
        {
            "processor_mode": processor,
            "hidden_dim": hidden,
            "batch_size": batch,
            "success": True,
            "peak_process_rss_bytes": 512 * 1024**2,
        }
        for processor in ("local", "triangle")
        for hidden in (16, 32, 64)
        for batch in (1, 2, 4, 8)
    ]
    summary = {
        "label_rows": label_rows,
        "model_rows": model_rows,
        "gate": {
            "decision": "innovation2_real_spn_pair_state_label_bank_not_ready"
        },
    }
    output = tmp_path / "e42.svg"
    render_real_spn_transfer_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E42" in svg
    assert "64×64 pair-state" in svg
    assert "禁止神经训练" in svg
    assert "不是神经性能" in svg
