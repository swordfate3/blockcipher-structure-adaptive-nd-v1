from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from blockcipher_nd.cli.plot_innovation2_selected_output_architecture_confirmation import (
    render_architecture_confirmation,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_architecture_confirmation import (
    ArchitectureConfirmationConfig,
    adjudicate_confirmation,
    candidate_from_phase_a_gate,
    confirmation_model_specs,
    confirmation_parameter_counts,
    prepare_confirmation_data,
    train_confirmation_matrix,
    validate_confirmation_contract,
)
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    REMOTE_RUN_ID as PHASE_A_REMOTE_RUN_ID,
)


def _phase_a_gate(candidate: str = "rescnn") -> dict[str, object]:
    return {
        "run_id": PHASE_A_REMOTE_RUN_ID,
        "status": "pass",
        "decision": "innovation2_selected8_architecture_candidate_requires_confirmation",
        "protocol_checks": {"valid": True},
        "execution_checks": {"complete": True},
        "metrics": {
            "selected_candidate_for_phase_b": candidate,
            "candidate_gates": {candidate: {"passed": True}},
        },
    }


def _tiny_config(candidate: str = "rescnn") -> ArchitectureConfirmationConfig:
    return ArchitectureConfirmationConfig(
        candidate_architecture=candidate,
        run_id="test-opa2",
        train_rows=8,
        test_rows=8,
        mlp_hidden_dim=16,
        lstm_hidden_dim=4,
        lstm_layers=1,
        rescnn_channels=4,
        rescnn_blocks=1,
        transformer_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        present_spn_dim=4,
        present_spn_blocks=1,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
        maximum_parameter_gap=100.0,
    )


def test_phase_a_gate_is_the_only_candidate_authority() -> None:
    assert candidate_from_phase_a_gate(_phase_a_gate("present_spn")) == "present_spn"

    blocked = _phase_a_gate()
    blocked["decision"] = "innovation2_selected8_mlp_anchor_retained_after_screen"
    with pytest.raises(ValueError, match="did not authorize"):
        candidate_from_phase_a_gate(blocked)
    failed_candidate = _phase_a_gate()
    failed_candidate["metrics"]["candidate_gates"]["rescnn"]["passed"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="did not pass"):
        candidate_from_phase_a_gate(failed_candidate)
    invalid_protocol = _phase_a_gate()
    invalid_protocol["protocol_checks"] = {"valid": False}
    with pytest.raises(ValueError, match="protocol_checks"):
        candidate_from_phase_a_gate(invalid_protocol)


def test_confirmation_config_is_frozen_to_non_mlp_seed3() -> None:
    formal = ArchitectureConfirmationConfig.phase_b_confirmation("transformer")

    assert formal.seed == 3
    assert formal.rounds == 3
    assert formal.train_rows == 1 << 17
    assert formal.test_rows == 1 << 16
    assert formal.epochs == 100
    assert formal.batch_size == 250
    with pytest.raises(ValueError, match="non-MLP"):
        ArchitectureConfirmationConfig.smoke("mlp")


def test_four_rows_use_shared_true_and_matched_shuffle_roles() -> None:
    specs = confirmation_model_specs(ArchitectureConfirmationConfig.smoke("lstm"))

    assert specs == (
        ("selected8_mlp_true_output", "mlp", False),
        ("selected8_mlp_label_shuffle", "mlp", True),
        ("selected8_lstm_true_output", "lstm", False),
        ("selected8_lstm_label_shuffle", "lstm", True),
    )


def test_seed3_contract_and_small_four_row_training_are_replayable(tmp_path: Path) -> None:
    config = _tiny_config()
    phase_a_gate = _phase_a_gate()
    data = prepare_confirmation_data(config, tmp_path)
    checks = validate_confirmation_contract(config, data, phase_a_gate)
    training = train_confirmation_matrix(config, data, tmp_path)
    gate = adjudicate_confirmation(config, checks, training)

    assert all(checks.values())
    assert len(training["rows"]) == 32
    assert len(training["history"]) == 4
    assert len(training["checkpoints"]) == 4
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("local_smoke_passed")
    assert gate["next_action"]["reopens_op12"] is False


def test_formal_parameter_budgets_remain_matched() -> None:
    for candidate in ("lstm", "rescnn", "transformer", "present_spn"):
        config = ArchitectureConfirmationConfig.smoke(candidate)
        counts = confirmation_parameter_counts(config)
        assert abs(counts[candidate] - counts["mlp"]) / counts["mlp"] <= 0.03


def test_confirmation_gate_requires_candidate_gain_and_matched_control() -> None:
    config = ArchitectureConfirmationConfig.phase_b_confirmation("rescnn")
    rows = []
    for architecture, true_auc, shuffle_auc in (
        ("mlp", 0.520, 0.501),
        ("rescnn", 0.525, 0.500),
    ):
        for shuffled, auc in ((False, true_auc), (True, shuffle_auc)):
            model = f"selected8_{architecture}_{'label_shuffle' if shuffled else 'true_output'}"
            for bit in config.selected_msb_indices:
                rows.append(
                    {
                        "model": model,
                        "architecture": architecture,
                        "msb_index": bit,
                        "threshold_accuracy": 0.515 if not shuffled else 0.5,
                        "majority_accuracy": 0.5,
                        "accuracy_minus_majority": 0.015 if not shuffled else 0.0,
                        "auc": auc,
                        "mse": 0.24,
                        "invalid_numpy_rint_rate": 0.0,
                    }
                )
    training = {
        "rows": rows,
        "summaries": [{"model": str(index)} for index in range(4)],
        "history": [
            {"model": model, "epoch": epoch}
            for model in range(4)
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(4)],
    }

    gate = adjudicate_confirmation(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_selected8_architecture_priority_independently_confirmed"
    assert gate["metrics"]["priority_passed"] is True


def test_confirmation_plot_explains_candidate_control_and_scope(tmp_path: Path) -> None:
    config = ArchitectureConfirmationConfig.smoke("rescnn")
    rows = []
    for model_index, model in enumerate(
        (
            "selected8_mlp_true_output",
            "selected8_mlp_label_shuffle",
            "selected8_rescnn_true_output",
            "selected8_rescnn_label_shuffle",
        )
    ):
        for bit_index, bit in enumerate(config.selected_msb_indices):
            rows.append(
                {
                    "model": model,
                    "msb_index": bit,
                    "auc": 0.5 + 0.002 * model_index + 0.001 * bit_index,
                    "accuracy_minus_majority": 0.001 * model_index,
                }
            )
    summary = {
        "metadata": {
            "mode": "smoke",
            "candidate_architecture": "rescnn",
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "bit_rows": rows,
        "gate": {
            "decision": "innovation2_selected8_architecture_confirmation_local_smoke_passed",
            "metrics": {"mean_candidate_minus_mlp_auc": 0.004},
        },
    }
    output = tmp_path / "curves.svg"

    render_architecture_confirmation(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "第四密钥匹配控制确认" in svg
    assert "位置保持ResCNN" in svg
    assert "标签打乱" in svg
    assert "不是四轮证据" in svg


def test_confirmation_training_cli_import_does_not_require_matplotlib() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import blockcipher_nd.cli.run_innovation2_selected_output_architecture_confirmation
print('import=pass')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import=pass"


def test_result_index_names_opa2_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opa2_present_r3_selected8_rescnn_smoke_20260721"
    )

    assert "OPA2" in name
    assert "第四密钥" in name
    assert "匹配控制确认" in name
