from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.cli.plot_innovation2_selected_output_topology_attribution import (
    render_topology_attribution,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputPresentSpn,
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_topology_attribution import (
    OPA2_REMOTE_RUN_ID,
    TOPOLOGY_SPECS,
    TopologyAttributionConfig,
    adjudicate_topology,
    authorize_from_opa2_gate,
    prepare_topology_data,
    topology_parameter_counts,
    train_topology_matrix,
    validate_topology_contract,
)


def _opa2_gate(mean_auc: float = 0.99) -> dict[str, object]:
    return {
        "run_id": OPA2_REMOTE_RUN_ID,
        "status": "pass",
        "decision": "innovation2_selected8_architecture_priority_independently_confirmed",
        "candidate_architecture": "present_spn",
        "protocol_checks": {"valid": True},
        "execution_checks": {"complete": True},
        "metrics": {
            "priority_passed": True,
            "architectures": {"present_spn": {"mean_true_auc": mean_auc}},
        },
    }


def _tiny_config() -> TopologyAttributionConfig:
    return TopologyAttributionConfig(
        run_id="test-opa3",
        train_rows=8,
        test_rows=8,
        mlp_hidden_dim=16,
        lstm_hidden_dim=4,
        lstm_layers=1,
        present_spn_dim=4,
        present_spn_blocks=1,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
    )


def test_three_topology_mappings_are_frozen_permutations() -> None:
    exact = _present_topology_mapping("exact")
    identity = _present_topology_mapping("identity")
    wrong = _present_topology_mapping("wrong")

    assert TOPOLOGY_SPECS == (
        ("present_spn_exact_p_true_output", "present_spn_exact_p"),
        ("present_spn_identity_p_true_output", "present_spn_identity_p"),
        ("present_spn_wrong_p_true_output", "present_spn_wrong_p"),
    )
    assert identity.tolist() == list(range(64))
    assert torch.all(wrong != exact)
    assert all(
        sorted(mapping.tolist()) == list(range(64))
        for mapping in (exact, identity, wrong)
    )
    for destination_msb, source_msb in enumerate(exact.tolist()):
        source_integer = 63 - source_msb
        destination_integer = 63 - destination_msb
        assert Present80.permutation_layer(1 << source_integer) == 1 << destination_integer


def test_present_spn_rejects_non_permutation_mapping() -> None:
    with pytest.raises(ValueError, match="64-position permutation"):
        SelectedOutputPresentSpn(
            token_dim=4,
            blocks=1,
            source_for_destination=torch.zeros(64, dtype=torch.long),
        )


def test_topology_models_have_identical_parameters_and_output_shape() -> None:
    config = _tiny_config()
    counts = topology_parameter_counts(config)

    assert len(set(counts.values())) == 1
    for mode in ("exact", "identity", "wrong"):
        model = SelectedOutputPresentSpn(
            token_dim=4,
            blocks=1,
            source_for_destination=_present_topology_mapping(mode),
        )
        assert model(torch.zeros((2, 64))).shape == (2, 8)


def test_opa2_gate_is_the_only_formal_authority() -> None:
    assert authorize_from_opa2_gate(_opa2_gate(0.98)) == pytest.approx(0.98)

    blocked = _opa2_gate()
    blocked["status"] = "hold"
    with pytest.raises(ValueError, match="did not authorize"):
        authorize_from_opa2_gate(blocked)
    invalid = _opa2_gate()
    invalid["metrics"]["priority_passed"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="priority gate"):
        authorize_from_opa2_gate(invalid)


def test_tiny_topology_matrix_is_complete_and_replayable(tmp_path: Path) -> None:
    config = _tiny_config()
    opa2_gate = _opa2_gate(0.5)
    data = prepare_topology_data(config, tmp_path)
    checks = validate_topology_contract(config, data, opa2_gate)
    training = train_topology_matrix(config, data, tmp_path)
    gate = adjudicate_topology(config, checks, training, opa2_gate)

    assert all(checks.values())
    assert len(training["rows"]) == 24
    assert len(training["history"]) == 3
    assert len(training["checkpoints"]) == 3
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("local_smoke_passed")
    assert gate["next_action"]["reopens_controlled_r4_gate"] is False


def test_formal_gate_requires_exact_topology_gain_and_opa2_reproduction() -> None:
    config = TopologyAttributionConfig.formal(device="cpu")
    aucs = {
        "present_spn_exact_p_true_output": 0.900,
        "present_spn_identity_p_true_output": 0.600,
        "present_spn_wrong_p_true_output": 0.550,
    }
    rows = []
    for model, architecture in TOPOLOGY_SPECS:
        for bit in config.selected_msb_indices:
            rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "msb_index": bit,
                    "threshold_accuracy": 0.85,
                    "majority_accuracy": 0.5,
                    "accuracy_minus_majority": 0.35,
                    "auc": aucs[model],
                    "mse": 0.1,
                    "invalid_numpy_rint_rate": 0.0,
                }
            )
    training = {
        "rows": rows,
        "summaries": [{"model": model} for model in aucs],
        "history": [
            {"model": model, "epoch": epoch}
            for model in aucs
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(3)],
    }

    gate = adjudicate_topology(config, {"valid": True}, training, _opa2_gate(0.9))

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_selected8_present_topology_independently_attributed"
    assert gate["metrics"]["attributed_bit_count"] == 8
    assert gate["next_action"]["reopens_controlled_r4_gate"] is True


def test_topology_plot_has_plain_chinese_scope_and_controls(tmp_path: Path) -> None:
    config = _tiny_config()
    rows = []
    for model_index, (model, architecture) in enumerate(TOPOLOGY_SPECS):
        for bit_index, bit in enumerate(config.selected_msb_indices):
            rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "msb_index": bit,
                    "auc": 0.55 - 0.01 * model_index + 0.001 * bit_index,
                }
            )
    summary = {
        "metadata": {
            "mode": "smoke",
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "bit_rows": rows,
        "gate": {
            "decision": "innovation2_selected8_topology_attribution_local_smoke_passed",
            "metrics": {"exact_minus_best_control_mean_auc": 0.01},
        },
    }
    output = tmp_path / "curves.svg"

    render_topology_attribution(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "真实P-layer拓扑归因" in svg
    assert "Identity" in svg
    assert "固定错误P-layer" in svg
    assert "不是四轮证据" in svg


def test_training_cli_import_does_not_require_matplotlib() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import blockcipher_nd.cli.run_innovation2_selected_output_topology_attribution
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


def test_result_index_names_opa3_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opa3_present_r3_selected8_topology_attribution_smoke_20260722"
    )

    assert "OPA3" in name
    assert "真实P-layer" in name
    assert "错误拓扑" in name
