from __future__ import annotations

from copy import deepcopy
import json

import matplotlib.pyplot as plt

from blockcipher_nd.cli.plot_uknit_family_multicipher_path_contribution_k1ar import (
    render_k1ar_svg,
)

from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_path_contribution_k1ar import (
    CHECKPOINT_FAMILIES,
    EXPECTED_CIPHERS,
    FRESH_SPLITS,
    REPLICAS,
    adjudicate,
    load_and_validate_config,
    load_authority,
)


def test_k1ar_config_and_bound_sources_are_exact() -> None:
    config = load_and_validate_config()
    readiness, datasets, checkpoints, controls, dataset_rows, checks = (
        load_authority(config)
    )
    assert config["evaluation"]["expected_rows"] == 24
    assert set(checkpoints) == set(CHECKPOINT_FAMILIES)
    assert all(set(rows) == set(REPLICAS) for rows in checkpoints.values())
    assert all(len(rows) == 36 for rows in controls.values())
    assert len(datasets) == 18
    assert dataset_rows
    assert all(
        int(cipher["input_bits"]) // int(cipher["pair_bits"]) == 4
        for cipher in readiness["ciphers"]
    )
    assert all(checks.values())


def _synthetic_rows(*, midori_delta: float, non_midori_delta: float) -> list[dict]:
    rows = []
    for family in CHECKPOINT_FAMILIES:
        for replica in REPLICAS:
            for cipher in EXPECTED_CIPHERS:
                for split in FRESH_SPLITS:
                    baseline_gain = 0.02
                    delta = midori_delta if cipher == "midori64" else non_midori_delta
                    gain = baseline_gain if family == "equal_loss_k1ao" else baseline_gain + delta
                    rows.append(
                        {
                            "checkpoint_family": family,
                            "replica": replica,
                            "cipher_key": cipher,
                            "split": split,
                            "transition_gain_auc": gain,
                            "transition_delta_auc": 0.55 + gain,
                            "transition_to_base_rms_ratio": 0.1,
                            "mean_signed_transition_probability": 0.01 + gain,
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "state_immutable": True,
                            "full_auc_replay_delta": 0.0,
                            "edge_auc_replay_delta": 0.0,
                            "max_abs_full_forward_replay_delta": 0.0,
                            "max_abs_edge_forward_replay_delta": 0.0,
                        }
                    )
    return rows


def test_k1ar_gate_supports_stable_heterogeneous_transition_demand() -> None:
    gate = adjudicate(
        source_checks={"sources_exact": True},
        rows=_synthetic_rows(midori_delta=0.02, non_midori_delta=-0.01),
    )
    assert gate["status"] == "pass"
    assert gate["heterogeneous_transition_demand_supported"] is True
    assert gate["midori_direction_pass_count"] == 4
    assert gate["non_midori_direction_pass_count"] == 8
    assert "structure-derived" in gate["next_action"]


def test_k1ar_gate_holds_when_all_ciphers_move_together() -> None:
    gate = adjudicate(
        source_checks={"sources_exact": True},
        rows=_synthetic_rows(midori_delta=0.02, non_midori_delta=0.02),
    )
    assert gate["status"] == "hold"
    assert gate["heterogeneous_transition_demand_supported"] is False
    assert "projection geometry" in gate["next_action"]


def test_k1ar_gate_fails_closed_on_replay_drift() -> None:
    rows = _synthetic_rows(midori_delta=0.02, non_midori_delta=-0.01)
    broken = deepcopy(rows)
    broken[0]["full_auc_replay_delta"] = 1e-3
    gate = adjudicate(source_checks={"sources_exact": True}, rows=broken)
    assert gate["status"] == "fail"
    assert "all_forward_replays_exact" in gate["failed_protocol_checks"]


def test_k1ar_plot_writes_four_panel_chinese_svg(tmp_path, monkeypatch) -> None:
    gate = adjudicate(
        source_checks={"sources_exact": True},
        rows=_synthetic_rows(midori_delta=0.02, non_midori_delta=-0.01),
    )
    output = tmp_path / "curves.svg"
    seen = {}
    original = plt.Figure.savefig

    def capture(self, *args, **kwargs):
        seen["axes"] = len(self.axes)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    report = render_k1ar_svg(gate, output)
    text = output.read_text(encoding="utf-8")
    assert seen["axes"] == 4
    assert report["comparison_panels"] == 12
    assert "三种 SPN 是否需要不同的结构分支强度" in text
    assert "2048/class/cipher" in text
    assert "不是正式规模" in text
    assert json.dumps(report, ensure_ascii=False)
