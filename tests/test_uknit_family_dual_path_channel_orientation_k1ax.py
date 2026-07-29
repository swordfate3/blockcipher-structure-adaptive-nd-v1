from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_uknit_family_dual_path_channel_orientation_k1ax import (
    render_k1ax_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_channel_orientation_k1ax import (
    CONDITIONS,
    EXPECTED_ROWS,
    adjudicate_audit,
    collect_panel_metrics,
    load_and_validate_config,
    load_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    build_candidate,
)


def test_k1ax_config_and_frozen_authority_are_exact() -> None:
    config = load_and_validate_config()
    (
        readiness,
        _k1av,
        dataset_rows,
        datasets,
        structures,
        controls,
        summary_rows,
        checkpoints,
        checks,
    ) = load_authority(config)

    assert config["evaluation"]["training_performed"] is False
    assert config["evaluation"]["optimizer_steps"] == 0
    assert config["evaluation"]["expected_rows"] == EXPECTED_ROWS
    assert "16-pair expansion" in config["blocked_actions"]
    assert len(readiness["ciphers"]) == 3
    assert len(dataset_rows) == len(datasets) == 18
    assert len(structures) == len(controls) == len(summary_rows) == 3
    assert set(checkpoints) == {0, 1}
    assert all(checks.values()), checks


def test_k1ax_path_decomposition_replays_frozen_model() -> None:
    config = load_and_validate_config()
    (
        readiness,
        k1av,
        _dataset_rows,
        datasets,
        structures,
        controls,
        _summary_rows,
        checkpoints,
        checks,
    ) = load_authority(config)
    assert all(checks.values())
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    model = build_candidate(
        cipher_configs["uknit64"], readiness["model"], k1av["model"]
    )
    model.load_state_dict(checkpoints[0]["state_dict"], strict=True)
    dataset = datasets[("uknit64", 3, "same_key_fresh")]
    indices = np.concatenate(
        (
            np.flatnonzero(dataset.labels == 0)[:32],
            np.flatnonzero(dataset.labels == 1)[:32],
        )
    )
    tiny_dataset = replace(
        dataset,
        features=np.array(dataset.features[indices], copy=True),
        labels=np.array(dataset.labels[indices], copy=True),
    )

    rows = collect_panel_metrics(
        model=model,
        dataset=tiny_dataset,
        structure=structures["uknit64"],
        summaries=controls["uknit64"],
        replica=0,
        cipher="uknit64",
        seed=3,
        split="same_key_fresh",
        checkpoint=checkpoints[0],
        batch_size=64,
        device="cpu",
    )

    assert len(rows) == 4
    assert {row["condition"] for row in rows} == set(CONDITIONS)
    assert all(row["maximum_full_forward_replay_delta"] <= 1e-7 for row in rows)
    assert all(row["training_performed"] is False for row in rows)
    assert all(row["optimizer_steps"] == 0 for row in rows)
    assert all(row["state_immutable"] is True for row in rows)


def test_k1ax_gate_prioritizes_routing_harm_cancellation_and_invalid(
    tmp_path: Path,
) -> None:
    config = load_and_validate_config()
    checkpoints = _synthetic_checkpoints(tmp_path)
    aligned = _synthetic_rows()

    unresolved = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        rows=aligned,
        checkpoints=checkpoints,
    )
    assert unresolved["status"] == "hold"
    assert "mechanism_unresolved" in unresolved["decision"]

    misrouted = deepcopy(aligned)
    for row in misrouted:
        if row["condition"] == "sbox_only_mismatch":
            row["effective_edge_gate"] = 0.13
            row["effective_transition_gate"] = 0.11
    routing = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        rows=misrouted,
        checkpoints=checkpoints,
    )
    assert routing["status"] == "pass"
    assert "component_routing_misalignment_supported" in routing["decision"]
    assert "K1-AY" in routing["next_action"]

    harmful = deepcopy(aligned)
    changed = 0
    for row in harmful:
        if row["condition"] == "correct_descriptor" and changed < 3:
            row["mean_signed_transition_full_context_probability"] = -0.001
            changed += 1
    harm = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        rows=harmful,
        checkpoints=checkpoints,
    )
    assert "learned_path_harm_supported" in harm["decision"]

    cancelling = deepcopy(aligned)
    changed = 0
    for row in cancelling:
        if row["condition"] == "correct_descriptor" and changed < 3:
            row["path_cancellation_fraction"] = 0.6
            changed += 1
    cancellation = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        rows=cancelling,
        checkpoints=checkpoints,
    )
    assert "path_cancellation_supported" in cancellation["decision"]

    invalid_rows = deepcopy(aligned)
    invalid_rows[0]["maximum_full_forward_replay_delta"] = 1e-4
    invalid = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        rows=invalid_rows,
        checkpoints=checkpoints,
    )
    assert invalid["status"] == "invalid"
    assert "full_forward_replay_exact" in invalid["failed_protocol_checks"]


def test_k1ax_plot_writes_four_panel_chinese_svg(tmp_path: Path) -> None:
    gate = adjudicate_audit(
        config=load_and_validate_config(),
        source_checks={"source": True},
        rows=_synthetic_rows(),
        checkpoints=_synthetic_checkpoints(tmp_path),
    )
    output = tmp_path / "curves.svg"

    report = render_k1ax_svg(gate, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["audit_panels"] == 12
    assert "共享结构编码器把两类语义混入了错误门控" in text
    assert "不授权16 pairs" in text


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    gates = {
        "correct_descriptor": (0.10, 0.10),
        "full_mismatch": (0.12, 0.12),
        "sbox_only_mismatch": (0.101, 0.12),
        "linear_only_mismatch": (0.12, 0.101),
    }
    for replica in (0, 1):
        for cipher in EXPECTED_CIPHERS:
            for split in FRESH_SPLITS:
                for condition in CONDITIONS:
                    edge, transition = gates[condition]
                    rows.append(
                        {
                            "replica": replica,
                            "cipher_key": cipher,
                            "split": split,
                            "condition": condition,
                            "effective_edge_gate": edge,
                            "effective_transition_gate": transition,
                            "mean_signed_edge_standalone_probability": 0.001,
                            "mean_signed_transition_standalone_probability": 0.001,
                            "mean_signed_edge_full_context_probability": 0.001,
                            "mean_signed_transition_full_context_probability": 0.001,
                            "path_opposition_fraction": 0.1,
                            "path_cancellation_fraction": 0.1,
                            "maximum_full_forward_replay_delta": 0.0,
                            "path_aucs": {
                                "pure_base": 0.60,
                                "edge_only": 0.61,
                                "transition_only": 0.61,
                                "full_dual_path": 0.62,
                            },
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "runtime_structure_held_correct": True,
                            "runtime_structure_cipher_key": cipher,
                            "state_immutable": True,
                        }
                    )
    return rows


def _synthetic_checkpoints(tmp_path: Path) -> dict[int, dict[str, object]]:
    checkpoints = {}
    for replica in (0, 1):
        path = tmp_path / f"replica{replica}.pt"
        path.write_bytes(f"checkpoint-{replica}".encode("ascii"))
        checkpoints[replica] = {"path": str(path), "sha256": file_sha256(path)}
    return checkpoints
