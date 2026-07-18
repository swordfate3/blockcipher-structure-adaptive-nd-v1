from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_cross_spn_r3_profile_operator_synthesis import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.cross_spn_r3_profile_operator_synthesis import (
    FROZEN_SOURCES,
    adjudicate_method_synthesis,
    validate_sources,
)


def _training_rows(seed: int, true_auc: float) -> list[dict[str, object]]:
    return [
        {
            "relation_mode": "independent",
            "seed": seed,
            "epochs_completed": 30,
            "validation_auc": true_auc - 0.25,
        },
        {
            "relation_mode": "true",
            "seed": seed,
            "epochs_completed": 30,
            "validation_auc": true_auc,
        },
        {
            "relation_mode": "corrupted",
            "seed": seed,
            "epochs_completed": 30,
            "validation_auc": true_auc - 0.10,
        },
    ]


def _method_source(name: str, true_auc: float) -> dict[str, object]:
    spec = FROZEN_SOURCES[name]
    gate = {
        "run_id": spec.run_id,
        "status": spec.status,
        "decision": spec.decision,
        "protocol_checks": {"valid": True},
        "metrics": {
            "contract": {
                "input_dim": 13,
                "output_shape": [4, 64],
                "parameter_counts_match": True,
                "parameter_counts": {
                    mode: 4_795 for mode in ("independent", "true", "corrupted")
                },
            },
            "seed0_rows": _training_rows(0, true_auc),
            "seed1_rows": _training_rows(1, true_auc - 0.01),
        },
    }
    metadata = {
        "task": f"innovation2_{name}_r3_only_profile_replication",
        "config": {"epochs": 30, "hidden_dim": 32, "steps": 2},
        "profile_source_run_id": f"{name}_strict_profile_source",
    }
    if name == "gift":
        metadata["checkpoint_transfer"] = False
    return {
        "gate": gate,
        "metadata": metadata,
        "hashes": {"gate.json": "a" * 64, "metadata.json": "b" * 64},
    }


def _gate_source(name: str, metrics: dict[str, object] | None = None) -> dict[str, object]:
    spec = FROZEN_SOURCES[name]
    return {
        "gate": {
            "run_id": spec.run_id,
            "status": spec.status,
            "decision": spec.decision,
            "metrics": metrics or {},
        },
        "hashes": {"gate.json": "c" * 64},
    }


def _sources() -> dict[str, dict[str, object]]:
    return {
        "present": _method_source("present", 0.95),
        "gift": _method_source("gift", 0.91),
        "skinny_r7": _gate_source("skinny_r7"),
        "skinny_r8": _gate_source("skinny_r8"),
        "skinny_adjacent": _gate_source("skinny_adjacent"),
        "skinny_bottom_row": _gate_source("skinny_bottom_row"),
        "skinny_single_cell": _gate_source("skinny_single_cell"),
        "real_spn": _gate_source(
            "real_spn", {"ready_label_family_count": 0}
        ),
    }


def test_e80_confirms_two_cipher_method_and_blocks_skinny_training() -> None:
    sources = _sources()
    gate, rows = adjudicate_method_synthesis("e80-test", sources)

    assert all(validate_sources(sources).values())
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_cross_spn_r3_profile_method_confirmed_"
        "skinny_labels_not_ready"
    )
    assert gate["metrics"]["confirmed_real_spn_count"] == 2
    assert gate["next_action"]["training"] is False
    assert [row["cipher"] for row in rows] == [
        "PRESENT-80",
        "GIFT-64",
        "SKINNY-64/64",
    ]


def test_e80_holds_when_one_cipher_topology_margin_is_too_small() -> None:
    sources = _sources()
    present_rows = sources["present"]["gate"]["metrics"]["seed1_rows"]
    by_mode = {row["relation_mode"]: row for row in present_rows}
    by_mode["corrupted"]["validation_auc"] = (
        by_mode["true"]["validation_auc"] - 0.01
    )

    gate, _ = adjudicate_method_synthesis("e80-test", sources)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_cross_spn_r3_profile_method_not_confirmed"


def test_e80_fails_on_frozen_source_hash_mismatch() -> None:
    sources = _sources()
    sources["skinny_r8"]["hashes"]["gate.json"] = "short"

    gate, _ = adjudicate_method_synthesis("e80-test", sources)

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation2_cross_spn_method_synthesis_protocol_invalid"


def test_plot_writes_clear_chinese_e80_svg(tmp_path: Path) -> None:
    gate, rows = adjudicate_method_synthesis("e80-test", _sources())
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(
        json.dumps({"gate": gate, "rows": rows}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E80" in svg
    assert "禁止把两个面板的AUC高低" in svg
    assert "SKINNY标签条件" in svg
    assert "论文kernel复现不等于训练标签就绪" in svg


def test_e80_fixture_mutation_does_not_leak_between_tests() -> None:
    first = _sources()
    second = deepcopy(first)
    first["present"]["gate"]["status"] = "hold"
    assert second["present"]["gate"]["status"] == "pass"
