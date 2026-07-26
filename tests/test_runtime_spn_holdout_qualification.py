from __future__ import annotations

import json
from pathlib import Path

import pytest

from blockcipher_nd.tasks.innovation1.runtime_spn_holdout_qualification import (
    CANDIDATES,
    adjudicate_holdout_qualification,
    atomic_gf2_relation_types,
    load_and_validate_holdout_qualification_config,
    run_holdout_qualification_audit,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    _load_structures,
    load_and_validate_holdout_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_holdout_qualification_a7_20260726.json"
)


def test_frozen_a7_config_and_atomic_relation_counts() -> None:
    config = load_and_validate_holdout_qualification_config(
        CONFIG,
        project_root=ROOT,
    )
    base = load_and_validate_holdout_config(
        ROOT / config["protocol"]["config_path"]
    )
    structures = _load_structures(base)

    assert tuple(config["protocol"]["candidate_order"]) == CANDIDATES
    assert len(atomic_gf2_relation_types(structures["rectangle80"])) == 4
    assert len(atomic_gf2_relation_types(structures["uknit64"])) == 16
    assert len(atomic_gf2_relation_types(structures["dialga128"])) == 16


def test_a7_recomputes_frozen_evidence_and_selects_dialga() -> None:
    config = load_and_validate_holdout_qualification_config(
        CONFIG,
        project_root=ROOT,
    )

    payload = run_holdout_qualification_audit(
        config=config,
        project_root=ROOT,
    )

    assert payload["validation"]["status"] == "pass"
    assert payload["gate"]["status"] == "pass"
    assert payload["gate"]["selected_holdout"] == "dialga128"
    assert payload["gate"]["eligible_candidates"] == ["dialga128"]
    assert payload["gate"]["per_candidate"]["rectangle80"][
        "technically_qualified"
    ]
    assert payload["gate"]["per_candidate"]["rectangle80"][
        "previous_whole_cipher_holdout"
    ]
    assert not payload["gate"]["per_candidate"]["uknit64"][
        "technically_qualified"
    ]
    assert len(payload["rows"]) == 6
    assert {row["training_performed"] for row in payload["rows"]} == {False}
    assert {row["new_data_generated"] for row in payload["rows"]} == {False}
    assert payload["structure_profiles"]["dialga128"][
        "atomic_gf2_coverage"
    ] == 1.0
    assert payload["structure_profiles"]["dialga128"][
        "exact_source_sbox_overlap"
    ] == 0


def test_a7_rejects_frozen_evidence_hash_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["evidence"]["rectangle80"]["gate"]["sha256"] = "0" * 64
    drifted = tmp_path / "a7.json"
    drifted.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen evidence hash drifted"):
        load_and_validate_holdout_qualification_config(
            drifted,
            project_root=ROOT,
        )


def test_a7_gate_stops_when_no_unused_candidate_qualifies() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = []
    for candidate in CANDIDATES:
        for seed in (0, 1):
            rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "correct_auc": 0.54,
                    "corrupted_auc": 0.50,
                    "no_topology_auc": 0.50,
                    "correct_minus_corrupted": 0.04,
                    "correct_minus_no_topology": 0.04,
                    "atomic_gf2_coverage": 1.0,
                    "cell_relabel_max_error": 0.0,
                    "evidence_valid": True,
                    "previous_whole_cipher_holdout": False,
                    "target_atomic_gf2_types": 16,
                    "covered_atomic_gf2_types": 16,
                    "target_unique_sboxes": 1,
                    "exact_source_sbox_overlap": 0,
                }
            )

    gate = adjudicate_holdout_qualification(
        config=config,
        rows=rows,
        validation={"status": "pass"},
    )

    assert gate["status"] == "hold"
    assert gate["selected_holdout"] is None
    assert gate["decision"].endswith("none_selected")
