from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.uknit_family_canonical_factorization import (
    RUN_ID,
    apply_sbox_factor,
    apply_uknit_linear_factor,
    load_and_validate_factorization_config,
    recover_uknit_linear_factors,
    recover_uknit_sbox_factors,
    run_canonical_factorization_audit,
    write_factorization_artifacts,
)
from blockcipher_nd.ciphers.spn.uknit import (
    UKNIT_SBOX_TABLES,
    uknit_linear_layer,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_family_canonical_factorization_k0_20260727.json"
)


def test_k0_config_freezes_zero_training_and_public_evidence() -> None:
    config = load_and_validate_factorization_config(CONFIG)

    assert config["run_id"] == RUN_ID
    assert config["audit"] == {
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
    }
    assert len(config["uknit"]["full_vectors"]) == 4
    assert len(config["uknit"]["prefix_zero_states"]) == 11
    assert len(config["dialga"]["full_vectors"]) == 4
    assert len(config["dialga"]["trace"]["states"]) == 16


def test_all_uknit_components_reconstruct_from_canonical_primitives() -> None:
    sbox_factors = recover_uknit_sbox_factors()
    linear_factors = recover_uknit_linear_factors()

    assert sum(len(round_factors) for round_factors in sbox_factors) == 192
    assert len(linear_factors) == 11
    assert all(
        apply_sbox_factor(value, factor) == table[value]
        for round_tables, round_factors in zip(
            UKNIT_SBOX_TABLES, sbox_factors, strict=True
        )
        for table, factor in zip(round_tables, round_factors, strict=True)
        for value in range(16)
    )
    assert all(
        apply_uknit_linear_factor(1 << bit, factor)
        == uknit_linear_layer(1 << bit, round_index)
        for round_index, factor in enumerate(linear_factors)
        for bit in range(64)
    )


def test_k0_audit_passes_all_exact_gates_and_wrong_controls() -> None:
    config = load_and_validate_factorization_config(CONFIG)
    payload = run_canonical_factorization_audit(config)

    assert len(payload["results"]) == 2
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["errors"] == []
    assert all(payload["validation"]["checks"].values())
    assert payload["gate"]["status"] == "pass"
    assert payload["gate"]["training_rows"] == 0
    assert payload["gate"]["optimizer_steps"] == 0
    assert payload["gate"]["remote"] is False
    assert {row["cipher"] for row in payload["results"]} == {
        "uKNIT-BC",
        "Dialga-128",
    }


def test_k0_artifact_writer_emits_exact_tables_without_chart(tmp_path: Path) -> None:
    config = load_and_validate_factorization_config(CONFIG)
    payload = run_canonical_factorization_audit(config)

    write_factorization_artifacts(payload, tmp_path)

    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "validation.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "gate.json").read_text())["run_id"] == RUN_ID
    assert json.loads((tmp_path / "summary.json").read_text())["training_rows"] == 0
    assert not (tmp_path / "curves.svg").exists()
