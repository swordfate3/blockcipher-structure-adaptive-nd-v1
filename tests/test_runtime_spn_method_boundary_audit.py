from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blockcipher_nd.tasks.innovation1.runtime_spn_method_boundary_audit import (
    EVIDENCE_IDS,
    RUN_ID,
    load_and_validate_audit_config,
    run_method_boundary_audit,
    write_method_boundary_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_method_boundary_c2_20260726.json"
)


def test_frozen_c2_config_contract_is_valid() -> None:
    config = load_and_validate_audit_config(CONFIG, project_root=ROOT)

    assert config["run_id"] == RUN_ID
    assert config["audit"]["training_rows"] == 0
    assert config["audit"]["optimizer_steps"] == 0
    assert config["audit"]["remote"] is False
    assert set(config["evidence"]) == set(EVIDENCE_IDS)
    assert "gift_r2g_seed0" in config["evidence"]
    assert "gift_r2f_seed0" not in config["evidence"]


def test_c2_freezes_supported_and_unsupported_method_boundary(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_and_validate_audit_config(config_path, project_root=tmp_path)

    payload = run_method_boundary_audit(config=config, project_root=tmp_path)
    statuses = {row["requirement_id"]: row["status"] for row in payload["results"]}

    assert statuses == {
        "R1": "supported",
        "R2": "supported",
        "R3": "supported",
        "R4": "supported",
        "R5": "supported",
        "R6": "partial",
        "R7": "contradicted",
        "R8": "contradicted",
        "R9": "supported",
        "R10": "contradicted",
        "R11": "contradicted",
        "R12": "contradicted",
    }
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["evidence_count"] == 13
    assert payload["validation"]["requirement_count"] == 12
    assert payload["gate"]["status"] == "pass"
    assert payload["gate"]["method_status"] == "partial"
    assert payload["gate"]["universal_runtime_spn_supported"] is False
    assert payload["gate"]["decision"].endswith("method_boundary_frozen")

    output_root = tmp_path / "artifacts"
    write_method_boundary_artifacts(payload=payload, output_root=output_root)

    result_rows = (output_root / "results.jsonl").read_text().splitlines()
    assert len(result_rows) == 12
    assert (output_root / "validation.json").is_file()
    assert (output_root / "gate.json").is_file()
    assert (output_root / "summary.json").is_file()
    assert not (output_root / "curves.svg").exists()


def test_c2_rejects_evidence_digest_drift(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_and_validate_audit_config(config_path, project_root=tmp_path)
    evidence_path = tmp_path / config["evidence"]["runtime_r0"]["path"]
    evidence_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_method_boundary_audit(config=config, project_root=tmp_path)


def test_c2_rejects_nonzero_optimizer_contract(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["audit"]["optimizer_steps"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer_steps"):
        load_and_validate_audit_config(config_path, project_root=tmp_path)


def _write_fixture(root: Path) -> Path:
    payloads = _fixture_payloads()
    evidence_specs = {}
    evidence_root = root / "evidence"
    evidence_root.mkdir(parents=True)
    for evidence_id in EVIDENCE_IDS:
        payload = payloads[evidence_id]
        path = evidence_root / f"{evidence_id}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence_specs[evidence_id] = {
            "path": str(path.relative_to(root)),
            "expected_run_id": payload["run_id"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provenance": f"fixture {evidence_id}",
        }
    config = {
        "run_id": RUN_ID,
        "audit": {
            "training_rows": 0,
            "optimizer_steps": 0,
            "remote": False,
            "control_margin": 0.005,
        },
        "evidence": evidence_specs,
    }
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def _fixture_payloads() -> dict[str, dict]:
    protocol = {"contract": True}
    one_to_one = {
        evidence_id: {
            "run_id": evidence_id,
            "status": "pass",
            "protocol_checks": protocol,
            "aucs": {"true": 0.65},
            "margins": {
                "true_minus_corrupted": 0.05,
                "true_minus_independent": 0.10,
            },
        }
        for evidence_id in (
            "gift_r2g_seed0",
            "gift_r2g_seed1",
            "present_t1_seed0",
            "present_t1_seed1",
        )
    }
    responsiveness = {
        f"seed{seed}_{cipher}": True
        for seed in (0, 1)
        for cipher in ("gift", "present", "skinny", "rectangle", "dialga")
    }
    false_semantics = {
        "dialga_identity_margin": False,
        "dialga_input_permuted_margin": False,
        "source_identity_margin": False,
        "source_input_permuted_margin": False,
    }
    payloads = {
        "runtime_r0": {
            "run_id": "runtime_r0",
            "readiness_checks": {
                "shared_parameter_geometry_stable": True,
                "runtime_structure_absent_from_state": True,
                "variable_width_and_pair_shapes_valid": True,
                "cell_relabel_equivariance": True,
                "four_runtime_structures_covered": True,
                "exact_gf2_inverses_valid": True,
                "permutation_and_general_gf2_supported": True,
                "permutation_gather_matches_gf2": True,
            },
        },
        **one_to_one,
        "skinny_t2a": {
            "run_id": "skinny_t2a",
            "status": "pass",
            "category_counts": {"general_gf2": {"passed": 5, "total": 5}},
        },
        "skinny_rtg3a": {
            "run_id": "skinny_rtg3a",
            "status": "pass",
            "samples_per_class": 1_000_000,
            "protocol_checks": protocol,
            "sources": [
                {
                    "seed": seed,
                    "status": "pass",
                    "aucs": {"true": 0.65},
                    "margins": {
                        "true_minus_corrupted": 0.05,
                        "true_minus_independent": 0.14,
                    },
                }
                for seed in (0, 1)
            ],
        },
        "rectangle_h1": {
            "run_id": "rectangle_h1",
            "full_pass": False,
            "per_seed": {
                "0": {"checks": {"target_controls": True}},
                "1": {"checks": {"target_controls": False}},
            },
        },
        "uknit_a6": {
            "run_id": "uknit_a6",
            "protocol_valid": True,
            "full_pass": False,
            "functional_pass": False,
            "target_training_rows": 0,
            "target_optimizer_steps": 0,
            "per_seed": {
                "0": {"functional_pass": False},
                "1": {"functional_pass": False},
            },
        },
        "dialga_a8": {
            "run_id": "dialga_a8",
            "full_pass": False,
            "per_seed": {
                "0": {"checks": {"target_topology_margins": True}},
                "1": {"checks": {"target_topology_margins": True}},
            },
        },
        "sbox_s1": {
            "run_id": "sbox_s1",
            "responsiveness": responsiveness,
            "research_checks": {
                "descriptor_responsive_every_seed_cipher": True,
                "source_macro_sbox_identifiable": False,
                "dialga_holdout_sbox_identifiable": False,
            },
        },
        "sbox_s2": {
            "run_id": "sbox_s2",
            "protocol_valid": True,
            "full_pass": False,
            "per_seed": {
                "0": {"checks": false_semantics},
                "1": {"checks": false_semantics},
            },
        },
        "topology_c1": {
            "run_id": "topology_c1",
            "full_pass": False,
            "per_seed": {
                "0": {
                    "checks": {
                        "corrupted_topology_margin": True,
                        "no_topology_margin": True,
                    }
                },
                "1": {
                    "checks": {
                        "corrupted_topology_margin": True,
                        "no_topology_margin": True,
                    }
                },
            },
        },
    }
    return payloads
