from __future__ import annotations

import json
import pickle
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_generalized_integral_relation_contract import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ATM_COMMIT,
    ATM_EXPECTED_RESULT_FILES,
    audit_atm_results,
)
from blockcipher_nd.tasks.innovation2.generalized_integral_relation_contract import (
    GeneralizedRelationContractConfig,
    audit_relation_overlap,
    evaluate_generalized_relation_contract,
    inspect_present_key_model,
)


def _write_atm_fixture(root: Path) -> Path:
    present = root / "Ciphers/PRESENT"
    results = present / "Results"
    construction = root / "Construction"
    results.mkdir(parents=True)
    construction.mkdir(parents=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "PRESENT_sbox = object()\n",
                    "PRESENT_bitpermutation = list(range(64))\n",
                    "splits = [(1,5,3), (2,4,3)]\n",
                    "for split in splits:\n",
                    "    f1 = construct_iterated_cipher([PRESENT_roundfunction]*split[0], [2**64-1]*split[0] + [0])\n",
                    "    f2 = construct_iterated_cipher([PRESENT_roundfunction]*split[1], [2**64-1]*(split[1]+1))\n",
                    "    f3 = construct_iterated_cipher([PRESENT_roundfunction]*split[2], [0] + [2**64-1]*split[2])\n",
                ],
            }
        ]
    }
    (present / "PRESENT.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (construction / "IteratedCipher.py").write_text(
        "def f(key_masks):\n    for km in key_masks:\n        f.connect_to_key(0, 0)\n",
        encoding="utf-8",
    )
    common = frozenset({(1, 1)})
    for index, name in enumerate(ATM_EXPECTED_RESULT_FILES):
        payload = {common, frozenset({(index + 2, 1 << (index % 4))})}
        (results / name).write_bytes(pickle.dumps(payload))
    return results


def test_key_model_audit_detects_independent_round_keys(tmp_path: Path) -> None:
    _write_atm_fixture(tmp_path)

    result = inspect_present_key_model(tmp_path)

    assert all(result["checks"].values())
    assert result["semantics"]["independent_round_key_model"] is True
    assert result["semantics"]["present80_key_schedule_implemented"] is False
    assert result["semantics"]["actual_present80_key_model_match"] is False


def test_relation_overlap_rejects_file_identity_split_and_has_no_negatives(
    tmp_path: Path,
) -> None:
    results = _write_atm_fixture(tmp_path)

    overlap = audit_relation_overlap(results)

    assert overlap["metrics"]["files"] == 8
    assert overlap["metrics"]["deduplicated_relations"] == 9
    assert overlap["metrics"]["relations_common_to_all_files"] == 1
    assert overlap["metrics"]["relations_unique_to_one_file"] == 8
    assert overlap["checks"]["file_split_is_relation_disjoint"] is False
    assert overlap["metrics"]["proven_key_dependent_negative_relations"] == 0


def test_contract_holds_when_positive_basis_has_no_valid_negative_or_key_schedule(
    tmp_path: Path,
) -> None:
    results = _write_atm_fixture(tmp_path)
    atm = audit_atm_results(results, published_dimension=9)
    key_model = inspect_present_key_model(tmp_path)
    overlap = audit_relation_overlap(results)
    config = GeneralizedRelationContractConfig(
        run_id="e56_smoke",
        mode="smoke",
        minimum_positives=8,
        minimum_negatives=8,
    )

    result = evaluate_generalized_relation_contract(
        config,
        actual_commit=ATM_COMMIT,
        atm_audit=atm,
        key_model=key_model,
        overlap=overlap,
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_generalized_relation_label_contract_not_ready"
    )
    assert result["gate"]["generalized_relation_checks"][
        "deduplicated_positive_width_at_least_256"
    ]
    assert not result["gate"]["generalized_relation_checks"][
        "proven_negative_width_at_least_256"
    ]
    assert not result["gate"]["original_target_checks"][
        "actual_present80_master_key_schedule"
    ]
    assert result["gate"]["next_action"]["training"] is False


def test_plot_exposes_positive_negative_and_key_model_boundaries(tmp_path: Path) -> None:
    summary = {
        "source_contract": {
            "key_model": {
                "key_model": "independent round keys; no PRESENT-80 key schedule"
            }
        },
        "metrics": {
            "deduplicated_relations": 470,
            "relations_common_to_all_files": 305,
            "relations_unique_to_one_file": 15,
            "proven_key_dependent_negative_relations": 0,
            "minimum_negative_relations": 256,
        },
        "relation_overlap": {
            "files": [
                {"relations": 338 + index, "common_to_all_files": 305}
                for index in range(8)
            ]
        },
        "gate": {
            "decision": "innovation2_generalized_relation_label_contract_not_ready",
            "generalized_relation_checks": {
                "positive_membership_semantics_defined": True,
                "deduplicated_positive_width_at_least_256": True,
                "proven_negative_width_at_least_256": False,
                "negative_witnesses_replayable": False,
                "public_file_split_relation_disjoint": False,
            },
            "original_target_checks": {
                "actual_present80_master_key_schedule": False,
                "constant_zero_or_one_known": False,
                "original_linear_mask_balance_mapping_complete": False,
            },
        },
    }
    source = tmp_path / "summary.json"
    output = tmp_path / "curves.svg"
    source.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(source), "--output", str(output)]) == 0
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E56" in svg
    assert "不是PRESENT-80平衡标签" in svg
    assert "严格负类为0" in svg
