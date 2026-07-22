from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from blockcipher_nd.cli.audit_innovation2_spn_rescnn_head_identifiability import (
    main as audit_main,
)
from blockcipher_nd.evaluation.result_index import DECISION_LABELS, display_name_for_run
from blockcipher_nd.tasks.innovation2.spn_rescnn_head_identifiability import (
    RUN_ID,
    SpnResCnnHeadIdentifiabilityConfig,
    absorb_final_position_permutation,
    audit_spn_rescnn_head_identifiability,
)


def test_absorbed_head_exactly_matches_final_position_routing() -> None:
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, 3, 8, generator=generator, dtype=torch.float64)
    weight = torch.randn(4, 3, 8, generator=generator, dtype=torch.float64)
    bias = torch.randn(4, generator=generator, dtype=torch.float64)
    permutation = torch.tensor([2, 5, 0, 7, 1, 4, 6, 3])

    routed = torch.nn.functional.linear(
        hidden.index_select(2, permutation).flatten(1), weight.flatten(1), bias
    )
    absorbed = absorb_final_position_permutation(weight, permutation)
    direct = torch.nn.functional.linear(hidden.flatten(1), absorbed.flatten(1), bias)

    torch.testing.assert_close(routed, direct, rtol=0.0, atol=1e-12)


def test_opn1_audits_actual_opc1_head_and_limits_claim_scope() -> None:
    audit = audit_spn_rescnn_head_identifiability(
        SpnResCnnHeadIdentifiabilityConfig()
    )

    assert len(audit["rows"]) == 3
    assert all(audit["gate"]["protocol_checks"].values())
    assert all(audit["gate"]["execution_checks"].values())
    assert audit["gate"]["status"] == "pass"
    assert audit["gate"]["decision"] == (
        "innovation2_spn_rescnn_final_routing_absorbable_by_global_head"
    )
    assert audit["gate"]["metrics"]["head_in_features"] == 252 * 64
    assert audit["gate"]["metrics"]["head_out_features"] == 8
    assert audit["gate"]["metrics"]["absorbable_routed_stage_indices_zero_based"] == [
        2
    ]
    assert audit["gate"]["metrics"][
        "not_proven_absorbable_routed_stage_indices_zero_based"
    ] == [0, 1]
    assert audit["gate"]["next_action"]["opc1_unchanged"] is True
    assert audit["gate"]["next_action"]["remote_training_authorized"] is False


def test_opn1_cli_writes_checksummed_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "audit"

    assert audit_main(["--output-root", str(output)]) == 0
    assert {
        "results.jsonl",
        "summary.json",
        "gate.json",
        "metadata.json",
        "progress.jsonl",
        "artifact_manifest.json",
        "validation.json",
    }.issubset(path.name for path in output.iterdir())
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for row in manifest:
        artifact = output / row["path"]
        assert row["bytes"] == artifact.stat().st_size
        assert row["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "pass"
    assert all(validation["checks"].values())


def test_result_index_names_and_explains_opn1() -> None:
    decision = "innovation2_spn_rescnn_final_routing_absorbable_by_global_head"

    assert "OPN1" in display_name_for_run(RUN_ID)
    assert "输出头" in display_name_for_run(RUN_ID)
    assert "最后一次" in DECISION_LABELS[decision]
