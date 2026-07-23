from __future__ import annotations

import csv
import json

from blockcipher_nd.cli import audit_runtime_spn_semantic_equivalence as cli
from blockcipher_nd.tasks.innovation1.runtime_spn_semantic_equivalence import (
    RuntimeSpnSemanticEquivalenceConfig,
    audit_runtime_spn_semantic_equivalence,
)


def test_runtime_spn_same_weight_semantics_are_equivalent() -> None:
    audit = audit_runtime_spn_semantic_equivalence(
        RuntimeSpnSemanticEquivalenceConfig(
            run_id="unit",
            batch_rows=3,
            pairs_per_sample=2,
        )
    )

    assert audit["gate"]["status"] == "pass"
    assert audit["gate"]["first_divergent_stage"] is None
    assert audit["gate"]["maximum_absolute_error"] <= 1e-6
    assert all(row["within_tolerance"] for row in audit["rows"])
    assert audit["gate"]["next_action"]["remote_training_authorized"] is False
    assert audit["gate"]["next_action"]["present_transfer_authorized"] is False
    assert audit["gate"]["next_action"][
        "rerun_frozen_two_seed_gate_required_after_representation_repair"
    ] is True


def test_runtime_spn_semantic_equivalence_cli_writes_complete_artifacts(
    tmp_path,
) -> None:
    assert cli.main(["--output-root", str(tmp_path), "--run-id", "cli-unit"]) == 0

    required = {
        "results.jsonl",
        "progress.jsonl",
        "validation.json",
        "gate.json",
        "summary.json",
        "stage_errors.csv",
        "curves.svg",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    gate = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    with (tmp_path / "stage_errors.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["stage"] == "msb_to_lsb_conversion"
    assert rows[-1]["stage"] == "final_logits"
    svg = (tmp_path / "curves.svg").read_text(encoding="utf-8")
    assert "逐阶段语义等价审计" in svg
    assert "无训练的确定性审计" in svg
