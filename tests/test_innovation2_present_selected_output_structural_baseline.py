from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_selected_output_structural_baseline import (
    main as audit_main,
)
from blockcipher_nd.evaluation.result_index import DECISION_LABELS, display_name_for_run
from blockcipher_nd.tasks.innovation2.present_selected_output_structural_baseline import (
    RUN_ID,
    PresentSelectedOutputStructuralBaselineConfig,
    audit_present_selected_output_structural_baseline,
    sbox_coordinate_metrics,
)


def test_present_sbox_coordinate_metrics_are_exact() -> None:
    metrics = [sbox_coordinate_metrics(bit) for bit in range(4)]

    assert [row["weight"] for row in metrics] == [8, 8, 8, 8]
    assert [row["anf_degree"] for row in metrics] == [2, 3, 3, 3]
    assert [row["nonlinearity"] for row in metrics] == [4, 4, 4, 4]
    assert [row["anf_terms"] for row in metrics] == [4, 7, 8, 8]


def test_structural_baseline_audits_all_bits_and_selected_positions() -> None:
    audit = audit_present_selected_output_structural_baseline(
        PresentSelectedOutputStructuralBaselineConfig()
    )

    assert len(audit["rows"]) == 64
    assert sum(row["selected"] for row in audit["rows"]) == 8
    assert all(audit["gate"]["protocol_checks"].values())
    assert all(audit["gate"]["execution_checks"].values())
    assert all(audit["gate"]["coarse_baseline_checks"].values())
    assert audit["gate"]["status"] == "pass"
    assert audit["gate"]["metrics"]["round1_input_cone_widths"] == [4]
    assert audit["gate"]["metrics"]["round2_input_cone_widths"] == [16]
    assert audit["gate"]["metrics"]["round3_input_cone_widths"] == [64]
    assert audit["gate"]["metrics"]["selected_sbox_output_bits_lsb"] == [1, 3]
    assert audit["gate"]["next_action"]["opc1_unchanged"] is True


def test_cli_writes_checksummed_structural_baseline_artifacts(tmp_path: Path) -> None:
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
    assert len(rows) == 64
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


def test_result_index_names_and_explains_structural_baseline() -> None:
    decision = (
        "innovation2_present_r3_selected_output_not_explained_by_"
        "coarse_structure_baselines"
    )

    assert "OPM1" in display_name_for_run(RUN_ID)
    assert "结构基线" in display_name_for_run(RUN_ID)
    assert "不足以解释" in DECISION_LABELS[decision]
