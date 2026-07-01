from __future__ import annotations

import json
import sys

from blockcipher_training_accelerator.benchmark import run_benchmark


def test_run_benchmark_writes_timing_report(tmp_path):
    report_path = tmp_path / "timing.json"

    report = run_benchmark(
        [sys.executable, "-c", "print('bench ok')"],
        label="unit",
        report_path=report_path,
        cwd=tmp_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.returncode == 0
    assert payload["label"] == "unit"
    assert payload["status"] == "passed"
    assert payload["duration_seconds"] >= 0.0
    assert payload["command"][0] == sys.executable
