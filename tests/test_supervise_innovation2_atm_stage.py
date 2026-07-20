from __future__ import annotations

import json
import sys
from pathlib import Path

from blockcipher_nd.cli.supervise_innovation2_atm_stage import main


def test_stage_supervisor_records_success_and_output(tmp_path: Path) -> None:
    marker_root = tmp_path / "markers"
    stdout = tmp_path / "logs/stdout.txt"
    stderr = tmp_path / "logs/stderr.txt"
    status = main(
        [
            "--timeout-seconds",
            "10",
            "--stage-id",
            "fixture_success",
            "--marker-root",
            str(marker_root),
            "--stdout",
            str(stdout),
            "--stderr",
            str(stderr),
            "--",
            sys.executable,
            "-c",
            "print('stage-ok')",
        ]
    )
    assert status == 0
    assert stdout.read_text(encoding="utf-8").strip() == "stage-ok"
    marker = json.loads(
        (marker_root / "fixture_success_done.marker").read_text(encoding="utf-8")
    )
    assert marker["status"] == "done"
    assert marker["return_code"] == 0


def test_stage_supervisor_kills_timed_out_process_tree(tmp_path: Path) -> None:
    marker_root = tmp_path / "markers"
    status = main(
        [
            "--timeout-seconds",
            "0.1",
            "--stage-id",
            "fixture_timeout",
            "--marker-root",
            str(marker_root),
            "--stdout",
            str(tmp_path / "stdout.txt"),
            "--stderr",
            str(tmp_path / "stderr.txt"),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ]
    )
    assert status == 124
    marker = json.loads(
        (marker_root / "fixture_timeout_timeout.marker").read_text(
            encoding="utf-8"
        )
    )
    assert marker["status"] == "timeout"
    assert marker["timeout_seconds"] == 0.1
