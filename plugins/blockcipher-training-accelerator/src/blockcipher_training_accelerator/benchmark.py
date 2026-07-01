from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


@dataclass(frozen=True)
class BenchmarkReport:
    label: str
    command: list[str]
    cwd: str
    started_at: str
    finished_at: str
    duration_seconds: float
    returncode: int
    status: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def run_benchmark(
    command: list[str],
    *,
    label: str,
    report_path: Path,
    cwd: Path,
    fail_on_nonzero: bool = False,
) -> BenchmarkReport:
    if not command:
        raise ValueError("command must not be empty")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    finished = time.perf_counter()
    finished_at = utc_now()

    report = BenchmarkReport(
        label=label,
        command=command,
        cwd=str(cwd),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(finished - started, 6),
        returncode=int(completed.returncode),
        status="passed" if completed.returncode == 0 else "failed",
    )
    report_path.write_text(
        json.dumps(report.to_json_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if fail_on_nonzero and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return report


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
