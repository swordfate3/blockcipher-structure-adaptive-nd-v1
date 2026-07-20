from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_generation import (
    E103_DECISION,
    E103_GATE_SHA256,
    PRESENT_BIT_PERMUTATION,
    RUN_ID,
    SOURCE_HASHES,
    execute_phase,
    search_config,
)


def test_e104_search_config_freezes_only_missing_r9_split() -> None:
    config = search_config()
    assert config.run_id == RUN_ID
    assert config.input_size == config.output_size == 64
    assert config.is_permutation is True
    assert config.num_workers == 36
    assert dict(config.oracle_parameters) == {
        "rounds": "9",
        "split": "3,3,3",
        "limit": "1024",
        "state_bits": "64",
        "key_model": "independent_64bit_round_keys",
        "qmc_shim": "single_process_cp_sat",
    }


def test_e104_present_permutation_matches_published_mapping() -> None:
    assert len(PRESENT_BIT_PERMUTATION) == 64
    assert len(set(PRESENT_BIT_PERMUTATION)) == 64
    assert PRESENT_BIT_PERMUTATION == tuple(
        (16 * bit) % 63 if bit < 63 else 63 for bit in range(64)
    )


def test_e104_source_drift_fails_before_environment_build(tmp_path: Path) -> None:
    atm_root = tmp_path / "atm"
    for relative in SOURCE_HASHES:
        path = atm_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("drift", encoding="utf-8")
    result = execute_phase(
        tmp_path / "output",
        atm_root=atm_root,
        e103_gate={"status": "pass", "decision": E103_DECISION},
        e103_gate_sha256=E103_GATE_SHA256,
        actual_atm_commit="0" * 40,
        mode="readiness",
    )
    assert result["gate"]["status"] == "fail"
    assert result["gate"]["decision"] == (
        "innovation2_present_r9_split333_source_protocol_invalid"
    )
    assert result["gate"]["next_action"]["long_search_open"] is False


def test_e103_anchor_records_original_artifact_hash() -> None:
    anchor = json.loads(
        Path(
            "configs/experiment/innovation2/"
            "innovation2_present_sbox4_r3_real_atm_compatibility_gate_anchor.json"
        ).read_text(encoding="utf-8")
    )
    assert anchor["artifact_sha256"] == E103_GATE_SHA256
    assert anchor["status"] == "pass"
    assert anchor["decision"] == E103_DECISION


def test_windows_py310_runtime_lock_is_exact_and_complete() -> None:
    lines = Path(
        "configs/runtime/innovation2_atm_windows_py310_requirements.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert lines == [
        "numpy==2.2.6",
        "pybind11==3.0.4",
        "ortools==9.15.6755",
        "python-sat==1.9.dev6",
        "galois==0.4.11",
    ]


def test_e104_remote_scripts_use_run_owned_paths_and_bounded_stages() -> None:
    root = Path("scripts/generated/remote")
    setup = (root / "setup_innovation2_present_r9_atm_split333_20260720.cmd").read_text(
        encoding="utf-8"
    )
    pipeline = (
        root / "run_innovation2_present_r9_atm_split333_pipeline_20260720.cmd"
    ).read_text(encoding="utf-8")
    launcher = (
        root / "launch_innovation2_present_r9_atm_split333_pipeline_20260720.cmd"
    ).read_text(encoding="utf-8")
    combined = setup + pipeline + launcher
    assert "cmd.exe /k" not in combined
    assert "EnableDelayedExpansion" not in combined
    assert "!" not in combined
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in setup
    assert "set BOOTSTRAP=G:\\lxy\\blockcipher-structure-adaptive-nd-v1-clean" in setup
    assert "set HOME=%RUN_HOME%" in setup
    assert "set USERPROFILE=%RUN_HOME%" in setup
    assert "set TEMP=%RUN_TEMP%" in setup
    assert "set TMP=%RUN_TEMP%" in setup
    assert "git config --global --add safe.directory %BOOTSTRAP%" in setup
    assert "set SOURCE_COMMIT=%%A" in setup
    assert "git -C %SOURCE% merge --ff-only %SOURCE_COMMIT%" in setup
    assert "set ACTUAL_SOURCE_COMMIT=%%A" in setup
    assert 'if not "%ACTUAL_SOURCE_COMMIT%"=="%SOURCE_COMMIT%"' in setup
    assert "set VENV=%RUN_ROOT%\\venv" in setup
    assert setup.count("-m venv --system-site-packages %VENV%") == 2
    assert 'import torch; print(torch.__version__)' in setup
    assert "if errorlevel 1 rmdir /s /q %VENV%" in setup
    assert "set VCVARS64=C:\\Program Files\\Microsoft Visual Studio" in setup
    assert 'call "%VCVARS64%"' in setup
    assert "set INCLUDE >> %LOGS%\\compiler_environment.txt" in setup
    assert "set LIB >> %LOGS%\\compiler_environment.txt" in setup
    assert "set PYTHONPATH=%SOURCE%\\src" in setup
    assert "set PYTHONPATH=%SOURCE%\\src" in pipeline
    assert "VC\\Auxiliary\\Build\\vcvars64.bat" in setup
    assert 'call "%VCVARS%"' in setup
    assert "--stage-id readiness --marker-root %LOGS%" in setup
    assert setup.count("status --porcelain") == 2
    assert "git -C %ATM_ROOT% cat-file -e %ATM_COMMIT%" in setup
    assert "if errorlevel 1 git -C %ATM_ROOT% fetch origin" in setup
    assert "bitarrays/bitset*.pyd" in setup
    assert "**/__pycache__/" in setup
    assert "*.pyc" in setup
    assert "set LOCK=%RUN_ROOT%\\pipeline.lock" in pipeline
    assert "mkdir %LOCK% 2>nul" in pipeline
    assert "if errorlevel 1 exit /b 9" in pipeline
    assert pipeline.count("rmdir /s /q %LOCK%") == 4
    assert pipeline.count("--timeout-seconds 600") == 2
    assert pipeline.count("--timeout-seconds 43200") == 3
    assert pipeline.count("--mode probe") == 2
    assert pipeline.count("--mode search") == 3
    assert 'cmd.exe /c %PIPELINE%' in launcher
    assert "/RU SYSTEM" in launcher
    assert "schtasks /Run" in launcher


def test_e104_windows_scripts_preserve_remote_and_scheduler_contract() -> None:
    roots = Path("scripts/generated/remote")
    names = (
        "setup_innovation2_present_r9_atm_split333_20260720.cmd",
        "run_innovation2_present_r9_atm_split333_pipeline_20260720.cmd",
        "launch_innovation2_present_r9_atm_split333_pipeline_20260720.cmd",
    )
    texts = [(roots / name).read_text(encoding="utf-8") for name in names]
    combined = "\n".join(texts)
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in combined
    assert "cmd.exe /c" in texts[2]
    assert "cmd.exe /k" not in combined
    assert "setlocal EnableDelayedExpansion" not in combined
    assert "!" not in combined
    assert "--timeout-seconds 600" in combined
    assert combined.count("--timeout-seconds 43200") == 3
    assert "--mode probe" in combined
    assert "--mode search" in combined
    assert "--no-deps --editable %SOURCE%" in combined
    assert "del %LOGS%\\setup_failed.marker" in combined
    assert "del %LOGS%\\pipeline_failed.marker" in combined
    assert "(3,3,3)" not in combined
    assert "schtasks /Run" in texts[2]


def test_e104_python_wrappers_are_windows_spawn_safe() -> None:
    for name in (
        "scripts/run-innovation2-present-r9-atm-split333-generation",
        "scripts/supervise-innovation2-atm-stage",
    ):
        text = Path(name).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in text


def test_e104_module_import_does_not_require_torch() -> None:
    code = """
import importlib.abc
import sys

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch intentionally unavailable")
        return None

sys.meta_path.insert(0, RejectTorch())
from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_generation import SPLIT
assert SPLIT == (3, 3, 3)
assert "torch" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
