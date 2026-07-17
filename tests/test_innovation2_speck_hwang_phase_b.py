from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_speck_hwang_phase_b import (
    _load_timing_evidence,
    adjudicate_phase_b,
)


def _rows(*, seconds: float = 10.0, memory: int = 1024) -> list[dict]:
    return [
        {
            "rounds": rounds,
            "paper_basis_valid_for_key": True,
            "elapsed_seconds": seconds,
            "peak_memory_bytes": memory,
        }
        for rounds in (6, 7)
    ]


def _gate(rows: list[dict], **overrides):
    params = {
        "run_id": "phase-b",
        "rows": rows,
        "cuda_available": True,
        "device_count": 1,
        "device_name": "A6000",
        "completed": True,
        "resume_rows_generated": 0,
        "official_vector_matches": True,
        "active_bits_exact": True,
        "parity_shape": (2, 1),
    }
    params.update(overrides)
    return adjudicate_phase_b(**params)


def test_e25_phase_b_gate_passes_complete_fast_single_key_timing() -> None:
    gate = _gate(_rows())
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_speck_hwang_phase_b_single_key_timing_ready"


def test_e25_phase_b_gate_holds_slow_direct_enumeration() -> None:
    gate = _gate(_rows(seconds=1801.0))
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_speck_hwang_phase_b_direct_enumeration_not_scalable"


def test_e25_phase_b_gate_fails_missing_cuda_or_mask_validation() -> None:
    rows = _rows()
    rows[1]["paper_basis_valid_for_key"] = False
    gate = _gate(rows, cuda_available=False, device_count=0)
    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation2_speck_hwang_phase_b_protocol_invalid"


def test_e25_phase_b_remote_scripts_preserve_windows_and_monitor_contracts() -> None:
    run = Path(
        "configs/remote/generated/run_i2_speck32_hwang_phase_b_singlekey_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    launch = Path(
        "configs/remote/generated/launch_i2_speck32_hwang_phase_b_singlekey_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    monitor = Path(
        "configs/remote/generated/monitor_i2_speck32_hwang_phase_b_singlekey_gpu0_20260717.sh"
    ).read_text(encoding="utf-8")
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in launch + run
    assert "!" not in launch + run
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launch + run
    assert "--chunk-size 16777216" in run
    assert "completed.npy" in run
    assert "source_expected_commit.txt" in launch + run
    assert "/RU SYSTEM /RL HIGHEST" in launch
    assert "logs/${RUN_ID}_done.marker" in monitor
    assert "sleep 60" in monitor


def test_e25_phase_b_recovers_timing_evidence_after_process_restart(tmp_path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        '{"event":"speck_parity_row_done","rounds":6,"elapsed_seconds":12.5,"peak_memory_bytes":99}\n'
        '{"event":"speck_parity_chunk_done","rounds":7}\n'
        '{"event":"speck_parity_row_done","rounds":7,"elapsed_seconds":15.0,"peak_memory_bytes":101}\n',
        encoding="utf-8",
    )
    timings, peaks = _load_timing_evidence(progress)
    assert timings == {6: 12.5, 7: 15.0}
    assert peaks == {6: 99, 7: 101}
