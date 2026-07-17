import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli import audit_innovation2_speck_hwang_contexts as cli
from blockcipher_nd.tasks.innovation2 import speck_hwang_contexts as contexts
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_kernel_basis
from blockcipher_nd.tasks.innovation2.speck_hwang_contexts import (
    CONTEXTS,
    ENUMERATED_CONTEXTS,
    PHASE_C_RUN_ID,
    PHASE_C_SOURCE_COMMIT,
    SpeckHwangContextConfig,
    collect_context_parity_rows,
    evaluate_context_audit,
    fixed_plaintext_for_context,
    load_phase_c_anchor,
    verify_context_partition_fixture,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    SPECK32_ACTIVE_BITS,
    SpeckParityCacheConfig,
    _cache_metadata,
    hwang_speck_basis_masks,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import make_phase_c_keys


def _config() -> SpeckHwangContextConfig:
    return SpeckHwangContextConfig(run_id="e26")


def _rows_with_kernel(basis: tuple[int, ...], *, count: int = 32) -> np.ndarray:
    orthogonal = gf2_kernel_basis(np.asarray(basis, dtype=np.uint64), width=32)
    rows = list(orthogonal)
    while len(rows) < count:
        rows.append(orthogonal[(len(rows) - len(orthogonal)) % len(orthogonal)])
    return np.asarray(rows, dtype=np.uint32)


def _matrix_with_kernel(basis: tuple[int, ...]) -> np.ndarray:
    half = _rows_with_kernel(basis)
    return np.concatenate((half, half))


def _invariant_context_rows() -> np.ndarray:
    r6 = _matrix_with_kernel(hwang_speck_basis_masks(6))
    r7 = _matrix_with_kernel(hwang_speck_basis_masks(7))
    return np.stack([np.stack((r6, r7)) for _ in CONTEXTS])


def _evaluate(rows: np.ndarray, **overrides):
    config = _config()
    params = {
        "keys": make_phase_c_keys(config.phase_c_config()),
        "context_parity_rows": rows,
        "baseline_valid": True,
        "caches_completed": {"01": True, "10": True, "11_direct": True},
        "resume_rows_generated": {"01": 0, "10": 0, "11_direct": 0},
        "direct_context11_checks": [True, True],
        "partition_fixture_valid": True,
        "cuda_available": True,
        "device_count": 1,
        "timing_rows": 258,
    }
    params.update(overrides)
    return evaluate_context_audit(config, **params)


def test_e26_freezes_all_fixed_context_encodings() -> None:
    assert CONTEXTS == ("00", "01", "10", "11")
    assert ENUMERATED_CONTEXTS == ("01", "10")
    assert [fixed_plaintext_for_context(value) for value in CONTEXTS] == [
        0x00,
        0x20,
        0x40,
        0x60,
    ]
    with pytest.raises(ValueError, match="unsupported"):
        fixed_plaintext_for_context("100")


def test_e26_small_real_speck_partition_fixture_matches() -> None:
    assert verify_context_partition_fixture() is True


def test_e26_loads_only_hash_matched_phase_c_anchor(monkeypatch, tmp_path) -> None:
    config = _config()
    keys = make_phase_c_keys(config.phase_c_config())
    root = tmp_path / "phase-c"
    (root / "cache/anchor").mkdir(parents=True)
    parity_rows = _invariant_context_rows()[0]
    np.save(root / "cache/anchor/parity_rows.npy", parity_rows)
    np.save(
        root / "cache/anchor/completed.npy",
        np.ones(parity_rows.shape, dtype=np.bool_),
    )
    cache_config = SpeckParityCacheConfig(
        run_id=f"{PHASE_C_RUN_ID}:anchor",
        rounds=(6, 7),
        keys=keys,
        active_bits=SPECK32_ACTIVE_BITS,
        fixed_plaintext=0,
        chunk_size=config.chunk_size,
        backend=config.backend,
        device=config.device,
    )
    metadata_path = root / "cache/anchor/metadata.json"
    metadata_path.write_text(
        json.dumps(_cache_metadata(cache_config), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "source_expected_commit.txt").write_text(
        PHASE_C_SOURCE_COMMIT + "\n", encoding="utf-8"
    )
    parity_path = root / "cache/anchor/parity_rows.npy"
    monkeypatch.setattr(
        contexts, "PHASE_C_PARITY_SHA256", hashlib.sha256(parity_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        contexts,
        "PHASE_C_METADATA_SHA256",
        hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    )

    loaded = load_phase_c_anchor(config, phase_c_root=root)
    np.testing.assert_array_equal(loaded["parity_rows"], parity_rows)
    assert loaded["keys"] == keys

    parity_rows[0, 0] ^= np.uint32(1)
    np.save(parity_path, parity_rows)
    with pytest.raises(ValueError, match="parity SHA256 mismatch"):
        load_phase_c_anchor(config, phase_c_root=root)


def test_e26_collects_two_contexts_derives_fourth_and_checks_direct(
    monkeypatch, tmp_path
) -> None:
    config = _config()
    baseline = _invariant_context_rows()[0]
    keys = make_phase_c_keys(config.phase_c_config())
    context01 = np.full_like(baseline, np.uint32(0x11111111))
    context10 = np.full_like(baseline, np.uint32(0x22222222))
    derived11 = baseline ^ context01 ^ context10
    seen: list[str] = []

    monkeypatch.setattr(
        contexts,
        "load_phase_c_anchor",
        lambda *args, **kwargs: {
            "keys": keys,
            "parity_rows": baseline,
            "metadata": {},
            "parity_path": tmp_path / "p.npy",
            "metadata_path": tmp_path / "m.json",
            "completed_path": tmp_path / "c.npy",
        },
    )

    def fake_cached(parity_config, *, cache_root, progress_callback):
        context = parity_config.run_id.split("context", 1)[1]
        seen.append(context)
        if context == "01":
            rows = context01
        elif context == "10":
            rows = context10
        else:
            rows = derived11[:, :1]
        return {
            "parity_rows": rows,
            "completed": np.ones(rows.shape, dtype=np.bool_),
            "metadata": {"run_id": parity_config.run_id},
            "rows_generated": 0,
        }

    monkeypatch.setattr(contexts, "run_cached_speck_parity_rows", fake_cached)
    result = collect_context_parity_rows(
        config,
        phase_c_root=tmp_path,
        cache_root=tmp_path / "cache",
    )

    assert seen == ["01", "01", "10", "10", "11_direct", "11_direct"]
    np.testing.assert_array_equal(result["context_parity_rows"][3], derived11)
    assert result["direct_context11_checks"] == [True, True]
    assert result["resume_rows_generated"] == {"01": 0, "10": 0, "11_direct": 0}


def test_e26_gate_passes_context_invariant_paper_kernels() -> None:
    result = _evaluate(_invariant_context_rows())
    gate = result["gate"]
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_speck_hwang_context_invariant"
    assert gate["metrics"]["exact_paper_span_context_rounds"] == 8
    assert gate["metrics"]["distinct_joint_signatures_by_round"] == {
        "6": 1,
        "7": 1,
    }
    assert all(gate["readiness_checks"].values())


def test_e26_gate_passes_context_dependent_stable_kernels() -> None:
    rows = _invariant_context_rows()
    alternate = _matrix_with_kernel((0x00000001,))
    rows[2, 1] = alternate
    rows[3, 1] = alternate
    result = _evaluate(rows)
    gate = result["gate"]
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_speck_hwang_context_dependent_stable"
    assert gate["signal_checks"]["context_invariant"] is False
    assert gate["signal_checks"]["context_dependent_stable"] is True


def test_e26_gate_holds_context_family_without_stable_nontrivial_kernels() -> None:
    full_rank_half = np.asarray([1 << bit for bit in range(32)], dtype=np.uint32)
    full_rank = np.concatenate((full_rank_half, full_rank_half))
    rows = np.stack([np.stack((full_rank, full_rank)) for _ in CONTEXTS])
    result = _evaluate(rows)
    assert result["gate"]["status"] == "hold"
    assert (
        result["gate"]["decision"]
        == "innovation2_speck_hwang_context_family_not_stable"
    )


def test_e26_gate_fails_protocol_or_derivation_evidence() -> None:
    result = _evaluate(
        _invariant_context_rows(),
        direct_context11_checks=[True, False],
        timing_rows=257,
    )
    assert result["gate"]["status"] == "fail"
    assert result["gate"]["decision"] == "innovation2_speck_hwang_context_protocol_invalid"


def test_e26_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    keys = make_phase_c_keys(config.phase_c_config())
    context_rows = _invariant_context_rows()
    baseline_dir = tmp_path / "source-baseline"
    baseline_dir.mkdir()
    parity_path = baseline_dir / "parity_rows.npy"
    completed_path = baseline_dir / "completed.npy"
    metadata_path = baseline_dir / "metadata.json"
    np.save(parity_path, context_rows[0])
    np.save(completed_path, np.ones(context_rows[0].shape, dtype=np.bool_))
    metadata_path.write_text("{}\n", encoding="utf-8")
    collected = {
        "keys": keys,
        "context_parity_rows": context_rows,
        "direct_context11_rows": context_rows[3, :, :1],
        "direct_context11_checks": [True, True],
        "baseline": {
            "parity_path": parity_path,
            "completed_path": completed_path,
            "metadata_path": metadata_path,
        },
        "cache_metadata": {
            "01": {},
            "10": {},
            "11_direct": {},
        },
        "completed": {"01": True, "10": True, "11_direct": True},
        "first_rows_generated": {"01": 128, "10": 128, "11_direct": 2},
        "resume_rows_generated": {"01": 0, "10": 0, "11_direct": 0},
    }
    timing = {
        (context, rounds, key_index): {
            "elapsed_seconds": 7.0,
            "peak_memory_bytes": 1024,
        }
        for context, rounds, key_count in (
            ("01", 6, 64),
            ("01", 7, 64),
            ("10", 6, 64),
            ("10", 7, 64),
            ("11_direct", 6, 1),
            ("11_direct", 7, 1),
        )
        for key_index in range(key_count)
    }
    monkeypatch.setattr(cli, "collect_context_parity_rows", lambda *a, **k: collected)
    monkeypatch.setattr(cli, "verify_context_partition_fixture", lambda: True)
    monkeypatch.setattr(cli, "_load_timing_evidence", lambda path: timing.copy())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    output_root = tmp_path / "result"

    assert (
        cli.main(
            [
                "--run-id",
                config.run_id,
                "--output-root",
                str(output_root),
                "--phase-c-root",
                str(tmp_path / "unused"),
                "--device",
                "cuda",
            ]
        )
        == 0
    )
    assert {
        "baseline_phase_c",
        "context_parity_rows.npy",
        "direct_context11_rows.npy",
        "gate.json",
        "kernel_basis.csv",
        "keys.csv",
        "metadata.json",
        "progress.jsonl",
        "results.jsonl",
        "summary.json",
    } <= {path.name for path in output_root.iterdir()}
    gate = json.loads((output_root / "gate.json").read_text())
    assert gate["decision"] == "innovation2_speck_hwang_context_invariant"


def test_e26_remote_scripts_preserve_cache_and_windows_contracts() -> None:
    run = Path(
        "configs/remote/generated/run_i2_speck32_hwang_contexts_32plus32_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    launch = Path(
        "configs/remote/generated/launch_i2_speck32_hwang_contexts_32plus32_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    monitor = Path(
        "configs/remote/generated/monitor_i2_speck32_hwang_contexts_32plus32_gpu0_20260717.sh"
    ).read_text(encoding="utf-8")
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in run + launch
    assert "!" not in run + launch
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run + launch
    assert "--phase-c-root" in run
    assert "--chunk-size 16777216" in run
    assert 'xcopy /E /I /Y "%ARTIFACT_DIR%\\cache"' in run
    assert "baseline_phase_c" in run
    assert "source_expected_commit.txt" in run + launch
    assert "/RU SYSTEM /RL HIGHEST" in launch
    assert "logs/${RUN_ID}_done.marker" in monitor
    assert "sleep 60" in monitor
