import hashlib
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_innovation2_speck_hwang_phase_c import (
    _load_timing_evidence,
    _summarize_runtime,
)
from blockcipher_nd.cli.plot_innovation2_speck_hwang_phase_c import (
    main as plot_main,
)
from blockcipher_nd.cli.validate_innovation2_speck_hwang_phase_c import (
    main as validate_main,
    validate_phase_c_archive,
)
from blockcipher_nd.tasks.innovation2 import speck_hwang_phase_c as phase_c
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_kernel_basis
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import hwang_speck_basis_masks
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    SpeckParityCacheConfig,
    _cache_metadata,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import (
    CONTROL_ACTIVE_BITS,
    KEY_GENERATION_OFFSET,
    SpeckHwangPhaseCConfig,
    collect_phase_c_parity_rows,
    evaluate_phase_c,
    make_phase_c_keys,
)


def _config(**overrides) -> SpeckHwangPhaseCConfig:
    values = {"run_id": "phase-c", "device": "cuda"}
    values.update(overrides)
    return SpeckHwangPhaseCConfig(**values)


def _rows_with_kernel(basis: tuple[int, ...], *, count: int = 32) -> np.ndarray:
    orthogonal = gf2_kernel_basis(np.asarray(basis, dtype=np.uint64), width=32)
    assert len(orthogonal) <= count
    rows = list(orthogonal)
    while len(rows) < count:
        rows.append(orthogonal[(len(rows) - len(orthogonal)) % len(orthogonal)])
    return np.asarray(rows, dtype=np.uint32)


def _passing_matrices() -> tuple[np.ndarray, np.ndarray]:
    r6_half = _rows_with_kernel(hwang_speck_basis_masks(6))
    r7_half = _rows_with_kernel(hwang_speck_basis_masks(7))
    anchor = np.stack(
        (np.concatenate((r6_half, r6_half)), np.concatenate((r7_half, r7_half)))
    )
    full_rank = np.asarray([1 << bit for bit in range(32)], dtype=np.uint32)
    control = np.stack((np.concatenate((full_rank, full_rank)),))
    return anchor, control


def _evaluate(
    *,
    config: SpeckHwangPhaseCConfig | None = None,
    anchor: np.ndarray | None = None,
    control: np.ndarray | None = None,
    timing_rows: int = 192,
):
    config = config or _config()
    passing_anchor, passing_control = _passing_matrices()
    return evaluate_phase_c(
        config,
        keys=make_phase_c_keys(config),
        anchor_parity_rows=passing_anchor if anchor is None else anchor,
        control_parity_rows=passing_control if control is None else control,
        completed={"anchor": True, "control": True},
        resume_rows_generated={"anchor": 0, "control": 0},
        cuda_available=True,
        device_count=1,
        timing_rows=timing_rows,
    )


def test_e25_phase_c_freezes_key_split_and_position_control() -> None:
    config = _config()
    keys = make_phase_c_keys(config)
    assert config.total_keys == 64
    assert KEY_GENERATION_OFFSET == 25031
    assert len(keys) == len(set(keys)) == 64
    assert set(keys[:32]).isdisjoint(keys[32:])
    assert set(CONTROL_ACTIVE_BITS) == set(range(32)) - {0, 1}


def test_e25_phase_c_gate_passes_exact_hwang_kernels_and_negative_control() -> None:
    result = _evaluate()
    gate = result["gate"]
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_speck_hwang_phase_c_kernel_reproduced"
    assert all(gate["readiness_checks"].values())
    assert all(gate["signal_checks"].values())
    rows = {(row["role"], row["rounds"]): row for row in result["rows"]}
    assert rows[("anchor", 6)]["joint_rank"] == 23
    assert rows[("anchor", 6)]["joint_nullity"] == 9
    assert rows[("anchor", 7)]["joint_rank"] == 31
    assert rows[("anchor", 7)]["joint_nullity"] == 1
    assert rows[("control", 7)]["joint_nullity"] == 0


def test_e25_phase_c_accepts_joint_exact_kernel_when_key_halves_are_underrank() -> None:
    anchor, control = _passing_matrices()
    r7_row_space = gf2_kernel_basis(
        np.asarray(hwang_speck_basis_masks(7), dtype=np.uint64), width=32
    )
    discovery = list(r7_row_space[:30])
    validation = list(r7_row_space[1:])
    discovery.extend(discovery[: 32 - len(discovery)])
    validation.extend(validation[: 32 - len(validation)])
    anchor[1] = np.asarray(discovery + validation, dtype=np.uint32)

    result = _evaluate(anchor=anchor, control=control)
    rows = {(row["role"], row["rounds"]): row for row in result["rows"]}
    assert rows[("anchor", 7)]["discovery_rank"] == 30
    assert rows[("anchor", 7)]["validation_rank"] == 30
    assert rows[("anchor", 7)]["joint_rank"] == 31
    assert result["gate"]["status"] == "pass"


def test_e25_phase_c_holds_when_hwang_mask_survives_position_control() -> None:
    anchor, _ = _passing_matrices()
    control = np.stack((anchor[1],))
    result = _evaluate(anchor=anchor, control=control)
    assert result["gate"]["status"] == "hold"
    assert (
        result["gate"]["decision"]
        == "innovation2_speck_hwang_phase_c_position_control_not_specific"
    )


def test_e25_phase_c_holds_when_anchor_kernel_is_not_reproduced() -> None:
    anchor, control = _passing_matrices()
    anchor[1, 0] ^= np.uint32(1 << 2)
    result = _evaluate(anchor=anchor, control=control)
    assert result["gate"]["status"] == "hold"
    assert (
        result["gate"]["decision"]
        == "innovation2_speck_hwang_phase_c_kernel_not_reproduced"
    )


def test_e25_phase_c_fails_missing_complete_timing_evidence() -> None:
    result = _evaluate(timing_rows=191)
    assert result["gate"]["status"] == "fail"
    assert (
        result["gate"]["decision"]
        == "innovation2_speck_hwang_phase_c_protocol_invalid"
    )


def test_e25_phase_c_collection_uses_exact_role_configs_and_resume(monkeypatch, tmp_path) -> None:
    config = _config(discovery_keys=2, validation_keys=2, chunk_size=17)
    seen = []

    def fake_cached(parity_config, *, cache_root, progress_callback):
        seen.append((parity_config, cache_root))
        shape = (len(parity_config.rounds), len(parity_config.keys))
        return {
            "parity_rows": np.zeros(shape, dtype=np.uint32),
            "completed": np.ones(shape, dtype=np.bool_),
            "metadata": {"run_id": parity_config.run_id},
            "rows_generated": 0,
        }

    monkeypatch.setattr(phase_c, "run_cached_speck_parity_rows", fake_cached)
    result = collect_phase_c_parity_rows(config, cache_root=tmp_path)
    assert [entry[0].rounds for entry in seen] == [(6, 7), (6, 7), (7,), (7,)]
    assert seen[0][0].active_bits == tuple(bit for bit in range(32) if bit not in {5, 6})
    assert seen[2][0].active_bits == CONTROL_ACTIVE_BITS
    assert seen[0][1] == tmp_path / "anchor"
    assert seen[2][1] == tmp_path / "control"
    assert result["resume_rows_generated"] == {"anchor": 0, "control": 0}


def test_e25_phase_c_recovers_and_summarizes_row_timing(tmp_path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "event": "speck_parity_row_done",
                        "role": "anchor",
                        "rounds": 6,
                        "key_index": 0,
                        "elapsed_seconds": 7.0,
                        "peak_memory_bytes": 100,
                    }
                ),
                json.dumps(
                    {
                        "event": "speck_parity_row_done",
                        "role": "anchor",
                        "rounds": 6,
                        "key_index": 1,
                        "elapsed_seconds": 9.0,
                        "peak_memory_bytes": 120,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = _load_timing_evidence(progress)
    summary = _summarize_runtime(evidence)
    assert len(evidence) == 2
    assert summary["anchor_r6"] == {
        "timed_rows": 2,
        "total_elapsed_seconds": 16.0,
        "mean_elapsed_seconds": 8.0,
        "max_elapsed_seconds": 9.0,
        "max_peak_memory_bytes": 120,
    }


def test_e25_phase_c_remote_scripts_preserve_windows_and_monitor_contracts() -> None:
    run = Path(
        "configs/remote/generated/run_i2_speck32_hwang_phase_c_32plus32_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    launch = Path(
        "configs/remote/generated/launch_i2_speck32_hwang_phase_c_32plus32_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    monitor = Path(
        "configs/remote/generated/monitor_i2_speck32_hwang_phase_c_32plus32_gpu0_20260717.sh"
    ).read_text(encoding="utf-8")
    assert "cmd.exe /c" in launch
    assert "cmd.exe /k" not in launch + run
    assert "!" not in launch + run
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launch + run
    assert "--discovery-keys 32" in run
    assert "--validation-keys 32" in run
    assert "--chunk-size 16777216" in run
    assert "anchor\\completed.npy" in run
    assert "control\\completed.npy" in run
    assert "source_expected_commit.txt" in launch + run
    assert "/RU SYSTEM /RL HIGHEST" in launch
    assert "logs/${RUN_ID}_done.marker" in monitor
    assert "sleep 60" in monitor


def test_e25_phase_c_plot_cli_writes_chinese_explanatory_svg(tmp_path) -> None:
    result = _evaluate()
    results_path = tmp_path / "results.jsonl"
    gate_path = tmp_path / "gate.json"
    output_path = tmp_path / "curves.svg"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["rows"]),
        encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(result["gate"], sort_keys=True) + "\n", encoding="utf-8"
    )

    assert (
        plot_main(
            [
                "--results",
                str(results_path),
                "--gate",
                str(gate_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E25 Phase C" in svg
    assert "32把发现密钥 + 32把全新验证密钥" in svg
    assert "预注册裁决门" in svg


def test_e25_phase_c_local_validator_recomputes_verified_archive(tmp_path) -> None:
    source_commit = "700ac88a4c250fb43ff076ce043c79a575faf95d"
    root = _write_valid_archive(tmp_path, source_commit=source_commit)

    assert (
        validate_main(
            [
                "--artifact-root",
                str(root),
                "--expected-source-commit",
                source_commit,
            ]
        )
        == 0
    )
    validation = json.loads((root / "validation.local.json").read_text())
    local_gate = json.loads((root / "gate.local.json").read_text())
    assert validation["status"] == "pass"
    assert validation["timing_rows"] == 192
    assert validation["errors"] == []
    assert local_gate["status"] == "pass"
    assert local_gate["local_validation"]["manifest_verified"] is True


def test_e25_phase_c_local_validator_rejects_manifest_tampering(tmp_path) -> None:
    source_commit = "700ac88a4c250fb43ff076ce043c79a575faf95d"
    root = _write_valid_archive(tmp_path, source_commit=source_commit)
    results_path = root / "results.jsonl"
    results_path.write_text(
        results_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    validation, gate = validate_phase_c_archive(
        root, expected_source_commit=source_commit
    )
    assert validation["status"] == "fail"
    assert "SHA256 mismatch: results.jsonl" in validation["errors"]
    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation2_speck_hwang_phase_c_protocol_invalid"


def _write_valid_archive(tmp_path: Path, *, source_commit: str) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    config = _config()
    anchor, control = _passing_matrices()
    result = _evaluate(config=config, anchor=anchor, control=control)
    keys = make_phase_c_keys(config)
    (root / "cache/anchor").mkdir(parents=True)
    (root / "cache/control").mkdir(parents=True)
    np.save(root / "cache/anchor/parity_rows.npy", anchor)
    np.save(root / "cache/control/parity_rows.npy", control)
    np.save(root / "cache/anchor/completed.npy", np.ones(anchor.shape, dtype=np.bool_))
    np.save(
        root / "cache/control/completed.npy", np.ones(control.shape, dtype=np.bool_)
    )
    anchor_config = SpeckParityCacheConfig(
        run_id=f"{config.run_id}:anchor",
        rounds=(6, 7),
        keys=keys,
        active_bits=tuple(bit for bit in range(32) if bit not in {5, 6}),
        fixed_plaintext=0,
        chunk_size=config.chunk_size,
        backend=config.backend,
        device=config.device,
    )
    control_config = SpeckParityCacheConfig(
        run_id=f"{config.run_id}:control",
        rounds=(7,),
        keys=keys,
        active_bits=CONTROL_ACTIVE_BITS,
        fixed_plaintext=0,
        chunk_size=config.chunk_size,
        backend=config.backend,
        device=config.device,
    )
    (root / "cache/anchor/metadata.json").write_text(
        json.dumps(_cache_metadata(anchor_config), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "cache/control/metadata.json").write_text(
        json.dumps(_cache_metadata(control_config), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps(result["metadata"], sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "gate.json").write_text(
        json.dumps(result["gate"], sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {"resume_rows_generated": {"anchor": 0, "control": 0}},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["rows"]),
        encoding="utf-8",
    )
    (root / "kernel_basis.csv").write_text(
        "run_id,role,rounds,split,basis_index,mask_hex,mask_weight,basis_valid\n",
        encoding="utf-8",
    )
    (root / "keys.csv").write_text(
        "key_index,split,key_hex\n"
        + "".join(
            f"{index},{'discovery' if index < 32 else 'validation'},0x{key:016X}\n"
            for index, key in enumerate(keys)
        ),
        encoding="utf-8",
    )
    progress_rows = []
    for role, rounds_values in (("anchor", (6, 7)), ("control", (7,))):
        for rounds in rounds_values:
            for key_index in range(64):
                progress_rows.append(
                    {
                        "event": "speck_parity_row_done",
                        "role": role,
                        "rounds": rounds,
                        "key_index": key_index,
                        "elapsed_seconds": 7.0,
                        "peak_memory_bytes": 939524096,
                    }
                )
    (root / "progress.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress_rows),
        encoding="utf-8",
    )
    (root / "source_expected_commit.txt").write_text(
        source_commit + "\n", encoding="utf-8"
    )
    (root / "retrieved_from_verified_result_branch.marker").touch()
    manifest_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "retrieved_from_verified_result_branch.marker"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
            for path in manifest_files
        ),
        encoding="utf-8",
    )
    return root
