from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pytest

from blockcipher_nd.tasks.innovation2 import speck_hwang_topology_pairs as topology
from blockcipher_nd.cli.audit_innovation2_speck_hwang_topology_pairs import main as audit_main
from blockcipher_nd.cli import validate_innovation2_speck_hwang_topology_pairs as validator
from blockcipher_nd.cli.plot_innovation2_speck_hwang_topology_pairs import (
    render_topology_pair_svg,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import make_phase_c_keys


def _roles(selected_specs: tuple[tuple[str, int], ...]):
    roles = {
        f"{family}_lane{lane:02d}_screen"
        for family in topology.FAMILIES
        for lane in topology.LANES
    } | {
        f"{family}_lane{lane:02d}_validation" for family, lane in selected_specs
    }
    return {role: True for role in roles}, {role: 0 for role in roles}


def _evaluate(
    true_candidates: tuple[int, ...],
    control_candidates: tuple[int, ...],
    run_id: str = "topology-test",
):
    config = topology.SpeckHwangTopologyPairConfig(run_id=run_id)
    keys = make_phase_c_keys(config.phase_c_config())
    paper_mask = topology.hwang_speck_basis_masks(7)[0]
    unbalanced = np.uint32(paper_mask & -paper_mask)
    screen = np.full((2, 16, 8), unbalanced, dtype=np.uint32)
    for lane in true_candidates:
        screen[0, lane] = 0
    for lane in control_candidates:
        screen[1, lane] = 0
    candidates = {
        "ror7_add_aligned": true_candidates,
        "offset_minus_one": control_candidates,
    }
    selected = topology.select_family_candidates(candidates)
    selected_specs = tuple(
        (family, lane) for family in topology.FAMILIES for lane in selected[family]
    )
    validation = np.zeros((len(selected_specs), 56), dtype=np.uint32)
    completed, resumed = _roles(selected_specs)
    timing = 256 + len(selected_specs) * 56
    return topology.evaluate_topology_pairs(
        config,
        keys=keys,
        screen_parity_rows=screen,
        candidates=candidates,
        selected=selected,
        selected_specs=selected_specs,
        validation_parity_rows=validation,
        baseline_valid=True,
        caches_completed=completed,
        resume_rows_generated=resumed,
        cuda_available=True,
        device_count=1,
        timing_rows=timing,
    )


def test_topology_pair_definitions_follow_speck_ror7_lane() -> None:
    assert topology.topology_pair_bits("ror7_add_aligned", 0) == (0, 23)
    assert topology.topology_pair_bits("ror7_add_aligned", 15) == (15, 22)
    assert topology.topology_pair_bits("offset_minus_one", 0) == (0, 22)
    assert len(topology.active_bits_for_topology_pair("ror7_add_aligned", 3)) == 30
    with pytest.raises(ValueError, match="unsupported topology"):
        topology.topology_pair_bits("other", 0)


def test_candidate_selection_caps_each_family_without_cross_fill() -> None:
    selected = topology.select_family_candidates(
        {
            "ror7_add_aligned": (0, 1, 2, 3, 4, 5),
            "offset_minus_one": (7,),
        }
    )
    assert selected == {
        "ror7_add_aligned": (0, 1, 2, 3),
        "offset_minus_one": (7,),
    }


def test_aligned_family_gate_passes_with_two_stable_true_lanes() -> None:
    result = _evaluate((0, 1), ())
    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_speck_topology_aligned_family"
    )
    assert result["gate"]["metrics"]["true_stable_count"] == 2
    assert result["gate"]["metrics"]["control_stable_count"] == 0
    assert all(result["gate"]["readiness_checks"].values())


def test_control_signal_blocks_topology_specific_claim() -> None:
    result = _evaluate((0, 1, 2), (0,))
    assert result["gate"]["decision"] == (
        "innovation2_speck_topology_pair_not_specific"
    )


def test_single_true_lane_is_too_narrow_and_zero_is_no_signal() -> None:
    assert _evaluate((0,), ())["gate"]["decision"] == (
        "innovation2_speck_topology_pair_too_narrow"
    )
    assert _evaluate((), ())["gate"]["decision"] == (
        "innovation2_speck_topology_pair_no_signal"
    )


def test_collector_validates_at_most_four_per_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = topology.SpeckHwangTopologyPairConfig(run_id="collect-test")
    keys = make_phase_c_keys(config.phase_c_config())
    baseline = {"keys": keys, "anchor": {}, "control": {}}
    monkeypatch.setattr(
        topology,
        "load_phase_c_position_baselines",
        lambda *args, **kwargs: baseline,
    )
    paper_mask = topology.hwang_speck_basis_masks(7)[0]
    unbalanced = np.uint32(paper_mask & -paper_mask)
    true_hits = {0, 1, 2, 3, 4}
    control_hits = {7}
    calls: list[tuple[str, int, int]] = []

    def fake_cached(cache_config, *, cache_root, progress_callback=None):
        family = "ror7_add_aligned" if "ror7_add_aligned" in cache_config.run_id else "offset_minus_one"
        lane = int(cache_config.run_id.split("lane")[-1].split("_")[0])
        length = len(cache_config.keys)
        calls.append((family, lane, length))
        hit = lane in (true_hits if family == "ror7_add_aligned" else control_hits)
        value = np.uint32(0 if hit else unbalanced)
        array = np.full((1, length), value, dtype=np.uint32)
        occurrence = calls.count((family, lane, length))
        return {
            "parity_rows": array,
            "completed": np.ones(array.shape, dtype=np.bool_),
            "metadata": {"family": family, "lane": lane},
            "rows_generated": length if occurrence == 1 else 0,
        }

    monkeypatch.setattr(topology, "run_cached_speck_parity_rows", fake_cached)
    collected = topology.collect_topology_pair_rows(
        config,
        phase_c_root=tmp_path / "phase_c",
        cache_root=tmp_path / "cache",
    )
    assert collected["candidates"]["ror7_add_aligned"] == (0, 1, 2, 3, 4)
    assert collected["selected"]["ror7_add_aligned"] == (0, 1, 2, 3)
    assert collected["selected"]["offset_minus_one"] == (7,)
    assert collected["validation_parity_rows"].shape == (5, 56)
    assert all(value == 0 for value in collected["resume_rows_generated"].values())


def test_readiness_cli_reuses_verified_phase_c_without_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = topology.SpeckHwangTopologyPairConfig(run_id="readiness-test")
    keys = make_phase_c_keys(config.phase_c_config())
    monkeypatch.setattr(
        "blockcipher_nd.cli.audit_innovation2_speck_hwang_topology_pairs.load_phase_c_position_baselines",
        lambda *args, **kwargs: {"keys": keys},
    )
    output = tmp_path / "readiness"
    assert audit_main(
        [
            "--run-id",
            "readiness-test",
            "--output-root",
            str(output),
            "--phase-c-root",
            str(tmp_path / "phase-c"),
            "--readiness-only",
        ]
    ) == 0
    payload = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["phase_c_keys"] == 64
    assert payload["training_performed"] is False


def test_topology_plot_requires_and_renders_all_32_rows(tmp_path: Path) -> None:
    result = _evaluate((0, 1), ())
    output = tmp_path / "curves.svg"
    render_topology_pair_svg(result["rows"], result["gate"], output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E27-N" in svg
    assert "64-key稳定位置" in svg
    assert "这不是神经训练" in svg
    with pytest.raises(ValueError, match="requires 16 lanes"):
        render_topology_pair_svg(result["rows"][:-1], result["gate"], output)


def test_independent_validator_recomputes_complete_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = "topology-archive-test"
    source_commit = "a" * 40
    config = topology.SpeckHwangTopologyPairConfig(run_id=run_id)
    keys = make_phase_c_keys(config.phase_c_config())
    result = _evaluate((0, 1), (), run_id=run_id)
    paper_mask = topology.hwang_speck_basis_masks(7)[0]
    unbalanced = np.uint32(paper_mask & -paper_mask)
    screen = np.full((2, 16, 8), unbalanced, dtype=np.uint32)
    screen[0, 0] = 0
    screen[0, 1] = 0
    selected = {
        "ror7_add_aligned": (0, 1),
        "offset_minus_one": (),
    }
    selected_specs = (("ror7_add_aligned", 0), ("ror7_add_aligned", 1))
    validation = np.zeros((2, 56), dtype=np.uint32)
    completed, resumed = _roles(selected_specs)
    root = tmp_path / run_id
    root.mkdir()

    def write_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    (root / "source_expected_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    (root / "git_revision.txt").write_text(source_commit + "\n", encoding="utf-8")
    (root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["rows"]),
        encoding="utf-8",
    )
    write_json(root / "metadata.json", result["metadata"])
    write_json(root / "gate.json", result["gate"])
    write_json(
        root / "summary.json",
        {"resume_rows_generated": resumed, "completed": completed},
    )
    write_json(
        root / "selected_candidates.json",
        {family: list(values) for family, values in selected.items()},
    )
    np.save(root / "screen_parity_rows.npy", screen)
    np.save(root / "validation_parity_rows.npy", validation)
    (root / "kernel_basis.csv").write_text("run_id\n", encoding="utf-8")
    with (root / "keys.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key_index", "split", "key_hex"])
        writer.writeheader()
        for index, key in enumerate(keys):
            writer.writerow(
                {
                    "key_index": index,
                    "split": "discovery" if index < 32 else "validation",
                    "key_hex": f"0x{key:016X}",
                }
            )

    progress = []
    for family in topology.FAMILIES:
        for lane in topology.LANES:
            progress.extend(
                {
                    "event": "speck_parity_row_done",
                    "family": family,
                    "lane": lane,
                    "phase": "screen",
                    "key_index": key_index,
                    "elapsed_seconds": 1.0,
                }
                for key_index in range(8)
            )
    for family, lane in selected_specs:
        progress.extend(
            {
                "event": "speck_parity_row_done",
                "family": family,
                "lane": lane,
                "phase": "validation",
                "key_index": key_index,
                "elapsed_seconds": 1.0,
            }
            for key_index in range(56)
        )
    (root / "progress.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress),
        encoding="utf-8",
    )

    for family_index, family in enumerate(topology.FAMILIES):
        for lane in topology.LANES:
            phases = ("screen", "validation") if (family, lane) in selected_specs else ("screen",)
            for phase in phases:
                cache = root / "cache" / family / f"lane{lane:02d}" / phase
                cache.mkdir(parents=True)
                expected = (
                    screen[family_index, lane]
                    if phase == "screen"
                    else validation[selected_specs.index((family, lane))]
                )
                np.save(cache / "parity_rows.npy", expected[None, :])
                np.save(cache / "completed.npy", np.ones((1, expected.size), dtype=np.bool_))
                phase_keys = keys[:8] if phase == "screen" else keys[8:]
                role = f"{family}_lane{lane:02d}_{phase}"
                write_json(
                    cache / "metadata.json",
                    {
                        "active_bits": list(topology.active_bits_for_topology_pair(family, lane)),
                        "assignments_per_key": 1 << 30,
                        "backend": config.backend,
                        "chunk_size": config.chunk_size,
                        "cipher": "SPECK32/64",
                        "device": config.device,
                        "fixed_plaintext": "0x00000000",
                        "keys": [f"0x{key:016X}" for key in phase_keys],
                        "output_bit_order": "LSB-first",
                        "rounds": [7],
                        "run_id": f"{run_id}:{role}",
                    },
                )

    baseline_hashes = {}
    for role in ("anchor", "control"):
        baseline = root / "baseline_phase_c" / role
        baseline.mkdir(parents=True)
        np.save(baseline / "parity_rows.npy", np.zeros((1, 64), dtype=np.uint32))
        np.save(baseline / "completed.npy", np.ones((1, 64), dtype=np.bool_))
        write_json(baseline / "metadata.json", {"role": role})
        baseline_hashes[role] = (
            hashlib.sha256((baseline / "parity_rows.npy").read_bytes()).hexdigest(),
            hashlib.sha256((baseline / "metadata.json").read_bytes()).hexdigest(),
        )
    monkeypatch.setattr(validator, "PHASE_C_ANCHOR_PARITY_SHA256", baseline_hashes["anchor"][0])
    monkeypatch.setattr(validator, "PHASE_C_ANCHOR_METADATA_SHA256", baseline_hashes["anchor"][1])
    monkeypatch.setattr(validator, "PHASE_C_CONTROL_PARITY_SHA256", baseline_hashes["control"][0])
    monkeypatch.setattr(validator, "PHASE_C_CONTROL_METADATA_SHA256", baseline_hashes["control"][1])

    files = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    validation_payload, local_gate = validator.validate_archive(
        root, expected_source_commit=source_commit
    )
    assert validation_payload["status"] == "pass"
    assert validation_payload["timing_rows"] == 368
    assert validation_payload["remote_gate_matches_recomputation"] is True
    assert local_gate["decision"] == "innovation2_speck_topology_aligned_family"


def test_remote_scripts_obey_windows_cache_and_monitor_rules() -> None:
    run_script = Path(
        "configs/remote/generated/run_i2_speck32_hwang_topology_pairs_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    launch_script = Path(
        "configs/remote/generated/launch_i2_speck32_hwang_topology_pairs_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = Path(
        "configs/remote/generated/monitor_i2_speck32_hwang_topology_pairs_gpu0_20260717.sh"
    ).read_text(encoding="utf-8")
    postprocess = Path(
        "configs/remote/generated/postprocess_i2_speck32_hwang_topology_pairs_gpu0_20260717.sh"
    ).read_text(encoding="utf-8")
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run_script
    assert "results_archive\\%RUN_ID%" in run_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in launch_script
    assert "EnableDelayedExpansion" not in run_script
    assert "selected_candidates.json" in run_script
    assert "git_revision.txt" in run_script
    assert "monitor_remote_results.py" not in monitor_script
    assert "retrieved_from_verified_result_branch.marker" in monitor_script
    assert "validate-innovation2-speck-hwang-topology-pairs" in postprocess
    assert "plot-innovation2-speck-hwang-topology-pairs" in postprocess
    assert "visual_qa_pending.marker" in postprocess
