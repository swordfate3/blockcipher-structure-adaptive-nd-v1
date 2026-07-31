from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.plot_uknit_r6_pair_amplification_k1bv import render
from blockcipher_nd.cli.package_uknit_r6_pair_amplification_k1bv import (
    RESULT_FILES,
    SOURCE_FILES,
    package_archive,
)
from blockcipher_nd.tasks.innovation1.uknit_r6_pair_amplification_k1bv import (
    CONDITIONS,
    EXPECTED_PARAMETER_COUNT,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    result_protocol_frozen,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiment/innovation1/innovation1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4.csv"
CONFIG = ROOT / "configs/remote/innovation1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4_gpu0_20260731.json"
RUN_ID = "i1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4_20260731"
GENERATED = ROOT / "configs/remote/generated"


def test_plan_and_zero_training_readiness_are_frozen() -> None:
    tasks = read_tasks(PLAN)
    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    assert {(task["seed"], task["pairs_per_sample"]) for task in tasks} == {
        (3, 4), (3, 16), (4, 4), (4, 16)
    }
    readiness = build_readiness(tasks)
    assert readiness["status"] == "pass"
    assert readiness["training_performed"] is False
    assert readiness["metrics"]["parameter_counts"] == {
        "exact4": EXPECTED_PARAMETER_COUNT,
        "exact16": EXPECTED_PARAMETER_COUNT,
        "wrong16": EXPECTED_PARAMETER_COUNT,
    }
    assert readiness["metrics"]["input_bits"] == {
        "exact4": 512, "exact16": 2048, "wrong16": 2048
    }


def test_gate_requires_pair_gain_and_correct_sbox_gap_on_both_seeds() -> None:
    strong = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=_rows({3: (0.52, 0.58, 0.54), 4: (0.51, 0.57, 0.53)}),
        progress_events=_progress(), source_checks={"source": True},
    )
    assert strong["status"] == "pass"
    assert strong["tier"] == "strong"
    assert strong["remote_scale"] == "authorize_65536_per_class_confirmation"

    weak = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=_rows({3: (0.50, 0.525, 0.505), 4: (0.49, 0.515, 0.500)}),
        progress_events=_progress(), source_checks={"source": True},
    )
    assert weak["tier"] == "weak"
    assert weak["remote_scale"] == "fresh_seed_submedium_only"

    held_rows = _rows({3: (0.50, 0.525, 0.522), 4: (0.49, 0.515, 0.500)})
    held = adjudicate(
        tasks=read_tasks(PLAN), result_rows=held_rows,
        progress_events=_progress(), source_checks={"source": True},
    )
    assert held["status"] == "hold"
    assert held["remote_scale"] == "no"


def test_remote_checkpoint_path_is_bound_without_local_file_probe() -> None:
    rows = _rows({3: (0.52, 0.58, 0.54), 4: (0.51, 0.57, 0.53)})
    assert all(not Path(row["training"]["checkpoint_output"]).is_file() for row in rows)
    assert result_protocol_frozen(rows)


def test_remote_assets_are_safe_and_ready() -> None:
    readiness = remote_readiness_report(CONFIG)
    assert readiness["status"] == "pass"
    run = (GENERATED / f"run_{RUN_ID}.cmd").read_text(encoding="utf-8")
    launch = (GENERATED / f"launch_{RUN_ID}.cmd").read_text(encoding="utf-8")
    monitor = (GENERATED / f"monitor_{RUN_ID}.sh").read_text(encoding="utf-8")
    assert "!" not in run + launch
    assert "cmd.exe /k" not in run + launch
    assert "cmd.exe /c" in launch
    assert 'if not "%PHYSICAL_GPU%"=="0"' in run + launch
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run + launch
    assert "--expected-rows 6" in run
    assert "--dataset-cache-root" in run
    assert "Hostname=ssh.github.com -p 443" in run + launch
    assert "git clone --no-checkout" in launch
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "visual_qa_pending.marker" in monitor


def test_plot_uses_plain_chinese_r6_pair_explanation(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=_rows({3: (0.52, 0.58, 0.54), 4: (0.51, 0.57, 0.53)}),
        progress_events=_progress(), source_checks={"source": True},
    )
    gate_path = tmp_path / "gate.json"
    output = tmp_path / "curves.svg"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    report = render(gate_path, output)
    svg = output.read_text(encoding="utf-8")
    assert report["panels"] == 2
    assert "uKNIT 第6轮增加密文对是否带来信号" in svg
    assert "4对与16对" in svg
    assert "错误S盒" in svg
    assert "0.5约等于随机猜" in svg


def test_archive_binds_six_checkpoints_and_eight_cache_payloads(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoints = run_root / "checkpoints"
    cache_root = run_root / "cache"
    for root in (source_root, results_root, logs_root, checkpoints, cache_root):
        root.mkdir(parents=True)
    for name in RESULT_FILES:
        payload = "".join("{}\n" for _ in range(6)) if name == "results.jsonl" else "{}\n"
        (results_root / name).write_text(payload, encoding="utf-8")
    (logs_root / "progress.jsonl").write_text('{"event":"run_done"}\n', encoding="utf-8")
    for index in range(6):
        (checkpoints / f"row{index:02d}.pt").write_bytes(b"checkpoint")
    for index in range(8):
        cache = cache_root / f"cache{index:02d}"
        cache.mkdir()
        (cache / "features.npy").write_bytes(b"features")
        (cache / "labels.npy").write_bytes(b"labels")
        (cache / "metadata.json").write_text(json.dumps({
            "generation_chunk_size": 1024, "generation_workers": 1,
            "total_rows": 16, "input_bits": 512 if index < 4 else 2048,
        }), encoding="utf-8")
    for source in SOURCE_FILES:
        path = source_root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    revision = "a" * 40
    actual = logs_root / "revision.txt"
    expected = run_root / "expected.txt"
    actual.write_text(revision + "\n", encoding="utf-8")
    expected.write_text(revision + "\n", encoding="utf-8")
    archive = source_root / "results_archive" / RUN_ID
    report = package_archive(
        run_root=run_root, source_root=source_root,
        source_commit_file=actual, expected_source_commit_file=expected,
        archive_root=archive,
    )
    assert report["result_rows"] == 6
    assert report["checkpoint_count"] == 6
    assert report["cache_count"] == 8
    manifest = json.loads((archive / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 6
    assert all(entry["bytes"] > 0 and len(entry["sha256"]) == 64 for entry in manifest["checkpoints"])


def _rows(values: dict[int, tuple[float, float, float]]) -> list[dict[str, object]]:
    rows = []
    for task in read_tasks(PLAN):
        pairs = int(task["pairs_per_sample"])
        if task["model_key"] == CONDITIONS["wrong16"]:
            condition, index = "wrong16", 2
        elif pairs == 4:
            condition, index = "exact4", 0
        else:
            condition, index = "exact16", 1
        auc = values[int(task["seed"])][index]
        rows.append({
            "cipher_key": "uknit64", "rounds": 6, "seed": task["seed"],
            "model": task["model_key"], "samples_per_class": 2048,
            "pairs_per_sample": pairs, "input_difference": 0x0000400000000000,
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
            "runtime_structure_round_start": 4,
            "runtime_structure_loaded_rounds": 2,
            "metrics": {"auc": auc, "accuracy": 0.5, "loss": 0.25},
            "history": [{"epoch": epoch} for epoch in range(1, 11)],
            "training": {
                "input_bits": pairs * 128, "train_rows": 4096,
                "validation_rows": 2048, "epochs": 10, "epochs_ran": 10,
                "batch_size": 64, "device": "cuda", "selected_checkpoint": "best",
                "restore_best_checkpoint": True,
                "checkpoint_output": rf"G:\lxy\runs\k1bv\seed{task['seed']}_{condition}.pt",
            },
        })
    return rows


def _progress() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for seed in (3, 4):
        for pairs in (4, 16):
            for split in ("train", "validation"):
                base = {
                    "seed": seed, "split": split,
                    "cache_path": rf"G:\lxy\runs\k1bv\cache\s{seed}\p{pairs}\{split}",
                    "chunk_size": 1024, "workers": 1,
                }
                events.append({"event": "cache_start", **base})
                events.append({"event": "cache_done", **base})
                if pairs == 16:
                    events.append({"event": "cache_reuse", "model": CONDITIONS["wrong16"], **base})
    events.append({"event": "run_done"})
    return deepcopy(events)
