from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.package_dialga_r4_dfc1 import SOURCE_FILES, main as package_main
from blockcipher_nd.tasks.innovation1.dialga_r4_dfc1 import (
    ARCHITECTURES,
    EXPECTED_CACHE_CREATIONS,
    EXPECTED_CACHE_REUSES,
    EXPECTED_FINAL_TEST_REPEATS,
    EXPECTED_FINAL_TEST_ROWS,
    EXPECTED_PARAMETER_COUNTS,
    FINAL_TEST_KEY,
    RUN_ID,
    adjudicate,
    candidate_protocol_frozen,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.dialga_r4_dfc1_launch import (
    DMC2_GATE,
    DMC2_GATE_SHA256,
    REMOTE_CONFIG,
    adjudicate_launch,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_dfc1_r4_1000000_seed0_seed1.csv"
)
GENERATED = ROOT / "configs/remote/generated"
RUN_SCRIPT = GENERATED / f"run_{RUN_ID}.cmd"
LAUNCH_SCRIPT = GENERATED / f"launch_{RUN_ID}.cmd"
MONITOR_SCRIPT = GENERATED / f"monitor_{RUN_ID}.sh"


def test_dfc1_plan_and_remote_readiness_are_frozen() -> None:
    tasks = read_tasks(PLAN)
    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    assert {task["train_samples_total"] for task in tasks} == {2_000_000}
    assert {task["validation_samples_total"] for task in tasks} == {500_000}
    assert {task["final_test_samples_total"] for task in tasks} == {
        EXPECTED_FINAL_TEST_ROWS
    }
    assert {task["final_test_repeats"] for task in tasks} == {
        EXPECTED_FINAL_TEST_REPEATS
    }
    assert {task["final_test_key"] for task in tasks} == {FINAL_TEST_KEY}

    readiness = remote_readiness_report(ROOT / REMOTE_CONFIG)
    assert readiness["status"] == "pass", readiness["errors"]
    assert readiness["expected_rows"] == 6
    assert readiness["max_samples_per_class"] == 1_000_000


def test_dfc1_gate_requires_each_validation_and_final_test_gate() -> None:
    rows = _result_rows()
    gate = _adjudicate(rows)
    assert gate["status"] == "pass"
    assert gate["formal_scale"] == "project_formal_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed_validation = deepcopy(rows)
    _row(failed_validation, seed=1, architecture="correct")["metrics"]["auc"] = 0.89
    held_validation = _adjudicate(failed_validation)
    assert held_validation["status"] == "hold"
    assert held_validation["formal_scale"] == "no"
    assert "seed1_correct_auc" in held_validation["failed_research_checks"]

    failed_repeat = deepcopy(rows)
    final = _row(failed_repeat, seed=0, architecture="correct")["final_evaluation"]
    final["metrics_by_repeat"][3]["auc"] = 0.89
    held_repeat = _adjudicate(failed_repeat)
    assert held_repeat["status"] == "hold"
    assert "seed0_final4_correct_auc" in held_repeat["failed_research_checks"]


def test_dfc1_protocol_rejects_incomplete_final_test_and_cache_contract() -> None:
    rows = _result_rows()
    rows[0]["final_evaluation"]["metrics_by_repeat"].pop()
    invalid = _adjudicate(rows)
    assert invalid["status"] == "invalid"
    assert "result_protocol_frozen" in invalid["failed_protocol_checks"]

    cache_invalid = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=_result_rows(),
        progress_events=_progress_events()[:-1],
        source_checks=_source_checks(),
    )
    assert cache_invalid["status"] == "invalid"
    assert (
        "twenty_eight_parameter_matched_cache_reuses"
        in cache_invalid["failed_protocol_checks"]
    )


def test_dfc1_launch_authority_and_generated_assets_are_fail_closed() -> None:
    authority = {
        "dmc2_gate_sha256_exact": True,
        "dmc2_gate_json_readable": True,
        "dmc2_gate_exact_pass": True,
        "dmc2_formal_preregistration_authorized": True,
        "dmc2_protocol_checks_complete": True,
        "dmc2_research_checks_complete": True,
    }
    commit = "c" * 40
    gate = adjudicate_launch(
        source_commit=commit,
        remote_main_sha=commit,
        readiness_status="pass",
        authority=authority,
        source_commit_valid=True,
        remote_main_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        protected_worktree_clean=True,
    )
    assert gate["launch_authorized"] is True
    assert gate["decision"] == "innovation1_dialga_dfc1_remote_launch_authorized"

    authority_path = ROOT / DMC2_GATE
    assert authority_path.is_file()
    assert hashlib.sha256(authority_path.read_bytes()).hexdigest() == DMC2_GATE_SHA256

    run = RUN_SCRIPT.read_text(encoding="utf-8")
    launch = LAUNCH_SCRIPT.read_text(encoding="utf-8")
    monitor = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "!" not in run + launch
    assert "cmd.exe /k" not in run + launch
    assert "cmd.exe /c" in launch
    assert 'if not "%PHYSICAL_GPU%"=="0"' in run + launch
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run
    assert "--dataset-cache-root" in run
    assert "--dataset-cache-chunk-size 1024" in run
    assert "--dataset-cache-workers 1" in run
    assert "--checkpoint-root" in run
    assert "--expected-rows 6" in run
    assert "scripts\\gate-dialga-r4-dfc1" in run
    assert "scripts\\package-dialga-r4-dfc1" in run
    assert "gate-dialga-r4-dmc2" not in run + monitor
    assert "package-dialga-r4-dmc2" not in run + monitor
    assert 'git add -f "results_archive\\%RUN_ID%"' in run
    assert "I1_DIALGA_DFC1_S0S1_GPU0" in launch
    assert "_launched.marker\" exit /b 9" in launch
    assert "_started.marker\" exit /b 9" in launch
    assert "_done.marker\" exit /b 9" in launch
    assert "_failed.marker\" exit /b 9" in launch
    assert "git clone --no-checkout" in launch
    assert "git checkout --detach" in launch
    assert "Hostname=ssh.github.com -p 443" in run + launch
    assert "innovation1_dialga_dfc1_remote_launch_authorized" in monitor
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "scripts/gate-dialga-r4-dfc1" in monitor
    assert "scripts/index-results" in monitor
    assert "visual_qa_pending.marker" in monitor
    assert "verified_result_incomplete_trying_raw_fallback" in monitor
    assert "if retrieve_archive raw" in monitor
    assert "unavailable or incomplete" in monitor
    assert "results_archive/${RUN_ID}/." not in monitor
    assert 'results_archive/${RUN_ID}" "${staging}/"' in monitor

    archive_log_path = (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        f"{RUN_ID}\\source\\results_archive\\{RUN_ID}\\logs\\failed.marker"
    )
    assert len(archive_log_path) < 260
    archive_checkpoint_path = (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        f"{RUN_ID}\\source\\results_archive\\{RUN_ID}\\checkpoints\\checkpoint_05.pt"
    )
    assert len(archive_checkpoint_path) < 240


def test_dfc1_archive_requires_six_checkpoints_and_fourteen_caches(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoints = run_root / "checkpoints"
    caches = run_root / "cache"
    for root in (source_root, results_root, logs_root, checkpoints, caches):
        root.mkdir(parents=True)
    for name in (
        "results.jsonl",
        "validation-plan.json",
        "validation.json",
        "gate.json",
        "summary.json",
        "history.csv",
    ):
        content = "".join("{}\n" for _ in range(6)) if name == "results.jsonl" else "{}\n"
        (results_root / name).write_text(content, encoding="utf-8")
    (logs_root / "progress.jsonl").write_text('{"event":"run_done"}\n', encoding="utf-8")
    (logs_root / f"{RUN_ID}_failed.marker").write_text("historical\n", encoding="utf-8")
    for index in range(6):
        (checkpoints / f"row{index}.pt").write_bytes(b"checkpoint")
    splits = ("train", "validation", *(f"final_test_{i}" for i in range(1, 6)))
    for seed in (0, 1):
        for split in splits:
            cache = caches / "dialga128" / "r4" / split / f"seed{seed}"
            cache.mkdir(parents=True)
            (cache / "features.npy").write_bytes(b"features")
            (cache / "labels.npy").write_bytes(b"labels")
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "generation_chunk_size": 1024,
                        "generation_workers": 1,
                        "total_rows": EXPECTED_FINAL_TEST_ROWS,
                        "input_bits": 1024,
                    }
                ),
                encoding="utf-8",
            )
    for source in SOURCE_FILES:
        path = source_root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    sha = "a" * 40
    actual = logs_root / f"{RUN_ID}_git_revision.txt"
    expected = run_root / "source_expected_commit.txt"
    actual.write_text(sha + "\n", encoding="utf-8")
    expected.write_text(sha + "\n", encoding="utf-8")
    archive = source_root / "results_archive" / RUN_ID
    assert package_main(
        [
            "--run-root",
            str(run_root),
            "--source-root",
            str(source_root),
            "--source-commit-file",
            str(actual),
            "--expected-source-commit-file",
            str(expected),
            "--archive-root",
            str(archive),
        ]
    ) == 0
    assert len(list((archive / "checkpoints").glob("*.pt"))) == 6
    cache_manifest = json.loads((archive / "cache_manifest.json").read_text())
    assert cache_manifest["count"] == EXPECTED_CACHE_CREATIONS
    assert {entry["split"] for entry in cache_manifest["caches"]} == set(splits)
    assert (archive / "plan.csv").is_file()
    assert (archive / "remote_config.json").is_file()
    assert (archive / "experiment_plan.md").is_file()
    assert (archive / "logs" / "failed.marker").is_file()
    assert (archive / "SHA256SUMS").is_file()


def test_dfc1_remote_postprocessing_does_not_import_plotting_modules() -> None:
    for script in ("scripts/gate-dialga-r4-dfc1", "scripts/package-dialga-r4-dfc1"):
        code = (
            "import builtins,runpy,sys;"
            "real_import=builtins.__import__;"
            "builtins.__import__=lambda name,*a,**k: "
            "(_ for _ in ()).throw(RuntimeError('plot import blocked')) "
            "if name.startswith(('matplotlib','seaborn')) else real_import(name,*a,**k);"
            f"sys.argv=[{script!r},'--help'];"
            f"runpy.run_path({script!r},run_name='__main__')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def _adjudicate(rows: list[dict[str, object]]) -> dict[str, object]:
    return adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_events=_progress_events(),
        source_checks=_source_checks(),
    )


def _source_checks() -> dict[str, bool]:
    return {
        "source_revision_matches_launch_pin": True,
        "six_nonempty_checkpoints_present": True,
    }


def _row(
    rows: list[dict[str, object]], *, seed: int, architecture: str
) -> dict[str, object]:
    model = ARCHITECTURES[architecture]
    return next(row for row in rows if row["seed"] == seed and row["model"] == model)


def _result_rows() -> list[dict[str, object]]:
    validation_aucs = {
        0: {"correct": 0.970, "corrupted": 0.955, "autond": 0.510},
        1: {"correct": 0.972, "corrupted": 0.957, "autond": 0.505},
    }
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for architecture, model in ARCHITECTURES.items():
            final_aucs = {
                "correct": [0.969, 0.968, 0.971, 0.970, 0.972],
                "corrupted": [0.952, 0.951, 0.954, 0.953, 0.955],
                "autond": [0.508, 0.509, 0.507, 0.510, 0.506],
            }[architecture]
            metrics_by_repeat = [
                {
                    "repeat": index + 1,
                    "seed": seed + 50_000 + index,
                    "final_test_key": FINAL_TEST_KEY,
                    "samples_total": EXPECTED_FINAL_TEST_ROWS,
                    "positive_rows": EXPECTED_FINAL_TEST_ROWS // 2,
                    "negative_rows": EXPECTED_FINAL_TEST_ROWS // 2,
                    "auc": auc,
                    "accuracy": auc - 0.01,
                }
                for index, auc in enumerate(final_aucs)
            ]
            rows.append(
                {
                    "model": model,
                    "rounds": 4,
                    "seed": seed,
                    "samples_per_class": 1_000_000,
                    "pairs_per_sample": 4,
                    "input_difference": 0x40,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "final_test_repeats": EXPECTED_FINAL_TEST_REPEATS,
                    "final_test_samples_total": EXPECTED_FINAL_TEST_ROWS,
                    "final_test_key": FINAL_TEST_KEY,
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNTS[architecture],
                    "metrics": {"auc": validation_aucs[seed][architecture]},
                    "training": {
                        "input_bits": 1024,
                        "train_rows": 2_000_000,
                        "validation_rows": 500_000,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "selected_checkpoint": "best",
                        "restore_best_checkpoint": True,
                        "checkpoint_output": f"checkpoints/seed{seed}_{architecture}.pt",
                    },
                    "final_evaluation": {
                        "repeats": EXPECTED_FINAL_TEST_REPEATS,
                        "samples_total_per_repeat": EXPECTED_FINAL_TEST_ROWS,
                        "final_test_key": FINAL_TEST_KEY,
                        "seeds": [seed + 50_000 + i for i in range(5)],
                        "metrics_by_repeat": metrics_by_repeat,
                    },
                }
            )
    return rows


def _progress_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    splits = ("train", "validation", *(f"final_test_{i}" for i in range(1, 6)))
    for seed in (0, 1):
        for split in splits:
            events.append({"event": "cache_start", "seed": seed, "split": split})
            events.extend(
                {"event": "cache_reuse", "seed": seed, "split": split}
                for _ in range(2)
            )
    assert sum(event["event"] == "cache_start" for event in events) == 14
    assert sum(event["event"] == "cache_reuse" for event in events) == EXPECTED_CACHE_REUSES
    return events
