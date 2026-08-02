from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.package_dialga_r4_dmc2 import SOURCE_FILES, main as package_main
from blockcipher_nd.tasks.innovation1.dialga_r4_dmc2 import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    RUN_ID,
    adjudicate,
    candidate_protocol_frozen,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.dialga_r4_dmc2_launch import (
    REMOTE_CONFIG,
    adjudicate_launch,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_dmc2_r4_262144_seed0_seed1.csv"
)
GENERATED = ROOT / "configs/remote/generated"
RUN_SCRIPT = GENERATED / f"run_{RUN_ID}.cmd"
LAUNCH_SCRIPT = GENERATED / f"launch_{RUN_ID}.cmd"
MONITOR_SCRIPT = GENERATED / f"monitor_{RUN_ID}.sh"


def test_dmc2_plan_and_remote_readiness_are_frozen() -> None:
    tasks = read_tasks(PLAN)
    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    readiness = remote_readiness_report(ROOT / REMOTE_CONFIG)
    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 6
    assert readiness["max_samples_per_class"] == 262144


def test_dmc2_gate_requires_both_seed_signals_and_margins(tmp_path: Path) -> None:
    rows = _result_rows()
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_events=_progress_events(),
        source_checks={
            "source_revision_matches_launch_pin": True,
            "six_nonempty_checkpoints_present": True,
        },
    )
    assert gate["status"] == "pass"
    assert gate["formal_scale"] == "authorized_dfc1_preregistration"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = deepcopy(rows)
    for row in failed:
        if row["seed"] == 1 and row["model"] == ARCHITECTURES["correct"]:
            row["metrics"]["auc"] = 0.89
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=failed,
        progress_events=_progress_events(),
        source_checks={
            "source_revision_matches_launch_pin": True,
            "six_nonempty_checkpoints_present": True,
        },
    )
    assert held["status"] == "hold"
    assert held["formal_scale"] == "no"


def test_dmc2_launch_gate_and_generated_assets_are_fail_closed() -> None:
    authority = {
        "dmc1_gate_sha256_exact": True,
        "dmc1_gate_json_readable": True,
        "dmc1_gate_exact_pass": True,
        "dmc1_scale_authorized": True,
        "dmc1_protocol_checks_complete": True,
        "dmc1_research_checks_complete": True,
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
    assert "_started.marker\" exit /b 9" in launch
    assert "git clone --no-checkout" in launch
    assert "git checkout --detach" in launch
    assert "Hostname=ssh.github.com -p 443" in run + launch
    assert "cmd.exe /c" in launch
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "scripts/gate-dialga-r4-dmc2" in monitor
    assert "scripts/index-results" in monitor
    assert "visual_qa_pending.marker" in monitor
    assert "verified_result_incomplete_trying_raw_fallback" in monitor
    assert "if retrieve_archive raw" in monitor
    assert "unavailable or incomplete" in monitor

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


def test_dmc2_archive_requires_six_checkpoints_and_four_caches(
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
    for index in range(4):
        split = "train" if index % 2 == 0 else "validation"
        cache = caches / "dialga128" / "r4" / split / f"cache{index}"
        cache.mkdir(parents=True)
        (cache / "features.npy").write_bytes(b"features")
        (cache / "labels.npy").write_bytes(b"labels")
        (cache / "metadata.json").write_text(
            json.dumps(
                {
                    "generation_chunk_size": 1024,
                    "generation_workers": 1,
                    "total_rows": 16,
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
            "--run-root", str(run_root),
            "--source-root", str(source_root),
            "--source-commit-file", str(actual),
            "--expected-source-commit-file", str(expected),
            "--archive-root", str(archive),
        ]
    ) == 0
    assert len(list((archive / "checkpoints").glob("*.pt"))) == 6
    assert (archive / "plan.csv").is_file()
    assert (archive / "remote_config.json").is_file()
    assert (archive / "experiment_plan.md").is_file()
    assert (archive / "logs" / "failed.marker").is_file()
    assert (archive / "SHA256SUMS").is_file()


def test_dmc2_remote_postprocessing_does_not_import_plotting_modules() -> None:
    for script in ("scripts/gate-dialga-r4-dmc2", "scripts/package-dialga-r4-dmc2"):
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


def _result_rows() -> list[dict[str, object]]:
    aucs = {
        0: {"correct": 0.970, "corrupted": 0.955, "autond": 0.510},
        1: {"correct": 0.972, "corrupted": 0.957, "autond": 0.505},
    }
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for architecture, model in ARCHITECTURES.items():
            rows.append(
                {
                    "model": model,
                    "rounds": 4,
                    "seed": seed,
                    "samples_per_class": 262144,
                    "pairs_per_sample": 4,
                    "input_difference": 0x40,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "final_test_repeats": 0,
                    "final_test_samples_total": None,
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNTS[architecture],
                    "metrics": {"auc": aucs[seed][architecture]},
                    "training": {
                        "input_bits": 1024,
                        "train_rows": 524288,
                        "validation_rows": 131072,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "selected_checkpoint": "best",
                        "restore_best_checkpoint": True,
                        "checkpoint_output": f"checkpoints/seed{seed}_{architecture}.pt",
                    },
                }
            )
    return rows


def _progress_events() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for split in ("train", "validation"):
            rows.append({"event": "cache_start", "seed": seed, "split": split})
            rows.extend(
                {"event": "cache_reuse", "seed": seed, "split": split}
                for _ in range(2)
            )
    return rows
