from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import torch

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.package_uknit_r5_k1ca_closeout import (
    SOURCE_FILES,
    main as package_main,
)
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_r5_invariant_autond_closeout_k1ca import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    RUN_ID,
    adjudicate,
    candidate_protocol_frozen,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_invariant_autond_closeout_k1ca_launch import (
    REMOTE_CONFIG,
    adjudicate_launch,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_uknit_r5_k1ca_invariant_autond_closeout_262144_seed3_seed4.csv"
)
GENERATED = ROOT / "configs/remote/generated"
RUN_SCRIPT = GENERATED / f"run_{RUN_ID}.cmd"
LAUNCH_SCRIPT = GENERATED / f"launch_{RUN_ID}.cmd"
MONITOR_SCRIPT = GENERATED / f"monitor_{RUN_ID}.sh"


def test_k1ca_plan_and_remote_readiness_are_frozen() -> None:
    tasks = read_tasks(PLAN)
    assert len(tasks) == 4
    assert candidate_protocol_frozen(tasks)
    readiness = remote_readiness_report(ROOT / REMOTE_CONFIG)
    assert readiness["status"] == "pass", readiness
    assert readiness["expected_rows"] == 4
    assert readiness["max_samples_per_class"] == 262144


def test_k1ca_models_accept_same_four_pair_input_and_counts_match() -> None:
    task = read_tasks(PLAN)[0]
    fixture = torch.zeros((2, 512), dtype=torch.float32)
    for architecture, model_key in ARCHITECTURES.items():
        model = build_model(
            model_key,
            input_bits=512,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options=dict(task["model_options"]),
        )
        assert tuple(model(fixture).shape) == (2, 1)
        assert (
            model_metadata(model)["trainable_parameter_count"]
            == EXPECTED_PARAMETER_COUNTS[architecture]
        )


def test_k1ca_gate_requires_both_seed_floors_and_autond_margins() -> None:
    rows = _result_rows()
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_events=_progress_events(),
        source_checks={
            "source_revision_matches_launch_pin": True,
            "four_nonempty_checkpoints_present": True,
        },
    )
    assert gate["status"] == "pass"
    assert gate["experiment_stage_after_valid_result"] == "closed"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = deepcopy(rows)
    for row in failed:
        if row["seed"] == 4 and row["model"] == ARCHITECTURES["autond_dbitnet"]:
            row["metrics"]["auc"] = 0.90
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=failed,
        progress_events=_progress_events(),
        source_checks={
            "source_revision_matches_launch_pin": True,
            "four_nonempty_checkpoints_present": True,
        },
    )
    assert held["status"] == "hold"
    assert held["experiment_stage_after_valid_result"] == "closed"
    assert "cache-reusing K1-CB" in held["next_action"]
    assert "without changing K1-CA" in held["next_action"]


def test_k1ca_cache_gate_rejects_extra_final_test_or_missing_reuse() -> None:
    events = _progress_events()
    valid = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=_result_rows(),
        progress_events=events,
        source_checks={
            "source_revision_matches_launch_pin": True,
            "four_nonempty_checkpoints_present": True,
        },
    )
    assert valid["status"] == "pass"

    with_final = deepcopy(events)
    with_final.insert(
        -1,
        {
            "event": "cache_start",
            "seed": 10003,
            "split": "final_test_1",
            "cache_path": _run_path("cache/final_test_1/extra"),
        },
    )
    invalid = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=_result_rows(),
        progress_events=with_final,
        source_checks={
            "source_revision_matches_launch_pin": True,
            "four_nonempty_checkpoints_present": True,
        },
    )
    assert invalid["status"] == "invalid"
    assert "zero_final_test_cache_events" in invalid["failed_protocol_checks"]

    missing_reuse = deepcopy(events)
    reuse_index = next(
        index
        for index in range(len(missing_reuse))
        if missing_reuse[index].get("event") == "cache_reuse"
    )
    del missing_reuse[reuse_index]
    invalid_reuse = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=_result_rows(),
        progress_events=missing_reuse,
        source_checks={
            "source_revision_matches_launch_pin": True,
            "four_nonempty_checkpoints_present": True,
        },
    )
    assert invalid_reuse["status"] == "invalid"
    assert "four_autond_cache_reuses_exact" in invalid_reuse["failed_protocol_checks"]


def test_k1ca_launch_gate_and_generated_assets_are_fail_closed() -> None:
    commit = "c" * 40
    gate = adjudicate_launch(
        source_commit=commit,
        remote_main_sha=commit,
        readiness_status="pass",
        authority={"k1u_selected_invariant": True},
        paper_contract_frozen=True,
        four_row_plan_frozen=True,
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
    assert "--hidden-bits 32" in run
    assert "--dataset-cache-root" in run
    assert "--dataset-cache-chunk-size 1024" in run
    assert "--dataset-cache-workers 1" in run
    assert "--expected-rows 4" in run
    assert "--final-test" not in run
    assert "scripts\\gate-uknit-r5-k1ca-closeout" in run
    assert "scripts\\package-uknit-r5-k1ca-closeout" in run
    assert "schtasks /Change" in launch and "/DISABLE" in launch
    assert "git clone --no-checkout" in launch
    assert "git checkout --detach" in launch
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "scripts/gate-uknit-r5-k1ca-closeout" in monitor
    assert "scripts/index-results" in monitor
    assert "visual_qa_pending.marker" in monitor
    assert "verified_result_incomplete_trying_raw_fallback" in monitor
    assert "results_archive/${RUN_ID}/." not in monitor
    assert 'results_archive/${RUN_ID}" "${staging}/"' in monitor

    archive_log_path = (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        f"{RUN_ID}\\source\\results_archive\\{RUN_ID}\\logs\\failed.marker"
    )
    assert len(archive_log_path) < 260


def test_k1ca_archive_requires_four_checkpoints_and_four_caches(tmp_path: Path) -> None:
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
        content = (
            "".join("{}\n" for _ in range(4)) if name == "results.jsonl" else "{}\n"
        )
        (results_root / name).write_text(content, encoding="utf-8")
    (logs_root / "progress.jsonl").write_text(
        '{"event":"run_done"}\n', encoding="utf-8"
    )
    (logs_root / f"{RUN_ID}_failed.marker").write_text("historical\n", encoding="utf-8")
    for index in range(4):
        (checkpoints / f"row{index}.pt").write_bytes(b"checkpoint")
    for index in range(4):
        split = "train" if index % 2 == 0 else "validation"
        cache = caches / "uknit64" / "r5" / split / f"cache{index}"
        cache.mkdir(parents=True)
        (cache / "features.npy").write_bytes(b"features")
        (cache / "labels.npy").write_bytes(b"labels")
        (cache / "metadata.json").write_text(
            json.dumps(
                {
                    "generation_chunk_size": 1024,
                    "generation_workers": 1,
                    "total_rows": 16,
                    "input_bits": 512,
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
    assert (
        package_main(
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
        )
        == 0
    )
    assert len(list((archive / "checkpoints").glob("*.pt"))) == 4
    assert (archive / "plan.csv").is_file()
    assert (archive / "remote_config.json").is_file()
    assert (archive / "experiment_plan.md").is_file()
    assert (archive / "logs" / "failed.marker").is_file()
    assert (archive / "SHA256SUMS").is_file()


def test_k1ca_remote_postprocessing_does_not_import_plotting_modules() -> None:
    for script in (
        "scripts/gate-uknit-r5-k1ca-closeout",
        "scripts/package-uknit-r5-k1ca-closeout",
    ):
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
        3: {"invariant_structure_expert": 0.970, "autond_dbitnet": 0.505},
        4: {"invariant_structure_expert": 0.968, "autond_dbitnet": 0.510},
    }
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for architecture, model in ARCHITECTURES.items():
            row: dict[str, object] = {
                "model": model,
                "rounds": 5,
                "seed": seed,
                "samples_per_class": 262144,
                "pairs_per_sample": 4,
                "input_difference": 0x0000400000000000,
                "difference_profile": "uknit64_k1q_cell11_r5",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "final_test_repeats": 0,
                "final_test_samples_total": None,
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNTS[architecture],
                "metrics": {
                    "auc": aucs[seed][architecture],
                    "accuracy": 0.95,
                    "loss": 0.1,
                },
                "history": [{"epoch": epoch} for epoch in range(1, 11)],
                "training": {
                    "input_bits": 512,
                    "train_rows": 524288,
                    "validation_rows": 131072,
                    "train_positive_rows": 262144,
                    "train_negative_rows": 262144,
                    "validation_positive_rows": 65536,
                    "validation_negative_rows": 65536,
                    "train_dataset_storage": "disk",
                    "validation_dataset_storage": "disk",
                    "dataset_cache_root": _run_path("cache"),
                    "dataset_cache_chunk_size": 1024,
                    "dataset_cache_workers": 1,
                    "device": "cuda",
                    "batch_size": 64,
                    "epochs": 10,
                    "epochs_ran": 10,
                    "checkpoint_metric": "val_auc",
                    "restore_best_checkpoint": True,
                    "selected_checkpoint": "best",
                    "best_checkpoint_metric": aucs[seed][architecture],
                    "checkpoint_output": _run_path(
                        f"checkpoints/seed{seed}_{architecture}.pt"
                    ),
                },
                "validation": {"samples_total": 131072, "samples_per_class": 65536},
            }
            if architecture == "invariant_structure_expert":
                row.update(
                    {
                        "runtime_structure_descriptor_sha256": "b74f9cc28b5fc28637b179f45ded67dec1a3d5dca04ca2eccb176ec790fbefd2",
                        "runtime_structure_round_start": 3,
                        "runtime_structure_loaded_rounds": 2,
                        "runtime_structure_mode": "invariant",
                        "runtime_structure_window_control": "invariant",
                    }
                )
            rows.append(row)
    return rows


def _progress_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for seed in (3, 4):
        for split in ("train", "validation"):
            path = _run_path(f"cache/uknit64/r5/{split}/seed{seed}")
            events.append(
                {
                    "event": "cache_start",
                    "seed": seed,
                    "model": ARCHITECTURES["invariant_structure_expert"],
                    "split": split,
                    "cache_path": path,
                    "chunk_size": 1024,
                    "workers": 1,
                }
            )
            events.append(
                {
                    "event": "cache_done",
                    "seed": seed,
                    "model": ARCHITECTURES["invariant_structure_expert"],
                    "split": split,
                    "cache_path": path,
                }
            )
            events.append(
                {
                    "event": "cache_reuse",
                    "seed": seed,
                    "model": ARCHITECTURES["autond_dbitnet"],
                    "split": split,
                    "cache_path": path,
                    "chunk_size": 1024,
                    "workers": 1,
                }
            )
    events.append({"event": "run_done"})
    return events


def _run_path(relative: str) -> str:
    normalized = relative.replace("/", "\\")
    return f"G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\{RUN_ID}\\{normalized}"
