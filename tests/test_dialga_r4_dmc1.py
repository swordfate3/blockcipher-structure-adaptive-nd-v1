from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.package_dialga_r4_dmc1 import (
    RESULT_FILES,
    SOURCE_FILES,
    package_archive,
)
from blockcipher_nd.cli.plot_dialga_r4_dmc1 import render_dmc1_svg
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.dialga_r4_dmc1 import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    RUN_ID,
    adjudicate,
    candidate_protocol_frozen,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.dialga_r4_dmc1_launch import (
    PLAN,
    REMOTE_CONFIG,
    adjudicate_launch,
)


ROOT = Path(__file__).resolve().parents[1]
REMOTE_RUN_ID = RUN_ID
GENERATED = ROOT / "configs/remote/generated"
RUN_SCRIPT = GENERATED / f"run_{REMOTE_RUN_ID}.cmd"
LAUNCH_SCRIPT = GENERATED / f"launch_{REMOTE_RUN_ID}.cmd"
MONITOR_SCRIPT = GENERATED / f"monitor_{REMOTE_RUN_ID}.sh"


def test_dmc1_plan_models_and_remote_readiness_are_frozen() -> None:
    tasks = read_tasks(ROOT / PLAN)
    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    counts: dict[str, int] = {}
    for architecture, model_key in ARCHITECTURES.items():
        task = next(task for task in tasks if task["model_key"] == model_key)
        model = build_model(
            model_key,
            input_bits=1024,
            hidden_bits=64,
            pair_bits=256,
            structure="SPN",
            model_options=task["model_options"],
        )
        counts[architecture] = sum(parameter.numel() for parameter in model.parameters())
        output = model(torch.zeros(2, 1024))
        assert output.shape[0] == 2
    assert counts == EXPECTED_PARAMETER_COUNTS
    readiness = remote_readiness_report(ROOT / REMOTE_CONFIG)
    assert readiness["status"] == "pass", readiness
    assert readiness["expected_rows"] == 6
    assert readiness["max_samples_per_class"] == 65536


def test_dmc1_gate_requires_both_seed_signal_and_control_margins(tmp_path: Path) -> None:
    rows = _result_rows(tmp_path)
    gate = adjudicate(
        tasks=read_tasks(ROOT / PLAN),
        result_rows=rows,
        progress_events=_progress_events(),
        source_checks={"source_revision_matches_launch_pin": True},
    )
    assert gate["status"] == "pass"
    assert gate["remote_scale"] == "authorized_262144_per_class"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = deepcopy(rows)
    for row in failed:
        if row["seed"] == 1 and row["model"] == ARCHITECTURES["corrupted"]:
            row["metrics"]["auc"] = 0.952
    held = adjudicate(
        tasks=read_tasks(ROOT / PLAN),
        result_rows=failed,
        progress_events=_progress_events(),
        source_checks={"source_revision_matches_launch_pin": True},
    )
    assert held["status"] == "hold"
    assert held["remote_scale"] == "no"


def test_dmc1_plot_has_clear_chinese_context(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(ROOT / PLAN),
        result_rows=_result_rows(tmp_path),
        progress_events=_progress_events(),
        source_checks={"source_revision_matches_launch_pin": True},
    )
    output = tmp_path / "curves.svg"
    report = render_dmc1_svg(gate, output)
    svg = output.read_text(encoding="utf-8")
    assert report["panels"] == 2
    assert "Dialga 第4轮异构拓扑中等规模验证" in svg
    assert "65536/class" in svg
    assert "跨密钥验证" in svg
    assert "正确拓扑相对两类控制的优势" in svg


def test_dmc1_launch_gate_and_generated_assets_are_fail_closed() -> None:
    authority = {
        "d1_gate_exact_pass": True,
        "d2_gate_exact_pass": True,
        "digests_exact": True,
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
    unpublished = adjudicate_launch(
        source_commit=commit,
        remote_main_sha="d" * 40,
        readiness_status="pass",
        authority=authority,
        source_commit_valid=True,
        remote_main_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        protected_worktree_clean=True,
    )
    assert unpublished["should_ssh"] is True
    assert unpublished["ssh_allowed"] is False

    run = RUN_SCRIPT.read_text(encoding="utf-8")
    launch = LAUNCH_SCRIPT.read_text(encoding="utf-8")
    monitor = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "!" not in run + launch
    assert "cmd.exe /k" not in run + launch
    assert "cmd.exe /c" in launch
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run + launch
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run
    assert "--dataset-cache-root" in run
    assert "--dataset-cache-chunk-size 1024" in run
    assert "--dataset-cache-workers 1" in run
    assert "--expected-rows 6" in run
    assert 'if not "%PHYSICAL_GPU%"=="1"' in run + launch
    assert "git clone --no-checkout" in launch
    assert "git checkout --detach" in launch
    assert "Hostname=ssh.github.com -p 443" in run + launch
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "retrieved_from_verified_result_branch.marker" in monitor
    assert "fallback_retrieved.marker" in monitor
    assert "visual_qa_pending.marker" in monitor


def test_dmc1_archive_requires_six_checkpoints_and_four_caches(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoints = run_root / "checkpoints"
    caches = run_root / "cache"
    for root in (source_root, results_root, logs_root, checkpoints, caches):
        root.mkdir(parents=True)
    for name in RESULT_FILES:
        payload = "".join("{}\n" for _ in range(6)) if name == "results.jsonl" else "{}\n"
        (results_root / name).write_text(payload, encoding="utf-8")
    (logs_root / "progress.jsonl").write_text('{"event":"run_done"}\n', encoding="utf-8")
    (logs_root / f"{RUN_ID}_gpu_info.txt").write_text("gpu\n", encoding="utf-8")
    for index in range(6):
        (checkpoints / f"row{index}.pt").write_bytes(b"checkpoint")
    for index in range(4):
        split = "train" if index < 2 else "validation"
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
    report = package_archive(
        run_root=run_root,
        source_root=source_root,
        source_commit_file=actual,
        expected_source_commit_file=expected,
        archive_root=archive,
    )
    assert report["result_rows"] == 6
    assert report["checkpoint_count"] == 6
    assert report["cache_count"] == 4
    assert (archive / "SHA256SUMS").is_file()
    assert len(list((archive / "checkpoints").glob("*.pt"))) == 6


def _result_rows(tmp_path: Path) -> list[dict[str, object]]:
    aucs = {
        0: {"correct": 0.960, "corrupted": 0.940, "autond": 0.520},
        1: {"correct": 0.955, "corrupted": 0.945, "autond": 0.530},
    }
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for architecture, model in ARCHITECTURES.items():
            checkpoint = tmp_path / f"seed{seed}_{architecture}.pt"
            checkpoint.write_bytes(b"checkpoint")
            rows.append(
                {
                    "model": model,
                    "rounds": 4,
                    "seed": seed,
                    "samples_per_class": 65536,
                    "pairs_per_sample": 4,
                    "input_difference": 0x40,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNTS[architecture],
                    "metrics": {"auc": aucs[seed][architecture]},
                    "training": {
                        "input_bits": 1024,
                        "train_rows": 131072,
                        "validation_rows": 32768,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "selected_checkpoint": "best",
                        "checkpoint_output": str(checkpoint),
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
