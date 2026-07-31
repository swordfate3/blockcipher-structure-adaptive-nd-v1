from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.package_uknit_r5_architecture_medium_k1bt import (
    RESULT_FILES,
    SOURCE_FILES,
    package_archive,
)
from blockcipher_nd.cli.plot_uknit_r5_architecture_medium_k1bt import render_k1bt_svg
from blockcipher_nd.tasks.innovation1.uknit_r5_architecture_medium_k1bt import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    RUN_ID,
    adjudicate,
    candidate_protocol_frozen,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_architecture_medium_k1bt_launch import (
    K1BS_PLAN,
    K1BT_PLAN,
    _plans_match,
    adjudicate_launch,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / K1BT_PLAN
REMOTE_RUN_ID = RUN_ID
GENERATED = ROOT / "configs/remote/generated"
RUN_SCRIPT = GENERATED / f"run_{REMOTE_RUN_ID}.cmd"
LAUNCH_SCRIPT = GENERATED / f"launch_{REMOTE_RUN_ID}.cmd"
MONITOR_SCRIPT = GENERATED / f"monitor_{REMOTE_RUN_ID}.sh"
REMOTE_CONFIG = ROOT / "configs/remote/innovation1_uknit_k1bt_architecture_medium_65536_seed3_seed4_gpu1_20260731.json"


def test_k1bt_plan_and_remote_readiness_are_frozen() -> None:
    tasks = read_tasks(PLAN)
    assert len(tasks) == 4
    assert candidate_protocol_frozen(tasks)
    assert _plans_match(ROOT / K1BS_PLAN, ROOT / K1BT_PLAN)
    readiness = remote_readiness_report(REMOTE_CONFIG)
    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 4
    assert readiness["max_samples_per_class"] == 65536


def test_k1bt_gate_requires_both_seed_signals_and_margins(tmp_path: Path) -> None:
    rows = _result_rows(tmp_path)
    gate = adjudicate(
        tasks=read_tasks(PLAN),
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
        if row["seed"] == 4 and row["model"] == ARCHITECTURES["uknit_structure_expert"]:
            row["metrics"]["auc"] = 0.54
    held = adjudicate(
        tasks=read_tasks(PLAN), result_rows=failed,
        progress_events=_progress_events(),
        source_checks={"source_revision_matches_launch_pin": True},
    )
    assert held["status"] == "hold"
    assert held["remote_scale"] == "no"


def test_k1bt_plot_has_clear_chinese_context(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN), result_rows=_result_rows(tmp_path),
        progress_events=_progress_events(),
        source_checks={"source_revision_matches_launch_pin": True},
    )
    output = tmp_path / "curves.svg"
    report = render_k1bt_svg(gate, output)
    svg = output.read_text(encoding="utf-8")
    assert report["panels"] == 2
    assert "uKNIT 第5轮中等规模网络对比" in svg
    assert "65536/class" in svg
    assert "跨密钥验证" in svg
    assert "结构专家相对通用基线的优势" in svg


def test_k1bt_launch_gate_and_generated_assets_are_fail_closed() -> None:
    authority = {
        "gate_identity_exact": True,
        "validation_exact_pass": True,
        "all_protocol_checks_pass": True,
        "all_research_checks_pass": True,
        "digests_exact": True,
    }
    commit = "c" * 40
    gate = adjudicate_launch(
        source_commit=commit, remote_main_sha=commit, readiness_status="pass",
        k1bs_authority=authority, plans_match_selected_models_and_scale_only=True,
        source_commit_valid=True, remote_main_valid=True, source_commit_exists=True,
        source_assets_committed=True, source_assets_match=True,
        protected_worktree_clean=True,
    )
    assert gate["launch_authorized"] is True
    unpublished = adjudicate_launch(
        source_commit=commit, remote_main_sha="d" * 40, readiness_status="pass",
        k1bs_authority=authority, plans_match_selected_models_and_scale_only=True,
        source_commit_valid=True, remote_main_valid=True, source_commit_exists=True,
        source_assets_committed=True, source_assets_match=True,
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
    assert "--dataset-cache-root" in run
    assert "--dataset-cache-chunk-size 1024" in run
    assert "--dataset-cache-workers 1" in run
    assert "--expected-rows 4" in run
    assert 'if not "%PHYSICAL_GPU%"=="1"' in run + launch
    assert "git clone --no-checkout" in launch
    assert "git checkout --detach" in launch
    assert "Hostname=ssh.github.com -p 443" in run + launch
    assert 'sed \'s/\\r$//\' SHA256SUMS | sha256sum -c -' in monitor
    assert "retrieved_from_verified_result_branch.marker" in monitor
    assert "fallback_retrieved.marker" in monitor
    assert "visual_qa_pending.marker" in monitor
    assert '"${FALLBACK_ROOT}"\n' in monitor
    assert '\n+touch "${MONITOR_ROOT}/monitor.log"' not in monitor


def test_k1bt_archive_requires_four_checkpoints_and_caches(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoints = run_root / "checkpoints"
    caches = run_root / "cache"
    for root in (source_root, results_root, logs_root, checkpoints, caches):
        root.mkdir(parents=True)
    for name in RESULT_FILES:
        (results_root / name).write_text("".join("{}\n" for _ in range(4)) if name == "results.jsonl" else "{}\n", encoding="utf-8")
    (logs_root / "progress.jsonl").write_text('{"event":"run_done"}\n', encoding="utf-8")
    (logs_root / f"{RUN_ID}_gpu_info.txt").write_text("gpu\n", encoding="utf-8")
    for index in range(4):
        (checkpoints / f"row{index}.pt").write_bytes(b"checkpoint")
        split = "train" if index < 2 else "validation"
        cache = caches / "uknit64" / "r5" / split / f"cache{index}"
        cache.mkdir(parents=True)
        (cache / "features.npy").write_bytes(b"features")
        (cache / "labels.npy").write_bytes(b"labels")
        (cache / "metadata.json").write_text(json.dumps({"generation_chunk_size": 1024, "generation_workers": 1, "total_rows": 16, "input_bits": 2048}), encoding="utf-8")
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
    report = package_archive(run_root=run_root, source_root=source_root, source_commit_file=actual, expected_source_commit_file=expected, archive_root=archive)
    assert report["result_rows"] == 4
    assert report["checkpoint_count"] == 4
    assert report["cache_count"] == 4
    assert (archive / "SHA256SUMS").is_file()
    assert len(list((archive / "checkpoints").glob("*.pt"))) == 4


def _result_rows(tmp_path: Path) -> list[dict[str, object]]:
    aucs = {3: {"uknit_structure_expert": 0.81, "autond_dbitnet": 0.52}, 4: {"uknit_structure_expert": 0.84, "autond_dbitnet": 0.53}}
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for architecture, model in ARCHITECTURES.items():
            checkpoint = tmp_path / f"seed{seed}_{architecture}.pt"
            checkpoint.write_bytes(b"checkpoint")
            rows.append({
                "model": model, "rounds": 5, "seed": seed,
                "samples_per_class": 65536, "pairs_per_sample": 16,
                "input_difference": 0x0000400000000000,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNTS[architecture],
                "metrics": {"auc": aucs[seed][architecture]},
                "training": {"input_bits": 2048, "train_rows": 131072,
                    "validation_rows": 32768, "epochs": 10, "epochs_ran": 10,
                    "selected_checkpoint": "best", "checkpoint_output": str(checkpoint)},
            })
    return rows


def _progress_events() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for split in ("train", "validation"):
            rows.append({"event": "cache_start", "seed": seed, "split": split})
            rows.append({"event": "cache_reuse", "seed": seed, "split": split})
    return rows
