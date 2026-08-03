from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from blockcipher_nd.cli.check_uknit_r5_k1cb_cache import main as cache_check_main
from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.package_uknit_r5_k1cb_paper_comparison import (
    SOURCE_FILES,
    main as package_main,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_r5_published_comparison_k1cb import (
    ARCHITECTURES,
    EXPECTED_CACHE_REUSES,
    EXPECTED_PARAMETER_COUNTS,
    K1CA_RUN_ID,
    cache_progress_checks,
    plan_protocol_frozen,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_published_comparison_k1cb_launch import (
    adjudicate_launch,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_r5_k1cb_published_comparison_262144_seed3_seed4.csv"
)
REMOTE_CONFIG = ROOT / (
    "configs/remote/"
    "innovation1_uknit_k1cb_published_comparison_262144_seed3_seed4_gpu0_20260803.json"
)
RUN_SCRIPT = ROOT / (
    "configs/remote/generated/"
    "run_i1_uknit_r5_k1cb_published_comparison_262144_s3s4_20260803.cmd"
)
LAUNCH_SCRIPT = ROOT / (
    "configs/remote/generated/"
    "launch_i1_uknit_r5_k1cb_published_comparison_262144_s3s4_20260803.cmd"
)
MONITOR_SCRIPT = ROOT / (
    "configs/remote/generated/"
    "monitor_i1_uknit_r5_k1cb_published_comparison_262144_s3s4_20260803.sh"
)


def test_k1cb_plan_is_same_scale_six_row_paper_comparison() -> None:
    tasks = read_tasks(PLAN)
    assert plan_protocol_frozen(tasks)
    assert len(tasks) == 6
    assert {(task["seed"], task["model_key"]) for task in tasks} == {
        (seed, model) for seed in (3, 4) for model in ARCHITECTURES.values()
    }
    assert all(task["samples_per_class"] == 262_144 for task in tasks)
    assert all(task["validation_samples_total"] == 131_072 for task in tasks)
    assert all(task["pairs_per_sample"] == 4 for task in tasks)
    assert all(task["target_epochs"] == 10 for task in tasks)
    assert all(task["final_test_repeats"] == 0 for task in tasks)


def test_k1cb_published_models_accept_exact_four_pair_input() -> None:
    tasks = read_tasks(PLAN)
    first_by_model = {task["model_key"]: task for task in tasks if task["seed"] == 3}
    for architecture, model_key in ARCHITECTURES.items():
        task = first_by_model[model_key]
        model = build_model(
            model_key,
            input_bits=512,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options=task["model_options"],
        )
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == (EXPECTED_PARAMETER_COUNTS[architecture])
        )
        assert model(torch.zeros(2, 512)).shape == (2, 1)


def test_k1cb_published_models_complete_a_tiny_training_step() -> None:
    tasks = read_tasks(PLAN)
    first_by_model = {task["model_key"]: task for task in tasks if task["seed"] == 3}
    features = torch.zeros((2, 512), dtype=torch.float32)
    labels = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    for model_key in ARCHITECTURES.values():
        task = first_by_model[model_key]
        model = build_model(
            model_key,
            input_bits=512,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options=task["model_options"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(features), labels)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()


def test_k1cb_cache_gate_requires_twelve_reuses_and_zero_generation() -> None:
    cache_root = (
        f"G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\{K1CA_RUN_ID}\\cache"
    )
    events = [
        {
            "event": "cache_reuse",
            "seed": seed,
            "model": model,
            "split": split,
            "cache_path": f"{cache_root}\\uknit64\\r5\\{split}\\seed-{seed}",
            "chunk_size": 1024,
            "workers": 1,
        }
        for seed in (3, 4)
        for model in ARCHITECTURES.values()
        for split in ("train", "validation")
    ]
    events.append({"event": "run_done"})
    checks = cache_progress_checks(events)
    assert len(events) - 1 == EXPECTED_CACHE_REUSES
    assert all(checks.values())

    generated = [*events, {"event": "cache_start", "split": "train"}]
    assert not cache_progress_checks(generated)["zero_cache_generation_events"]


def test_k1cb_cache_preflight_fails_without_creating_missing_source(
    tmp_path: Path,
) -> None:
    source_cache_root = tmp_path / "missing-k1ca-cache"
    output = tmp_path / "cache-audit.json"
    status = cache_check_main(
        [
            "--plan",
            str(PLAN),
            "--source-cache-root",
            str(source_cache_root),
            "--output",
            str(output),
        ]
    )
    assert status == 4
    assert output.is_file()
    assert not source_cache_root.exists()


def test_k1cb_remote_assets_enforce_paper_comparison_and_reuse_only() -> None:
    readiness = remote_readiness_report(REMOTE_CONFIG)
    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 6

    run = RUN_SCRIPT.read_text(encoding="utf-8")
    launch = LAUNCH_SCRIPT.read_text(encoding="utf-8")
    monitor = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "!" not in run + launch
    assert "cmd.exe /k" not in run + launch
    assert "cmd.exe /c" in launch
    assert 'if not "%PHYSICAL_GPU%"=="0"' in run + launch
    assert "scripts\\check-uknit-r5-k1cb-cache" in run
    assert run.index("scripts\\check-uknit-r5-k1cb-cache") < run.index("scripts\\train")
    assert '--dataset-cache-root "%SOURCE_CACHE_ROOT%"' in run
    assert "--expected-rows 6" in run
    assert "--final-test" not in run
    assert "scripts\\gate-uknit-r5-k1cb-paper-comparison" in run
    assert "scripts\\package-uknit-r5-k1cb-paper-comparison" in run
    assert "schtasks /Change" in launch and "/DISABLE" in launch
    assert "git clone --no-checkout" in launch
    assert "git checkout --detach" in launch
    assert "waiting_for_source_k1ca" in monitor
    assert "source_gate_valid" in monitor
    assert "scripts/check-uknit-r5-k1cb-paper-comparison-launch" in monitor
    assert "innovation1_uknit_k1cb_remote_launch_authorized" in monitor
    assert "g.get('should_ssh') is True" in monitor
    assert "g.get('ssh_allowed') is True" in monitor
    assert "g.get('launch_authorized') is True" in monitor
    assert monitor.index("prepare_launch_gate") < monitor.index('scp "${LAUNCHER}"')
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "--source-cache-audit" in monitor
    assert "scripts/index-results" in monitor
    assert "visual_qa_pending.marker" in monitor


def test_k1cb_launch_gate_is_fail_closed_and_publication_bound() -> None:
    commit = "c" * 40
    gate = adjudicate_launch(
        source_commit=commit,
        remote_main_sha=commit,
        readiness_status="pass",
        authority={"k1ca_complete": True},
        paper_contract_frozen=True,
        six_row_plan_frozen=True,
        source_commit_valid=True,
        remote_main_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        protected_worktree_clean=True,
    )
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_uknit_k1cb_remote_launch_authorized"
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is True
    assert gate["launch_authorized"] is True

    unpublished = adjudicate_launch(
        source_commit=commit,
        remote_main_sha="d" * 40,
        readiness_status="pass",
        authority={"k1ca_complete": True},
        paper_contract_frozen=True,
        six_row_plan_frozen=True,
        source_commit_valid=True,
        remote_main_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        protected_worktree_clean=True,
    )
    assert unpublished["should_ssh"] is True
    assert unpublished["ssh_allowed"] is False
    assert unpublished["launch_authorized"] is False

    missing_source = adjudicate_launch(
        source_commit=commit,
        remote_main_sha=commit,
        readiness_status="pass",
        authority={"k1ca_complete": False},
        paper_contract_frozen=True,
        six_row_plan_frozen=True,
        source_commit_valid=True,
        remote_main_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        protected_worktree_clean=True,
    )
    assert missing_source["should_ssh"] is False
    assert missing_source["launch_authorized"] is False


def test_k1cb_package_embeds_k1ca_source_evidence(tmp_path: Path) -> None:
    run_root = tmp_path / "k1cb"
    source_root = run_root / "source"
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoints_root = run_root / "checkpoints"
    source_k1ca_root = tmp_path / "k1ca"
    source_k1ca_results = source_k1ca_root / "results"
    source_k1ca_archive = source_k1ca_root / "source" / "results_archive" / K1CA_RUN_ID
    for path in (
        source_root,
        results_root,
        logs_root,
        checkpoints_root,
        source_k1ca_results,
        source_k1ca_archive,
    ):
        path.mkdir(parents=True)

    for name in (
        "results.jsonl",
        "validation-plan.json",
        "validation.json",
        "gate.json",
        "summary.json",
        "history.csv",
    ):
        content = (
            "".join("{}\n" for _ in range(6)) if name == "results.jsonl" else "{}\n"
        )
        (results_root / name).write_text(content, encoding="utf-8")
    (results_root / "source_cache_audit.json").write_text(
        json.dumps({"status": "pass", "checks": {"all": True}}) + "\n",
        encoding="utf-8",
    )
    (logs_root / "progress.jsonl").write_text(
        '{"event":"run_done"}\n', encoding="utf-8"
    )
    for index in range(6):
        (checkpoints_root / f"row{index}.pt").write_bytes(b"checkpoint")
    for source in SOURCE_FILES:
        path = source_root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")

    (source_k1ca_results / "gate.json").write_text("{}\n", encoding="utf-8")
    (source_k1ca_results / "results.jsonl").write_text("{}\n", encoding="utf-8")
    (source_k1ca_archive / "cache_manifest.json").write_text("{}\n", encoding="utf-8")
    (source_k1ca_archive / "run_manifest.json").write_text("{}\n", encoding="utf-8")

    sha = "a" * 40
    actual = logs_root / "git_revision.txt"
    expected = run_root / "source_expected_commit.txt"
    actual.write_text(sha + "\n", encoding="utf-8")
    expected.write_text(sha + "\n", encoding="utf-8")
    archive = source_root / "results_archive" / "k1cb"
    assert (
        package_main(
            [
                "--run-root",
                str(run_root),
                "--source-root",
                str(source_root),
                "--source-k1ca-root",
                str(source_k1ca_root),
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
    assert len(list((archive / "checkpoints").glob("*.pt"))) == 6
    assert (archive / "source_k1ca" / "gate.json").is_file()
    assert (archive / "source_k1ca" / "results.jsonl").is_file()
    assert (archive / "source_k1ca" / "cache_manifest.json").is_file()
    assert (archive / "SHA256SUMS").is_file()


def test_k1cb_remote_postprocessing_does_not_import_plotting_modules() -> None:
    for script in (
        "scripts/check-uknit-r5-k1cb-cache",
        "scripts/check-uknit-r5-k1cb-paper-comparison-launch",
        "scripts/gate-uknit-r5-k1cb-paper-comparison",
        "scripts/package-uknit-r5-k1cb-paper-comparison",
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
