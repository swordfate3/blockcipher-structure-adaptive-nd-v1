from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.package_uknit_family_ctspn_k1u import (
    RESULT_FILES,
    SOURCE_FILES,
    package_k1u_archive,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1u import RUN_ID
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1u_launch import (
    K1T_PLAN,
    K1U_PLAN,
    _plans_match_scale_only,
    adjudicate_k1u_launch,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "configs/remote/generated"
RUN_SCRIPT = (
    GENERATED
    / "run_i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728.cmd"
)
LAUNCH_SCRIPT = (
    GENERATED
    / "launch_i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728.cmd"
)
MONITOR_SCRIPT = (
    GENERATED
    / "monitor_i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728.sh"
)
REMOTE_CONFIG = (
    ROOT
    / "configs/remote/innovation1_uknit_k1u_position_residual_medium_65536_seed3_seed4_gpu1_20260728.json"
)


def _authority() -> dict[str, object]:
    return {
        "gate_identity_exact": True,
        "validation_exact_pass": True,
        "all_protocol_checks_pass": True,
        "all_research_checks_pass": True,
        "visual_qa_passed": True,
        "digests": {},
        "digests_exact": True,
    }


def _gate(*, published: bool = True, authority: dict[str, object] | None = None):
    source_commit = "c" * 40
    remote_main_sha = source_commit if published else "d" * 40
    return adjudicate_k1u_launch(
        source_commit=source_commit,
        remote_main_sha=remote_main_sha,
        readiness_status="pass",
        k1t_authority=authority or _authority(),
        plans_match_scale_only=True,
        source_commit_valid=True,
        remote_main_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        protected_paths_unchanged=True,
        protected_worktree_clean=True,
    )


def test_k1u_launch_gate_authorizes_only_exact_published_source() -> None:
    gate = _gate()

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("k1u_remote_launch_authorized")
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is True
    assert gate["launch_authorized"] is True

    unpublished = _gate(published=False)
    assert unpublished["status"] == "hold"
    assert unpublished["should_ssh"] is True
    assert unpublished["ssh_allowed"] is False
    assert unpublished["launch_authorized"] is False


def test_k1u_launch_gate_fails_closed_on_authority_drift() -> None:
    authority = _authority()
    authority["digests_exact"] = False

    gate = _gate(authority=authority)

    assert gate["status"] == "fail"
    assert gate["should_ssh"] is False
    assert gate["launch_authorized"] is False


def test_real_k1u_plan_is_k1t_with_only_medium_scale_identity_change() -> None:
    assert _plans_match_scale_only(ROOT / K1T_PLAN, ROOT / K1U_PLAN)


def test_k1u_generated_assets_are_fail_closed_and_remote_owned() -> None:
    run = RUN_SCRIPT.read_text(encoding="utf-8")
    launch = LAUNCH_SCRIPT.read_text(encoding="utf-8")
    monitor = MONITOR_SCRIPT.read_text(encoding="utf-8")
    config = json.loads(REMOTE_CONFIG.read_text(encoding="utf-8"))

    assert "!" not in run
    assert "!" not in launch
    assert "set -o pipefail" in monitor
    assert "cmd.exe /k" not in run + launch
    assert "cmd.exe /c" in launch
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run + launch
    assert "--epochs 10" in run
    assert "--hidden-bits 32" in run
    assert "--dataset-cache-chunk-size 1024" in run
    assert "--dataset-cache-workers 1" in run
    assert "--expected-rows 6" in run
    assert "package-uknit-family-ctspn-k1u" in run
    assert "raw_ready.marker" in run
    assert 'if not "%PHYSICAL_GPU%"=="1"' in run + launch
    assert "git clone --no-checkout" in launch
    assert "git status --porcelain" in launch
    assert "git checkout --detach" in launch
    assert "source_expected_commit.txt" in launch
    assert "retrieved_from_verified_result_branch.marker" in monitor
    assert "RAW_RETRIEVAL_NOTICE.txt" in monitor
    assert 'git fetch --force origin' in monitor
    assert 'git archive "${result_ref}" "results_archive/${RUN_ID}"' in monitor
    assert 'if [[ "${mode}" == "verified" ]]' in monitor
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "raw_evidence_supplement" in monitor
    assert "RAW_EVIDENCE_SUPPLEMENT_NOTICE.txt" in monitor
    assert "/cache/uknit64/r5/validation" in monitor
    assert "plot-uknit-family-ctspn-k1u" in monitor
    assert config["physical_gpu"] == 1
    assert config["result_sync"] == "local_tmux_monitor_scp_fallback"


def test_k1u_archive_packager_requires_six_checkpoints_and_four_caches(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoint_root = run_root / "checkpoints"
    cache_root = run_root / "cache"
    for root in (source_root, results_root, logs_root, checkpoint_root, cache_root):
        root.mkdir(parents=True)

    for name in RESULT_FILES:
        path = results_root / name
        if name == "results.jsonl":
            path.write_text("".join("{}\n" for _ in range(6)), encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
    (logs_root / "progress.jsonl").write_text(
        '{"event":"run_done"}\n', encoding="utf-8"
    )
    (logs_root / f"{RUN_ID}_gpu_info.txt").write_text("gpu\n", encoding="utf-8")
    for index in range(6):
        (checkpoint_root / f"row{index:04d}.pt").write_bytes(b"checkpoint")
    for index in range(4):
        split = "train" if index < 2 else "validation"
        payload = cache_root / "uknit64" / "r5" / split / f"cache{index}"
        payload.mkdir(parents=True)
        (payload / "features.npy").write_bytes(b"features")
        (payload / "labels.npy").write_bytes(b"labels")
        (payload / "metadata.json").write_text(
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
    for source_file in SOURCE_FILES:
        path = source_root / source_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    source_sha = "a" * 40
    source_commit_file = logs_root / f"{RUN_ID}_git_revision.txt"
    expected_commit_file = run_root / "source_expected_commit.txt"
    source_commit_file.write_text(source_sha + "\n", encoding="utf-8")
    expected_commit_file.write_text(source_sha + "\n", encoding="utf-8")
    archive_root = source_root / "results_archive" / RUN_ID

    report = package_k1u_archive(
        run_root=run_root,
        source_root=source_root,
        source_commit_file=source_commit_file,
        expected_source_commit_file=expected_commit_file,
        archive_root=archive_root,
    )

    assert report["result_rows"] == 6
    assert report["checkpoint_count"] == 6
    assert report["cache_count"] == 4
    assert report["archived_checkpoint_count"] == 6
    assert report["archived_validation_cache_count"] == 2
    assert (archive_root / "SHA256SUMS").is_file()
    assert len(list((archive_root / "cache_metadata").glob("*.json"))) == 4
    assert len(list((archive_root / "checkpoints").glob("*.pt"))) == 6
    assert len(list((archive_root / "validation_cache").rglob("features.npy"))) == 2
    assert len(list((archive_root / "validation_cache").rglob("labels.npy"))) == 2
    assert not list((archive_root / "validation_cache").glob("**/train/**"))
