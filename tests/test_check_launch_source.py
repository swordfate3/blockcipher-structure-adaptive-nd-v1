from __future__ import annotations

from blockcipher_nd.cli.check_launch_source import launch_source_report_from_status


def test_launch_source_gate_rejects_unpushed_commits():
    report = launch_source_report_from_status("## main...origin/main [ahead 38]\n")

    assert report["status"] == "fail"
    assert report["branch"] == "main"
    assert report["upstream"] == "origin/main"
    assert report["ahead"] == 38
    assert report["behind"] == 0
    assert report["should_push"] is True
    assert "unpushed_commits" in report["errors"]


def test_launch_source_gate_passes_clean_published_branch():
    report = launch_source_report_from_status("## main...origin/main\n")

    assert report["status"] == "pass"
    assert report["ahead"] == 0
    assert report["behind"] == 0
    assert report["dirty"] is False
    assert report["should_push"] is False
    assert report["errors"] == []


def test_launch_source_gate_accepts_dotted_branch_names():
    report = launch_source_report_from_status("## release/v1.2...origin/release/v1.2\n")

    assert report["status"] == "pass"
    assert report["branch"] == "release/v1.2"
    assert report["upstream"] == "origin/release/v1.2"
    assert report["errors"] == []


def test_launch_source_gate_rejects_dirty_or_untracked_worktree():
    report = launch_source_report_from_status(
        "\n".join(
            [
                "## main...origin/main",
                " M src/blockcipher_nd/cli/check_launch_source.py",
                "?? tmp.txt",
            ]
        )
        + "\n"
    )

    assert report["status"] == "fail"
    assert report["dirty"] is True
    assert "dirty_worktree" in report["errors"]


def test_launch_source_gate_rejects_missing_upstream():
    report = launch_source_report_from_status("## experiment\n")

    assert report["status"] == "fail"
    assert report["branch"] == "experiment"
    assert report["upstream"] is None
    assert "missing_upstream" in report["errors"]
