from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.package_dialga_r4_dmc1 import package_archive
from blockcipher_nd.tasks.innovation1.dialga_r4_dfc1 import (
    EXPECTED_CACHE_CREATIONS,
    EXPECTED_RESULT_ROWS,
    RUN_ID,
)


SOURCE_FILES = (
    "configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_dfc1_r4_1000000_seed0_seed1.csv",
    "configs/remote/innovation1_dialga_dfc1_r4_formal_1000000_seed0_seed1_gpu0_20260802.json",
    "docs/experiments/innovation1-dialga128-runtime-e4-dfc1-r4-formal-plan.md",
)
SOURCE_ARCHIVE_NAMES = ("plan.csv", "remote_config.json", "experiment_plan.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the remote Dialga DFC1 archive.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = package_archive(
        run_root=args.run_root,
        source_root=args.source_root,
        source_commit_file=args.source_commit_file,
        expected_source_commit_file=args.expected_source_commit_file,
        archive_root=args.archive_root,
        run_id=RUN_ID,
        expected_result_rows=EXPECTED_RESULT_ROWS,
        expected_cache_creations=EXPECTED_CACHE_CREATIONS,
        source_files=SOURCE_FILES,
        source_archive_names=SOURCE_ARCHIVE_NAMES,
        short_checkpoint_names=True,
        claim_scope=(
            "remote 1000000/class Dialga prefix-r4 project-formal evaluation "
            "with five fresh third-key final tests per model/seed; not full-round, "
            "key-recovery, SOTA, paper reproduction, or universal-SPN evidence"
        ),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
