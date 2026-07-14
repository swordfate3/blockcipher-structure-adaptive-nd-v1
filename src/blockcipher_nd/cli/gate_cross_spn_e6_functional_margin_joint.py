from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.gate_cross_spn_e5_source_objective_joint import (
    render_joint_gate_svg,
    write_joint_summary_csv,
)
from blockcipher_nd.planning.cross_spn_e6_functional_margin_gate import (
    gate_cross_spn_e6_functional_margin_joint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly gate E6-R0 target seeds.")
    parser.add_argument("--seed2-gate", required=True, type=Path)
    parser.add_argument("--seed3-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--plot-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.seed2_gate, args.seed3_gate)
    ]
    report = gate_cross_spn_e6_functional_margin_joint(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report["status"] == "pass":
        if args.summary_csv is not None:
            write_joint_summary_csv(report, args.summary_csv)
        if args.plot_output is not None:
            render_joint_gate_svg(
                report,
                args.plot_output,
                title="创新1 E6-R0：功能性拓扑边际源目标的迁移门控",
                subtitle=(
                    "PRESENT-80 r7 源 seed 0 → GIFT-64 r6｜训练 8,192/类｜"
                    "目标只训练 1 轮｜误差线为配对 bootstrap 95% CI"
                ),
                verdict=(
                    "裁决以 gate.json 为准：仅当两颗 seed 均超过 off、placebo "
                    "和 scratch，才允许 source seed 1；否则停止远程扩展。"
                ),
            )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]
