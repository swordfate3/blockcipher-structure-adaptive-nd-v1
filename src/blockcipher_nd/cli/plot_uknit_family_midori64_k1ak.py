from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from blockcipher_nd.cli.plot_uknit_family_midori64_k1ai import (
    render_midori_structure_heatmaps,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese Midori64 K1-AK transition chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1ak_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ak_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    report = render_midori_structure_heatmaps(
        gate,
        output,
        title=("创新1 K1-AK：按 cell 统计 S盒输入/输出转移，能否稳定辨认正确结构"),
        subtitle=(
            "Midori64 第4轮、cell8 差分、每样本4对密文、2048/class、"
            "seed6/7；仅替换 K1-AA 的紧凑直方图读出。"
        ),
        conclusion=_decision_text(gate),
        reading_guide=(
            "左图比较四种运行时结构的 fresh AUC；右图放大正确结构相对"
            "错误 S盒、错误扩散和无结构的净优势。"
        ),
    )
    return {**report, "cell_shared_sbox_transition": True}


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("sbox_transition_residual_supported"):
        return "结论：新读出在两颗 seed、两种 fresh 密钥范围内同时保留信号并稳定辨认正确 S盒。"
    if decision.endswith("sbox_transition_discrimination_failed"):
        return "结论：信号与扩散归因保留，但正确 S盒仍未在所有新样本组合上稳定领先。"
    if decision.endswith("sbox_supported_anchor_retention_failed"):
        return "结论：正确 S盒归因成立，但新读出损失了 K1-AI 的同预算信号，需要检查融合优化。"
    if decision.endswith("signal_or_diffusion_retention_failed"):
        return "结论：新读出破坏了原有信号或扩散归因，不保留该结构。"
    return "协议无效：先修复源证据、数据、训练或控制绑定，当前指标不能解释。"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ak_svg"]
