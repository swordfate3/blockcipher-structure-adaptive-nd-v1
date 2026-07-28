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
        description="Render the Chinese Midori64 K1-AJ same-checkpoint chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1aj_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1aj_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    report = render_midori_structure_heatmaps(
        gate,
        output,
        title=("创新1 K1-AJ：同一组 Midori64 网络权重是否真正依赖正确 S盒和扩散结构"),
        subtitle=(
            "固定 K1-AI 正确结构最佳权重，全程零训练；每颗 seed 下只替换"
            "运行时 S盒或扩散结构。"
        ),
        conclusion=_decision_text(gate),
        reading_guide=(
            "左图看同一检查点在四种运行时结构下的 AUC；右图放大正确结构的因果净优势。"
        ),
    )
    return {**report, "same_checkpoint": True, "training_performed": False}


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("same_checkpoint_semantic_use_supported"):
        return (
            "结论：同一组权重稳定依赖正确 S盒和正确扩散；K1-AI 的缺口来自独立训练捷径。"
        )
    if decision.endswith("diffusion_causal_sbox_discrimination_failed"):
        return "结论：同一组权重稳定依赖正确扩散；S盒会改变预测，但尚未在所有新样本上稳定领先。"
    if decision.endswith("sbox_causal_linear_discrimination_failed"):
        return "结论：同一组权重依赖正确 S盒，但没有稳定辨认正确扩散；先拆分结构分支。"
    if decision.endswith("structure_independent_path_dominates"):
        return "结论：同一组权重未稳定依赖两类结构；结构无关路径可能主导当前预测。"
    return "协议无效：先修复源证据、检查点、数据或重放绑定，当前指标不能解释。"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1aj_svg"]
