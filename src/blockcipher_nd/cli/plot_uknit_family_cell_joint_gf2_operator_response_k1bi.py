from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from blockcipher_nd.cli.plot_uknit_family_exact_gf2_operator_response_k1bh import (
    render_k1bh_svg,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BI cell-joint GF(2) response chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bi_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bi_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    report = render_k1bh_svg(
        gate,
        output,
        figure_title=(
            "创新1 K1-BI：保留 cell 内四比特联合取值后，正确扩散算子是否独有"
        ),
        subtitle=(
            "同一批4-pair密文、四个精确GF(2)视图；按运行时cell重建0–15取值直方图，"
            "错误算子仍不得重新拟合。"
        ),
        decision_summary=_decision_text(gate),
        footer=(
            "唯一变化是独立bit均值改为运行时cell的16类联合响应；这是本地零神经参数机制审计，"
            "不是正式训练或SOTA结果。"
        ),
        control_title="双向控制：标签打乱 AUC 是否仍显著偏离随机",
    )
    return {
        **report,
        "experiment": "K1-BI",
        "representation": "runtime_cell_joint_16_value_histogram",
    }


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("cell_joint_topology_signal_supported"):
        return "裁决：cell联合响应在三种密码上都认出正确拓扑；下一步才设计共享的位置保持神经残差。"
    if decision.endswith("shuffle_attribution_not_supported"):
        return "裁决：双向标签打乱控制未通过；固定当前特征，先建立多次打乱的方向不变零分布。"
    if decision.endswith("cell_joint_signal_unstable"):
        return "裁决：cell联合响应仍不能稳定保留uKNIT信号；停止纯线性路线，转入S盒感知五阶段cell原语。"
    if decision.endswith("anchor_regression"):
        return "裁决：cell联合响应损失Midori/Dialga锚点；先审计cell重建和Fisher方差处理。"
    if decision.endswith("not_topology_identifying"):
        return "裁决：cell联合响应可预测但不能稳定识别正确拓扑；先审计错误算子等价性。"
    return "裁决：协议无效；只修复来源绑定、cell重建、评分器复用或产物完整性后原样重跑。"


__all__ = ["main", "render_k1bi_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
