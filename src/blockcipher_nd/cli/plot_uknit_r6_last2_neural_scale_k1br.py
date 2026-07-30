from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


LABELS = {
    "exact_position_histogram_residual": "正确 S 盒 + 原生位置",
    "wrong_sbox_position_histogram_residual": "错误 S 盒控制",
    "invariant_histogram_residual": "位置抹除控制",
}
COLORS = {
    "exact_position_histogram_residual": "#007C83",
    "wrong_sbox_position_histogram_residual": "#C4473A",
    "invariant_histogram_residual": "#6A5A9C",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the uKNIT r6 K1-BR result.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def render_k1br_svg(gate: dict[str, Any], output: Path) -> dict[str, Any]:
    aucs = gate.get("aucs", {})
    minimum = min(0.49, *(float(value) for value in aucs.values()))
    maximum = max(0.56, *(float(value) for value in aucs.values()))
    pad = max(0.005, (maximum - minimum) * 0.12)
    lo, hi = minimum - pad, maximum + pad
    x0, x1 = 280.0, 1020.0

    def scale(value: float) -> float:
        return x0 + (float(value) - lo) / (hi - lo) * (x1 - x0)

    rows = []
    for index, condition in enumerate(LABELS):
        value = float(aucs.get(condition, 0.5))
        y = 230 + index * 105
        rows.append(
            f'<text x="40" y="{y + 7}" class="label">{html.escape(LABELS[condition])}</text>'
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="track"/>'
            f'<circle cx="{scale(value):.2f}" cy="{y}" r="11" fill="{COLORS[condition]}"/>'
            f'<text x="{scale(value):.2f}" y="{y - 22}" text-anchor="middle" class="value">AUC {value:.6f}</text>'
        )
    chance = scale(0.5)
    title = "uKNIT 6轮：扩大数据后能否学到可归因的区分信号"
    subtitle = "262144/类训练，seed3，4对密文，10轮训练；单种子大规模诊断，不是正式结论"
    tier = html.escape(str(gate.get("tier", "unknown")))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="610" viewBox="0 0 1120 610">
<style>
text {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: #202124; letter-spacing: 0; }}
.title {{ font-size: 28px; font-weight: 700; }} .subtitle {{ font-size: 17px; fill: #53565A; }}
.label {{ font-size: 18px; font-weight: 600; }} .value {{ font-size: 15px; font-weight: 700; }}
.track {{ stroke: #D7D9DC; stroke-width: 4; }} .axis {{ font-size: 14px; fill: #5F6368; }}
</style>
<rect width="1120" height="610" fill="#FFFFFF"/>
<text x="40" y="55" class="title">{title}</text>
<text x="40" y="91" class="subtitle">{subtitle}</text>
<line x1="{chance:.2f}" y1="170" x2="{chance:.2f}" y2="470" stroke="#35383B" stroke-width="2" stroke-dasharray="7 7"/>
<text x="{chance:.2f}" y="155" text-anchor="middle" class="axis">随机水平 0.500</text>
{"".join(rows)}
<text x="40" y="545" class="subtitle">裁决层级：{tier}；正确归因还要求领先错误 S 盒控制。</text>
<text x="{x0}" y="505" text-anchor="middle" class="axis">{lo:.3f}</text>
<text x="{x1}" y="505" text-anchor="middle" class="axis">{hi:.3f}</text>
</svg>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return {
        "status": "rendered_visual_qa_pending",
        "width": 1120,
        "height": 610,
        "series": len(rows),
        "axis_min": lo,
        "axis_max": hi,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1br_svg(gate, args.output)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
