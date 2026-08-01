from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

from blockcipher_nd.tasks.innovation1.runtime_spn_trainable_post_expert_adapter_k1by13 import (
    EXPECTED_SEEDS,
    STRUCTURE_MARGIN,
)


CONDITION_LABELS = {
    "anchor_correct": "原始正确结构锚点",
    "adapter_correct": "可训练适配器：正确边",
    "adapter_affine": "可训练适配器：仿射错误边",
    "adapter_shuffled": "可训练适配器：打乱边",
}
CONDITION_COLORS = {
    "anchor_correct": "#4D4D4D",
    "adapter_correct": "#0072B2",
    "adapter_affine": "#D55E00",
    "adapter_shuffled": "#009E73",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Chinese K1-BY13 adapter comparison figure."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by13_svg(gate, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def render_k1by13_svg(
    gate: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    width = 1800
    height = 1080
    left = 140
    chart_top = 210
    chart_height = 520
    chart_bottom = chart_top + chart_height
    chart_width = 1520
    y_min = 0.45
    y_max = 1.0
    conditions = tuple(CONDITION_LABELS)
    seed_results = gate.get("seed_results", {})

    def y(value: float) -> float:
        bounded = min(max(value, y_min), y_max)
        return chart_bottom - (bounded - y_min) / (y_max - y_min) * chart_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;}'
        '.title{font-size:34px;font-weight:700;}.subtitle{font-size:21px;fill:#444;}'
        '.axis{font-size:18px;fill:#333;}.label{font-size:19px;fill:#222;}'
        '.value{font-size:18px;font-weight:700;}.note{font-size:18px;fill:#333;}'
        '.decision{font-size:20px;font-weight:700;}</style>',
        '<text x="900" y="58" text-anchor="middle" class="title">'
        'PRESENT-80 七轮：零初始化可训练结构适配器</text>',
        '<text x="900" y="98" text-anchor="middle" class="subtitle">'
        '比较正确边、仿射错误边与固定打乱边；2048/class，16 个密文对</text>',
        '<text x="900" y="134" text-anchor="middle" class="subtitle">'
        '主指标为跨密钥验证 AUC；每颗种子必须独立通过结构优势门槛</text>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        tick_y = y(tick)
        elements.append(
            f'<line x1="{left}" y1="{tick_y:.1f}" x2="{left + chart_width}" '
            'y2="{:.1f}" stroke="#D9D9D9" stroke-width="1"/>'.format(tick_y)
        )
        elements.append(
            f'<text x="{left - 20}" y="{tick_y + 6:.1f}" text-anchor="end" '
            f'class="axis">{tick:.2f}</text>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{chart_top}" x2="{left}" '
            f'y2="{chart_bottom}" stroke="#222" stroke-width="2"/>',
            f'<line x1="{left}" y1="{chart_bottom}" x2="{left + chart_width}" '
            f'y2="{chart_bottom}" stroke="#222" stroke-width="2"/>',
            '<text x="38" y="470" transform="rotate(-90 38 470)" '
            'text-anchor="middle" class="label">验证 AUC</text>',
        ]
    )
    group_width = chart_width / len(EXPECTED_SEEDS)
    bar_width = 115
    gap = 35
    cluster_width = len(conditions) * bar_width + (len(conditions) - 1) * gap
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        values = seed_results.get(str(seed), {}).get("auc_by_condition", {})
        group_left = left + seed_index * group_width
        cluster_left = group_left + (group_width - cluster_width) / 2
        for index, condition in enumerate(conditions):
            value = float(values.get(condition, y_min))
            x = cluster_left + index * (bar_width + gap)
            top = y(value)
            bar_height = max(chart_bottom - top, 1.0)
            elements.extend(
                [
                    f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_width}" '
                    f'height="{bar_height:.1f}" fill="{CONDITION_COLORS[condition]}"/>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{top - 12:.1f}" '
                    f'text-anchor="middle" class="value">{value:.6f}</text>',
                ]
            )
        center = group_left + group_width / 2
        elements.append(
            f'<text x="{center:.1f}" y="{chart_bottom + 46}" '
            f'text-anchor="middle" class="label">随机种子 {seed}</text>'
        )

    legend_y = 820
    for index, condition in enumerate(conditions):
        x = 160 + index * 400
        elements.extend(
            [
                f'<rect x="{x}" y="{legend_y}" width="28" height="28" '
                f'fill="{CONDITION_COLORS[condition]}"/>',
                f'<text x="{x + 42}" y="{legend_y + 22}" class="label">'
                f'{escape(CONDITION_LABELS[condition])}</text>',
            ]
        )

    margins = []
    for seed in EXPECTED_SEEDS:
        values = seed_results.get(str(seed), {}).get("correct_minus_control", {})
        margins.append(
            f'种子{seed}：正确-锚点 {float(values.get("anchor_correct", 0.0)):+.6f}，'
            f'正确-仿射 {float(values.get("adapter_affine", 0.0)):+.6f}，'
            f'正确-打乱 {float(values.get("adapter_shuffled", 0.0)):+.6f}'
        )
    elements.extend(
        [
            f'<text x="140" y="900" class="note">结构优势门槛：正确-仿射和正确-打乱均需 '
            f'≥ +{STRUCTURE_MARGIN:.3f}</text>',
            f'<text x="140" y="938" class="note">{escape(margins[0])}</text>',
            f'<text x="140" y="976" class="note">{escape(margins[1])}</text>',
            f'<text x="140" y="1026" class="decision">裁决：'
            f'{escape(str(gate.get("decision", "尚未裁决")))}</text>',
            '</svg>',
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements), encoding="utf-8")
    return {
        "status": "pass",
        "output": str(output),
        "width": width,
        "height": height,
        "seeds": list(EXPECTED_SEEDS),
        "conditions": list(conditions),
    }


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by13_svg"]
