from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


CONDITIONS = (
    "candidate",
    "anchor",
    "repeat_last",
    "rotated",
    "corrupted",
    "no_topology",
)
CONDITION_LABELS = {
    "candidate": "CT-SPN 正确顺序",
    "anchor": "Runtime-E4 锚点",
    "repeat_last": "重复末层",
    "rotated": "旋转顺序",
    "corrupted": "错误拓扑",
    "no_topology": "无拓扑",
}
CONDITION_COLORS = {
    "candidate": "#0F766E",
    "anchor": "#2563EB",
    "repeat_last": "#D97706",
    "rotated": "#7C3AED",
    "corrupted": "#DC2626",
    "no_topology": "#64748B",
}
CIPHER_LABELS = {
    "uknit64": "uKNIT-BC 五轮",
    "dialga128": "Dialga-128 四轮",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese CT-SPN K1 AUC and attribution-margin chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = _read_json(args.gate)
    render_ctspn_k1_svg(gate, args.output)
    report = {
        "status": "rendered_pending_visual_qa",
        "run_id": gate.get("run_id"),
        "gate_status": gate.get("status"),
        "decision": gate.get("decision"),
        "output": str(args.output),
        "visual_qa_required": True,
    }
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def render_ctspn_k1_svg(gate: Mapping[str, Any], output: Path) -> None:
    seed_results = _validated_seed_results(gate)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
        figure.subplots_adjust(
            left=0.15,
            right=0.97,
            top=0.76,
            bottom=0.09,
            hspace=0.5,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 K1：规范化线性层顺序是否真正帮助 uKNIT 类 SPN 区分",
            x=0.07,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.905,
            "相同数据与训练预算；候选和锚点分别训练，结构控制复用对应最佳权重且不重新训练。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.07,
            0.855,
            f"裁决：{_decision_text(gate)}",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.07,
            0.81,
            "AUC 图越靠右越好；边际图必须同时超过 0.005 虚线，任一密码或 seed 失败均不通过。",
            ha="left",
            fontsize=9.5,
            color="#6B7280",
        )

        for column, cipher in enumerate(("uknit64", "dialga128")):
            _plot_auc_panel(axes[0, column], cipher, seed_results[cipher])
            _plot_margin_panel(axes[1, column], cipher, seed_results[cipher])

        handles = [
            plt.Line2D(
                [0],
                [0],
                color="#111827",
                marker=marker,
                linestyle="none",
                markersize=7,
                label=label,
            )
            for marker, label in (("o", "seed0"), ("s", "seed1"))
        ]
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.965),
            frameon=False,
            ncol=2,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def _plot_auc_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, float]],
) -> None:
    values = _auc_values(seeds)
    y_positions = list(reversed(range(len(CONDITIONS))))
    for index, condition in enumerate(CONDITIONS):
        y = y_positions[index]
        color = CONDITION_COLORS[condition]
        seed0 = values["0"][condition]
        seed1 = values["1"][condition]
        axis.plot(
            (seed0, seed1),
            (y - 0.12, y + 0.12),
            color=color,
            linewidth=1.6,
            alpha=0.55,
        )
        for value, offset, marker in ((seed0, -0.12, "o"), (seed1, 0.12, "s")):
            axis.scatter(
                value,
                y + offset,
                color=color,
                marker=marker,
                s=48,
                zorder=3,
            )
            axis.annotate(
                f"{value:.4f}",
                (value, y + offset),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.3,
                color="#374151",
            )
    flat = [value for seed in values.values() for value in seed.values()]
    lower = max(0.0, min(0.5, min(flat)) - 0.012)
    upper = min(1.0, max(flat) + max(0.035, (max(flat) - lower) * 0.2))
    axis.set_xlim(lower, upper)
    axis.set_yticks(y_positions, [CONDITION_LABELS[key] for key in CONDITIONS])
    axis.set_title(f"{CIPHER_LABELS[cipher]}：冻结验证 AUC", loc="left", fontweight="bold")
    axis.set_xlabel("AUC")
    axis.axvline(0.5, color="#9CA3AF", linestyle=(0, (3, 3)), linewidth=1)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)


def _plot_margin_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, float]],
) -> None:
    margin_keys = (
        "anchor",
        "repeat_last",
        "rotated",
        "corrupted",
        "no_topology",
    )
    y_positions = list(reversed(range(len(margin_keys))))
    margins = {
        seed: {
            key: float(values[f"candidate_minus_{key}"])
            for key in margin_keys
        }
        for seed, values in seeds.items()
    }
    for index, key in enumerate(margin_keys):
        y = y_positions[index]
        color = CONDITION_COLORS[key]
        seed0 = margins["0"][key]
        seed1 = margins["1"][key]
        axis.plot(
            (seed0, seed1),
            (y - 0.12, y + 0.12),
            color=color,
            linewidth=1.6,
            alpha=0.55,
        )
        for value, offset, marker in ((seed0, -0.12, "o"), (seed1, 0.12, "s")):
            axis.scatter(
                value,
                y + offset,
                color=color,
                marker=marker,
                s=48,
                zorder=3,
            )
            axis.annotate(
                f"{value:+.4f}",
                (value, y + offset),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.3,
                color="#374151",
            )
    flat = [value for seed in margins.values() for value in seed.values()]
    span = max(0.02, max(flat) - min(flat))
    axis.set_xlim(min(-0.01, min(flat) - 0.15 * span), max(0.02, max(flat) + 0.3 * span))
    axis.set_yticks(
        y_positions,
        [f"候选 - {CONDITION_LABELS[key]}" for key in margin_keys],
    )
    axis.set_title(f"{CIPHER_LABELS[cipher]}：正确顺序的净优势", loc="left", fontweight="bold")
    axis.set_xlabel("AUC 差值")
    axis.axvline(0.0, color="#9CA3AF", linewidth=1)
    axis.axvline(0.005, color="#047857", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)


def _auc_values(
    seeds: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        seed: {
            "candidate": float(values["candidate_auc"]),
            "anchor": float(values["anchor_auc"]),
            "repeat_last": float(values["candidate_repeat_last_auc"]),
            "rotated": float(values["candidate_rotated_auc"]),
            "corrupted": float(values["candidate_corrupted_auc"]),
            "no_topology": float(values["candidate_no_topology_auc"]),
        }
        for seed, values in seeds.items()
    }


def _validated_seed_results(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    raw = gate.get("seed_results")
    if not isinstance(raw, dict) or set(raw) != {"uknit64", "dialga128"}:
        raise ValueError("K1 gate must contain uKNIT and Dialga seed results")
    required = {
        "candidate_auc",
        "anchor_auc",
        "candidate_repeat_last_auc",
        "candidate_rotated_auc",
        "candidate_corrupted_auc",
        "candidate_no_topology_auc",
        "candidate_minus_anchor",
        "candidate_minus_repeat_last",
        "candidate_minus_rotated",
        "candidate_minus_corrupted",
        "candidate_minus_no_topology",
    }
    result: dict[str, dict[str, dict[str, float]]] = {}
    for cipher, seeds in raw.items():
        if not isinstance(seeds, dict) or set(seeds) != {"0", "1"}:
            raise ValueError(f"K1 {cipher} must contain seed0 and seed1")
        result[cipher] = {}
        for seed, values in seeds.items():
            if not isinstance(values, dict) or not required.issubset(values):
                raise ValueError(f"K1 {cipher} seed{seed} metrics are incomplete")
            if any(not isinstance(values[key], (int, float)) for key in required):
                raise ValueError(f"K1 {cipher} seed{seed} metrics must be numeric")
            result[cipher][seed] = {key: float(values[key]) for key in required}
    return result


def _decision_text(gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status", "unknown"))
    if status == "pass":
        return "通过，正确规范顺序在两种密码和两颗 seed 上均满足绝对 AUC 与归因边际"
    if status == "hold":
        return "暂缓，至少一个密码、seed 或结构控制未达到预注册门槛"
    return "协议无效，必须修复证据绑定后按原计划重跑"


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "fail": "#B91C1C"}.get(
        status, "#374151"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
