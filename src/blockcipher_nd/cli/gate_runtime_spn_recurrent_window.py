from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window import (
    adjudicate_runtime_spn_recurrent_window,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adjudicate the frozen two-seed uKNIT Runtime-E4 recurrent-window gate."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = args.run_root / "results.jsonl"
    rows = _read_jsonl(results_path)
    gate = adjudicate_runtime_spn_recurrent_window(
        run_id=args.run_id,
        rows=rows,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": gate["cipher"],
        "training_performed": True,
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 10,
        "seeds": [0, 1],
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    render_recurrent_window_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        {
            "event": "recurrent_window_gate_done",
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_recurrent_window_svg(
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    roles = ("anchor", "candidate", "repeat_last", "corrupted", "no_topology")
    role_labels = (
        "最后一轮锚点",
        "真实异构双窗口",
        "重复末轮双窗口",
        "错误线性拓扑",
        "无线性拓扑",
    )
    margin_keys = (
        "candidate_minus_anchor",
        "candidate_minus_repeat_last",
        "candidate_minus_corrupted",
        "candidate_minus_no_topology",
    )
    margin_labels = (
        "相对最后一轮锚点",
        "相对重复末轮",
        "相对错误拓扑",
        "相对无拓扑",
    )
    seeds = (0, 1)
    colors = ("#2563EB", "#D97706")
    seed_results = gate["seed_results"]
    aucs = {
        seed: [float(seed_results[str(seed)][f"{role}_auc"]) for role in roles]
        for seed in seeds
    }
    margins = {
        seed: [float(seed_results[str(seed)][key]) for key in margin_keys]
        for seed in seeds
    }
    if gate["status"] == "fail":
        conclusion = "协议检查未通过；不得解释模型差异，先修复协议。"
    elif gate["status"] == "pass":
        conclusion = "两颗种子均通过绝对信号与四项归因门，可进入预注册的同检查点窗口交换审计。"
    else:
        conclusion = (
            "双种子门未通过：seed1成立，seed0近随机且低于错误/无拓扑；"
            "停止扩样，转向结构原语分工的本地架构设计。"
        )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.6))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.67,
            bottom=0.22,
            wspace=0.28,
        )
        figure.suptitle(
            "创新1 U3：uKNIT 五轮异构双窗口运行时 SPN 裁决",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=17.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            "训练 2,048/类，验证 1,024/类，4 对密文/样本，10 epochs；虚线为预注册门槛。",
            ha="left",
            va="top",
            color="#475569",
        )
        figure.text(
            0.075,
            0.83,
            f"结论：{conclusion}",
            ha="left",
            va="top",
            fontsize=11.0,
            fontweight="bold",
            color="#9F1239" if gate["status"] != "pass" else "#166534",
        )

        role_axis = axes[0]
        x = np.arange(len(roles), dtype=float)
        bar_width = 0.34
        for offset, seed, color in zip((-0.18, 0.18), seeds, colors):
            bars = role_axis.bar(
                x + offset,
                aucs[seed],
                width=bar_width,
                color=color,
                label=f"seed{seed}",
                zorder=3,
            )
            role_axis.bar_label(
                bars,
                labels=[f"{value:.4f}" for value in aucs[seed]],
                padding=3,
                fontsize=9.0,
                rotation=90,
            )
        role_axis.axhline(0.5, color="#64748B", linewidth=1.2, linestyle=":")
        role_axis.axhline(
            float(gate["thresholds"]["candidate_auc"]),
            color="#BE123C",
            linewidth=1.2,
            linestyle="--",
            label="候选 AUC 门槛 0.520",
        )
        role_values = [value for seed in seeds for value in aucs[seed]]
        role_axis.set_ylim(
            min(0.485, min(role_values) - 0.006),
            max(0.535, max(role_values) + 0.009),
        )
        role_axis.set_xticks(x, role_labels, rotation=16, ha="right")
        role_axis.set_ylabel("最佳验证 AUC")
        role_axis.set_title("五个等预算角色的最终结果", loc="left", fontweight="bold")
        role_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
        role_axis.legend(frameon=False, ncol=2, loc="upper left")

        margin_axis = axes[1]
        mx = np.arange(len(margin_keys), dtype=float)
        for offset, seed, color in zip((-0.18, 0.18), seeds, colors):
            bars = margin_axis.bar(
                mx + offset,
                margins[seed],
                width=bar_width,
                color=color,
                label=f"seed{seed}",
                zorder=3,
            )
            margin_axis.bar_label(
                bars,
                labels=[f"{value:+.4f}" for value in margins[seed]],
                padding=3,
                fontsize=9.0,
                rotation=90,
            )
        margin_axis.axhline(0.0, color="#64748B", linewidth=1.2)
        margin_axis.axhline(
            float(gate["thresholds"]["candidate_minus_each_control"]),
            color="#BE123C",
            linewidth=1.2,
            linestyle="--",
            label="归因门槛 +0.005",
        )
        margin_values = [value for seed in seeds for value in margins[seed]]
        margin_axis.set_ylim(
            min(-0.02, min(margin_values) - 0.006),
            max(0.04, max(margin_values) + 0.009),
        )
        margin_axis.set_xticks(mx, margin_labels, rotation=16, ha="right")
        margin_axis.set_ylabel("候选 AUC 差值")
        margin_axis.set_title("真实双窗口相对对照的增益", loc="left", fontweight="bold")
        margin_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
        margin_axis.legend(frameon=False, loc="upper left")

        figure.text(
            0.075,
            0.075,
            "状态说明：hold 表示该异构递归窗口在两颗种子上不稳定；不等于 Runtime-E4 或一般 GF(2) 路线整体失败。",
            ha="left",
            color="#475569",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", dpi=160)
        plt.close(figure)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
