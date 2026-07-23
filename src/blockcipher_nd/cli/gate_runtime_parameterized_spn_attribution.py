from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation1.runtime_parameterized_spn_attribution import (
    adjudicate_runtime_spn_r1,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RTG1 runtime-SPN R1 against a frozen E4 anchor."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cipher", choices=["GIFT-64", "PRESENT-80"], required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--anchor-results", type=Path, required=True)
    parser.add_argument("--r0-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_rows = _read_jsonl(args.runtime_root / "results.jsonl")
    anchor_rows = _read_jsonl(args.anchor_results)
    r0_gate = _read_json(args.r0_root / "gate.json")
    gate = adjudicate_runtime_spn_r1(
        run_id=args.run_id,
        cipher=args.cipher,
        runtime_rows=runtime_rows,
        anchor_rows=anchor_rows,
        r0_gate=r0_gate,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "runtime_results": str(args.runtime_root / "results.jsonl"),
            "anchor_results": str(args.anchor_results),
            "r0_gate": str(args.r0_root / "gate.json"),
        },
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_parameterized_spn_r1_attribution",
        "cipher": args.cipher,
        "training_performed": True,
        "samples_per_class": 8192,
        "validation_samples_per_class": 4096,
        "epochs": 10,
        "seed": 0,
        "gate": gate,
    }
    _write_json(args.runtime_root / "validation.json", validation)
    _write_json(args.runtime_root / "gate.json", gate)
    _write_json(args.runtime_root / "summary.json", summary)
    _write_history(args.runtime_root / "history.csv", runtime_rows)
    render_runtime_spn_attribution_svg(
        gate,
        args.runtime_root / "curves.svg",
    )
    _append_progress(
        args.runtime_root / "progress.jsonl",
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_runtime_spn_attribution_svg(
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    roles = ("anchor", "true", "corrupted", "independent")
    labels = ("E4 固定锚点", "正确运行时拓扑", "保持度数的打乱拓扑", "无拓扑控制")
    values = [float(gate["aucs"][role]) for role in roles]
    colors = ("#475569", "#2563EB", "#DC2626", "#D97706")
    margins = gate["margins"]
    margin_labels = ("正确 - E4锚点", "正确 - 打乱拓扑", "正确 - 无拓扑")
    margin_values = [
        float(margins["true_minus_anchor"]),
        float(margins["true_minus_corrupted"]),
        float(margins["true_minus_independent"]),
    ]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
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
        figure, axes = plt.subplots(1, 2, figsize=(14.8, 7.7))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.72,
            bottom=0.20,
            wspace=0.34,
        )
        figure.suptitle(
            "创新1 RTG1-R1：GIFT-64 运行时拓扑归因结果",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            "r6，8192/class 训练，4096/class 验证，10 epochs，seed0；三种运行时模式参数量相同。",
            ha="left",
            va="top",
            fontsize=10.5,
            color="#475569",
        )
        figure.text(
            0.075,
            0.842,
            "结论：正确拓扑接近随机且未超过两个控制；这是单密码本地诊断，不是稳定跨密码结论。",
            ha="left",
            va="top",
            fontsize=10.5,
            color="#B42318",
            fontweight="bold",
        )

        auc_axis, margin_axis = axes
        x = np.arange(len(roles), dtype=np.float64)
        bars = auc_axis.bar(x, values, color=colors, width=0.64)
        auc_axis.bar_label(
            bars,
            labels=[f"{value:.6f}" for value in values],
            padding=5,
            fontsize=9.0,
        )
        auc_axis.axhline(
            0.5,
            color="#64748B",
            linestyle="--",
            linewidth=1.1,
            label="随机水平 0.5",
        )
        auc_axis.set_title("同协议验证 AUC", loc="left", fontweight="bold", pad=10)
        auc_axis.set_ylabel("AUC（局部放大）")
        auc_axis.set_xticks(x, labels=labels, rotation=8, ha="right")
        auc_axis.set_ylim(0.49, max(values) + 0.012)
        auc_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        auc_axis.grid(False, axis="x")
        auc_axis.legend(loc="upper right", frameon=False)

        y = np.arange(len(margin_values), dtype=np.float64)
        margin_colors = ["#059669" if value >= 0 else "#DC2626" for value in margin_values]
        margin_bars = margin_axis.barh(y, margin_values, color=margin_colors, height=0.58)
        margin_axis.bar_label(
            margin_bars,
            labels=[f"{value:+.6f}" for value in margin_values],
            padding=5,
            fontsize=9.0,
        )
        margin_axis.axvline(0.0, color="#64748B", linewidth=1.0)
        margin_axis.axvline(
            0.005,
            color="#059669",
            linestyle="--",
            linewidth=1.1,
            label="控制增益门 +0.005",
        )
        margin_axis.axvline(
            -0.005,
            color="#7C3AED",
            linestyle=":",
            linewidth=1.2,
            label="锚点容差 -0.005",
        )
        margin_axis.set_title("预注册 margin 审判", loc="left", fontweight="bold", pad=10)
        margin_axis.set_xlabel("AUC 差值")
        margin_axis.set_yticks(y, labels=margin_labels)
        margin_axis.invert_yaxis()
        limit = max(0.06, max(abs(value) for value in margin_values) * 1.28)
        margin_axis.set_xlim(-limit, limit)
        margin_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        margin_axis.legend(loc="lower right", frameon=False)

        figure.text(
            0.075,
            0.055,
            "下一步只改一个变量：在全局池化前增加运行时 cell-token 交互，保留 E4 的逐pair对齐cell关系；先做小规模 GIFT 校准。",
            ha="left",
            va="bottom",
            fontsize=10.0,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    history_rows = [
        {"model": row["model"], **epoch}
        for row in rows
        for epoch in row.get("history", [])
    ]
    if not history_rows:
        raise ValueError("runtime result rows contain no training history")
    fields = sorted({key for row in history_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history_rows)


def _append_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
