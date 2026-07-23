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
    adjudicate_runtime_spn_r2a_e4_attribution,
    adjudicate_runtime_spn_r2f_late_attribution,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RTG1-R2a runtime E4 topology attribution."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r1d-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    gate = adjudicate_runtime_spn_r2a_e4_attribution(
        run_id=args.run_id,
        rows=rows,
        r1d_gate=_read_json(args.r1d_root / "gate.json"),
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "results": str(args.run_root / "results.jsonl"),
            "r1d_gate": str(args.r1d_root / "gate.json"),
        },
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_e4_attribution_r2a",
        "cipher": "GIFT-64",
        "training_performed": True,
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 5,
        "seed": 0,
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    _write_history(args.run_root / "history.csv", rows)
    render_runtime_e4_attribution_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def parse_late_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RTG1-R2f late-conditioned topology attribution."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r1d-root", type=Path, required=True)
    parser.add_argument("--r2e-root", type=Path, required=True)
    return parser.parse_args(argv)


def main_late(argv: list[str] | None = None) -> int:
    args = parse_late_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    seed = _single_seed(rows)
    gate = adjudicate_runtime_spn_r2f_late_attribution(
        run_id=args.run_id,
        rows=rows,
        r1d_gate=_read_json(args.r1d_root / "gate.json"),
        r2e_gate=_read_json(args.r2e_root / "gate.json"),
        expected_seed=seed,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "results": str(args.run_root / "results.jsonl"),
            "r1d_gate": str(args.r1d_root / "gate.json"),
            "r2e_gate": str(args.r2e_root / "gate.json"),
        },
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_late_attribution_r2f",
        "cipher": "GIFT-64",
        "training_performed": True,
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 5,
        "seed": seed,
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    _write_history(args.run_root / "history.csv", rows)
    render_runtime_e4_attribution_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_runtime_e4_attribution_svg(
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    roles = ("true", "corrupted", "independent")
    labels = ("正确线性拓扑", "打乱线性拓扑", "无线性拓扑")
    values = [float(gate["aucs"][role]) for role in roles]
    margin_labels = ("正确 - R1d锚点", "正确 - 打乱", "正确 - 无拓扑")
    margins = [
        float(gate["margins"]["true_minus_r1d_anchor"]),
        float(gate["margins"]["true_minus_corrupted"]),
        float(gate["margins"]["true_minus_independent"]),
    ]
    passed = gate["status"] == "pass"
    run_id = str(gate["run_id"])
    seed = int(gate.get("seed", 0))
    if "r2c" in run_id:
        stage = "R2c 全bit打乱控制"
    elif "r2g" in run_id:
        stage = "R2g cell内bit角色对齐修复"
    elif "r2f" in run_id:
        stage = "R2f 晚期S盒三控制"
    elif "r2b" in run_id:
        stage = "R2b 同预算容量"
    elif "bitorderfix" in run_id:
        stage = "R2a 位序修复"
    else:
        stage = "R2a 初次执行"
    if gate["status"] == "fail":
        conclusion = "协议检查未通过；结果不得用于拓扑归因，必须先修复并原预算重跑。"
    elif passed and "r2g" in run_id and seed == 0:
        conclusion = "修复后正确拓扑超过两种控制并保住主干；下一步只做同预算seed1复验。"
    elif passed and "r2g" in run_id:
        conclusion = "修复后两颗seed均通过四门；下一步只做同预算PRESENT迁移。"
    elif passed and "r2f" in run_id and seed == 0:
        conclusion = "正确拓扑超过两种控制且保留主干信号；下一步只做同预算seed1复验。"
    elif passed and "r2f" in run_id:
        conclusion = "两颗seed均支持真实拓扑优势；下一步只做同预算第二种SPN迁移。"
    elif passed:
        conclusion = "正确拓扑同时保留主干信号并超过两种控制，可进入下一预注册门。"
    elif (
        gate["research_checks"]["true_exceeds_corrupted_by_0p005"]
        and gate["research_checks"]["true_exceeds_independent_by_0p005"]
        and not gate["research_checks"]["true_within_r1d_anchor_tolerance"]
    ):
        conclusion = "正确拓扑已超过两种控制，但主干保真门未过；保持 hold，不扩大样本。"
    else:
        conclusion = "正确拓扑未同时超过两种控制；停止扩样，继续本地重构线性拓扑交互。"
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
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.3))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.70,
            bottom=0.19,
            wspace=0.34,
        )
        figure.suptitle(
            f"创新1 RTG1-{stage}：运行时 E4 线性拓扑归因",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            f"GIFT-64 r6，2048/class训练，1024/class验证，5 epochs，seed{seed}；cell与S盒元数据保持不变。",
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
            color="#047857" if passed else "#B42318",
            fontweight="bold",
        )

        auc_axis, margin_axis = axes
        x = np.arange(len(values), dtype=np.float64)
        bars = auc_axis.bar(
            x,
            values,
            width=0.58,
            color=("#2563EB", "#DC2626", "#64748B"),
        )
        auc_axis.bar_label(
            bars,
            labels=[f"{value:.6f}" for value in values],
            padding=5,
            fontsize=9.0,
        )
        auc_axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1.0)
        auc_axis.axhline(
            0.52,
            color="#059669",
            linestyle=":",
            linewidth=1.2,
            label="最低信号门 0.520",
        )
        auc_axis.set_title("同预算验证 AUC", loc="left", fontweight="bold")
        auc_axis.set_ylabel("AUC（局部放大）")
        auc_axis.set_xticks(x, labels=labels)
        auc_axis.set_ylim(min(0.49, min(values) - 0.005), max(0.525, max(values) + 0.006))
        auc_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        auc_axis.legend(loc="upper right", frameon=False)

        y = np.arange(len(margins), dtype=np.float64)
        colors = [
            "#059669" if margins[0] >= -0.005 else "#DC2626",
            *("#059669" if margin >= 0.005 else "#DC2626" for margin in margins[1:]),
        ]
        bars = margin_axis.barh(y, margins, height=0.48, color=colors)
        for bar, margin in zip(bars, margins, strict=True):
            if margin <= -0.004:
                x_position = margin / 2.0
                horizontal_alignment = "center"
                color = "#FFFFFF"
            elif margin < 0.0:
                x_position = margin - 0.001
                horizontal_alignment = "right"
                color = "#334155"
            else:
                x_position = margin + 0.001
                horizontal_alignment = "left"
                color = "#334155"
            margin_axis.text(
                x_position,
                bar.get_y() + bar.get_height() / 2.0,
                f"{margin:+.6f}",
                ha=horizontal_alignment,
                va="center",
                fontsize=9.0,
                color=color,
                fontweight="bold" if margin <= -0.004 else "normal",
            )
        margin_axis.axvline(0.0, color="#64748B", linewidth=1.0)
        margin_axis.axvline(
            0.005,
            color="#059669",
            linestyle=":",
            linewidth=1.2,
            label="归因门 +0.005",
        )
        margin_axis.axvline(
            -0.005,
            color="#D97706",
            linestyle=":",
            linewidth=1.2,
            label="锚点容差 -0.005",
        )
        margin_axis.set_title("主干保真与拓扑贡献", loc="left", fontweight="bold")
        margin_axis.set_xlabel("AUC 差值")
        margin_axis.set_yticks(y, labels=margin_labels)
        margin_axis.set_xlim(min(-0.015, min(margins) - 0.005), max(0.02, max(margins) + 0.006))
        margin_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        margin_axis.legend(loc="lower right", frameon=False)

        figure.text(
            0.075,
            0.055,
            "证据边界：单密码、单seed、小规模运行时拓扑归因校准；不是正式训练或稳定跨密码结论。",
            ha="left",
            va="bottom",
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


def _single_seed(rows: list[dict[str, Any]]) -> int:
    seeds = {int(row["seed"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError(f"expected one seed across all rows, got {sorted(seeds)}")
    return seeds.pop()


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
