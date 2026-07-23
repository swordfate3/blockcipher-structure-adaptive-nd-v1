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

from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation1.runtime_parameterized_spn_attribution import (
    adjudicate_runtime_spn_r2d_sbox_scale,
    adjudicate_runtime_spn_r2e_sbox_location,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate the RTG1-R2d external S-box context scale calibration."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r2c-root", type=Path, required=True)
    parser.add_argument("--r1d-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    gate = adjudicate_runtime_spn_r2d_sbox_scale(
        run_id=args.run_id,
        candidate_rows=rows,
        r2c_rows=_read_jsonl(args.r2c_root / "results.jsonl"),
        r2c_gate=_read_json(args.r2c_root / "gate.json"),
        r1d_gate=_read_json(args.r1d_root / "gate.json"),
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "results": str(args.run_root / "results.jsonl"),
            "r2c_results": str(args.r2c_root / "results.jsonl"),
            "r2c_gate": str(args.r2c_root / "gate.json"),
            "r1d_gate": str(args.r1d_root / "gate.json"),
        },
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_sbox_scale_r2d",
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
    render_sbox_scale_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {"run_id": args.run_id, "status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def parse_location_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate the RTG1-R2e external S-box injection location."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r2c-root", type=Path, required=True)
    parser.add_argument("--r2d-root", type=Path, required=True)
    parser.add_argument("--r1d-root", type=Path, required=True)
    return parser.parse_args(argv)


def main_location(argv: list[str] | None = None) -> int:
    args = parse_location_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    gate = adjudicate_runtime_spn_r2e_sbox_location(
        run_id=args.run_id,
        candidate_rows=rows,
        r2c_rows=_read_jsonl(args.r2c_root / "results.jsonl"),
        r2c_gate=_read_json(args.r2c_root / "gate.json"),
        r2d_gate=_read_json(args.r2d_root / "gate.json"),
        r1d_gate=_read_json(args.r1d_root / "gate.json"),
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "results": str(args.run_root / "results.jsonl"),
            "r2c_results": str(args.r2c_root / "results.jsonl"),
            "r2c_gate": str(args.r2c_root / "gate.json"),
            "r2d_gate": str(args.r2d_root / "gate.json"),
            "r1d_gate": str(args.r1d_root / "gate.json"),
        },
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_sbox_location_r2e",
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
    render_sbox_location_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {"run_id": args.run_id, "status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_sbox_scale_svg(gate: dict[str, Any], output_path: Path) -> None:
    values = (
        float(gate["aucs"]["scale_1p0_baseline"]),
        float(gate["aucs"]["scale_0p1_candidate"]),
        float(gate["aucs"]["r1d_equivariant_anchor"]),
    )
    margins = (
        float(gate["margins"]["candidate_minus_scale1_baseline"]),
        float(gate["margins"]["candidate_minus_r1d_anchor"]),
    )
    passed = gate["status"] == "pass"
    if gate["status"] == "fail":
        conclusion = "协议检查未通过；修复前不解释S盒强度差异。"
    elif passed:
        conclusion = "0.1强度恢复到E4锚点容差内；下一步按同预算重跑三种拓扑控制。"
    else:
        conclusion = "0.1强度仍未恢复E4锚点；停止调强度，不扩样本或seed。"

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.19, wspace=0.32
        )
        figure.suptitle(
            "创新1 RTG1-R2d：外部 S 盒上下文强度校准",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            "GIFT-64 r6，2048/class训练，1024/class验证，5 epochs，seed0；只改变S盒残差系数1.0→0.1。",
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
        bars = auc_axis.bar(
            range(3), values, width=0.58, color=("#64748B", "#2563EB", "#0F766E")
        )
        auc_axis.bar_label(
            bars, labels=[f"{value:.6f}" for value in values], padding=5
        )
        auc_axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1.0)
        auc_axis.axhline(
            0.52, color="#059669", linestyle=":", linewidth=1.2, label="信号门 0.520"
        )
        auc_axis.set_title("同预算验证 AUC", loc="left", fontweight="bold")
        auc_axis.set_ylabel("AUC（局部放大）")
        auc_axis.set_xticks(range(3), labels=("强度 1.0", "强度 0.1", "R1d锚点"))
        auc_axis.set_ylim(min(0.50, min(values) - 0.004), max(values) + 0.005)
        auc_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        auc_axis.legend(loc="upper left", frameon=False)

        bars = margin_axis.barh(
            range(2),
            margins,
            height=0.46,
            color=("#2563EB", "#059669" if margins[1] >= -0.005 else "#DC2626"),
        )
        margin_axis.bar_label(
            bars, labels=[f"{margin:+.6f}" for margin in margins], padding=6
        )
        margin_axis.axvline(0.0, color="#64748B", linewidth=1.0)
        margin_axis.axvline(
            -0.005,
            color="#D97706",
            linestyle=":",
            linewidth=1.2,
            label="锚点容差 -0.005",
        )
        margin_axis.set_title("候选相对差值", loc="left", fontweight="bold")
        margin_axis.set_xlabel("AUC 差值")
        margin_axis.set_yticks(
            range(2), labels=("0.1 - 1.0", "0.1 - R1d锚点")
        )
        margin_axis.set_xlim(
            min(-0.012, min(margins) - 0.004), max(0.012, max(margins) + 0.004)
        )
        margin_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        margin_axis.legend(loc="lower right", frameon=False)

        figure.text(
            0.075,
            0.055,
            "证据边界：单密码、单seed、小规模单变量校准；尚未重新验证打乱拓扑和无拓扑控制。",
            ha="left",
            va="bottom",
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def render_sbox_location_svg(gate: dict[str, Any], output_path: Path) -> None:
    values = (
        float(gate["aucs"]["early_add_baseline"]),
        float(gate["aucs"]["late_pair_candidate"]),
        float(gate["aucs"]["r1d_equivariant_anchor"]),
    )
    margins = (
        float(gate["margins"]["candidate_minus_early_baseline"]),
        float(gate["margins"]["candidate_minus_r1d_anchor"]),
    )
    passed = gate["status"] == "pass"
    if gate["status"] == "fail":
        conclusion = "协议检查未通过；修复前不解释S盒注入位置差异。"
    elif passed:
        conclusion = "晚期S盒条件化恢复E4锚点容差；下一步同预算重跑三种拓扑控制。"
    else:
        conclusion = "晚期S盒条件化仍未恢复E4锚点；停止位置实验并审计融合机制。"

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.19, wspace=0.32
        )
        figure.suptitle(
            "创新1 RTG1-R2e：外部 S 盒注入位置校准",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            "GIFT-64 r6，2048/class训练，1024/class验证，5 epochs，seed0；只把S盒上下文从cell前移到pair后。",
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
        bars = auc_axis.bar(
            range(3), values, width=0.58, color=("#64748B", "#2563EB", "#0F766E")
        )
        auc_axis.bar_label(
            bars, labels=[f"{value:.6f}" for value in values], padding=5
        )
        auc_axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1.0)
        auc_axis.axhline(
            0.52, color="#059669", linestyle=":", linewidth=1.2, label="信号门 0.520"
        )
        auc_axis.set_title("同预算验证 AUC", loc="left", fontweight="bold")
        auc_axis.set_ylabel("AUC（局部放大）")
        auc_axis.set_xticks(range(3), labels=("早期注入", "晚期注入", "R1d锚点"))
        auc_axis.set_ylim(min(0.50, min(values) - 0.004), max(values) + 0.005)
        auc_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        auc_axis.legend(loc="upper left", frameon=False)

        colors = (
            "#059669" if margins[0] >= 0.0 else "#DC2626",
            "#059669" if margins[1] >= -0.005 else "#DC2626",
        )
        bars = margin_axis.barh(range(2), margins, height=0.46, color=colors)
        margin_axis.bar_label(
            bars, labels=[f"{margin:+.6f}" for margin in margins], padding=6
        )
        margin_axis.axvline(0.0, color="#64748B", linewidth=1.0)
        margin_axis.axvline(
            -0.005,
            color="#D97706",
            linestyle=":",
            linewidth=1.2,
            label="锚点容差 -0.005",
        )
        margin_axis.set_title("晚期候选相对差值", loc="left", fontweight="bold")
        margin_axis.set_xlabel("AUC 差值")
        margin_axis.set_yticks(
            range(2), labels=("晚期 - 早期", "晚期 - R1d锚点")
        )
        margin_axis.set_xlim(
            min(-0.012, min(margins) - 0.004), max(0.012, max(margins) + 0.004)
        )
        margin_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        margin_axis.legend(loc="lower right", frameon=False)

        figure.text(
            0.075,
            0.055,
            "证据边界：单密码、单seed、小规模单变量校准；尚未用晚期条件化重新验证两种拓扑控制。",
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
