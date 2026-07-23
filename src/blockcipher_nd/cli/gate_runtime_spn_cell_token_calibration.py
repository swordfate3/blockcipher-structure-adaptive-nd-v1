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
    adjudicate_runtime_spn_r1a_cell_token,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate the RTG1-R1a GIFT cell-token calibration."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--r1-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.runtime_root / "results.jsonl")
    r1_gate = _read_json(args.r1_root / "gate.json")
    gate = adjudicate_runtime_spn_r1a_cell_token(
        run_id=args.run_id,
        rows=rows,
        r1_gate=r1_gate,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "runtime_results": str(args.runtime_root / "results.jsonl"),
            "r1_gate": str(args.r1_root / "gate.json"),
        },
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_cell_token_r1a_calibration",
        "cipher": "GIFT-64",
        "training_performed": True,
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 5,
        "seed": 0,
        "gate": gate,
    }
    _write_json(args.runtime_root / "validation.json", validation)
    _write_json(args.runtime_root / "gate.json", gate)
    _write_json(args.runtime_root / "summary.json", summary)
    _write_history(args.runtime_root / "history.csv", rows)
    render_cell_token_calibration_svg(gate, args.runtime_root / "curves.svg")
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


def render_cell_token_calibration_svg(
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    roles = ("current", "cell_true", "cell_corrupted")
    labels = ("旧主干\n正确拓扑", "cell-token\n正确拓扑", "cell-token\n打乱拓扑")
    values = [float(gate["aucs"][role]) for role in roles]
    margin_labels = ("cell-token正确 - 旧主干", "cell-token正确 - 打乱拓扑")
    margin_values = [
        float(gate["margins"]["cell_true_minus_current"]),
        float(gate["margins"]["cell_true_minus_corrupted"]),
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
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.4))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.72,
            bottom=0.20,
            wspace=0.34,
        )
        figure.suptitle(
            "创新1 RTG1-R1a：GIFT-64 cell-token 小规模校准",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            "r6，2048/class 训练，1024/class 验证，5 epochs，seed0；只改变全局池化前的 cell-token 交互。",
            ha="left",
            va="top",
            fontsize=10.5,
            color="#475569",
        )
        figure.text(
            0.075,
            0.842,
            "结论：cell-token 未改善旧主干，且正确拓扑低于打乱拓扑；停止扩样，转 E4 位置可识别性审计。",
            ha="left",
            va="top",
            fontsize=10.5,
            color="#B42318",
            fontweight="bold",
        )
        auc_axis, margin_axis = axes
        x = np.arange(len(roles), dtype=np.float64)
        bars = auc_axis.bar(
            x,
            values,
            color=("#475569", "#2563EB", "#DC2626"),
            width=0.62,
        )
        auc_axis.bar_label(
            bars,
            labels=[f"{value:.6f}" for value in values],
            padding=5,
            fontsize=9.0,
        )
        auc_axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1.1)
        auc_axis.axhline(
            0.52,
            color="#059669",
            linestyle=":",
            linewidth=1.2,
            label="最低信号门 0.520",
        )
        auc_axis.set_title("同预算验证 AUC", loc="left", fontweight="bold", pad=10)
        auc_axis.set_ylabel("AUC（局部放大）")
        auc_axis.set_xticks(x, labels=labels)
        auc_axis.set_ylim(0.49, 0.525)
        auc_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        auc_axis.grid(False, axis="x")
        auc_axis.legend(loc="upper left", frameon=False)

        y = np.arange(len(margin_values), dtype=np.float64)
        bars = margin_axis.barh(
            y,
            margin_values,
            color=["#059669" if value >= 0 else "#DC2626" for value in margin_values],
            height=0.56,
        )
        margin_axis.bar_label(
            bars,
            labels=[f"{value:+.6f}" for value in margin_values],
            padding=5,
            fontsize=9.0,
        )
        margin_axis.axvline(0.0, color="#64748B", linewidth=1.0)
        margin_axis.axvline(
            0.005,
            color="#D97706",
            linestyle="--",
            linewidth=1.1,
            label="拓扑门 +0.005",
        )
        margin_axis.axvline(
            0.010,
            color="#059669",
            linestyle=":",
            linewidth=1.2,
            label="架构增益门 +0.010",
        )
        margin_axis.set_title("预注册增益门", loc="left", fontweight="bold", pad=10)
        margin_axis.set_xlabel("AUC 差值")
        margin_axis.set_yticks(y, labels=margin_labels)
        margin_axis.invert_yaxis()
        margin_axis.set_xlim(-0.012, 0.014)
        margin_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        margin_axis.legend(loc="lower right", frameon=False)

        figure.text(
            0.075,
            0.055,
            "下一步：同预算比较 E4 typed-cell 原模型与禁用绝对位置嵌入的模型，判断运行时坐标描述是否必要。",
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
        raise ValueError("calibration result rows contain no training history")
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
