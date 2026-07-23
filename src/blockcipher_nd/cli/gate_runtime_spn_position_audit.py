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
    adjudicate_runtime_spn_r1b_position,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate the RTG1-R1b E4 position-identifiability audit."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r1a-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    r1a_gate = _read_json(args.r1a_root / "gate.json")
    gate = adjudicate_runtime_spn_r1b_position(
        run_id=args.run_id,
        rows=rows,
        r1a_gate=r1a_gate,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_e4_position_identifiability_r1b",
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
    render_position_audit_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {"run_id": args.run_id, "status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_position_audit_svg(gate: dict[str, Any], output_path: Path) -> None:
    values = [float(gate["aucs"][role]) for role in ("learned", "zero")]
    margin = float(gate["margins"]["learned_minus_zero"])
    supported = gate["decision"] == "innovation1_runtime_spn_position_identity_supported"
    conclusion = (
        "绝对位置是E4信号的重要来源；下一步设计运行时坐标描述。"
        if supported
        else "绝对位置未通过贡献门；下一步审计E4双视图编码，而不是添加坐标。"
    )
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
        figure.subplots_adjust(left=0.075, right=0.975, top=0.70, bottom=0.19, wspace=0.31)
        figure.suptitle(
            "创新1 RTG1-R1b：E4绝对位置可识别性审计",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            "GIFT-64 r6，2048/class训练，1024/class验证，5 epochs，seed0；仅关闭16-cell位置嵌入。",
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
            color="#166534" if supported else "#B42318",
            fontweight="bold",
        )

        auc_axis, margin_axis = axes
        bars = auc_axis.bar(
            (0, 1), values, width=0.56, color=("#2563EB", "#64748B")
        )
        auc_axis.bar_label(
            bars, labels=[f"{value:.6f}" for value in values], padding=5
        )
        auc_axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1.0)
        auc_axis.axhline(
            0.52, color="#059669", linestyle=":", linewidth=1.2, label="信号门 0.520"
        )
        auc_axis.set_title("同预算验证AUC", loc="left", fontweight="bold")
        auc_axis.set_ylabel("AUC（局部放大）")
        auc_axis.set_xticks((0, 1), ("E4位置嵌入\n正常学习", "同模型位置嵌入\n固定为零"))
        lower = min(0.49, min(values) - 0.005)
        upper = max(0.525, max(values) + 0.005)
        auc_axis.set_ylim(lower, upper)
        auc_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        auc_axis.legend(loc="upper left", frameon=False)

        bar = margin_axis.barh(
            (0,), (margin,), height=0.44, color="#059669" if margin >= 0.01 else "#DC2626"
        )
        margin_axis.bar_label(bar, labels=[f"{margin:+.6f}"], padding=6)
        margin_axis.axvline(0.0, color="#64748B", linewidth=1.0)
        margin_axis.axvline(
            0.01, color="#059669", linestyle=":", linewidth=1.2, label="贡献门 +0.010"
        )
        margin_axis.set_title("位置嵌入的独立贡献", loc="left", fontweight="bold")
        margin_axis.set_xlabel("正常E4 - 零位置E4（AUC）")
        margin_axis.set_yticks((0,), ("位置贡献",))
        margin_axis.set_xlim(min(-0.015, margin - 0.005), max(0.02, margin + 0.005))
        margin_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        margin_axis.legend(loc="lower right", frameon=False)

        figure.text(
            0.075,
            0.055,
            "证据边界：单密码、单seed、小规模架构审计；不等于运行时拓扑优于控制，也不授权远程扩样。",
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
