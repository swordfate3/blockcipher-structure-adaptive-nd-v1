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

from blockcipher_nd.tasks.innovation1.runtime_spn_present_transfer import (
    adjudicate_runtime_spn_present_transfer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RTG1-T1 PRESENT runtime-topology transfer."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    gate = adjudicate_runtime_spn_present_transfer(
        run_id=args.run_id, rows=rows, expected_seed=args.seed
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(args.run_root / "results.jsonl"),
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_present_transfer_t1",
        "cipher": "PRESENT-80",
        "training_performed": True,
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 5,
        "seed": args.seed,
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    _write_history(args.run_root / "history.csv", rows)
    render_present_transfer_svg(gate, args.run_root / "curves.svg")
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


def render_present_transfer_svg(gate: dict[str, Any], output: Path) -> None:
    labels = ("正确P层", "打乱P层", "无P层拓扑")
    values = [
        float(gate["aucs"][role])
        for role in ("true", "corrupted", "independent")
    ]
    margins = [
        float(gate["margins"]["true_minus_corrupted"]),
        float(gate["margins"]["true_minus_independent"]),
    ]
    passed = gate["status"] == "pass"
    seed = int(gate["seed"])
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
        figure, axes = plt.subplots(1, 2, figsize=(14.2, 7.4))
        figure.subplots_adjust(left=0.075, right=0.97, top=0.72, bottom=0.18, wspace=0.32)
        figure.suptitle(
            "创新1 RTG1-T1：运行时结构参数化主干迁移到 PRESENT",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            f"seed{seed}；同一主干参数形状，仅从外部替换 PRESENT 的 cell、S盒和 P层；三行预算相同。",
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        if passed and seed == 0:
            conclusion = "正确 PRESENT 拓扑通过信号门和两种控制门；下一步只做 seed1 原样复验。"
        elif passed:
            conclusion = "PRESENT 两颗 seed 均通过；下一步审计两密码稳定性，不扩大训练规模。"
        else:
            conclusion = "PRESENT 迁移未通过冻结门；停止复验和扩样，保留为诊断结果。"
        figure.text(
            0.075,
            0.845,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if passed else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )
        x = np.arange(3)
        bars = axes[0].bar(
            x, values, color=("#059669", "#DC2626", "#64748B"), width=0.62
        )
        axes[0].bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=5)
        axes[0].axhline(0.5, color="#334155", linestyle=":", linewidth=1.2)
        axes[0].set_ylim(min(0.48, min(values) - 0.01), max(0.54, max(values) + 0.02))
        axes[0].set_xticks(x, labels=labels)
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("三种外部拓扑输入", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        margin_x = np.arange(2)
        margin_bars = axes[1].bar(
            margin_x,
            margins,
            color=["#059669" if value >= 0.005 else "#DC2626" for value in margins],
            width=0.58,
        )
        axes[1].bar_label(
            margin_bars, labels=[f"{value:+.6f}" for value in margins], padding=5
        )
        axes[1].axhline(0.005, color="#1D4ED8", linestyle="--", linewidth=1.4)
        axes[1].axhline(0.0, color="#334155", linewidth=1.0)
        axes[1].set_xticks(margin_x, labels=("正确 - 打乱", "正确 - 无拓扑"))
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title("拓扑归因边际（门槛 +0.005）", loc="left", fontweight="bold")
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        figure.text(
            0.075,
            0.055,
            "本图是 PRESENT r7、2048/class 的本地迁移诊断，不是论文规模、SOTA 或 Zhang/Wang 复现。",
            ha="left",
            va="bottom",
            color="#334155",
            fontsize=10.0,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
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


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    history_rows = [
        {
            "model": row["model"],
            **history,
        }
        for row in rows
        for history in row.get("history", [])
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
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
