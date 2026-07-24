from __future__ import annotations

import argparse
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

from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_signal import (
    adjudicate_runtime_spn_skinny_signal,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RTG1-T2-B SKINNY fixed-key signal selection."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=("screen", "confirmation"))
    parser.add_argument("--expected-rounds", required=True, type=_rounds)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = args.run_root / "results.jsonl"
    rows = _read_jsonl(results_path)
    gate = adjudicate_runtime_spn_skinny_signal(
        run_id=args.run_id,
        rows=rows,
        expected_rounds=args.expected_rounds,
        expected_seed=args.seed,
        phase=args.phase,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_skinny_fixed_key_signal_t2b",
        "training_performed": True,
        "train_samples_per_class": 512,
        "validation_samples_per_class": 256,
        "epochs": 5,
        "seed": args.seed,
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    write_history_csv(results_path, args.run_root / "history.csv")
    render_skinny_signal_svg(gate, args.run_root / "curves.svg")
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


def render_skinny_signal_svg(gate: dict[str, Any], output: Path) -> None:
    rounds = sorted(int(value) for value in gate["aucs_by_round"])
    values = [float(gate["aucs_by_round"][str(value)]) for value in rounds]
    selected = gate["selected_round"]
    colors = ["#059669" if value == selected else "#2563EB" for value in rounds]
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
        figure.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.18, wspace=0.32)
        phase_label = "轮数筛选" if gate["phase"] == "screen" else "独立 seed 确认"
        figure.suptitle(
            f"创新1 RTG1-T2-B：SKINNY 固定密钥差分信号{phase_label}",
            x=0.08,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.08,
            0.895,
            f"seed{gate['seed']}；差分 0x2000；训练 512/class，验证 256/class；正确拓扑模型；5 epochs。",
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        if gate["status"] == "pass" and gate["phase"] == "screen":
            conclusion = f"按冻结规则选中最深过线的 r{selected}；下一步只做 r{selected} seed1 原样确认。"
        elif gate["status"] == "pass":
            conclusion = f"r{selected} 通过独立 seed 信号门；下一步才允许一般 GF(2) 拓扑三控制。"
        else:
            conclusion = "冻结信号门未通过；停止拓扑训练和机械扩样。"
        figure.text(
            0.08,
            0.845,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )
        x = np.arange(len(rounds))
        bars = axes[0].bar(x, values, color=colors, width=0.58)
        axes[0].bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=5)
        axes[0].axhline(0.55, color="#DC2626", linestyle="--", linewidth=1.4, label="信号门 0.55")
        axes[0].axhline(0.5, color="#334155", linestyle=":", linewidth=1.2, label="随机基线 0.50")
        axes[0].set_ylim(0.48, 1.04)
        axes[0].set_xticks(x, labels=[f"r{value}" for value in rounds])
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("各轮最终 AUC", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(loc="upper right", frameon=False)

        near_rounds = [value for value in rounds if value >= 7]
        near_values = [float(gate["aucs_by_round"][str(value)]) for value in near_rounds]
        near_x = np.arange(len(near_rounds))
        near_bars = axes[1].bar(
            near_x,
            near_values,
            color=["#059669" if value == selected else "#64748B" for value in near_rounds],
            width=0.58,
        )
        axes[1].bar_label(near_bars, labels=[f"{value:.6f}" for value in near_values], padding=5)
        axes[1].axhline(0.55, color="#DC2626", linestyle="--", linewidth=1.4)
        axes[1].axhline(0.5, color="#334155", linestyle=":", linewidth=1.2)
        upper = max(0.58, max(near_values, default=0.55) + 0.025)
        axes[1].set_ylim(0.48, upper)
        axes[1].set_xticks(near_x, labels=[f"r{value}" for value in near_rounds])
        axes[1].set_ylabel("验证 AUC（放大）")
        axes[1].set_title("高轮近随机区域放大", loc="left", fontweight="bold")
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        figure.text(
            0.08,
            0.055,
            "证据范围：本地固定密钥信号选择；不比较打乱/无拓扑，不代表拓扑优越性、论文复现或正式规模。",
            ha="left",
            va="bottom",
            color="#334155",
            fontsize=10.0,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def _rounds(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one round")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
