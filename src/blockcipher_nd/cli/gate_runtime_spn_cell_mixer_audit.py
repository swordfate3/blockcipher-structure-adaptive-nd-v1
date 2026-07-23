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
    adjudicate_runtime_spn_r1d_cell_mixer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate the RTG1-R1d E4 cell-mixer audit."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r1c-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    seed = _single_seed(rows)
    gate = adjudicate_runtime_spn_r1d_cell_mixer(
        run_id=args.run_id,
        rows=rows,
        r1c_gate=_read_json(args.r1c_root / "gate.json"),
        expected_seed=seed,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_e4_cell_mixer_r1d",
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
    render_cell_mixer_audit_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {"run_id": args.run_id, "status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_cell_mixer_audit_svg(gate: dict[str, Any], output_path: Path) -> None:
    values = [float(gate["aucs"][role]) for role in ("fixed", "equivariant")]
    margin = float(gate["margins"]["fixed_minus_equivariant"])
    decision = str(gate["decision"])
    seed = int(gate.get("seed", 0))
    if decision.endswith("fixed_cell_mixer_dependency_supported"):
        conclusion = "固定cell交互有独立贡献；下一步设计拓扑功能坐标。"
        conclusion_color = "#166534"
    elif "equivariant_e4" in decision and decision.endswith("supported"):
        conclusion = (
            "等变E4主干保留信号；下一步移植到运行时三控制归因。"
            if seed == 0
            else f"seed{seed}等变E4锚点有效；下一步复验晚期S盒三控制。"
        )
        conclusion_color = "#166534"
    else:
        conclusion = "两种cell mixer均未校准；停止拆E4并重设计消息传递。"
        conclusion_color = "#B42318"

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
            "创新1 RTG1-R1d：E4 cell混合方式审计",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            f"GIFT-64 r6，2048/class训练，1024/class验证，5 epochs，seed{seed}；两行参数数目完全相同。",
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
            color=conclusion_color,
            fontweight="bold",
        )

        auc_axis, margin_axis = axes
        bars = auc_axis.bar(
            (0, 1), values, width=0.56, color=("#2563EB", "#0F766E")
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
        auc_axis.set_xticks(
            (0, 1),
            ("固定16-cell\nToken-Mixer", "cell重标号等变\n共享Mixer"),
        )
        auc_axis.set_ylim(min(0.49, min(values) - 0.005), max(0.525, max(values) + 0.005))
        auc_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        auc_axis.legend(loc="upper left", frameon=False)

        bar = margin_axis.barh(
            (0,), (margin,), height=0.44, color="#059669" if margin >= 0.01 else "#DC2626"
        )
        margin_axis.bar_label(bar, labels=[f"{margin:+.6f}"], padding=6)
        margin_axis.axvline(0.0, color="#64748B", linewidth=1.0)
        margin_axis.axvline(
            0.01, color="#059669", linestyle=":", linewidth=1.2, label="固定交互门 +0.010"
        )
        margin_axis.set_title("固定cell交互的独立贡献", loc="left", fontweight="bold")
        margin_axis.set_xlabel("固定Mixer - 等变Mixer（AUC）")
        margin_axis.set_yticks((0,), ("固定交互贡献",))
        margin_axis.set_xlim(min(-0.015, margin - 0.005), max(0.02, margin + 0.005))
        margin_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        margin_axis.legend(loc="lower right", frameon=False)

        figure.text(
            0.075,
            0.055,
            "证据边界：单密码、单seed、小规模主干校准；尚未比较正确、打乱与无拓扑运行时控制。",
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
