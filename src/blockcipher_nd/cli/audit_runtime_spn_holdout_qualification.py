from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation1.runtime_spn_holdout_qualification import (
    CANDIDATES,
    CONDITIONS,
    DISPLAY_NAMES,
    load_and_validate_holdout_qualification_config,
    run_holdout_qualification_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen Runtime-SPN candidates for a valid next holdout."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_and_validate_holdout_qualification_config(
        config_path,
        project_root=project_root,
    )
    _write_progress(output_root / "progress.jsonl", "audit_start", config["run_id"])
    payload = run_holdout_qualification_audit(
        config=config,
        project_root=project_root,
    )
    _write_jsonl(output_root / "results.jsonl", payload["rows"])
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "gate.json", payload["gate"])
    _write_json(output_root / "summary.json", payload["summary"])
    _write_json(output_root / "structure-profiles.json", payload["structure_profiles"])
    render_holdout_qualification_svg(payload["gate"], payload["rows"], output_root / "curves.svg")
    _write_progress(
        output_root / "progress.jsonl",
        "audit_done",
        config["run_id"],
        status=payload["gate"]["status"],
        decision=payload["gate"]["decision"],
        selected_holdout=payload["gate"]["selected_holdout"],
    )
    print(json.dumps(payload["gate"], ensure_ascii=False, sort_keys=True))
    return 1 if payload["gate"]["status"] == "fail" else 0


def render_holdout_qualification_svg(
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = {
        "correct": "正确拓扑",
        "corrupted": "损坏拓扑",
        "no_topology": "无拓扑",
    }
    colors = {
        "correct": "#2563EB",
        "corrupted": "#D97706",
        "no_topology": "#64748B",
    }
    y_limits = {
        "rectangle80": (0.58, 0.72),
        "uknit64": (0.47, 0.55),
        "dialga128": (0.48, 1.00),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 7.4))
        figure.subplots_adjust(
            left=0.055,
            right=0.98,
            top=0.72,
            bottom=0.19,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 A7：下一次整密码留出资格审计",
            x=0.055,
            y=0.965,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.895,
            "只复算冻结证据，不训练、不生成新数据；候选必须两颗 seed 都有信号、依赖拓扑且原子 GF(2) 类型被源密码完整覆盖。",
            ha="left",
            color="#475569",
        )
        selected = gate.get("selected_holdout")
        decision_text = (
            f"裁决：选择 {DISPLAY_NAMES[selected]} 进入 A8 零目标训练行留出实验"
            if selected in DISPLAY_NAMES
            else "裁决：没有候选通过，停止新的整密码留出训练"
        )
        figure.text(
            0.055,
            0.825,
            decision_text,
            ha="left",
            color="#047857" if selected else "#B42318",
            fontweight="bold",
        )
        figure.text(
            0.98,
            0.765,
            "红色虚线：AUC 资格下限 0.55；结构 margin 下限为 +0.005。各候选使用独立纵轴以便阅读局部差异。",
            ha="right",
            color="#64748B",
            fontsize=9,
        )

        for axis, candidate in zip(axes, CANDIDATES, strict=True):
            candidate_rows = sorted(
                (row for row in rows if row["candidate"] == candidate),
                key=lambda row: row["seed"],
            )
            x = np.arange(2, dtype=np.float64)
            width = 0.23
            for index, condition in enumerate(CONDITIONS):
                values = [row[f"{condition}_auc"] for row in candidate_rows]
                bars = axis.bar(
                    x + (index - 1) * width,
                    values,
                    width,
                    label=labels[condition],
                    color=colors[condition],
                )
                axis.bar_label(
                    bars,
                    labels=[f"{value:.4f}" for value in values],
                    padding=3,
                    fontsize=8.5,
                )
            candidate_gate = gate["per_candidate"][candidate]
            status = (
                "入选"
                if candidate_gate["eligible"]
                else "已用过"
                if candidate_gate["technically_qualified"]
                and candidate_gate["previous_whole_cipher_holdout"]
                else "不合格"
            )
            axis.set_title(
                f"{DISPLAY_NAMES[candidate]}  ·  {status}",
                loc="left",
                fontweight="bold",
                color="#047857" if candidate_gate["eligible"] else "#334155",
            )
            axis.set_xticks(x, ("seed0", "seed1"))
            axis.set_ylim(*y_limits[candidate])
            axis.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)
            axis.axhline(0.55, color="#DC2626", linestyle=":", linewidth=1)
            axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
            first = candidate_rows[0]
            axis.text(
                0.0,
                -0.19,
                (
                    f"GF(2) 原子覆盖 {first['covered_atomic_gf2_types']}/"
                    f"{first['target_atomic_gf2_types']}  ·  "
                    f"精确 S 盒重合 {first['exact_source_sbox_overlap']}"
                ),
                transform=axis.transAxes,
                ha="left",
                color="#475569",
                fontsize=9,
            )
            if axis is axes[0]:
                axis.set_ylabel("验证 AUC（每个候选使用独立纵轴）")

        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=3,
            frameon=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches=None)
        plt.close(figure)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_progress(path: Path, event: str, run_id: str, **payload: Any) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
