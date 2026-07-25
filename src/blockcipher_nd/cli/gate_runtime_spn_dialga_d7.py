from __future__ import annotations

import argparse
import hashlib
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

from blockcipher_nd.cli.gate_runtime_spn_dialga_d6 import (
    audit_source_cache_reuse,
)
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    adjudicate_runtime_spn_dialga_d1,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d6 import SEEDS
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d7 import (
    adjudicate_runtime_spn_dialga_d7,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate Dialga-128 Runtime-E5 D7 r4 regression."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--d1-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = args.run_root / "results.jsonl"
    progress_path = args.run_root / "progress.jsonl"
    d1_results_path = args.d1_root / "results.jsonl"
    d1_gate_path = args.d1_root / "gate.json"
    d1_validation_path = args.d1_root / "validation.json"

    rows = _read_jsonl(results_path)
    progress_rows = _read_jsonl(progress_path)
    d1_rows = _read_jsonl(d1_results_path)
    persisted_d1_gate = _read_json(d1_gate_path)
    d1_validation = _read_json(d1_validation_path)
    replayed_d1_gate = adjudicate_runtime_spn_dialga_d1(
        run_id=str(persisted_d1_gate.get("run_id", "")),
        rows=d1_rows,
    )
    expected_cache_root = args.d1_root / "cache"
    cache_audit = audit_source_cache_reuse(
        progress_rows=progress_rows,
        expected_cache_root=expected_cache_root,
        source_label="d1",
    )
    gate = adjudicate_runtime_spn_dialga_d7(
        run_id=args.run_id,
        rows=rows,
        d1_rows=d1_rows,
        persisted_d1_gate=persisted_d1_gate,
        replayed_d1_gate=replayed_d1_gate,
        d1_validation=d1_validation,
        expected_cache_root=expected_cache_root,
        cache_audit=cache_audit,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
        "source_d1": {
            "root": str(args.d1_root),
            "results_sha256": _sha256(d1_results_path),
            "gate_sha256": _sha256(d1_gate_path),
            "validation_sha256": _sha256(d1_validation_path),
        },
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": "Dialga-128",
        "training_performed": True,
        "data_generation_performed": False,
        "train_samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 10,
        "seeds": list(SEEDS),
        "source_d1_root": str(args.d1_root),
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    write_history_csv(results_path, args.run_root / "history.csv")
    render_dialga_d7_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        progress_path,
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_dialga_d7_svg(gate: dict[str, Any], output_path: Path) -> None:
    roles = ("correct", "corrupted", "no_topology")
    role_labels = ("E5 正确拓扑", "E5 错误拓扑", "E5 无拓扑")
    role_colors = ("#059669", "#DC2626", "#64748B")
    x = np.arange(2, dtype=np.float64)
    auc_series = {
        "D1 E4 正确锚点": [
            float(gate["d1_correct_anchors"][f"seed{seed}"]) for seed in SEEDS
        ],
        **{
            label: [float(gate["aucs"][f"seed{seed}"][role]) for seed in SEEDS]
            for role, label in zip(roles, role_labels, strict=True)
        },
    }
    margin_keys = (
        "correct_minus_corrupted",
        "correct_minus_no_topology",
        "correct_minus_d1",
    )
    margin_labels = ("正确 - 错误", "正确 - 无拓扑", "正确 - D1锚点")
    margin_values = {
        seed: [
            float(gate["margins"][f"seed{seed}"][key]) for key in margin_keys
        ]
        for seed in SEEDS
    }
    gate_values = {
        seed: [
            float(gate["learned_topology_gates"][f"seed{seed}"][role]["bounded"])
            for role in roles
        ]
        for seed in SEEDS
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.8,
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
        figure, axes = plt.subplots(1, 3, figsize=(17.2, 7.0))
        figure.subplots_adjust(
            left=0.055, right=0.985, top=0.70, bottom=0.22, wspace=0.28
        )
        figure.suptitle(
            "创新1 D7：Dialga 四轮 E5 强信号回归",
            x=0.055,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.89,
            "复用 D1 四轮数据和预算，仅将 Runtime-E4 换成 Runtime-E5；检验 E5 是否保留已知机制。",
            ha="left",
            color="#475569",
            fontsize=10.5,
        )
        figure.text(
            0.055,
            0.81,
            f"裁决：{_decision_text(gate)}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
            fontsize=10.4,
        )

        width = 0.19
        colors = ("#2563EB", *role_colors)
        for index, ((label, values), color) in enumerate(
            zip(auc_series.items(), colors, strict=True)
        ):
            offset = (index - 1.5) * width
            bars = axes[0].bar(x + offset, values, width, label=label, color=color)
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=3,
                fontsize=7.7,
                rotation=90,
            )
        auc_flat = [value for values in auc_series.values() for value in values]
        axes[0].set_ylim(max(0.45, min(auc_flat) - 0.05), min(1.0, max(auc_flat) + 0.04))
        axes[0].set_xticks(x, ("seed0", "seed1"))
        axes[0].set_ylabel("最佳验证 AUC")
        axes[0].set_title("E5 与 D1 E4 锚点", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[0].legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            fontsize=7.8,
        )

        margin_x = np.arange(3, dtype=np.float64)
        pair_width = 0.34
        for seed, offset, color in (
            (0, -pair_width / 2, "#2563EB"),
            (1, pair_width / 2, "#D97706"),
        ):
            bars = axes[1].bar(
                margin_x + offset,
                margin_values[seed],
                pair_width,
                label=f"seed{seed}",
                color=color,
            )
            axes[1].bar_label(
                bars,
                labels=[f"{value:+.4f}" for value in margin_values[seed]],
                padding=3,
                fontsize=8.0,
                rotation=90,
            )
        axes[1].axhline(0.0, color="#334155", linewidth=1.0)
        axes[1].axhline(
            0.005,
            color="#059669",
            linestyle="--",
            linewidth=1.2,
            label="控制差门槛 +0.005",
        )
        axes[1].axhline(
            -0.010,
            color="#7C3AED",
            linestyle=":",
            linewidth=1.3,
            label="D1 保留门槛 -0.010",
        )
        margin_flat = margin_values[0] + margin_values[1] + [0.0, 0.005, -0.01]
        span = max(0.02, max(margin_flat) - min(margin_flat))
        axes[1].set_ylim(min(margin_flat) - 0.25 * span, max(margin_flat) + 0.45 * span)
        axes[1].set_xticks(margin_x, margin_labels)
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title("保留与拓扑控制边际", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper right", fontsize=8.0)

        role_x = np.arange(3, dtype=np.float64)
        for seed, offset, alpha in (
            (0, -pair_width / 2, 1.0),
            (1, pair_width / 2, 0.68),
        ):
            bars = axes[2].bar(
                role_x + offset,
                gate_values[seed],
                pair_width,
                label=f"seed{seed}",
                color=role_colors,
                alpha=alpha,
                edgecolor="#FFFFFF",
            )
            axes[2].bar_label(
                bars,
                labels=[f"{value:+.5f}" for value in gate_values[seed]],
                padding=3,
                fontsize=8.0,
                rotation=90,
            )
        axes[2].axhline(0.0, color="#334155", linewidth=1.0)
        gate_flat = gate_values[0] + gate_values[1] + [0.0]
        gate_span = max(0.002, max(gate_flat) - min(gate_flat))
        axes[2].set_ylim(
            min(gate_flat) - 0.35 * gate_span,
            max(gate_flat) + 0.55 * gate_span,
        )
        axes[2].set_xticks(role_x, ("正确拓扑", "错误拓扑", "无拓扑"))
        axes[2].set_ylabel("tanh(门控参数)")
        axes[2].set_title("训练后拓扑修正强度", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[2].legend(frameon=False, loc="upper right", fontsize=8.2)

        figure.text(
            0.055,
            0.035,
            "本结果只裁决 E5 是否保留 Dialga 四轮机制；无论通过与否，都不重新开放五轮 E5。",
            ha="left",
            color="#334155",
            fontsize=9.8,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("r4_regression_supported"):
        return "通过：E5 保留四轮强信号和两种正确拓扑优势；仅保留架构实现。"
    if decision.endswith("protocol_invalid"):
        return "协议无效：修复冻结检查前，当前数值不可解释。"
    return "未通过：E5 未稳定保留四轮 D1 机制，停止后续 E5 实验。"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["main", "parse_args", "render_dialga_d7_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
