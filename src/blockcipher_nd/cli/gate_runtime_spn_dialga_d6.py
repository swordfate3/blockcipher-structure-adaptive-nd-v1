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

from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    adjudicate_runtime_spn_dialga_d3,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d6 import (
    SEEDS,
    adjudicate_runtime_spn_dialga_d6,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate Dialga-128 Runtime-E5 D6 gated residuals."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--d3-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = args.run_root / "results.jsonl"
    progress_path = args.run_root / "progress.jsonl"
    d3_results_path = args.d3_root / "results.jsonl"
    d3_gate_path = args.d3_root / "gate.json"
    d3_validation_path = args.d3_root / "validation.json"

    rows = _read_jsonl(results_path)
    progress_rows = _read_jsonl(progress_path)
    d3_rows = _read_jsonl(d3_results_path)
    persisted_d3_gate = _read_json(d3_gate_path)
    d3_validation = _read_json(d3_validation_path)
    replayed_d3_gate = adjudicate_runtime_spn_dialga_d3(
        run_id=str(persisted_d3_gate.get("run_id", "")),
        rows=d3_rows,
    )
    expected_cache_root = args.d3_root / "cache"
    cache_audit = audit_d3_cache_reuse(
        progress_rows=progress_rows,
        expected_cache_root=expected_cache_root,
    )
    gate = adjudicate_runtime_spn_dialga_d6(
        run_id=args.run_id,
        rows=rows,
        d3_rows=d3_rows,
        persisted_d3_gate=persisted_d3_gate,
        replayed_d3_gate=replayed_d3_gate,
        d3_validation=d3_validation,
        expected_cache_root=expected_cache_root,
        cache_audit=cache_audit,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
        "source_d3": {
            "root": str(args.d3_root),
            "results_sha256": _sha256(d3_results_path),
            "gate_sha256": _sha256(d3_gate_path),
            "validation_sha256": _sha256(d3_validation_path),
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
        "source_d3_root": str(args.d3_root),
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    write_history_csv(results_path, args.run_root / "history.csv")
    render_dialga_d6_svg(gate, args.run_root / "curves.svg")
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


def audit_d3_cache_reuse(
    *,
    progress_rows: list[dict[str, Any]],
    expected_cache_root: Path,
) -> dict[str, Any]:
    expected_root = expected_cache_root.resolve()
    expected_cache_paths = {
        path.parent.resolve() for path in expected_cache_root.rglob("metadata.json")
    }
    cache_events = [
        row for row in progress_rows if row.get("stage") == "dataset_cache"
    ]
    reuse_events = [row for row in cache_events if row.get("event") == "cache_reuse"]
    generation_events = [
        row for row in cache_events if row.get("event") != "cache_reuse"
    ]
    observed_paths: set[Path] = set()
    malformed_paths = 0
    for event in reuse_events:
        value = event.get("cache_path")
        if not isinstance(value, str) or not value:
            malformed_paths += 1
            continue
        observed_paths.add(Path(value).resolve())
    index_split_pairs = {
        (int(row.get("index", -1)), str(row.get("split", "")))
        for row in reuse_events
    }
    expected_index_split_pairs = {
        (index, split) for index in range(1, 7) for split in ("train", "validation")
    }
    checks = {
        "four_d3_cache_directories_present": len(expected_cache_paths) == 4,
        "twelve_reuse_events_complete": len(reuse_events) == 12,
        "six_rows_both_splits_reused": index_split_pairs
        == expected_index_split_pairs,
        "no_dataset_generation_events": not generation_events,
        "all_reuse_paths_well_formed": malformed_paths == 0,
        "reuse_paths_are_exact_d3_cache_leaves": observed_paths
        == expected_cache_paths,
        "reuse_paths_stay_under_d3_cache_root": all(
            path.is_relative_to(expected_root) for path in observed_paths
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "expected_cache_root": str(expected_cache_root),
        "expected_cache_paths": sorted(str(path) for path in expected_cache_paths),
        "observed_cache_paths": sorted(str(path) for path in observed_paths),
        "reuse_event_count": len(reuse_events),
        "generation_event_count": len(generation_events),
    }


def render_dialga_d6_svg(gate: dict[str, Any], output_path: Path) -> None:
    roles = ("correct", "corrupted", "no_topology")
    role_labels = ("正确拓扑", "错误拓扑", "无拓扑")
    role_colors = ("#059669", "#DC2626", "#64748B")
    auc_values = {
        seed: [float(gate["aucs"][f"seed{seed}"][role]) for role in roles]
        for seed in SEEDS
    }
    margin_keys = (
        "correct_minus_corrupted",
        "correct_minus_no_topology",
        "correct_minus_d3",
    )
    margin_labels = ("正确 - 错误", "正确 - 无拓扑", "正确 - 旧D3")
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
            left=0.055, right=0.985, top=0.70, bottom=0.17, wspace=0.28
        )
        figure.suptitle(
            "创新1 D6：Dialga 五轮门控拓扑残差诊断",
            x=0.055,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.89,
            "固定 D3 数据、密钥、差分和训练预算；仅把网络改为“独立主干 + 有界拓扑修正”。",
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

        x = np.arange(3, dtype=np.float64)
        width = 0.34
        for seed, offset, alpha in ((0, -width / 2, 1.0), (1, width / 2, 0.68)):
            bars = axes[0].bar(
                x + offset,
                auc_values[seed],
                width,
                label=f"seed{seed}",
                color=role_colors,
                alpha=alpha,
                edgecolor="#FFFFFF",
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in auc_values[seed]],
                padding=3,
                fontsize=8.0,
                rotation=90,
            )
        auc_flat = auc_values[0] + auc_values[1] + [0.5, 0.52]
        axes[0].axhline(0.5, color="#475569", linestyle=":", linewidth=1.1)
        axes[0].axhline(
            0.52, color="#2563EB", linestyle="--", linewidth=1.2, label="正确拓扑门槛 0.52"
        )
        axes[0].set_ylim(max(0.45, min(auc_flat) - 0.03), min(1.0, max(auc_flat) + 0.07))
        axes[0].set_xticks(x, role_labels)
        axes[0].set_ylabel("最佳验证 AUC")
        axes[0].set_title("同预算三种拓扑角色", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.2)

        margin_x = np.arange(3, dtype=np.float64)
        for seed, offset, color in ((0, -width / 2, "#2563EB"), (1, width / 2, "#D97706")):
            bars = axes[1].bar(
                margin_x + offset,
                margin_values[seed],
                width,
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
            0.005, color="#059669", linestyle="--", linewidth=1.2, label="控制差门槛 +0.005"
        )
        axes[1].axhline(
            0.010, color="#7C3AED", linestyle=":", linewidth=1.3, label="超过旧D3 +0.010"
        )
        margin_flat = margin_values[0] + margin_values[1] + [0.0, 0.005, 0.01]
        span = max(0.02, max(margin_flat) - min(margin_flat))
        axes[1].set_ylim(min(margin_flat) - 0.25 * span, max(margin_flat) + 0.45 * span)
        axes[1].set_xticks(margin_x, margin_labels)
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title("必须逐 seed 同时通过的边际", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper left", fontsize=8.2)

        for seed, offset, alpha in ((0, -width / 2, 1.0), (1, width / 2, 0.68)):
            bars = axes[2].bar(
                x + offset,
                gate_values[seed],
                width,
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
        axes[2].set_xticks(x, role_labels)
        axes[2].set_ylabel("tanh(门控参数)")
        axes[2].set_title("训练后拓扑修正强度", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[2].legend(frameon=False, loc="upper right", fontsize=8.2)

        figure.text(
            0.055,
            0.055,
            "本结果是 Dialga-128 prefix-r5、2048/class 的本地架构诊断，不是正式规模或攻击结果。",
            ha="left",
            color="#334155",
            fontsize=9.8,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("gated_residual_supported"):
        return "通过：正确拓扑逐 seed 优于两种控制，并超过旧 D3。"
    if decision.endswith("base_improvement_without_topology_attribution"):
        return "暂留：独立主干有所改善，但拓扑归因边际不完整。"
    if decision.endswith("protocol_invalid"):
        return "协议无效：先修复冻结检查，当前数值不可解释。"
    return "未通过：五轮正确拓扑未在两颗 seed 上形成稳定优势。"


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


__all__ = ["audit_d3_cache_reuse", "main", "parse_args", "render_dialga_d6_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
