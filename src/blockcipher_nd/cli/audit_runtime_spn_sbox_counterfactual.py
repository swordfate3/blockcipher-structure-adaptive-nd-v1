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

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_counterfactual import (
    adjudicate_uknit_same_checkpoint_counterfactual,
    evaluate_same_checkpoint_pair,
    file_sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit uKNIT S-box assignment with frozen U1 checkpoints."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--u1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_rows = _read_jsonl(args.u1_root / "results.jsonl")
    result_rows: list[dict[str, Any]] = []
    _write_progress(args.output_root / "progress.jsonl", "run_start", args.run_id)

    descriptor_path = Path("configs/runtime/spn/uknit64.json")
    descriptor = load_runtime_spn_descriptor(
        descriptor_path,
        rounds=2,
        round_start=2,
    )
    correct_structure = descriptor.structure
    shuffled_structure = correct_structure.shuffled_sbox_assignments(20260724)

    for seed in (0, 1):
        dataset, feature_path, label_path = _load_validation_dataset(
            source_rows,
            seed,
        )
        for source_role in ("candidate", "anchor"):
            source = _source_row(source_rows, seed, source_role)
            checkpoint_path = Path(source["training"]["checkpoint_output"])
            pair_rows = evaluate_same_checkpoint_pair(
                seed=seed,
                source_role=source_role,
                model_options=dict(source["training"]["model_options"]),
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                correct_structure=correct_structure,
                shuffled_structure=shuffled_structure,
                checkpoint_sha256=file_sha256(checkpoint_path),
                feature_sha256=file_sha256(feature_path),
                label_sha256=file_sha256(label_path),
                descriptor_sha256=descriptor.sha256,
                batch_size=args.batch_size,
                device=args.device,
            )
            result_rows.extend(pair_rows)
            _write_progress(
                args.output_root / "progress.jsonl",
                "pair_done",
                args.run_id,
                seed=seed,
                source_role=source_role,
            )

    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    gate = adjudicate_uknit_same_checkpoint_counterfactual(
        run_id=args.run_id,
        rows=result_rows,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {"u1_results": str(args.u1_root / "results.jsonl")},
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": "uKNIT-BC",
        "training_performed": False,
        "source_run": str(args.u1_root),
        "gate": gate,
    }
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_counterfactual_svg(gate, args.output_root / "curves.svg")
    _write_progress(
        args.output_root / "progress.jsonl",
        "run_done",
        args.run_id,
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_counterfactual_svg(gate: dict[str, Any], output_path: Path) -> None:
    labels = ("seed0 候选", "seed0 锚点", "seed1 候选", "seed1 锚点")
    keys = ("seed0_candidate", "seed0_anchor", "seed1_candidate", "seed1_anchor")
    correct = [gate["pair_results"][key]["correct_auc"] for key in keys]
    shuffled = [gate["pair_results"][key]["shuffled_auc"] for key in keys]
    margins = [
        gate["pair_results"][f"seed{seed}_candidate"][
            "correct_minus_shuffled_auc"
        ]
        for seed in (0, 1)
    ]
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
        figure, axes = plt.subplots(1, 2, figsize=(13.6, 6.6))
        figure.subplots_adjust(left=0.075, right=0.97, top=0.72, bottom=0.16, wspace=0.3)
        figure.suptitle(
            "创新1 uKNIT-BC：同一权重下的 S盒归属反事实审计（U2-A）",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.885,
            "冻结 U1 最佳 checkpoint 与验证数据；每组只切换正确/打乱 S盒到 cell 的归属，不重新训练。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.075,
            0.82,
            f"裁决：{_decision_text(gate)}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )

        x = np.arange(len(labels), dtype=np.float64)
        width = 0.34
        left = axes[0].bar(
            x - width / 2,
            correct,
            width,
            label="正确归属",
            color="#2563EB",
        )
        right = axes[0].bar(
            x + width / 2,
            shuffled,
            width,
            label="归属打乱",
            color="#D97706",
        )
        axes[0].bar_label(left, labels=[f"{value:.4f}" for value in correct], padding=4)
        axes[0].bar_label(
            right,
            labels=[f"{value:.4f}" for value in shuffled],
            padding=15,
        )
        axes[0].set_title("同 checkpoint 验证 AUC", loc="left", fontweight="bold")
        axes[0].set_ylabel("AUC")
        axes[0].set_xticks(x, labels)
        axes[0].axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)
        lower = max(0.0, min(0.5, *correct, *shuffled) - 0.006)
        upper = min(1.0, max(*correct, *shuffled) + 0.009)
        if upper - lower < 0.05:
            upper = min(1.0, lower + 0.05)
        axes[0].set_ylim(lower, upper)
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper left")

        bars = axes[1].bar(
            np.arange(2),
            margins,
            width=0.55,
            color=("#059669" if margins[0] >= 0.005 else "#DC2626", "#059669" if margins[1] >= 0.005 else "#DC2626"),
        )
        axes[1].bar_label(bars, labels=[f"{value:+.6f}" for value in margins], padding=5)
        axes[1].axhline(0.0, color="#64748B", linewidth=1)
        axes[1].axhline(0.005, color="#2563EB", linestyle="--", linewidth=1.2, label="通过门槛 +0.005")
        axes[1].set_title("候选：正确归属 - 打乱归属", loc="left", fontweight="bold")
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_xticks(np.arange(2), ("seed0 候选", "seed1 候选"))
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper left")
        axes[1].margins(y=0.25)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    if gate["status"] == "pass":
        return "两颗 seed 都依赖正确归属，可设计同预算配对训练。"
    if gate["decision"].endswith("invariance_control_failed"):
        return "全局锚点不满足归属不变性，审计实现需先修复。"
    if gate["status"] == "fail":
        return "协议检查失败，AUC 不得用于模型判断。"
    return "当前 late_cell 权重未稳定使用归属信息，停止扩样并重设计交互。"


def _source_row(
    rows: list[dict[str, Any]],
    seed: int,
    source_role: str,
) -> dict[str, Any]:
    context = "late_cell" if source_role == "candidate" else "late_pair"
    matches = [
        row
        for row in rows
        if row.get("seed") == seed
        and row.get("runtime_structure_mode") == "true"
        and row.get("training", {}).get("model_options", {}).get("sbox_context_mode")
        == context
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one U1 source row for seed={seed} role={source_role}, got {len(matches)}"
        )
    return matches[0]


def _load_validation_dataset(
    source_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[DiskDifferentialDataset, Path, Path]:
    source = _source_row(source_rows, seed, "candidate")
    cache_root = Path(source["training"]["dataset_cache_root"])
    validation_root = cache_root / "uknit64" / "r4" / "validation"
    matches = list(validation_root.glob(f"seed-{10000 + seed}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one validation cache for seed {seed}, got {len(matches)}")
    cache_dir = matches[0]
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = DiskDifferentialDataset(
        features=np.load(feature_path, mmap_mode="r"),
        labels=np.load(label_path, mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, feature_path, label_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress(path: Path, event: str, run_id: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    "run_id": run_id,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
