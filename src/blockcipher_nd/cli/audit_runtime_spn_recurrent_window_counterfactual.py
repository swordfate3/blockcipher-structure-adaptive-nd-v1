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
from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window import (
    adjudicate_runtime_spn_recurrent_window,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window_counterfactual import (
    EXPECTED_CONDITIONS,
    FROZEN_MODEL_OPTIONS,
    U3_RUN_ID,
    adjudicate_same_checkpoint_window_panel,
    adjudicate_window_counterfactual_source,
    evaluate_same_checkpoint_window_panel,
    file_sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit U3 recurrent-window use with frozen candidate checkpoints."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--u3-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": args.u3_root / "results.jsonl",
        "gate": args.u3_root / "gate.json",
        "validation": args.u3_root / "validation.json",
        "plan_validation": args.u3_root / "plan_validation.json",
    }
    source_rows = _read_jsonl(paths["results"])
    persisted_gate = _read_json(paths["gate"])
    validation = _read_json(paths["validation"])
    plan_validation = _read_json(paths["plan_validation"])
    replayed_gate = adjudicate_runtime_spn_recurrent_window(
        run_id=U3_RUN_ID,
        rows=source_rows,
    )
    candidates = _candidate_rows(source_rows)
    candidate_rows_valid = len(candidates) == 2 and all(
        _candidate_row_matches_contract(row, seed)
        for seed, row in enumerate(candidates)
    )
    candidate_checkpoints_exist = candidate_rows_valid and all(
        Path(row["training"]["checkpoint_output"]).is_file() for row in candidates
    )
    source_gate = adjudicate_window_counterfactual_source(
        run_id=args.run_id,
        persisted_gate=persisted_gate,
        replayed_gate=replayed_gate,
        validation=validation,
        plan_validation=plan_validation,
        result_rows_count=len(source_rows),
        candidate_rows_valid=candidate_rows_valid,
        candidate_checkpoints_exist=candidate_checkpoints_exist,
        visual_qa_passed=(args.u3_root / "visual_qa_passed.marker").is_file(),
        plan_sha256=_safe_sha256(args.plan),
    )
    source_gate["source_paths"] = {
        key: str(path) for key, path in paths.items()
    } | {
        "plan": str(args.plan),
        "visual_qa": str(args.u3_root / "visual_qa_passed.marker"),
    }
    _write_json(args.output_root / "source_gate.json", source_gate)
    if not source_gate["execution_authorized"]:
        print(json.dumps(source_gate, ensure_ascii=False, sort_keys=True))
        return 4

    _write_progress(args.output_root / "progress.jsonl", "run_start", args.run_id)
    source_hashes = {key: file_sha256(path) for key, path in paths.items()}
    descriptor = load_runtime_spn_descriptor(
        "configs/runtime/spn/uknit64.json",
        rounds=2,
        round_start=3,
    )
    result_rows: list[dict[str, Any]] = []
    for seed, source in enumerate(candidates):
        dataset, feature_path, label_path, metadata_path = _load_validation_dataset(
            source
        )
        checkpoint_path = Path(source["training"]["checkpoint_output"])
        result_rows.extend(
            evaluate_same_checkpoint_window_panel(
                seed=seed,
                source_row=source,
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                correct_structure=descriptor.structure,
                checkpoint_sha256=file_sha256(checkpoint_path),
                feature_path=feature_path,
                feature_sha256=file_sha256(feature_path),
                label_path=label_path,
                label_sha256=file_sha256(label_path),
                metadata_path=metadata_path,
                metadata_sha256=file_sha256(metadata_path),
                descriptor_sha256=descriptor.sha256,
                source_hashes=source_hashes,
                batch_size=args.batch_size,
                device=args.device,
            )
        )
        _write_progress(
            args.output_root / "progress.jsonl",
            "seed_done",
            args.run_id,
            seed=seed,
        )

    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    gate = adjudicate_same_checkpoint_window_panel(
        run_id=args.run_id,
        rows=result_rows,
    )
    result_validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_gate": str(args.output_root / "source_gate.json"),
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": gate["cipher"],
        "training_performed": False,
        "source_run": str(args.u3_root),
        "samples_total_per_seed": 2048,
        "seeds": [0, 1],
        "conditions": list(EXPECTED_CONDITIONS),
        "gate": gate,
    }
    _write_json(args.output_root / "validation.json", result_validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_window_counterfactual_svg(gate, args.output_root / "curves.svg")
    _write_progress(
        args.output_root / "progress.jsonl",
        "run_done",
        args.run_id,
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_window_counterfactual_svg(
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    condition_labels = ("完整正确窗口", "重复末层", "错误拓扑", "无拓扑")
    colors = ("#2563EB", "#D97706", "#DC2626", "#64748B")
    seed_results = gate["seed_results"]
    aucs = {
        seed: tuple(
            seed_results[str(seed)][f"{condition}_auc"]
            for condition in EXPECTED_CONDITIONS
        )
        for seed in (0, 1)
    }
    margins = tuple(
        seed_results[str(seed)][f"full_minus_{condition}"]
        for seed in (0, 1)
        for condition in ("repeat_last", "corrupted", "no_topology")
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
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.2))
        figure.subplots_adjust(
            left=0.075,
            right=0.97,
            top=0.73,
            bottom=0.17,
            wspace=0.3,
        )
        figure.suptitle(
            "创新1 U4：同一权重是否真正使用完整的两轮 SPN 窗口",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            "冻结 U3 两颗 seed 的候选最佳权重和验证数据；只替换运行时窗口或拓扑关系，不重新训练。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.075,
            0.83,
            f"裁决：{_decision_text(gate)}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )

        x = np.arange(2, dtype=np.float64)
        width = 0.18
        for index, (label, color) in enumerate(
            zip(condition_labels, colors, strict=True)
        ):
            values = [aucs[seed][index] for seed in (0, 1)]
            bars = axes[0].bar(
                x + (index - 1.5) * width,
                values,
                width,
                label=label,
                color=color,
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=4 + index * 7,
                fontsize=8.5,
            )
        flat_aucs = [value for values in aucs.values() for value in values]
        lower = max(0.0, min(0.5, *flat_aucs) - 0.008)
        upper = min(1.0, max(*flat_aucs) + 0.018)
        if upper - lower < 0.06:
            upper = min(1.0, lower + 0.06)
        axes[0].set_ylim(lower, upper)
        axes[0].set_title("冻结 checkpoint 的验证 AUC", loc="left", fontweight="bold")
        axes[0].set_ylabel("AUC")
        axes[0].set_xticks(x, ("seed0", "seed1"))
        axes[0].axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper left", ncols=2)

        margin_labels = tuple(
            f"seed{seed}\n{label}"
            for seed in (0, 1)
            for label in ("对重复末层", "对错误拓扑", "对无拓扑")
        )
        bars = axes[1].bar(
            np.arange(6),
            margins,
            width=0.64,
            color=["#059669" if value >= 0.005 else "#DC2626" for value in margins],
        )
        axes[1].bar_label(
            bars,
            labels=[f"{value:+.6f}" for value in margins],
            padding=5,
            fontsize=8.5,
        )
        axes[1].axhline(0.0, color="#64748B", linewidth=1)
        axes[1].axhline(
            0.005,
            color="#2563EB",
            linestyle="--",
            linewidth=1.2,
            label="通过门槛 +0.005",
        )
        axes[1].set_title("完整窗口相对三个控制的 AUC 差值", loc="left", fontweight="bold")
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_xticks(np.arange(6), margin_labels)
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper left")
        axes[1].margins(y=0.25)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    if gate["status"] == "pass":
        return "两颗 seed 的同一权重都依赖完整正确窗口，可进入跨密码权重复用门。"
    if gate["status"] == "fail":
        return "协议或来源回放失败，AUC 不得用于研究判断。"
    return "同一权重下窗口优势未稳定保留，停止扩样并重设计局部交互。"


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = [
        row
        for row in rows
        if row.get("seed") in (0, 1)
        and row.get("model") == "runtime_spn_e4_equivariant_true"
        and row.get("training", {}).get("model_options") == FROZEN_MODEL_OPTIONS
    ]
    return sorted(matches, key=lambda row: int(row["seed"]))


def _candidate_row_matches_contract(row: dict[str, Any], seed: int) -> bool:
    training = row.get("training", {})
    validation = row.get("validation", {})
    return bool(
        row.get("seed") == seed
        and row.get("cipher_key") == "uknit64"
        and row.get("rounds") == 5
        and row.get("samples_per_class") == 2048
        and row.get("pairs_per_sample") == 4
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and validation.get("samples_per_class") == 1024
        and validation.get("samples_total") == 2048
        and training.get("selected_checkpoint") == "best"
        and training.get("restore_best_checkpoint") is True
        and training.get("validation_rows") == 2048
        and isinstance(training.get("checkpoint_output"), str)
        and bool(training.get("checkpoint_output"))
    )


def _load_validation_dataset(
    source: dict[str, Any],
) -> tuple[DiskDifferentialDataset, Path, Path, Path]:
    seed = int(source["seed"])
    training = source["training"]
    validation_root = (
        Path(training["dataset_cache_root"])
        / str(source["cipher_key"])
        / f"r{source['rounds']}"
        / "validation"
    )
    matches = list(validation_root.glob(f"seed-{10000 + seed}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one U3 validation cache for seed {seed}, got {len(matches)}"
        )
    cache_dir = matches[0]
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    dataset = DiskDifferentialDataset(
        features=np.load(feature_path, mmap_mode="r"),
        labels=np.load(label_path, mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, feature_path, label_path, metadata_path


def _safe_sha256(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return []
    return values if all(isinstance(value, dict) for value in values) else []


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
