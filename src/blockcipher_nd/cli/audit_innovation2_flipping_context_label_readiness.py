from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_context_label_readiness import (
    _read_csv,
    _read_json,
    _write_csv,
    _write_json,
    _write_progress,
    render_context_label_svg,
)
from blockcipher_nd.tasks.innovation2.integral_context_label_readiness import (
    FlippingContextLabelReadinessConfig,
    run_flipping_context_label_readiness_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit equal-prevalence flipping context-mask labels."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--membership-count", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = FlippingContextLabelReadinessConfig(
        run_id=args.run_id,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
        membership_count=args.membership_count,
    )
    source_gate = _read_json(args.source_root / "gate.json")
    source_metadata = _read_json(args.source_root / "metadata.json")
    source_basis_rows = _read_csv(args.source_root / "kernel_basis.csv")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_run_id": source_gate.get("run_id"),
            "membership_count": args.membership_count,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_flipping_context_label_readiness_audit(
        config,
        source_gate=source_gate,
        source_metadata=source_metadata,
        source_basis_rows=source_basis_rows,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "labels.csv", result["label_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_context_label_svg(
        result["rows"],
        result["label_rows"],
        result["gate"],
        args.output_root / "curves.svg",
        title="创新2 E17b：等流行率翻转-mask 标签捷径审计",
        subtitle=(
            "固定选择32个恰在4/16个 context 中平衡的 mask，共512行；"
            "不重新加密、不训练网络。"
        ),
        primary_stop=0.75,
        primary_stop_label="E17b 捷径 AUC 停止线 0.75",
        secondary_stop=None,
    )
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "positive_rate": result["gate"]["positive_rate"],
                "flipping_masks": result["gate"]["flipping_masks"],
                "distinct_context_label_signatures": result["gate"][
                    "distinct_context_label_signatures"
                ],
                "baseline_accuracies": result["gate"]["baseline_accuracies"],
                "baseline_aucs": result["gate"]["baseline_aucs"],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
