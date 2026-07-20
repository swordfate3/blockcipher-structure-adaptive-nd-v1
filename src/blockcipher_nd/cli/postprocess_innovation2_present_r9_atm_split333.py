from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.index_results import main as index_results_main
from blockcipher_nd.cli.plot_innovation2_present_r9_atm_source_heldout_ranking import (
    main as plot_e105_main,
)
from blockcipher_nd.cli.run_innovation2_present_r9_atm_source_heldout_ranking import (
    main as run_e105_main,
)
from blockcipher_nd.cli.validate_innovation2_present_r9_atm_split333_retrieval import (
    main as validate_e104_main,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate E104 and run frozen E105 source-heldout postprocessing."
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--verified-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--public-results-root", required=True, type=Path)
    parser.add_argument("--e105-output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation_code = validate_e104_main(
        [
            "--raw-root",
            str(args.raw_root),
            "--verified-root",
            str(args.verified_root),
        ]
    )
    if validation_code != 0:
        _write_report(
            args.raw_root / "postprocess.local.json",
            {
                "status": "fail",
                "stage": "e104_validation",
                "e105_started": False,
                "visual_qa_required": False,
            },
        )
        return validation_code

    evaluation_code = run_e105_main(
        [
            "evaluate",
            "--results-root",
            str(args.public_results_root),
            "--checkpoint-root",
            str(args.checkpoint_root),
            "--e104-root",
            str(args.verified_root / "results"),
            "--output-root",
            str(args.e105_output_root),
            "--device",
            args.device,
        ]
    )
    if evaluation_code != 0:
        report_path = args.e105_output_root / "postprocess.json"
        report = {
            "status": "fail",
            "stage": "e105_evaluation",
            "e104_verified": True,
            "result_index_refreshed": False,
            "visual_qa_required": False,
        }
        _write_report(report_path, report)
        index_code = index_results_main([])
        report["result_index_refreshed"] = index_code == 0
        _write_report(report_path, report)
        return evaluation_code if index_code == 0 else index_code

    plot_code = plot_e105_main(
        [
            "--summary",
            str(args.e105_output_root / "summary.json"),
            "--output",
            str(args.e105_output_root / "curves.svg"),
        ]
    )
    if plot_code != 0:
        report_path = args.e105_output_root / "postprocess.json"
        report = {
            "status": "fail",
            "stage": "e105_plot",
            "e104_verified": True,
            "e105_evaluated": True,
            "result_index_refreshed": False,
            "visual_qa_required": True,
        }
        _write_report(report_path, report)
        index_code = index_results_main([])
        report["result_index_refreshed"] = index_code == 0
        _write_report(report_path, report)
        return plot_code if index_code == 0 else index_code

    index_code = index_results_main([])
    report = {
        "status": "awaiting_visual_qa" if index_code == 0 else "fail",
        "stage": "visual_qa",
        "e104_verified": True,
        "e105_evaluated": True,
        "plot_generated": True,
        "result_index_refreshed": index_code == 0,
        "visual_qa_required": True,
        "visual_qa_passed": False,
        "next_action": (
            "render curves.svg to pixels and run visual-qa-redraw before final adjudication"
        ),
    }
    _write_report(args.e105_output_root / "postprocess.json", report)
    return 0 if index_code == 0 else index_code


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
