from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.difference_screen_gate import DEFAULT_REFERENCE_DIFFERENCE


DEFAULT_CONFIRMATION_SAMPLES_PER_CLASS = 262144


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lean PRESENT r9 input-difference confirmation CSV from a screen CSV."
    )
    parser.add_argument("--screen-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-difference", required=True)
    parser.add_argument("--reference-difference", default=DEFAULT_REFERENCE_DIFFERENCE)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_CONFIRMATION_SAMPLES_PER_CLASS)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args(argv)


def create_difference_confirmation_plan(
    *,
    screen_plan_path: Path,
    output_path: Path,
    selected_difference: str,
    reference_difference: str = DEFAULT_REFERENCE_DIFFERENCE,
    samples_per_class: int = DEFAULT_CONFIRMATION_SAMPLES_PER_CLASS,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    fieldnames, rows = _read_csv(screen_plan_path)
    by_difference = {_difference_id(row): row for row in rows}
    if len(by_difference) != len(rows):
        raise ValueError("screen plan contains duplicate difference_profile:difference_member rows")
    if reference_difference not in by_difference:
        raise ValueError(f"reference difference not found in screen plan: {reference_difference}")
    if selected_difference not in by_difference:
        raise ValueError(f"selected difference not found in screen plan: {selected_difference}")
    if selected_difference == reference_difference:
        raise ValueError("selected_difference must be a non-reference candidate")

    selected_slug = _slug(selected_difference)
    output_rows = [
        _confirmation_row(
            by_difference[reference_difference],
            architecture_rank=0,
            role="reference",
            selected_slug=selected_slug,
            samples_per_class=samples_per_class,
        ),
        _confirmation_row(
            by_difference[selected_difference],
            architecture_rank=1,
            role="selected",
            selected_slug=selected_slug,
            samples_per_class=samples_per_class,
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "status": "pass",
        "screen_plan": str(screen_plan_path),
        "output": str(output_path),
        "selected_difference": selected_difference,
        "reference_difference": reference_difference,
        "samples_per_class": samples_per_class,
        "rows": len(output_rows),
        "claim_scope": (
            "PRESENT r9 input-difference confirmation matrix; data-construction follow-up only, "
            "not formal or breakthrough evidence until completed and retrieved"
        ),
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"screen plan has no rows: {path}")
        return list(reader.fieldnames), rows


def _confirmation_row(
    row: dict[str, str],
    *,
    architecture_rank: int,
    role: str,
    selected_slug: str,
    samples_per_class: int,
) -> dict[str, str]:
    updated = dict(row)
    updated["architecture_rank"] = str(architecture_rank)
    updated["samples_per_class"] = str(samples_per_class)
    updated["family"] = f"present_nibble_invp_pair_consistency_r9_difference_confirmation_{role}_{selected_slug}"
    updated["network"] = f"{row.get('network', '').rstrip()}-262k-confirm-{role}".strip("-")
    updated["evidence"] = (
        f"MEDIUM {samples_per_class}/class r9 input-difference confirmation; "
        "fixed pair-consistency model; changes data construction only; not formal evidence"
    )
    return updated


def _difference_id(row: dict[str, str]) -> str:
    return f"{row.get('difference_profile', '')}:{row.get('difference_member', '')}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = create_difference_confirmation_plan(
        screen_plan_path=args.screen_plan,
        output_path=args.output,
        selected_difference=args.selected_difference,
        reference_difference=args.reference_difference,
        samples_per_class=args.samples_per_class,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
