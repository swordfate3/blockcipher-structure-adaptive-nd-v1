from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.plot_runtime_spn_permutation_control_identifiability_k1by4 import (
    render_k1by4_svg,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_permutation_control_identifiability_k1by4 import (
    CONFIG_PATH,
    ROOT,
    RUN_ID,
    load_and_validate_config,
    run_audit,
)


DEFAULT_OUTPUT = ROOT / "outputs/local_audit" / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit K1-BY4 PRESENT permutation-control representation identifiability."
        )
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(args.config)
    payload = run_audit(
        config,
        output_root=args.output_root,
        project_root=ROOT,
        device=args.device,
    )
    render_report = render_k1by4_svg(
        payload["gate"],
        args.output_root / "curves.svg",
    )
    (args.output_root / "visual_qa_render_report.json").write_text(
        json.dumps(render_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if payload["gate"]["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "main", "parse_args"]
