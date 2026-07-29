from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_exact_gf2_operator_response_k1bh import (
    render_k1bh_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_exact_gf2_operator_response_k1bh import (
    CONFIG_PATH,
    ROOT,
    RUN_ID,
    load_and_validate_config,
    run_audit,
)


DEFAULT_OUTPUT = ROOT / "outputs/local_audit" / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the K1-BH exact GF(2) operator-response audit."
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
    render_report = render_k1bh_svg(
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
