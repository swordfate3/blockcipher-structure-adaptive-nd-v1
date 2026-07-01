from __future__ import annotations

import json

from blockcipher_training_accelerator.runner import run_accelerated_matrix


def test_run_accelerated_matrix_writes_compatible_jsonl_with_accelerator_metadata(tmp_path):
    output = tmp_path / "results.jsonl"

    rows = run_accelerated_matrix(
        [
            "--ciphers",
            "speck32",
            "--models",
            "mlp",
            "--rounds",
            "1",
            "--seeds",
            "0",
            "--samples-per-class",
            "8",
            "--pairs-per-sample",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-bits",
            "8",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--speed-profile",
            "baseline",
        ]
    )

    assert len(rows) == 1
    payload = json.loads(output.read_text(encoding="utf-8").strip())
    assert payload["cipher_key"] == "speck32"
    assert "metrics" in payload
    assert payload["training"]["accelerator"]["profile"] == "baseline"
    assert payload["training"]["accelerator"]["duration_seconds"] >= 0.0
