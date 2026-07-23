from __future__ import annotations

import json

import numpy as np

from blockcipher_nd.cli import audit_runtime_spn_skinny_readiness as cli
from blockcipher_nd.planning.matrix import cipher_key_from_name
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_readiness import (
    SkinnyRuntimeReadinessConfig,
    run_skinny_runtime_readiness,
)


def test_skinny_is_registered_in_standard_cipher_and_plan_factories() -> None:
    cipher = build_cipher("skinny64", rounds=32, key=0xF5269826FC681238)

    assert cipher.encrypt(0x06034F957724D19D) == 0xBB39DFB2429B8AC7
    assert cipher_key_from_name("SKINNY-64/64") == "skinny64"
    assert default_difference("skinny64") == 0x40


def test_skinny_general_gf2_readiness_passes_every_frozen_gate(tmp_path) -> None:
    result = run_skinny_runtime_readiness(
        SkinnyRuntimeReadinessConfig(run_id="unit"),
        cache_root=tmp_path / "cache",
    )

    gate = result["gate"]
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_runtime_spn_skinny_general_gf2_data_ready"
    )
    assert gate["checks_passed"] == gate["checks_total"]
    assert gate["training_performed"] is False
    assert gate["empirical_topology_superiority_tested"] is False
    assert set(gate["category_counts"]) == {
        "cipher_factory",
        "strict_dataset",
        "cache_replay",
        "runtime_model",
        "general_gf2",
    }
    assert max(result["summary"]["linear_row_degrees"]) > 1
    for split, rows in (("train", 128), ("validation", 64)):
        features = np.load(tmp_path / "cache" / split / "features.npy")
        labels = np.load(tmp_path / "cache" / split / "labels.npy")
        assert features.shape == (rows, 512)
        assert labels.shape == (rows,)


def test_skinny_general_gf2_readiness_cli_writes_complete_artifacts(
    tmp_path,
) -> None:
    output = tmp_path / "readiness"
    exit_code = cli.main(["--run-id", "cli-unit", "--output-root", str(output)])

    assert exit_code == 0
    assert {
        "results.jsonl",
        "progress.jsonl",
        "metadata.json",
        "summary.json",
        "gate.json",
        "curves.svg",
        "cache",
    } <= {path.name for path in output.iterdir()}
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    svg = (output / "curves.svg").read_text(encoding="utf-8")
    assert "SKINNY 一般 GF(2) 数据通路就绪审计" in svg
    assert "本轮不训练神经网络" in svg
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress[0]["event"] == "run_start"
    assert progress[-1]["event"] == "run_done"
