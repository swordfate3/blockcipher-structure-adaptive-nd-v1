from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.engine.checkpoint_initialization import file_sha256
from blockcipher_nd.planning.cross_spn_e6_readiness import (
    E6_SOURCE_MODELS,
    E6_TARGET_MODELS,
    build_e6_readiness_manifest,
)


def test_e6_readiness_manifest_pins_all_source_checkpoints(tmp_path: Path) -> None:
    source_rows = {}
    for index, role in enumerate(("off", "candidate", "placebo"), start=1):
        checkpoint = tmp_path / f"source-{role}.pt"
        torch.save({"state_dict": {"weight": torch.tensor([float(index)])}}, checkpoint)
        source_rows[E6_SOURCE_MODELS[role]] = {
            "training": {"checkpoint_output": str(checkpoint)}
        }

    manifest = build_e6_readiness_manifest(
        source_rows,
        tmp_path / "results.jsonl",
    )

    assert manifest["version"] == 1
    assert set(manifest["targets"]) == set(E6_TARGET_MODELS.values())
    assert manifest["targets"][E6_TARGET_MODELS["scratch"]] == {
        "kind": "scratch",
        "target_mapping": "true",
    }
    for role in ("off", "candidate", "placebo"):
        target = manifest["targets"][E6_TARGET_MODELS[role]]
        checkpoint = Path(target["source_checkpoint"])
        assert target["source_model"] == E6_SOURCE_MODELS[role]
        assert target["source_checkpoint_sha256"] == file_sha256(checkpoint)
        assert target["source_samples_per_class"] == 64
        assert target["source_epochs"] == 3
