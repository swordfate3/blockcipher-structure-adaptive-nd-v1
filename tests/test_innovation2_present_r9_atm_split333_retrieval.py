from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blockcipher_nd.cli.validate_innovation2_present_r9_atm_split333_retrieval import (
    main,
)
from blockcipher_nd.cli.postprocess_innovation2_present_r9_atm_split333 import (
    main as postprocess_main,
    parse_args as parse_postprocess_args,
)
from blockcipher_nd.tasks.innovation2.atm_resumable_search_runner import (
    RUNNER_VERSION,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_generation import (
    ATM_COMMIT,
    SOURCE_HASHES,
    search_config,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_retrieval import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_SOURCE_COMMIT,
    Split333RetrievalConfig,
    validate_split333_retrieval,
)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(payload) + b"\n")


def _envelope(payload: dict[str, object]) -> dict[str, object]:
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }


def _complete_raw_retrieval(root: Path) -> Path:
    raw = root / "raw"
    logs = raw / "logs"
    results = raw / "results"
    search = results / "search_state"
    parameter_hash = search_config().parameter_hash()
    for marker in (
        logs / "pipeline_passed.marker",
        results / "probe_001_passed.marker",
        results / "probe_002_passed.marker",
    ):
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("pass\n", encoding="utf-8")
    (logs / "source_revision.txt").write_text(
        EXPECTED_SOURCE_COMMIT + "\n",
        encoding="utf-8",
    )
    (logs / "atm_revision.txt").write_text(ATM_COMMIT + "\n", encoding="utf-8")
    (logs / "source_status_after_sync.txt").write_text(
        "## main...origin/main\n",
        encoding="utf-8",
    )
    _write_json(results / "readiness_gate.json", {"status": "pass"})
    _write_json(results / "probe_gate.json", {"status": "pass"})
    _write_json(
        results / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_present_r9_split333_generation_passed",
        },
    )
    _write_json(results / "summary.json", {"relations": 1, "relation_rank": 1})
    relations = [[[1, 2]]]
    _write_json(results / "relations.json", {"relations": relations})
    _write_json(
        results / "generation_passed.marker",
        {"parameter_hash": parameter_hash, "relations": 1},
    )
    _write_json(results / "source_hashes.json", SOURCE_HASHES)
    _write_json(results / "model_contract.json", {"sha256": EXPECTED_MODEL_SHA256})
    _write_json(
        results / "probe_001.json",
        {
            "controlled_interruption": True,
            "new_durable_candidates": 1,
            "candidate_reuse_events_total": 7,
        },
    )
    _write_json(
        results / "probe_002.json",
        {
            "controlled_interruption": True,
            "new_durable_candidates": 1,
            "candidate_reuse_events_total": 1,
        },
    )
    _write_json(
        search / "metadata.json",
        {
            "parameter_hash": parameter_hash,
            "parameters": search_config().parameter_payload(),
        },
    )
    result_payload = {
        "parameter_hash": parameter_hash,
        "runner_version": RUNNER_VERSION,
        "relations": relations,
    }
    result_path = search / "result.json"
    _write_json(result_path, _envelope(result_payload))
    result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    _write_json(
        search / "complete.marker",
        {
            "parameter_hash": parameter_hash,
            "result_sha256": result_hash,
            "relation_count": 1,
        },
    )
    (search / "progress.jsonl").write_text(
        json.dumps({"event": "run_complete"}) + "\n",
        encoding="utf-8",
    )
    candidate_payload = {
        "layer": 0,
        "u": 1,
        "v": 2,
        "key_dependent": False,
        "support": [],
        "parameter_hash": parameter_hash,
    }
    _write_json(
        search / "candidate_results/layer_000__u_1__v_2.json",
        _envelope(candidate_payload),
    )
    return raw


def test_e104_retrieval_validation_replays_all_frozen_evidence(tmp_path: Path) -> None:
    raw = _complete_raw_retrieval(tmp_path)

    validation = validate_split333_retrieval(
        Split333RetrievalConfig(),
        raw_root=raw,
    )

    assert validation["status"] == "pass"
    assert validation["decision"] == (
        "innovation2_present_r9_split333_retrieval_verified"
    )
    assert all(validation["existence_checks"].values())
    assert all(validation["checks"].values())
    assert validation["metrics"]["relations"] == 1
    assert validation["metrics"]["relation_rank"] == 1
    assert validation["metrics"]["candidate_files"] == 1
    assert validation["next_action"]["e105_open"] is True


def test_e104_retrieval_validation_rejects_corrupt_candidate(tmp_path: Path) -> None:
    raw = _complete_raw_retrieval(tmp_path)
    candidate = next((raw / "results/search_state/candidate_results").glob("*.json"))
    candidate.write_text('{"payload":', encoding="utf-8")

    validation = validate_split333_retrieval(
        Split333RetrievalConfig(),
        raw_root=raw,
    )

    assert validation["status"] == "fail"
    assert validation["checks"]["candidate_cache_integrity"] is False
    assert validation["next_action"]["e105_open"] is False


def test_e104_retrieval_uses_latest_numbered_resume_probes(tmp_path: Path) -> None:
    raw = _complete_raw_retrieval(tmp_path)
    results = raw / "results"
    _write_json(
        results / "probe_963.json",
        {
            "controlled_interruption": True,
            "new_durable_candidates": 1,
            "candidate_reuse_events_total": 963,
        },
    )
    _write_json(
        results / "probe_964.json",
        {
            "controlled_interruption": True,
            "new_durable_candidates": 1,
            "candidate_reuse_events_total": 964,
        },
    )
    for suffix in (963, 964):
        (results / f"probe_{suffix}_passed.marker").write_text(
            "pass\n",
            encoding="utf-8",
        )
    _write_json(
        results / "probe_001.json",
        {"controlled_interruption": False, "new_durable_candidates": 0},
    )

    validation = validate_split333_retrieval(
        Split333RetrievalConfig(),
        raw_root=raw,
    )

    assert validation["status"] == "pass"
    assert validation["metrics"]["first_resume_probe"] == "probe_963"
    assert validation["metrics"]["second_resume_probe"] == "probe_964"


def test_e104_retrieval_cli_promotes_only_after_validation(tmp_path: Path) -> None:
    raw = _complete_raw_retrieval(tmp_path)
    verified = tmp_path / "verified"

    return_code = main(
        ["--raw-root", str(raw), "--verified-root", str(verified)]
    )

    assert return_code == 0
    assert (verified / "validation.json").is_file()
    assert (verified / "artifact_manifest.json").is_file()
    assert (verified / "results.jsonl").is_file()
    assert (verified / "logs/pipeline_passed.marker").is_file()
    assert (verified / "results/generation_passed.marker").is_file()
    assert not (verified / "RAW_RETRIEVAL_NOTICE.txt").exists()


def test_postprocess_cli_requires_separate_raw_verified_checkpoint_and_e105_roots() -> None:
    args = parse_postprocess_args(
        [
            "--raw-root",
            "raw",
            "--verified-root",
            "verified",
            "--checkpoint-root",
            "checkpoints",
            "--public-results-root",
            "public",
            "--e105-output-root",
            "e105",
        ]
    )
    assert args.raw_root == Path("raw")
    assert args.verified_root == Path("verified")
    assert args.checkpoint_root == Path("checkpoints")
    assert args.e105_output_root == Path("e105")


def test_postprocess_stops_before_e105_when_e104_is_incomplete(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    verified = tmp_path / "verified"
    e105 = tmp_path / "e105"

    return_code = postprocess_main(
        [
            "--raw-root",
            str(raw),
            "--verified-root",
            str(verified),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--public-results-root",
            str(tmp_path / "public"),
            "--e105-output-root",
            str(e105),
        ]
    )

    assert return_code == 1
    assert not verified.exists()
    assert not e105.exists()
    report = json.loads((raw / "postprocess.local.json").read_text(encoding="utf-8"))
    assert report["stage"] == "e104_validation"
    assert report["e105_started"] is False
