from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


RUNNER_VERSION = "atm-resumable-search-v1"

Coordinate = tuple[int, int]
Relation = tuple[Coordinate, ...]
OracleResult = tuple[bool, set[Coordinate]]
Oracle = Callable[[Coordinate], OracleResult]


class ControlledInterruption(RuntimeError):
    """Raised by fixture runs after a durable candidate boundary."""


class ParameterMismatchError(ValueError):
    """Raised when an existing run directory belongs to another protocol."""


class ArtifactIntegrityError(ValueError):
    """Raised when a completed run cannot be verified."""


@dataclass(frozen=True)
class ResumableSearchConfig:
    run_id: str
    input_size: int
    output_size: int
    is_permutation: bool
    num_workers: int
    oracle_id: str
    source_commit: str
    search_source_sha256: str
    oracle_parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id or not self.oracle_id:
            raise ValueError("run_id and oracle_id must be non-empty")
        if min(self.input_size, self.output_size, self.num_workers) < 1:
            raise ValueError("sizes and num_workers must be positive")
        if len(self.source_commit) != 40 or len(self.search_source_sha256) != 64:
            raise ValueError("source commit and hash must be full-length")
        names = [name for name, _ in self.oracle_parameters]
        if len(names) != len(set(names)):
            raise ValueError("oracle parameter names must be unique")

    def parameter_payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "runner_version": RUNNER_VERSION,
            "oracle_parameters": dict(self.oracle_parameters),
        }

    def parameter_hash(self) -> str:
        return _sha256_bytes(_canonical_json(self.parameter_payload()))


_WORKER_ORACLE: Oracle | None = None


def _initialize_worker(oracle: Oracle) -> None:
    global _WORKER_ORACLE
    _WORKER_ORACLE = oracle


def _invoke_worker(coordinate: Coordinate) -> tuple[Coordinate, OracleResult]:
    if _WORKER_ORACLE is None:
        raise RuntimeError("ATM oracle worker was not initialized")
    return coordinate, _WORKER_ORACLE(coordinate)


def run_resumable_integral_property_search(
    oracle: Oracle,
    *,
    config: ResumableSearchConfig,
    output_root: Path,
    interrupt_after_new_candidates: int | None = None,
) -> dict[str, Any]:
    if interrupt_after_new_candidates is not None and interrupt_after_new_candidates < 1:
        raise ValueError("interrupt_after_new_candidates must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = output_root / "candidate_results"
    candidate_root.mkdir(exist_ok=True)
    parameter_payload = config.parameter_payload()
    parameter_hash = config.parameter_hash()
    metadata = {
        "parameter_hash": parameter_hash,
        "parameters": parameter_payload,
        "candidate_artifact_contract": "atomic-checksummed-json-v1",
        "oracle_cache_resume_semantics": (
            "completed candidate results are durable; unfinished candidates and internal "
            "oracle caches may be recomputed after restart"
        ),
    }
    metadata_path = output_root / "metadata.json"
    progress_path = output_root / "progress.jsonl"
    started_path = output_root / "started.marker"
    complete_path = output_root / "complete.marker"
    result_path = output_root / "result.json"
    fresh = not metadata_path.exists()
    if fresh:
        _atomic_write_json(metadata_path, metadata)
    else:
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ParameterMismatchError("existing metadata is unreadable") from error
        if existing_metadata != metadata:
            raise ParameterMismatchError(
                "existing run metadata does not match the requested search parameters"
            )
    if complete_path.exists():
        completed = _load_completed_result(
            result_path=result_path,
            complete_path=complete_path,
            parameter_hash=parameter_hash,
        )
        _append_progress(
            progress_path,
            "completed_result_reused",
            parameter_hash=parameter_hash,
            relation_count=len(completed),
        )
        return {
            "relations": completed,
            "parameter_hash": parameter_hash,
            "new_candidate_calls": 0,
            "reused_candidate_results": 0,
            "rejected_candidate_artifacts": 0,
            "basis_candidates": 0,
            "wuv_candidates": 0,
            "key_dependent_candidates": 0,
            "layers_completed": 0,
            "completed_result_reused": True,
        }
    if not started_path.exists():
        _atomic_write_json(
            started_path,
            {"parameter_hash": parameter_hash, "runner_version": RUNNER_VERSION},
        )
        _append_progress(progress_path, "run_start", parameter_hash=parameter_hash)
    elif not fresh:
        _append_progress(progress_path, "resume_start", parameter_hash=parameter_hash)

    invmask = (1 << config.input_size) - 1
    input_modifiers = tuple((1 << bit) ^ invmask for bit in range(config.input_size))
    output_modifiers = tuple(1 << bit for bit in range(config.output_size))
    basis_layers: list[set[Coordinate]] = [set()]
    wuv: dict[Coordinate, set[Coordinate]] = {}
    metrics = {
        "new_candidate_calls": 0,
        "reused_candidate_results": 0,
        "rejected_candidate_artifacts": 0,
        "basis_candidates": 0,
        "wuv_candidates": 0,
        "key_dependent_candidates": 0,
        "layers_completed": 0,
    }
    while len(basis_layers) == 1 or basis_layers[-1]:
        if len(basis_layers) == 1 and not config.is_permutation:
            candidates = tuple((invmask, value) for value in output_modifiers)
        elif len(basis_layers) == 1:
            basis_layers.append(set())
            candidates = tuple(
                (left, right)
                for left in input_modifiers
                for right in output_modifiers
            )
        else:
            candidates = _next_weight_candidates(
                basis_layers[-1],
                input_modifiers=input_modifiers,
                output_modifiers=output_modifiers,
                invmask=invmask,
                target_weight=len(basis_layers),
            )
        candidates = tuple(sorted(set(candidates)))
        basis_layers.append(set())
        layer = len(basis_layers) - 1
        _append_progress(
            progress_path,
            "layer_start",
            layer=layer,
            candidates=len(candidates),
        )
        cached: dict[Coordinate, OracleResult] = {}
        missing: list[Coordinate] = []
        for coordinate in candidates:
            path = _candidate_path(candidate_root, layer, coordinate, config)
            loaded, rejected = _load_candidate_artifact(
                path,
                layer=layer,
                coordinate=coordinate,
                parameter_hash=parameter_hash,
            )
            if rejected:
                metrics["rejected_candidate_artifacts"] += 1
                _append_progress(
                    progress_path,
                    "candidate_artifact_rejected",
                    layer=layer,
                    u=coordinate[0],
                    v=coordinate[1],
                    file=path.name,
                )
            if loaded is None:
                missing.append(coordinate)
            else:
                cached[coordinate] = loaded
                metrics["reused_candidate_results"] += 1
                _append_progress(
                    progress_path,
                    "candidate_reused",
                    layer=layer,
                    u=coordinate[0],
                    v=coordinate[1],
                )
        for coordinate, raw_result in _evaluate_candidates(
            oracle,
            missing,
            num_workers=config.num_workers,
        ):
            result = _normalize_oracle_result(raw_result)
            path = _candidate_path(candidate_root, layer, coordinate, config)
            _write_candidate_artifact(
                path,
                layer=layer,
                coordinate=coordinate,
                result=result,
                parameter_hash=parameter_hash,
            )
            cached[coordinate] = result
            metrics["new_candidate_calls"] += 1
            _append_progress(
                progress_path,
                "candidate_completed",
                layer=layer,
                u=coordinate[0],
                v=coordinate[1],
            )
            if (
                interrupt_after_new_candidates is not None
                and metrics["new_candidate_calls"] >= interrupt_after_new_candidates
            ):
                _append_progress(
                    progress_path,
                    "controlled_interrupt",
                    layer=layer,
                    completed_new_candidates=metrics["new_candidate_calls"],
                )
                raise ControlledInterruption(
                    f"controlled interruption after {metrics['new_candidate_calls']} candidates"
                )
        for coordinate in candidates:
            key_dependent, support = cached[coordinate]
            if key_dependent:
                metrics["key_dependent_candidates"] += 1
            elif not support:
                basis_layers[-1].add(coordinate)
                metrics["basis_candidates"] += 1
            else:
                wuv[coordinate] = support
                metrics["wuv_candidates"] += 1
        metrics["layers_completed"] += 1
        _append_progress(
            progress_path,
            "layer_complete",
            layer=layer,
            basis_candidates=len(basis_layers[-1]),
            wuv_candidates=len(wuv),
        )

    relations = _build_relations(basis_layers, wuv)
    result_payload = {
        "parameter_hash": parameter_hash,
        "runner_version": RUNNER_VERSION,
        "relations": [
            [[left, right] for left, right in relation] for relation in relations
        ],
    }
    result_envelope = _artifact_envelope(result_payload)
    _atomic_write_json(result_path, result_envelope)
    result_sha256 = _sha256_file(result_path)
    verified = _read_artifact_envelope(result_path)
    if verified != result_payload:
        raise ArtifactIntegrityError("atomic result verification failed")
    _atomic_write_json(
        complete_path,
        {
            "parameter_hash": parameter_hash,
            "result_sha256": result_sha256,
            "relation_count": len(relations),
        },
    )
    _append_progress(
        progress_path,
        "run_complete",
        relation_count=len(relations),
        result_sha256=result_sha256,
    )
    return {
        "relations": relations,
        "parameter_hash": parameter_hash,
        **metrics,
        "completed_result_reused": False,
    }


def validate_completed_search_result(
    output_root: Path,
    *,
    config: ResumableSearchConfig,
) -> tuple[Relation, ...]:
    """Validate and load a completed result without invoking its oracle."""
    return _load_completed_result(
        result_path=output_root / "result.json",
        complete_path=output_root / "complete.marker",
        parameter_hash=config.parameter_hash(),
    )


def _next_weight_candidates(
    previous_basis: set[Coordinate],
    *,
    input_modifiers: Sequence[int],
    output_modifiers: Sequence[int],
    invmask: int,
    target_weight: int,
) -> tuple[Coordinate, ...]:
    candidates: set[Coordinate] = set()
    for left, right in previous_basis:
        for modifier in input_modifiers:
            candidate_left = left & modifier
            if (
                (candidate_left ^ invmask).bit_count() + right.bit_count()
                != target_weight
            ):
                continue
            if all(
                not (((candidate_left | check) ^ invmask) > 0)
                or (candidate_left ^ check ^ invmask, right) in previous_basis
                for check in input_modifiers
            ):
                candidates.add((candidate_left, right))
        for modifier in output_modifiers:
            candidate_right = right | modifier
            if (
                (left ^ invmask).bit_count() + candidate_right.bit_count()
                != target_weight
            ):
                continue
            if all(
                not ((candidate_right & check) > 0)
                or (left, candidate_right ^ check) in previous_basis
                for check in output_modifiers
            ):
                candidates.add((left, candidate_right))
    return tuple(sorted(candidates))


def _evaluate_candidates(
    oracle: Oracle,
    candidates: Sequence[Coordinate],
    *,
    num_workers: int,
) -> Iterator[tuple[Coordinate, OracleResult]]:
    if num_workers == 1:
        for coordinate in candidates:
            yield coordinate, oracle(coordinate)
        return
    context = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    with context.Pool(
        num_workers,
        initializer=_initialize_worker,
        initargs=(oracle,),
    ) as pool:
        yield from pool.imap_unordered(_invoke_worker, candidates, chunksize=1)


def _build_relations(
    basis_layers: Sequence[set[Coordinate]],
    wuv: dict[Coordinate, set[Coordinate]],
) -> tuple[Relation, ...]:
    relations: set[Relation] = {
        (coordinate,) for layer in basis_layers for coordinate in layer
    }
    if wuv:
        columns = tuple(sorted(wuv))
        support_coordinates = tuple(sorted(set().union(*wuv.values())))
        row_masks = [
            sum(1 << index for index, column in enumerate(columns) if support in wuv[column])
            for support in support_coordinates
        ]
        for vector in _gf2_nullspace(row_masks, width=len(columns)):
            relation = tuple(
                column for index, column in enumerate(columns) if vector & (1 << index)
            )
            if relation:
                relations.add(relation)
    return tuple(sorted(relations, key=lambda relation: (len(relation), relation)))


def _gf2_nullspace(rows: Iterable[int], *, width: int) -> tuple[int, ...]:
    reduced = [int(row) for row in rows if row]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(width):
        selected = next(
            (index for index in range(pivot_row, len(reduced)) if reduced[index] & (1 << column)),
            None,
        )
        if selected is None:
            continue
        reduced[pivot_row], reduced[selected] = reduced[selected], reduced[pivot_row]
        for index in range(len(reduced)):
            if index != pivot_row and reduced[index] & (1 << column):
                reduced[index] ^= reduced[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    free_columns = [column for column in range(width) if column not in pivot_columns]
    basis: list[int] = []
    for free in free_columns:
        vector = 1 << free
        for row_index, pivot in reversed(tuple(enumerate(pivot_columns))):
            if (reduced[row_index] & vector).bit_count() & 1:
                vector |= 1 << pivot
        basis.append(vector)
    return tuple(basis)


def _candidate_path(
    root: Path,
    layer: int,
    coordinate: Coordinate,
    config: ResumableSearchConfig,
) -> Path:
    width = max(1, (max(config.input_size, config.output_size) + 3) // 4)
    return root / (
        f"layer_{layer:03d}__u_{coordinate[0]:0{width}x}__v_{coordinate[1]:0{width}x}.json"
    )


def _normalize_oracle_result(raw: Any) -> OracleResult:
    if not isinstance(raw, tuple) or len(raw) != 2 or not isinstance(raw[0], bool):
        raise TypeError("ATM oracle must return (bool, iterable[(int, int)])")
    if raw[1] is None:
        if not raw[0]:
            raise TypeError("key-independent ATM oracle results must include a support iterable")
        return True, set()
    support: set[Coordinate] = set()
    for item in raw[1]:
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or not all(isinstance(value, int) and value >= 0 for value in item)
        ):
            raise TypeError("ATM oracle support coordinates must be non-negative integer pairs")
        support.add((int(item[0]), int(item[1])))
    return raw[0], support


def _write_candidate_artifact(
    path: Path,
    *,
    layer: int,
    coordinate: Coordinate,
    result: OracleResult,
    parameter_hash: str,
) -> None:
    payload = {
        "layer": layer,
        "u": coordinate[0],
        "v": coordinate[1],
        "key_dependent": result[0],
        "support": [list(item) for item in sorted(result[1])],
        "parameter_hash": parameter_hash,
    }
    _atomic_write_json(path, _artifact_envelope(payload))


def _load_candidate_artifact(
    path: Path,
    *,
    layer: int,
    coordinate: Coordinate,
    parameter_hash: str,
) -> tuple[OracleResult | None, bool]:
    if not path.exists():
        return None, False
    try:
        payload = _read_artifact_envelope(path)
        if payload.get("layer") != layer:
            raise ValueError("layer mismatch")
        if (payload.get("u"), payload.get("v")) != coordinate:
            raise ValueError("coordinate mismatch")
        if payload.get("parameter_hash") != parameter_hash:
            raise ValueError("parameter hash mismatch")
        result = _normalize_oracle_result(
            (payload.get("key_dependent"), payload.get("support"))
        )
    except (ArtifactIntegrityError, TypeError, ValueError, OSError):
        return None, True
    return result, False


def _artifact_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "payload": payload,
        "payload_sha256": _sha256_bytes(_canonical_json(payload)),
    }


def _read_artifact_envelope(path: Path) -> dict[str, Any]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        expected = envelope["payload_sha256"]
    except (json.JSONDecodeError, KeyError, TypeError, OSError) as error:
        raise ArtifactIntegrityError(f"invalid artifact envelope: {path}") from error
    actual = _sha256_bytes(_canonical_json(payload))
    if actual != expected:
        raise ArtifactIntegrityError(f"artifact checksum mismatch: {path}")
    return payload


def _load_completed_result(
    *,
    result_path: Path,
    complete_path: Path,
    parameter_hash: str,
) -> tuple[Relation, ...]:
    try:
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ArtifactIntegrityError("complete marker is unreadable") from error
    if marker.get("parameter_hash") != parameter_hash:
        raise ArtifactIntegrityError("complete marker parameter hash mismatch")
    if marker.get("result_sha256") != _sha256_file(result_path):
        raise ArtifactIntegrityError("completed result file hash mismatch")
    payload = _read_artifact_envelope(result_path)
    if payload.get("parameter_hash") != parameter_hash:
        raise ArtifactIntegrityError("result parameter hash mismatch")
    relations = tuple(
        tuple((int(left), int(right)) for left, right in relation)
        for relation in payload.get("relations", [])
    )
    if marker.get("relation_count") != len(relations):
        raise ArtifactIntegrityError("complete marker relation count mismatch")
    return relations


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = _canonical_json(payload) + b"\n"
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
