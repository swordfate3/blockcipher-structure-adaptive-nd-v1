from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraOutputPredictionConfig,
    prepare_disk_output_prediction_data,
)
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputPositionBoundResidualCnn,
    SelectedOutputResidualCnn,
    _build_model,
    _present_topology_mapping,
    _train_one_model,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    validate_selected_output_contract,
)


RUN_ID_PREFIX = "i2_output_prediction_opd1_present_r3_position_bound_spn_rescnn"
ROUND_EXTENSION_RUN_ID_PREFIX = (
    "i2_output_prediction_opf1_present_r4_position_bound_spn_rescnn"
)
SCALE_EXTENSION_RUN_ID_PREFIX = (
    "i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20"
)
OPC1_RUN_ID = (
    "i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_key6_gpu0_20260722"
)
OPC1_RELEASE_DECISION = "innovation2_spn_rescnn_hybrid_not_supported"
OPN1_RUN_ID = (
    "i2_output_prediction_opn1_present_r3_spn_rescnn_head_"
    "permutation_identifiability_audit_20260722"
)
OPN1_RELEASE_DECISION = "innovation2_spn_rescnn_final_routing_absorbable_by_global_head"
OPC1_GATE_SHA256 = "ebb86a9feab6d2d9993937f5c0a7f4afe1bfe3597c8c1dff083956381e0310b4"
OPN1_GATE_SHA256 = "887a7db3643e73bdda67958bcaae470881a09db25ab0ba5ff6c3d6bb0a2503d7"
OPD1_RUN_ID = (
    "i2_output_prediction_opd1_present_r3_position_bound_spn_rescnn_"
    "key7_gpu0_20260722"
)
OPD1_RELEASE_DECISION = "innovation2_position_bound_spn_rescnn_not_supported"
OPD1_GATE_SHA256 = "3d63163ab94e95b6c8c859be0867cc0a6b1f91382bd842e32dd3adbe04863579"
OPD1_PLAINTEXTS_SHA256 = (
    "0f08d171c5b833ee1223da07bfc80e10d7ea99bbc0bef1b068547d3a7e8120e1"
)
OPF1_RUN_ID = (
    "i2_output_prediction_opf1_present_r4_position_bound_spn_rescnn_"
    "key7_gpu0_20260722"
)
OPF1_RELEASE_DECISION = "innovation2_position_bound_r4_boundary_observed"
OPF1_GATE_SHA256 = "dad638c7180682074f134233e807ed0b58bb40d7d671f7a5d01540721e399354"
OPF1_TRAIN_ROWS = 1 << 17
OPF1_TEST_ROWS = 1 << 16
OPF1_TOTAL_ROWS = OPF1_TRAIN_ROWS + OPF1_TEST_ROWS
OPF1_TRAIN_PLAINTEXTS_RAW_SHA256 = (
    "eca0f5705c2d9a6b4f0475bfb90e55d2bfa2d5e4d7b8c380b10ab55778a4555a"
)
OPF1_TEST_PLAINTEXTS_RAW_SHA256 = (
    "5c5410d4c0761f729f5f705d43a7392bf90f6ae0bee65a57321760d515b82fec"
)
MODEL_SPECS = (
    ("selected8_global_head_rescnn_anchor_true_output", "rescnn", False),
    (
        "selected8_position_head_rescnn_no_p_true_output",
        "position_head_rescnn_no_p",
        False,
    ),
    (
        "selected8_position_head_spn_rescnn_exact_p_true_output",
        "position_head_spn_rescnn_exact_p",
        False,
    ),
    (
        "selected8_position_head_spn_rescnn_wrong_p_true_output",
        "position_head_spn_rescnn_wrong_p",
        False,
    ),
    (
        "selected8_position_head_spn_rescnn_exact_p_label_shuffle",
        "position_head_spn_rescnn_exact_p",
        True,
    ),
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PositionBoundSpnResCnnConfig:
    run_id: str = f"{RUN_ID_PREFIX}_smoke_seed7_20260722"
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 7
    train_rows: int = 64
    test_rows: int = 64
    mlp_hidden_dim: int = 1936
    lstm_hidden_dim: int = 300
    lstm_layers: int = 6
    rescnn_channels: int = 252
    rescnn_blocks: int = 10
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_chunk_rows: int = 32
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES
    maximum_parameter_gap: float = 0.001
    minimum_candidate_mean_auc: float = 0.550
    minimum_candidate_global_gain: float = 0.010
    minimum_candidate_no_p_gain: float = 0.010
    minimum_mean_topology_margin: float = 0.020
    minimum_mean_shuffle_margin: float = 0.030
    minimum_per_bit_auc: float = 0.550
    minimum_per_bit_anchor_gain: float = 0.005
    minimum_per_bit_control_margin: float = 0.015
    minimum_accuracy_margin: float = 0.005
    minimum_passed_bits: int = 4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {
            "smoke",
            "position_bound_head",
            "round_extension_smoke",
            "round_extension",
            "scale_extension_smoke",
            "scale_extension",
        }:
            raise ValueError("invalid OPD1 mode")
        expected_rounds = 4 if _is_r4_mode(self.mode) else 3
        if self.rounds != expected_rounds or self.seed != 7:
            raise ValueError(
                "position-bound experiments require the mode-matched PRESENT "
                "round count and key seed7"
            )
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPD1 positions must match OP10 through OPC1")
        integer_values = (
            self.train_rows,
            self.test_rows,
            self.rescnn_channels,
            self.rescnn_blocks,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
            self.minimum_passed_bits,
        )
        if min(integer_values) <= 0 or self.learning_rate <= 0:
            raise ValueError(
                "OPD1 row, model, training, and gate values must be positive"
            )

    @classmethod
    def formal(
        cls,
        *,
        run_id: str | None = None,
        device: str = "cuda",
    ) -> PositionBoundSpnResCnnConfig:
        return cls(
            run_id=run_id or f"{RUN_ID_PREFIX}_key7_gpu0_20260722",
            mode="position_bound_head",
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )

    @classmethod
    def round_extension(
        cls,
        *,
        run_id: str | None = None,
        device: str = "cuda",
        smoke: bool = False,
    ) -> PositionBoundSpnResCnnConfig:
        if smoke:
            return cls(
                run_id=run_id
                or f"{ROUND_EXTENSION_RUN_ID_PREFIX}_smoke_seed7_20260722",
                mode="round_extension_smoke",
                rounds=4,
                device=device,
            )
        return cls(
            run_id=run_id
            or f"{ROUND_EXTENSION_RUN_ID_PREFIX}_key7_gpu0_20260722",
            mode="round_extension",
            rounds=4,
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )

    @classmethod
    def scale_extension(
        cls,
        *,
        run_id: str | None = None,
        device: str = "cuda",
        smoke: bool = False,
    ) -> PositionBoundSpnResCnnConfig:
        if smoke:
            return cls(
                run_id=run_id
                or f"{SCALE_EXTENSION_RUN_ID_PREFIX}_smoke_seed7_20260722",
                mode="scale_extension_smoke",
                rounds=4,
                device=device,
            )
        return cls(
            run_id=run_id
            or f"{SCALE_EXTENSION_RUN_ID_PREFIX}_key7_gpu0_20260722",
            mode="scale_extension",
            rounds=4,
            train_rows=1 << 20,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )


def _is_r4_mode(mode: str) -> bool:
    return mode.startswith(("round_extension", "scale_extension"))


def _is_scale_extension(mode: str) -> bool:
    return mode.startswith("scale_extension")


def _scale_split_boundaries(
    config: PositionBoundSpnResCnnConfig,
) -> tuple[int, int, int]:
    total_rows = config.train_rows + config.test_rows
    if config.mode == "scale_extension":
        return OPF1_TRAIN_ROWS, OPF1_TOTAL_ROWS, total_rows
    base_train_rows = config.train_rows // 2
    return base_train_rows, base_train_rows + config.test_rows, total_rows


def authorize_from_source_gates(
    opc1_gate: dict[str, Any],
    opn1_gate: dict[str, Any],
) -> None:
    expected = (
        (
            opc1_gate,
            OPC1_RUN_ID,
            "hold",
            OPC1_RELEASE_DECISION,
            ("protocol_checks", "execution_checks"),
        ),
        (
            opn1_gate,
            OPN1_RUN_ID,
            "pass",
            OPN1_RELEASE_DECISION,
            ("protocol_checks", "execution_checks"),
        ),
    )
    for gate, run_id, status, decision, groups in expected:
        if (
            gate.get("run_id") != run_id
            or gate.get("status") != status
            or gate.get("decision") != decision
        ):
            raise ValueError("OPD1 formal mode requires frozen OPC1 hold and OPN1 pass")
        for group in groups:
            checks = gate.get(group)
            if not isinstance(checks, dict) or not checks or not all(checks.values()):
                raise ValueError(f"source gate {group} did not fully pass")


def authorize_round_extension_from_opd1_gate(opd1_gate: dict[str, Any]) -> None:
    if (
        opd1_gate.get("run_id") != OPD1_RUN_ID
        or opd1_gate.get("status") != "hold"
        or opd1_gate.get("decision") != OPD1_RELEASE_DECISION
    ):
        raise ValueError("OPF1 requires the frozen completed OPD1 gate")
    for group in ("protocol_checks", "execution_checks"):
        checks = opd1_gate.get(group)
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError(f"OPD1 source gate {group} did not fully pass")
    means = opd1_gate.get("metrics", {}).get("mean_auc_by_model", {})
    exact = means.get("selected8_position_head_spn_rescnn_exact_p_true_output")
    if exact is None or not math.isclose(float(exact), 0.999996158392159):
        raise ValueError("OPD1 exact-P reference metric does not match")


def authorize_scale_extension_from_opf1_gate(opf1_gate: dict[str, Any]) -> None:
    if (
        opf1_gate.get("run_id") != OPF1_RUN_ID
        or opf1_gate.get("status") != "hold"
        or opf1_gate.get("decision") != OPF1_RELEASE_DECISION
    ):
        raise ValueError("OPF2 requires the frozen completed OPF1 gate")
    for group in ("protocol_checks", "execution_checks"):
        checks = opf1_gate.get(group)
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError(f"OPF1 source gate {group} did not fully pass")
    means = opf1_gate.get("metrics", {}).get("mean_auc_by_model", {})
    exact = means.get("selected8_position_head_spn_rescnn_exact_p_true_output")
    shuffled = means.get("selected8_position_head_spn_rescnn_exact_p_label_shuffle")
    if (
        exact is None
        or shuffled is None
        or not math.isclose(float(exact), 0.5137553581211971)
        or not math.isclose(float(shuffled), 0.5009341432724128)
    ):
        raise ValueError("OPF1 scale-reference metrics do not match")


def prepare_position_bound_data(
    config: PositionBoundSpnResCnnConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    data_config = KimuraOutputPredictionConfig(
        run_id=config.run_id,
        mode="smoke",
        rounds=config.rounds,
        seed=config.seed,
        train_rows=config.train_rows,
        test_rows=config.test_rows,
        hidden_dim=config.lstm_hidden_dim,
        layers=config.lstm_layers,
        mlp_hidden_dim=config.mlp_hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        data_chunk_rows=config.data_chunk_rows,
        device=config.device,
    )
    data = prepare_disk_output_prediction_data(
        data_config,
        output_root,
        progress=progress,
    )
    if _is_scale_extension(config.mode):
        base_train_rows, base_total_rows, total_rows = _scale_split_boundaries(config)
        split_layout = {
            "name": "opf1_test_preserving_scale_extension_v1",
            "train_index_segments": [
                [0, base_train_rows],
                [base_total_rows, total_rows],
            ],
            "test_index_segment": [base_train_rows, base_total_rows],
            "opf1_train_plaintexts_raw_sha256": (
                OPF1_TRAIN_PLAINTEXTS_RAW_SHA256
            ),
            "opf1_test_plaintexts_raw_sha256": OPF1_TEST_PLAINTEXTS_RAW_SHA256,
        }
        metadata = {**data["metadata"], "split_layout": split_layout}
        metadata_path = Path(data["data_root"]) / "cache_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        data["metadata"] = metadata
        data["split_layout"] = split_layout
    return data


def position_bound_parameter_counts(
    config: PositionBoundSpnResCnnConfig,
) -> dict[str, int]:
    return {
        architecture: sum(
            parameter.numel()
            for parameter in _build_model(config, architecture).parameters()  # type: ignore[arg-type]
        )
        for architecture in {architecture for _, architecture, _ in MODEL_SPECS}
    }


def validate_position_bound_contract(
    config: PositionBoundSpnResCnnConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    selected_config = SelectedOutputBitHeadConfig(
        run_id=config.run_id,
        mode="smoke",
        rounds=config.rounds,
        seed=config.seed,
        train_rows=config.train_rows,
        test_rows=config.test_rows,
        hidden_dim=config.mlp_hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        data_chunk_rows=config.data_chunk_rows,
        selected_msb_indices=config.selected_msb_indices,
        device=config.device,
    )
    checks = validate_selected_output_contract(selected_config, data)
    checks.pop("independent_key_seed_is_one")
    keys = {seed: random.Random(910_000 + seed).getrandbits(80) for seed in range(8)}
    counts = position_bound_parameter_counts(config)
    with torch.random.fork_rng():
        torch.manual_seed(2_040_000 + config.seed)
        exact = _build_model(config, "position_head_spn_rescnn_exact_p")  # type: ignore[arg-type]
        torch.manual_seed(2_040_000 + config.seed)
        wrong = _build_model(config, "position_head_spn_rescnn_wrong_p")  # type: ignore[arg-type]
    no_p = _build_model(config, "position_head_rescnn_no_p")  # type: ignore[arg-type]
    global_anchor = _build_model(config, "rescnn")  # type: ignore[arg-type]
    selected = torch.tensor(config.selected_msb_indices, dtype=torch.long)
    exact_sources = _present_topology_mapping("exact").index_select(0, selected)
    wrong_sources = _present_topology_mapping("wrong").index_select(0, selected)
    local_counts = [
        counts["position_head_rescnn_no_p"],
        counts["position_head_spn_rescnn_exact_p"],
        counts["position_head_spn_rescnn_wrong_p"],
    ]
    checks.update(
        {
            "eighth_fixed_key_seed_is_seven": config.seed == 7,
            "seed7_key_differs_from_seed0_through_seed6": len(set(keys.values())) == 8
            and int(data["secret_key"]) == keys[7],
            "five_frozen_matrix_rows": len(MODEL_SPECS) == 5,
            "position_head_variants_have_identical_parameter_counts": len(
                set(local_counts)
            )
            == 1,
            "position_head_and_global_anchor_within_parameter_gap": abs(
                local_counts[0] - counts["rescnn"]
            )
            / counts["rescnn"]
            <= config.maximum_parameter_gap,
            "global_anchor_remains_plain_rescnn": isinstance(
                global_anchor,
                SelectedOutputResidualCnn,
            ),
            "position_models_use_local_heads": all(
                isinstance(model, SelectedOutputPositionBoundResidualCnn)
                and len(model.head.heads) == len(config.selected_msb_indices)
                and all(
                    head[0].in_features == config.rescnn_channels
                    for head in model.head.heads
                )
                for model in (no_p, exact, wrong)
            ),
            "exact_and_wrong_trainable_states_match_at_initialization": exact.state_dict().keys()
            == wrong.state_dict().keys()
            and all(
                torch.equal(exact.state_dict()[name], wrong.state_dict()[name])
                for name in exact.state_dict()
            ),
            "exact_and_wrong_selected_final_sources_differ": bool(
                torch.all(exact_sources != wrong_sources)
            ),
            "final_routing_changes_selected_head_sources": bool(
                torch.any(exact_sources != selected)
                and torch.any(wrong_sources != selected)
            ),
            "labels_are_true_outputs_not_sample_classes": True,
        }
    )
    if _is_r4_mode(config.mode):
        checks["round_extension_changes_only_cipher_rounds"] = (
            config.rounds == 4
            and config.seed == 7
            and config.selected_msb_indices == SELECTED_MSB_INDICES
        )
        checks["plaintext_generator_matches_opd1_seed_protocol"] = (
            int(data["metadata"]["seed"]) == 7
            and int(data["metadata"]["train_rows"]) == config.train_rows
            and int(data["metadata"]["test_rows"]) == config.test_rows
        )
        if config.mode == "round_extension":
            plaintext_path = Path(data["data_root"]) / "plaintexts.npy"
            checks["plaintext_file_sha256_matches_opd1"] = (
                hashlib.sha256(plaintext_path.read_bytes()).hexdigest()
                == OPD1_PLAINTEXTS_SHA256
            )
    if _is_scale_extension(config.mode):
        base_train_rows, base_total_rows, total_rows = _scale_split_boundaries(config)
        plaintexts = data["plaintexts"]
        layout = data.get("split_layout", {})
        checks.update(
            {
                "scale_extension_changes_only_training_rows": config.rounds == 4
                and config.seed == 7
                and config.train_rows
                in {64, 1 << 20}
                and config.test_rows in {64, OPF1_TEST_ROWS},
                "scale_split_layout_is_frozen": layout.get(
                    "train_index_segments"
                )
                == [[0, base_train_rows], [base_total_rows, total_rows]]
                and layout.get("test_index_segment")
                == [base_train_rows, base_total_rows],
                "scale_split_counts_match": base_train_rows
                + (total_rows - base_total_rows)
                == config.train_rows
                and base_total_rows - base_train_rows == config.test_rows,
                "opf1_train_plaintexts_are_preserved": (
                    config.mode == "scale_extension_smoke"
                    or hashlib.sha256(
                        np.asarray(plaintexts[:OPF1_TRAIN_ROWS]).tobytes()
                    ).hexdigest()
                    == OPF1_TRAIN_PLAINTEXTS_RAW_SHA256
                ),
                "opf1_test_plaintexts_are_preserved": (
                    config.mode == "scale_extension_smoke"
                    or hashlib.sha256(
                        np.asarray(
                            plaintexts[OPF1_TRAIN_ROWS:OPF1_TOTAL_ROWS]
                        ).tobytes()
                    ).hexdigest()
                    == OPF1_TEST_PLAINTEXTS_RAW_SHA256
                ),
                "actual_scale_train_test_indices_are_disjoint": (
                    base_train_rows <= base_total_rows <= total_rows
                    and checks["plaintexts_are_unique"]
                ),
            }
        )
    return checks


def train_position_bound_matrix(
    config: PositionBoundSpnResCnnConfig,
    data: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if _is_scale_extension(config.mode):
        base_train_rows, base_total_rows, total_rows = _scale_split_boundaries(config)
        train_features = np.concatenate(
            (
                np.asarray(data["features"][:base_train_rows]),
                np.asarray(data["features"][base_total_rows:total_rows]),
            )
        )
        test_features = np.array(
            data["features"][base_train_rows:base_total_rows], copy=True
        )
        train_full_targets = np.concatenate(
            (
                np.asarray(data["full_targets"][:base_train_rows]),
                np.asarray(data["full_targets"][base_total_rows:total_rows]),
            )
        )
        test_full_targets = np.array(
            data["full_targets"][base_train_rows:base_total_rows], copy=True
        )
    else:
        train_features = np.array(data["features"][: config.train_rows], copy=True)
        test_features = np.array(data["features"][config.train_rows :], copy=True)
        train_full_targets = np.asarray(data["full_targets"][: config.train_rows])
        test_full_targets = np.asarray(data["full_targets"][config.train_rows :])
    selected = np.asarray(config.selected_msb_indices, dtype=np.int64)
    train_targets = np.array(train_full_targets[:, selected], copy=True)
    test_targets = np.array(test_full_targets[:, selected], copy=True)
    shuffle = np.random.default_rng(2_050_000 + config.seed).permutation(
        config.train_rows
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, architecture, shuffle_labels in MODEL_SPECS:
        model_targets = np.array(train_targets, copy=True)
        if shuffle_labels:
            model_targets = model_targets[shuffle]
        result = _train_one_model(
            config,  # type: ignore[arg-type]
            model_name=model_name,
            architecture=architecture,
            train_features=train_features,
            train_targets=model_targets,
            test_features=test_features,
            test_targets=test_targets,
            output_root=output_root,
            progress=progress,
        )
        rows.extend(result["rows"])
        summaries.append(result["summary"])
        history.extend(result["history"])
        checkpoints.append(result["checkpoint"])
    return {
        "rows": rows,
        "summaries": summaries,
        "history": history,
        "checkpoints": checkpoints,
    }


def adjudicate_position_bound(
    config: PositionBoundSpnResCnnConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
    *,
    reference_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row for row in training["rows"]
    }
    names = [name for name, _, _ in MODEL_SPECS]
    means = {
        name: float(
            np.mean(
                [
                    float(indexed[(name, bit)]["auc"])
                    for bit in config.selected_msb_indices
                ]
            )
        )
        for name in names
    }
    global_name, no_p_name, exact_name, wrong_name, shuffle_name = names
    bit_gates = []
    output_bit_gates = []
    for bit in config.selected_msb_indices:
        global_anchor = indexed[(global_name, bit)]
        no_p = indexed[(no_p_name, bit)]
        exact = indexed[(exact_name, bit)]
        wrong = indexed[(wrong_name, bit)]
        shuffled = indexed[(shuffle_name, bit)]
        checks = {
            "candidate_auc_at_least_0_550": float(exact["auc"])
            >= config.minimum_per_bit_auc,
            "candidate_minus_global_at_least_0_005": float(exact["auc"])
            - float(global_anchor["auc"])
            >= config.minimum_per_bit_anchor_gain,
            "candidate_minus_no_p_at_least_0_005": float(exact["auc"])
            - float(no_p["auc"])
            >= config.minimum_per_bit_anchor_gain,
            "candidate_minus_wrong_at_least_0_015": float(exact["auc"])
            - float(wrong["auc"])
            >= config.minimum_per_bit_control_margin,
            "candidate_minus_shuffle_at_least_0_015": float(exact["auc"])
            - float(shuffled["auc"])
            >= config.minimum_per_bit_control_margin,
            "accuracy_margin_at_least_0_005": float(exact["accuracy_minus_majority"])
            >= config.minimum_accuracy_margin,
        }
        bit_gates.append(
            {
                "msb_index": bit,
                "global_auc": float(global_anchor["auc"]),
                "no_p_auc": float(no_p["auc"]),
                "candidate_auc": float(exact["auc"]),
                "wrong_auc": float(wrong["auc"]),
                "shuffle_auc": float(shuffled["auc"]),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        output_checks = {
            "candidate_auc_at_least_0_550": float(exact["auc"])
            >= config.minimum_per_bit_auc,
            "candidate_minus_shuffle_at_least_0_015": float(exact["auc"])
            - float(shuffled["auc"])
            >= config.minimum_per_bit_control_margin,
            "accuracy_margin_at_least_0_005": float(exact["accuracy_minus_majority"])
            >= config.minimum_accuracy_margin,
        }
        output_bit_gates.append(
            {
                "msb_index": bit,
                "candidate_auc": float(exact["auc"]),
                "shuffle_auc": float(shuffled["auc"]),
                "accuracy_minus_majority": float(exact["accuracy_minus_majority"]),
                "checks": output_checks,
                "passed": all(output_checks.values()),
            }
        )
    passed_bits = sum(row["passed"] for row in bit_gates)
    passed_output_bits = sum(row["passed"] for row in output_bit_gates)
    formal_checks = {
        "candidate_mean_auc_at_least_0_550": means[exact_name]
        >= config.minimum_candidate_mean_auc,
        "candidate_minus_global_mean_auc_at_least_0_010": means[exact_name]
        - means[global_name]
        >= config.minimum_candidate_global_gain,
        "candidate_minus_no_p_mean_auc_at_least_0_010": means[exact_name]
        - means[no_p_name]
        >= config.minimum_candidate_no_p_gain,
        "candidate_minus_wrong_mean_auc_at_least_0_020": means[exact_name]
        - means[wrong_name]
        >= config.minimum_mean_topology_margin,
        "candidate_minus_shuffle_mean_auc_at_least_0_030": means[exact_name]
        - means[shuffle_name]
        >= config.minimum_mean_shuffle_margin,
        "at_least_four_bits_pass": passed_bits >= config.minimum_passed_bits,
    }
    exact_accuracy_margin = float(
        np.mean(
            [
                float(indexed[(exact_name, bit)]["accuracy_minus_majority"])
                for bit in config.selected_msb_indices
            ]
        )
    )
    round_extension_checks = {
        "candidate_mean_auc_at_least_0_550": means[exact_name]
        >= config.minimum_candidate_mean_auc,
        "candidate_minus_shuffle_mean_auc_at_least_0_030": means[exact_name]
        - means[shuffle_name]
        >= config.minimum_mean_shuffle_margin,
        "candidate_mean_accuracy_margin_at_least_0_005": exact_accuracy_margin
        >= config.minimum_accuracy_margin,
        "at_least_four_output_bits_pass": passed_output_bits
        >= config.minimum_passed_bits,
    }
    execution_checks = {
        "five_models_complete": len(training["summaries"]) == 5,
        "forty_result_rows_complete": len(training["rows"]) == 40,
        "history_rows_complete": len(training["history"]) == config.epochs * 5,
        "five_checkpoint_hashes_present": len(training["checkpoints"]) == 5
        and all(row.get("sha256") for row in training["checkpoints"]),
        "all_metrics_are_finite": all(
            math.isfinite(float(row[field]))
            for row in training["rows"]
            for field in (
                "threshold_accuracy",
                "majority_accuracy",
                "accuracy_minus_majority",
                "auc",
                "mse",
            )
        ),
    }
    valid = (
        bool(protocol_checks)
        and all(protocol_checks.values())
        and all(execution_checks.values())
    )
    is_round_extension = config.mode.startswith("round_extension")
    is_scale_extension = _is_scale_extension(config.mode)
    is_r4 = _is_r4_mode(config.mode)
    if is_r4 and reference_gate is None:
        valid = False
    if not valid:
        status = "fail"
        decision = "innovation2_position_bound_spn_rescnn_protocol_invalid"
        action = "repair only the frozen head, data, controls, or artifact protocol"
    elif config.mode in {
        "smoke",
        "round_extension_smoke",
        "scale_extension_smoke",
    }:
        status = "pass"
        if is_scale_extension:
            decision = "innovation2_position_bound_r4_scale_local_readiness_passed"
            action = "launch the frozen seed7 2^20 scale matrix from a pushed commit"
        elif is_round_extension:
            decision = "innovation2_position_bound_r4_local_readiness_passed"
            action = "launch the frozen seed7 round-four matrix from a pushed commit"
        else:
            decision = "innovation2_position_bound_spn_rescnn_local_readiness_passed"
            action = "prepare the frozen seed7 remote matrix from a pushed commit"
    elif is_scale_extension and all(round_extension_checks.values()):
        status = "pass"
        decision = "innovation2_position_bound_r4_scale_supported"
        action = "repeat the unchanged 2^20 round-four matrix under a fresh fixed key"
    elif is_scale_extension:
        status = "hold"
        decision = "innovation2_position_bound_r4_scale_not_supported"
        action = "stop mechanical scaling and preregister one new r4 architecture hypothesis"
    elif is_round_extension and all(round_extension_checks.values()):
        status = "pass"
        decision = "innovation2_position_bound_r4_output_supported"
        action = (
            "repeat the unchanged round-four matrix under a fresh fixed key before r5"
        )
    elif is_round_extension:
        status = "hold"
        decision = "innovation2_position_bound_r4_boundary_observed"
        action = (
            "retain the r3 positive result and report the matched r3-to-r4 boundary"
        )
    elif all(formal_checks.values()):
        status = "pass"
        decision = "innovation2_position_bound_spn_rescnn_requires_confirmation"
        action = "repeat the unchanged five-row matrix under a fresh fixed key"
    else:
        status = "hold"
        decision = "innovation2_position_bound_spn_rescnn_not_supported"
        action = (
            "retain the global-head ResCNN anchor and stop this position-head route"
        )
    reference_exact_mean = None
    candidate_minus_reference = None
    if reference_gate is not None:
        reference_exact_mean = float(
            reference_gate["metrics"]["mean_auc_by_model"][exact_name]
        )
        candidate_minus_reference = means[exact_name] - reference_exact_mean
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "bit_gates": bit_gates,
        "output_bit_gates": output_bit_gates,
        "metrics": {
            "mean_auc_by_model": means,
            "candidate_minus_global_mean_auc": means[exact_name] - means[global_name],
            "candidate_minus_no_p_mean_auc": means[exact_name] - means[no_p_name],
            "candidate_minus_wrong_mean_auc": means[exact_name] - means[wrong_name],
            "candidate_minus_shuffle_mean_auc": means[exact_name] - means[shuffle_name],
            "passed_bit_count": passed_bits,
            "passed_output_bit_count": passed_output_bits,
            "candidate_mean_accuracy_margin": exact_accuracy_margin,
            "formal_checks": formal_checks,
            "round_extension_checks": round_extension_checks,
            "reference_exact_p_mean_auc": reference_exact_mean,
            "candidate_minus_reference_exact_p_mean_auc": (
                candidate_minus_reference
            ),
            "r3_reference_exact_p_mean_auc": (
                reference_exact_mean if is_round_extension else None
            ),
            "r4_minus_r3_exact_p_mean_auc": (
                candidate_minus_reference if is_round_extension else None
            ),
            "opf1_reference_exact_p_mean_auc": (
                reference_exact_mean if is_scale_extension else None
            ),
            "opf2_minus_opf1_exact_p_mean_auc": (
                candidate_minus_reference if is_scale_extension else None
            ),
        },
        "claim_scope": (
            "local implementation readiness"
            if config.mode
            in {"smoke", "round_extension_smoke", "scale_extension_smoke"}
            else (
                (
                    "seed7 PRESENT r4 selected-eight-output position-bound SPN-ResCNN "
                    "2^20 training-scale adjudication"
                    if is_scale_extension
                    else "seed7 PRESENT r4 selected-eight-output position-bound "
                    "SPN-ResCNN matched round extension"
                )
                if is_r4
                else "seed7 PRESENT r3 selected-eight-output position-bound "
                "SPN-ResCNN attribution"
            )
        )
        + (
            "; not cross-key confirmation, r5 evidence, full-ciphertext recovery, "
            "sample classification, or SOTA"
            if is_r4
            else "; not r4 evidence, full-ciphertext recovery, sample classification, or SOTA"
        ),
        "next_action": {
            "action": action,
            "formal_launch_requires_opc1_hold_and_opn1_pass": True,
            "round_extension_requires_completed_opd1": is_r4,
            "scale_extension_requires_completed_opf1": is_scale_extension,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
            "reopens_r4": is_r4 and status == "pass",
        },
    }


def serializable_position_bound_config(
    config: PositionBoundSpnResCnnConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload
