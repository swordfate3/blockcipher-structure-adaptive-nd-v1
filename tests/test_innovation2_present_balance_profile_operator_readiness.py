from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_innovation2_present_balance_profile_operator_readiness import (
    main as plot_main,
)
from blockcipher_nd.models.structure.spn.present_balance_profile_operator import (
    PresentBalanceProfileOperator,
    PresentBalanceProfileOperatorSpec,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    masked_binary_cross_entropy,
)


def present_player() -> np.ndarray:
    return np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)], dtype=np.int64
    )


def make_model(mode: str, player: np.ndarray | None = None) -> PresentBalanceProfileOperator:
    return PresentBalanceProfileOperator(
        PresentBalanceProfileOperatorSpec(
            input_dim=39,
            hidden_dim=16,
            steps=2,
            dropout=0.0,
            relation_mode=mode,
        ),
        torch.from_numpy(present_player() if player is None else player),
    )


def copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target_parameters = dict(target.named_parameters())
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            target_parameters[name].copy_(parameter)


def test_profile_operator_outputs_64_logits_and_masked_loss_matches() -> None:
    model = make_model("true")
    features = torch.randn(3, 64, 39)
    targets = torch.randint(0, 2, (3, 64), dtype=torch.float32)
    observed = torch.zeros(3, 64, dtype=torch.bool)
    observed[:, ::7] = True

    logits = model(features)
    masked = masked_binary_cross_entropy(logits, targets, observed)
    explicit = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[observed], targets[observed]
    )

    assert logits.shape == (3, 64)
    assert torch.allclose(masked, explicit)
    masked.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_relation_modes_have_equal_parameters_but_distinct_context() -> None:
    independent = make_model("independent")
    true = make_model("true")
    corrupted_player = np.roll(present_player(), 1)
    corrupted = make_model("corrupted", corrupted_player)
    copy_parameters(independent, true)
    copy_parameters(independent, corrupted)
    features = torch.randn(2, 64, 39)

    with torch.no_grad():
        true_logits = true(features)
        corrupted_logits = corrupted(features)

    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in (independent, true, corrupted)]
    assert len(set(counts)) == 1
    assert not torch.allclose(true_logits, corrupted_logits)


def test_profile_operator_is_equivariant_to_cell_relabeling() -> None:
    torch.manual_seed(7)
    player = present_player()
    cell_order = np.asarray([5, 2, 13, 0, 8, 15, 1, 10, 6, 12, 3, 14, 9, 4, 11, 7])
    permutation = np.empty(64, dtype=np.int64)
    for source_cell, target_cell in enumerate(cell_order):
        for lane in range(4):
            permutation[4 * source_cell + lane] = 4 * int(target_cell) + lane
    relabeled_player = np.empty_like(player)
    for source in range(64):
        relabeled_player[permutation[source]] = permutation[player[source]]
    original = make_model("true", player)
    relabeled = make_model("true", relabeled_player)
    copy_parameters(original, relabeled)
    features = torch.randn(2, 64, 39)
    permuted_features = torch.empty_like(features)
    permuted_features[:, permutation] = features

    with torch.no_grad():
        logits = original(features)
        relabeled_logits = relabeled(permuted_features)
    expected = torch.empty_like(logits)
    expected[:, permutation] = logits

    assert torch.max(torch.abs(relabeled_logits - expected)) <= 1e-6


def test_plot_writes_chinese_e66_svg(tmp_path: Path) -> None:
    rows = [
        {
            "relation_mode": mode,
            "train_auc": train_auc,
            "validation_auc": validation_auc,
            "parameter_count": 9000,
        }
        for mode, train_auc, validation_auc in (
            ("independent", 0.64, 0.61),
            ("corrupted", 0.62, 0.58),
            ("true", 0.68, 0.65),
        )
    ]
    summary = {
        "trained_rows": rows,
        "contract": {
            "masked_loss_explicit_max_abs_error": 0.0,
            "cell_relabel_max_abs_error": 1e-7,
            "true_corrupted_logit_max_abs_difference": 0.02,
        },
        "gate": {
            "decision": "innovation2_present_profile_operator_readiness_passed",
            "metrics": {
                "true_minus_independent": 0.04,
                "true_minus_corrupted": 0.07,
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert (
        plot_main(["--summary", str(summary_path), "--output", str(output_path)])
        == 0
    )
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E66" in svg
    assert "逐节点PRESENT平衡谱算子" in svg
