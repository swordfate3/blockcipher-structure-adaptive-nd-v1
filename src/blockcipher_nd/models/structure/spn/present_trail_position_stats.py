from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm


class PresentTrailPositionStatsPairSetDistinguisher(nn.Module):
    """PRESENT position-aware statistics for S-box-DDT beamstats pair sets.

    The model targets encodings such as
    ``present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits``.
    It keeps per-depth, per-word, and per-cell activity instead of collapsing
    the public trail evidence into only global density values.
    """

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 2496,
        base_channels: int = 32,
        nibble_bits: int = 4,
        trail_depth: int = 4,
        trail_words_per_depth: int = 9,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        stats_hidden_bits: int | None = None,
        metadata_bits: int = 0,
        active_conditioning: str = "none",
    ) -> None:
        super().__init__()
        if active_conditioning not in {"none", "relative_stats", "p_layer_relative_stats"}:
            raise ValueError(
                "PresentTrailPositionStats active_conditioning must be none, "
                "relative_stats, or p_layer_relative_stats"
            )
        if metadata_bits < 0:
            raise ValueError("PresentTrailPositionStats metadata_bits must be non-negative")
        base_input_bits = input_bits - metadata_bits
        if base_input_bits <= 0:
            raise ValueError("PresentTrailPositionStats metadata_bits must leave base feature bits")
        if base_input_bits % pair_bits != 0:
            raise ValueError("PresentTrailPositionStats input_bits must be a multiple of pair_bits")
        if pair_bits % 64 != 0:
            raise ValueError("PresentTrailPositionStats pair_bits must be a multiple of 64-bit PRESENT words")
        if pair_bits % nibble_bits != 0:
            raise ValueError("PresentTrailPositionStats pair_bits must be a multiple of nibble_bits")
        if base_input_bits // pair_bits < 2:
            raise ValueError("PresentTrailPositionStats needs at least two pairs per sample")
        words_per_pair = pair_bits // 64
        if words_per_pair <= trail_depth * trail_words_per_depth:
            raise ValueError("PresentTrailPositionStats requires prefix words before trail words")

        self.input_bits = input_bits
        self.base_input_bits = base_input_bits
        self.metadata_bits = metadata_bits
        self.active_conditioning = active_conditioning
        self.pair_bits = pair_bits
        self.pairs_per_sample = base_input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.words_per_pair = words_per_pair
        self.cells_per_word = 64 // nibble_bits
        self.trail_depth = trail_depth
        self.trail_words_per_depth = trail_words_per_depth
        self.prefix_words = self.words_per_pair - trail_depth * trail_words_per_depth
        if self.active_conditioning != "none" and self.metadata_bits != self.cells_per_word:
            raise ValueError(
                "PresentTrailPositionStats active conditioning requires one metadata bit per PRESENT cell"
            )
        if self.active_conditioning == "p_layer_relative_stats" and self.cells_per_word != 16:
            raise ValueError("PresentTrailPositionStats p_layer_relative_stats requires 16 PRESENT cells")
        if self.active_conditioning == "p_layer_relative_stats":
            self.register_buffer(
                "active_cell_permutations",
                torch.tensor(_present_p_layer_relative_cell_permutations(), dtype=torch.long),
                persistent=False,
            )
        self.stats_hidden_bits = stats_hidden_bits or max(64, base_channels * 8)
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.stats_feature_bits = self.position_stats_feature_bits(
            self.words_per_pair,
            self.cells_per_word,
            trail_depth,
            self.prefix_words,
            trail_words_per_depth,
        ) + metadata_bits
        classifier_hidden = max(64, base_channels * 4)
        self.classifier = nn.Sequential(
            build_norm(norm, self.stats_feature_bits),
            nn.Linear(self.stats_feature_bits, self.stats_hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.stats_hidden_bits, classifier_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    @staticmethod
    def position_stats_feature_bits(
        words_per_pair: int,
        cells_per_word: int,
        trail_depth: int,
        prefix_words: int,
        trail_words_per_depth: int,
    ) -> int:
        return (
            words_per_pair * cells_per_word * 2
            + words_per_pair * 4
            + cells_per_word * 4
            + trail_depth * trail_words_per_depth * cells_per_word * 3
            + trail_depth * cells_per_word * 4
            + trail_depth * trail_words_per_depth * 4
            + prefix_words * cells_per_word * 2
            + 16
        )

    def _position_statistics(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        base_features = features[:, : self.base_input_bits]
        cells = base_features.float().reshape(
            base_features.shape[0],
            self.pairs_per_sample,
            self.words_per_pair,
            self.cells_per_word,
            self.nibble_bits,
        )
        activity = cells.mean(dim=-1)
        if self.active_conditioning != "none":
            activity = self._active_relative_activity(activity, features)
        word_activity = activity.mean(dim=-1)
        cell_activity = activity.mean(dim=2)

        word_cell_mean = activity.mean(dim=1)
        word_cell_std = activity.std(dim=1, unbiased=False)

        word_mean = word_activity.mean(dim=1)
        word_std = word_activity.std(dim=1, unbiased=False)
        word_first_last = word_activity[:, -1] - word_activity[:, 0]
        word_span = word_activity.amax(dim=1) - word_activity.amin(dim=1)

        cell_mean = cell_activity.mean(dim=1)
        cell_std = cell_activity.std(dim=1, unbiased=False)
        cell_first_last = cell_activity[:, -1] - cell_activity[:, 0]
        cell_span = cell_activity.amax(dim=1) - cell_activity.amin(dim=1)

        prefix = activity[:, :, : self.prefix_words]
        trail = activity[:, :, self.prefix_words :].reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.trail_depth,
            self.trail_words_per_depth,
            self.cells_per_word,
        )

        depth_word_cell_mean = trail.mean(dim=1)
        depth_word_cell_std = trail.std(dim=1, unbiased=False)
        depth_word_cell_span = trail.amax(dim=1) - trail.amin(dim=1)

        depth_cell_activity = trail.mean(dim=3)
        depth_cell_mean = depth_cell_activity.mean(dim=1)
        depth_cell_std = depth_cell_activity.std(dim=1, unbiased=False)
        depth_cell_first_last = depth_cell_activity[:, -1] - depth_cell_activity[:, 0]
        depth_cell_span = depth_cell_activity.amax(dim=1) - depth_cell_activity.amin(dim=1)

        depth_word_activity = trail.mean(dim=4)
        depth_word_mean = depth_word_activity.mean(dim=1)
        depth_word_std = depth_word_activity.std(dim=1, unbiased=False)
        depth_word_first_last = depth_word_activity[:, -1] - depth_word_activity[:, 0]
        depth_word_span = depth_word_activity.amax(dim=1) - depth_word_activity.amin(dim=1)

        prefix_mean = prefix.mean(dim=1)
        prefix_std = prefix.std(dim=1, unbiased=False)

        global_stats = self._global_position_stats(activity, trail)

        return torch.cat(
            [
                word_cell_mean.flatten(1),
                word_cell_std.flatten(1),
                word_mean,
                word_std,
                word_first_last,
                word_span,
                cell_mean,
                cell_std,
                cell_first_last,
                cell_span,
                depth_word_cell_mean.flatten(1),
                depth_word_cell_std.flatten(1),
                depth_word_cell_span.flatten(1),
                depth_cell_mean.flatten(1),
                depth_cell_std.flatten(1),
                depth_cell_first_last.flatten(1),
                depth_cell_span.flatten(1),
                depth_word_mean.flatten(1),
                depth_word_std.flatten(1),
                depth_word_first_last.flatten(1),
                depth_word_span.flatten(1),
                prefix_mean.flatten(1),
                prefix_std.flatten(1),
                global_stats,
            ],
            dim=1,
        )

    def _active_relative_activity(
        self,
        activity: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        active_metadata = features[:, -self.metadata_bits :].float()
        active_indices = active_metadata.argmax(dim=1)
        if self.active_conditioning == "p_layer_relative_stats":
            source_cells = self.active_cell_permutations.to(activity.device).index_select(
                dim=0,
                index=active_indices,
            )
        else:
            relative_cells = torch.arange(self.cells_per_word, device=activity.device)
            source_cells = (active_indices[:, None] + relative_cells[None, :]) % self.cells_per_word
        gather_index = source_cells[:, None, None, :].expand(
            -1,
            self.pairs_per_sample,
            self.words_per_pair,
            -1,
        )
        return torch.gather(activity, dim=3, index=gather_index)

    def _global_position_stats(self, activity: torch.Tensor, trail: torch.Tensor) -> torch.Tensor:
        pair_density = activity.mean(dim=(2, 3))
        trail_density = trail.mean(dim=(2, 3, 4))
        low_cells = activity[..., :4].mean(dim=(2, 3))
        mid_cells = activity[..., 4:12].mean(dim=(2, 3))
        high_cells = activity[..., 12:].mean(dim=(2, 3))
        even_odd = (
            activity[..., ::2].mean(dim=(2, 3))
            - activity[..., 1::2].mean(dim=(2, 3))
        )
        return torch.stack(
            [
                pair_density.mean(dim=1),
                pair_density.std(dim=1, unbiased=False),
                pair_density[:, -1] - pair_density[:, 0],
                pair_density.amax(dim=1) - pair_density.amin(dim=1),
                trail_density.mean(dim=1),
                trail_density.std(dim=1, unbiased=False),
                trail_density[:, -1] - trail_density[:, 0],
                trail_density.amax(dim=1) - trail_density.amin(dim=1),
                low_cells.mean(dim=1),
                mid_cells.mean(dim=1),
                high_cells.mean(dim=1),
                even_odd.mean(dim=1),
                low_cells[:, -1] - low_cells[:, 0],
                mid_cells[:, -1] - mid_cells[:, 0],
                high_cells[:, -1] - high_cells[:, 0],
                even_odd[:, -1] - even_odd[:, 0],
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stats = self._position_statistics(features)
        if self.metadata_bits:
            stats = torch.cat([stats, features[:, -self.metadata_bits :].float()], dim=1)
        return self.classifier(stats)


def _present_p_layer_relative_cell_permutations() -> list[list[int]]:
    permutations: list[list[int]] = []
    for active_nibble in range(16):
        source_cell = 15 - active_nibble
        p_targets = _present_p_layer_target_cells(active_nibble)
        head: list[int] = []
        for cell in [source_cell, *p_targets]:
            if cell not in head:
                head.append(cell)
        tail = sorted(
            (cell for cell in range(16) if cell not in head),
            key=lambda cell: (abs(cell - source_cell), cell),
        )
        permutations.append(head + tail)
    return permutations


def _present_p_layer_target_cells(source_nibble: int) -> list[int]:
    targets: list[int] = []
    for bit_offset in range(4):
        source_bit = source_nibble * 4 + bit_offset
        target_bit = 63 if source_bit == 63 else (16 * source_bit) % 63
        target_cell = 15 - (target_bit // 4)
        if target_cell not in targets:
            targets.append(target_cell)
    return targets


__all__ = ["PresentTrailPositionStatsPairSetDistinguisher"]
