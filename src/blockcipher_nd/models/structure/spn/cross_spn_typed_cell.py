from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.models.common.components import (
    AttentionPooling,
    build_activation,
    build_norm,
)
from blockcipher_nd.models.structure.spn.token_mixer_pairset import (
    EquivariantSpnTokenMixerBlock,
    SpnTokenMixerBlock,
    SpnTokenMixerPairSetDistinguisher,
)
from blockcipher_nd.registry.cipher_factory import build_cipher


_SHUFFLED_MAPPING_SEED = 20260627
_SECOND_SHUFFLED_MAPPING_SEED = 20260715


def shuffled_permutation_indices(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(64, generator=generator)


def cipher_inverse_permutation_indices(
    cipher_key: str,
    mapping_mode: str,
) -> torch.Tensor:
    if mapping_mode == "raw":
        return torch.arange(64, dtype=torch.long)
    if mapping_mode == "shuffled":
        return shuffled_permutation_indices(_SHUFFLED_MAPPING_SEED)
    if mapping_mode != "true":
        raise ValueError(f"unsupported mapping_mode: {mapping_mode}")

    cipher = build_cipher(cipher_key, rounds=1, key=0)
    inverse_permutation = getattr(cipher, "inverse_permutation_layer", None)
    if cipher.block_bits != 64 or inverse_permutation is None:
        raise ValueError(
            f"cipher does not expose a 64-bit inverse permutation: {cipher_key}"
        )

    indices = [-1] * 64
    for source_msb_index in range(64):
        source_state = 1 << (63 - source_msb_index)
        mapped_state = inverse_permutation(source_state)
        if mapped_state <= 0 or mapped_state & (mapped_state - 1):
            raise ValueError(f"inverse permutation is not one-hot for {cipher_key}")
        target_msb_index = 63 - (mapped_state.bit_length() - 1)
        indices[target_msb_index] = source_msb_index
    if sorted(indices) != list(range(64)):
        raise ValueError(f"inverse permutation is not bijective for {cipher_key}")
    return torch.tensor(indices, dtype=torch.long)


class CrossSpnTypedCellPairSetDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        cipher_key: str,
        mapping_mode: str,
        pair_bits: int = 128,
        base_channels: int = 32,
        token_dim: int | None = None,
        mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        pooling: str = "attention_mean_max",
        dropout: float = 0.0,
        position_mode: str = "learned",
        view_encoder_mode: str = "separate",
        cell_mixer_mode: str = "fixed",
        topology_auxiliary_mode: str = "none",
        topology_auxiliary_scale: float = 0.1,
        topology_functional_margin: float = 0.01,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "CrossSpnTypedCell expects raw 128-bit ciphertext pairs"
            )
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of pair_bits")
        if mixer_depth < 1:
            raise ValueError("mixer_depth must be >= 1")
        if pooling not in {"attention", "attention_mean_max", "mean_max"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        if position_mode not in {"learned", "zero"}:
            raise ValueError(f"unsupported position_mode: {position_mode}")
        if view_encoder_mode not in {"separate", "shared_current"}:
            raise ValueError(
                f"unsupported view_encoder_mode: {view_encoder_mode}"
            )
        if cell_mixer_mode not in {"fixed", "equivariant"}:
            raise ValueError(f"unsupported cell_mixer_mode: {cell_mixer_mode}")
        if topology_auxiliary_mode not in {
            "none",
            "off",
            "true_vs_shuffled",
            "shuffled_vs_shuffled",
            "functional_true_vs_shuffled",
            "functional_shuffled_vs_shuffled",
        }:
            raise ValueError(
                "unsupported topology_auxiliary_mode: "
                f"{topology_auxiliary_mode}"
            )
        if topology_auxiliary_scale < 0.0:
            raise ValueError("topology_auxiliary_scale must be non-negative")
        if topology_functional_margin < 0.0:
            raise ValueError("topology_functional_margin must be non-negative")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.cipher_key = cipher_key
        self.mapping_mode = mapping_mode
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.pooling = pooling
        self.position_mode = position_mode
        self.view_encoder_mode = view_encoder_mode
        self.cell_mixer_mode = cell_mixer_mode
        self.embedding_bits = max(32, base_channels * 4)
        self.topology_auxiliary_mode = topology_auxiliary_mode
        self.topology_auxiliary_scale = topology_auxiliary_scale
        self.topology_functional_margin = topology_functional_margin
        self.register_buffer(
            "mapping_indices",
            cipher_inverse_permutation_indices(cipher_key, mapping_mode),
            persistent=False,
        )

        self.current_cell_encoder = nn.Sequential(
            nn.Linear(4, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.previous_cell_encoder = nn.Sequential(
            nn.Linear(4, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.typed_fusion = nn.Sequential(
            nn.Linear(self.token_dim * 2, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 16, self.token_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.mixer_blocks = nn.ModuleList(
            [
                (
                    SpnTokenMixerBlock
                    if cell_mixer_mode == "fixed"
                    else EquivariantSpnTokenMixerBlock
                )(
                    nibbles_per_pair=16,
                    token_dim=self.token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(mixer_depth)
            ]
        )
        self.sequence_norm = build_norm(norm, self.token_dim)
        self.pair_projection = nn.Sequential(
            nn.Linear(self.token_dim * 3, self.embedding_bits),
            build_activation(activation),
            nn.Dropout(dropout),
        )
        self.attention = AttentionPooling(
            self.embedding_bits,
            hidden_bits=max(32, base_channels * 4),
            activation=activation,
            norm=norm,
        )
        pooling_multiplier = 3 if pooling == "attention_mean_max" else 2 if pooling == "mean_max" else 1
        classifier_bits = self.embedding_bits * pooling_multiplier
        self.classifier = nn.Sequential(
            build_norm(norm, classifier_bits),
            nn.Linear(classifier_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )
        self.topology_auxiliary_head: nn.Module | None = None
        if topology_auxiliary_mode != "none":
            auxiliary_hidden_bits = max(32, base_channels * 2)
            self.topology_auxiliary_head = nn.Sequential(
                build_norm(norm, self.embedding_bits),
                nn.Linear(self.embedding_bits, auxiliary_hidden_bits),
                build_activation(activation),
                nn.Linear(auxiliary_hidden_bits, 1),
            )
            true_indices = cipher_inverse_permutation_indices(cipher_key, "true")
            shuffled_indices = shuffled_permutation_indices(
                _SHUFFLED_MAPPING_SEED
            )
            second_shuffled_indices = shuffled_permutation_indices(
                _SECOND_SHUFFLED_MAPPING_SEED
            )
            self.register_buffer(
                "topology_true_indices",
                true_indices,
                persistent=False,
            )
            self.register_buffer(
                "topology_shuffled_a_indices",
                shuffled_indices,
                persistent=False,
            )
            self.register_buffer(
                "topology_shuffled_b_indices",
                second_shuffled_indices,
                persistent=False,
            )
            if topology_auxiliary_mode == "true_vs_shuffled":
                positive_indices = true_indices
                negative_indices = shuffled_indices
            elif topology_auxiliary_mode == "shuffled_vs_shuffled":
                positive_indices = shuffled_indices
                negative_indices = second_shuffled_indices
            else:
                positive_indices = true_indices
                negative_indices = shuffled_indices
            self.register_buffer(
                "topology_auxiliary_positive_indices",
                positive_indices,
                persistent=False,
            )
            self.register_buffer(
                "topology_auxiliary_negative_indices",
                negative_indices,
                persistent=False,
            )
        self.last_attention_weights: torch.Tensor | None = None
        self.last_auxiliary_loss: torch.Tensor | None = None
        self.last_auxiliary_metrics: dict[str, torch.Tensor] = {}
        self._last_functional_logits: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def typed_cell_view(
        self,
        features: torch.Tensor,
        mapping_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
        selected_mapping = (
            self.mapping_indices if mapping_indices is None else mapping_indices
        ).to(difference.device)
        previous_difference = difference.index_select(2, selected_mapping)
        return (
            difference.reshape(features.shape[0], self.pairs_per_sample, 16, 4),
            previous_difference.reshape(
                features.shape[0], self.pairs_per_sample, 16, 4
            ),
        )

    def encode_pairs_with_mapping(
        self,
        features: torch.Tensor,
        mapping_indices: torch.Tensor,
    ) -> torch.Tensor:
        current, previous = self.typed_cell_view(features, mapping_indices)
        batch = features.shape[0]
        current = current.reshape(batch * self.pairs_per_sample, 16, 4)
        previous = previous.reshape(batch * self.pairs_per_sample, 16, 4)
        current_hidden = self.current_cell_encoder(current)
        previous_hidden = (
            self.previous_cell_encoder(previous)
            if self.view_encoder_mode == "separate"
            else self.current_cell_encoder(previous)
        )
        hidden = self.typed_fusion(
            torch.cat([current_hidden, previous_hidden], dim=2)
        )
        position_embedding = (
            self.position_embedding
            if self.position_mode == "learned"
            else self.position_embedding * 0.0
        )
        hidden = hidden + position_embedding
        for block in self.mixer_blocks:
            hidden = block(hidden)
        hidden = self.sequence_norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        activity = current.mean(dim=2, keepdim=True)
        active_embedding = torch.sum(hidden * activity, dim=1) / (
            activity.sum(dim=1).clamp_min(1.0)
        )
        return self.pair_projection(
            torch.cat([mean_embedding, max_embedding, active_embedding], dim=1)
        ).reshape(batch, self.pairs_per_sample, self.embedding_bits)

    def encode_pairs(self, features: torch.Tensor) -> torch.Tensor:
        return self.encode_pairs_with_mapping(features, self.mapping_indices)

    def topology_auxiliary_loss(self, features: torch.Tensor) -> torch.Tensor:
        if self.topology_auxiliary_head is None:
            raise RuntimeError("topology auxiliary head is not configured")
        positive_embeddings = self.encode_pairs_with_mapping(
            features,
            self.topology_auxiliary_positive_indices,
        ).mean(dim=1)
        negative_embeddings = self.encode_pairs_with_mapping(
            features,
            self.topology_auxiliary_negative_indices,
        ).mean(dim=1)
        positive_logits = self.topology_auxiliary_head(positive_embeddings).squeeze(1)
        negative_logits = self.topology_auxiliary_head(negative_embeddings).squeeze(1)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                positive_logits,
                torch.ones_like(positive_logits),
            )
            + F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits),
            )
        )
        return loss * self.topology_auxiliary_scale

    def classify_pair_embeddings(
        self,
        pair_embeddings: torch.Tensor,
        *,
        record_attention: bool = True,
    ) -> torch.Tensor:
        attention_embedding, attention_weights = self.attention(pair_embeddings)
        if record_attention:
            self.last_attention_weights = attention_weights.detach()
        if self.pooling == "attention":
            pooled = attention_embedding
        else:
            mean_embedding = pair_embeddings.mean(dim=1)
            max_embedding = pair_embeddings.max(dim=1).values
            if self.pooling == "mean_max":
                pooled = torch.cat([mean_embedding, max_embedding], dim=1)
            else:
                pooled = torch.cat(
                    [attention_embedding, mean_embedding, max_embedding], dim=1
                )
        return self.classifier(pooled)

    def logits_with_mapping(
        self,
        features: torch.Tensor,
        mapping_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.classify_pair_embeddings(
            self.encode_pairs_with_mapping(features, mapping_indices),
            record_attention=False,
        )

    def compute_auxiliary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_name: str,
    ) -> torch.Tensor | None:
        if self.topology_auxiliary_mode not in {
            "functional_true_vs_shuffled",
            "functional_shuffled_vs_shuffled",
        }:
            return self.last_auxiliary_loss
        if self._last_functional_logits is None:
            raise RuntimeError("functional topology logits are unavailable")
        shuffled_a_logits, shuffled_b_logits = self._last_functional_logits
        shuffled_a_loss = self._per_sample_classification_loss(
            shuffled_a_logits.squeeze(1), labels, loss_name
        )
        shuffled_b_loss = self._per_sample_classification_loss(
            shuffled_b_logits.squeeze(1), labels, loss_name
        )
        if self.topology_auxiliary_mode == "functional_true_vs_shuffled":
            preferred_loss = self._per_sample_classification_loss(
                logits,
                labels,
                loss_name,
            )
            comparison_loss = 0.5 * (shuffled_a_loss + shuffled_b_loss)
        else:
            preferred_loss = shuffled_a_loss
            comparison_loss = shuffled_b_loss
        margin_values = F.relu(
            self.topology_functional_margin
            + preferred_loss
            - comparison_loss
        )
        auxiliary_loss = self.topology_auxiliary_scale * margin_values.mean()
        self.last_auxiliary_loss = auxiliary_loss
        self.last_auxiliary_metrics = {
            "functional_preferred_loss": preferred_loss.detach().mean(),
            "functional_comparison_loss": comparison_loss.detach().mean(),
            "functional_loss_gap": (
                comparison_loss.detach().mean() - preferred_loss.detach().mean()
            ),
            "functional_margin_loss": margin_values.detach().mean(),
            "functional_violation_rate": (margin_values.detach() > 0.0)
            .float()
            .mean(),
        }
        return auxiliary_loss

    @staticmethod
    def _per_sample_classification_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_name: str,
    ) -> torch.Tensor:
        if loss_name == "mse":
            return F.mse_loss(torch.sigmoid(logits), labels, reduction="none")
        if loss_name == "bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
        raise ValueError(f"unsupported loss: {loss_name}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_embeddings = self.encode_pairs(features)
        logits = self.classify_pair_embeddings(pair_embeddings)
        self.last_auxiliary_metrics = {}
        self._last_functional_logits = None
        self.last_auxiliary_loss = (
            self.topology_auxiliary_loss(features)
            if self.training
            and self.topology_auxiliary_mode
            in {"true_vs_shuffled", "shuffled_vs_shuffled"}
            else None
        )
        if self.training and self.topology_auxiliary_mode in {
            "functional_true_vs_shuffled",
            "functional_shuffled_vs_shuffled",
        }:
            self._last_functional_logits = (
                self.logits_with_mapping(features, self.topology_shuffled_a_indices),
                self.logits_with_mapping(features, self.topology_shuffled_b_indices),
            )
        return logits


def _set_fixed_adapter(
    kwargs: dict[str, object],
    *,
    cipher_key: str,
    mapping_mode: str,
) -> None:
    requested_cipher = kwargs.get("cipher_key", cipher_key)
    requested_mapping = kwargs.get("mapping_mode", mapping_mode)
    if requested_cipher != cipher_key:
        raise ValueError(
            f"fixed cipher {cipher_key!r} received conflicting value "
            f"{requested_cipher!r}"
        )
    if requested_mapping != mapping_mode:
        raise ValueError(
            f"fixed mapping {mapping_mode!r} received conflicting value "
            f"{requested_mapping!r}"
        )
    kwargs["cipher_key"] = cipher_key
    kwargs["mapping_mode"] = mapping_mode


class PresentCrossSpnTypedCellTrueDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellShuffledDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="shuffled")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellRawDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="raw")
        super().__init__(*args, **kwargs)


def _set_fixed_topology_auxiliary_mode(
    kwargs: dict[str, object],
    mode: str,
) -> None:
    requested_mode = kwargs.get("topology_auxiliary_mode", mode)
    if requested_mode != mode:
        raise ValueError(
            f"fixed topology auxiliary mode {mode!r} received conflicting "
            f"value {requested_mode!r}"
        )
    kwargs["topology_auxiliary_mode"] = mode


class PresentCrossSpnTypedCellE5OffDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(kwargs, "off")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellE5TrueShuffledDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(kwargs, "true_vs_shuffled")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(kwargs, "shuffled_vs_shuffled")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellE6OffDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(kwargs, "off")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(
            kwargs,
            "functional_true_vs_shuffled",
        )
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(
            kwargs,
            "functional_shuffled_vs_shuffled",
        )
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellTrueDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="true")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellNoPositionDistinguisher(
    GiftCrossSpnTypedCellTrueDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        requested_mode = kwargs.get("position_mode", "zero")
        if requested_mode != "zero":
            raise ValueError(
                "fixed position mode 'zero' received conflicting value "
                f"{requested_mode!r}"
            )
        kwargs["position_mode"] = "zero"
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellSharedViewEncoderDistinguisher(
    GiftCrossSpnTypedCellNoPositionDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        requested_mode = kwargs.get("view_encoder_mode", "shared_current")
        if requested_mode != "shared_current":
            raise ValueError(
                "fixed view encoder mode 'shared_current' received conflicting value "
                f"{requested_mode!r}"
            )
        kwargs["view_encoder_mode"] = "shared_current"
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellEquivariantMixerDistinguisher(
    GiftCrossSpnTypedCellSharedViewEncoderDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        requested_mode = kwargs.get("cell_mixer_mode", "equivariant")
        if requested_mode != "equivariant":
            raise ValueError(
                "fixed cell mixer mode 'equivariant' received conflicting value "
                f"{requested_mode!r}"
            )
        kwargs["cell_mixer_mode"] = "equivariant"
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellShuffledDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="shuffled")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellRawDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="raw")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellTrueFromPresentTrueDistinguisher(
    GiftCrossSpnTypedCellTrueDistinguisher
):
    pass


class GiftCrossSpnTypedCellTrueFromPresentShuffledDistinguisher(
    GiftCrossSpnTypedCellTrueDistinguisher
):
    pass


class GiftCrossSpnTypedCellShuffledFromPresentTrueDistinguisher(
    GiftCrossSpnTypedCellShuffledDistinguisher
):
    pass


class GiftCrossSpnTypedCellE5Distinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(kwargs, "off")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellE5ScratchDistinguisher(
    GiftCrossSpnTypedCellE5Distinguisher
):
    pass


class GiftCrossSpnTypedCellE5FromPresentOffDistinguisher(
    GiftCrossSpnTypedCellE5Distinguisher
):
    pass


class GiftCrossSpnTypedCellE5FromPresentTrueShuffledDistinguisher(
    GiftCrossSpnTypedCellE5Distinguisher
):
    pass


class GiftCrossSpnTypedCellE5FromPresentShuffledPlaceboDistinguisher(
    GiftCrossSpnTypedCellE5Distinguisher
):
    pass


class GiftCrossSpnTypedCellE6Distinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="true")
        _set_fixed_topology_auxiliary_mode(kwargs, "off")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellE6ScratchDistinguisher(
    GiftCrossSpnTypedCellE6Distinguisher
):
    pass


class GiftCrossSpnTypedCellE6FromPresentOffDistinguisher(
    GiftCrossSpnTypedCellE6Distinguisher
):
    pass


class GiftCrossSpnTypedCellE6FromPresentFunctionalMarginDistinguisher(
    GiftCrossSpnTypedCellE6Distinguisher
):
    pass


class GiftCrossSpnTypedCellE6FromPresentShuffledPlaceboDistinguisher(
    GiftCrossSpnTypedCellE6Distinguisher
):
    pass


class GiftAlignedTokenMixerRawInputDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        token_dim: int | None = None,
        mixer_depth: int = 1,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        pooling: str = "topk_logsumexp",
        dropout: float = 0.0,
        top_k: int = 2,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "GiftAlignedTokenMixerRawInput expects raw 128-bit ciphertext pairs"
            )
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of pair_bits")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.register_buffer(
            "mapping_indices",
            cipher_inverse_permutation_indices("gift64", "true"),
            persistent=False,
        )
        self.delegate = SpnTokenMixerPairSetDistinguisher(
            input_bits=self.pairs_per_sample * 256,
            pair_bits=256,
            base_channels=base_channels,
            token_dim=token_dim,
            mixer_depth=mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            pooling=pooling,
            dropout=dropout,
            top_k=top_k,
            lse_temperature=lse_temperature,
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def aligned_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
        mapped_difference = difference.index_select(2, self.mapping_indices)
        return torch.cat(
            [pairs[:, :, 0], pairs[:, :, 1], difference, mapped_difference],
            dim=2,
        ).reshape(features.shape[0], self.pairs_per_sample * 256)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.delegate(self.aligned_view(features))


__all__ = [
    "CrossSpnTypedCellPairSetDistinguisher",
    "GiftAlignedTokenMixerRawInputDistinguisher",
    "GiftCrossSpnTypedCellE5Distinguisher",
    "GiftCrossSpnTypedCellE5FromPresentOffDistinguisher",
    "GiftCrossSpnTypedCellE5FromPresentShuffledPlaceboDistinguisher",
    "GiftCrossSpnTypedCellE5FromPresentTrueShuffledDistinguisher",
    "GiftCrossSpnTypedCellE5ScratchDistinguisher",
    "GiftCrossSpnTypedCellE6Distinguisher",
    "GiftCrossSpnTypedCellE6FromPresentFunctionalMarginDistinguisher",
    "GiftCrossSpnTypedCellE6FromPresentOffDistinguisher",
    "GiftCrossSpnTypedCellE6FromPresentShuffledPlaceboDistinguisher",
    "GiftCrossSpnTypedCellE6ScratchDistinguisher",
    "GiftCrossSpnTypedCellRawDistinguisher",
    "GiftCrossSpnTypedCellNoPositionDistinguisher",
    "GiftCrossSpnTypedCellSharedViewEncoderDistinguisher",
    "GiftCrossSpnTypedCellEquivariantMixerDistinguisher",
    "GiftCrossSpnTypedCellShuffledFromPresentTrueDistinguisher",
    "GiftCrossSpnTypedCellShuffledDistinguisher",
    "GiftCrossSpnTypedCellTrueFromPresentShuffledDistinguisher",
    "GiftCrossSpnTypedCellTrueFromPresentTrueDistinguisher",
    "GiftCrossSpnTypedCellTrueDistinguisher",
    "PresentCrossSpnTypedCellRawDistinguisher",
    "PresentCrossSpnTypedCellE5OffDistinguisher",
    "PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher",
    "PresentCrossSpnTypedCellE5TrueShuffledDistinguisher",
    "PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher",
    "PresentCrossSpnTypedCellE6OffDistinguisher",
    "PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher",
    "PresentCrossSpnTypedCellShuffledDistinguisher",
    "PresentCrossSpnTypedCellTrueDistinguisher",
    "cipher_inverse_permutation_indices",
    "shuffled_permutation_indices",
]
