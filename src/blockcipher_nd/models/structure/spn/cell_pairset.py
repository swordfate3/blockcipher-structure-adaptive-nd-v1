from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.structure.adaptive_dbitnet import StructureConditionedDBitNetEncoder


class SpnCellPairSetDBitNetDistinguisher(nn.Module):
    """SPN-focused PairSet DBitNet with explicit 4-bit cell encoding."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 192,
        base_channels: int = 32,
        cell_bits: int = 4,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("SpnCellPairSetDBitNet input_bits must be a multiple of pair_bits")
        if pair_bits % cell_bits != 0:
            raise ValueError("SpnCellPairSetDBitNet pair_bits must be a multiple of cell_bits")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.cell_bits = cell_bits
        self.cells_per_pair = pair_bits // cell_bits
        self.encoder = StructureConditionedDBitNetEncoder(
            pair_bits,
            base_channels=base_channels,
            structure="SPN",
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_bits, base_channels),
            nn.GELU(),
            nn.Linear(base_channels, base_channels),
            nn.GELU(),
        )
        self.cell_embedding_bits = base_channels * 4
        self.fused_pair_embedding_bits = self.encoder.embedding_bits + self.cell_embedding_bits
        self.cell_projection = nn.Sequential(
            nn.Linear(base_channels * 2, self.cell_embedding_bits),
            nn.GELU(),
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(self.fused_pair_embedding_bits),
            nn.Linear(self.fused_pair_embedding_bits, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fused_pair_embedding_bits * 3),
            nn.Linear(self.fused_pair_embedding_bits * 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        if structure != "SPN":
            return
        self.encoder.set_cipher_structure(structure)

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def _cell_embedding(self, pair_features: torch.Tensor) -> torch.Tensor:
        cells = pair_features.reshape(
            pair_features.shape[0] * self.cells_per_pair,
            self.cell_bits,
        )
        cell_embeddings = self.cell_encoder(cells).reshape(
            pair_features.shape[0],
            self.cells_per_pair,
            -1,
        )
        mean_embedding = cell_embeddings.mean(dim=1)
        max_embedding = cell_embeddings.max(dim=1).values
        return self.cell_projection(torch.cat([mean_embedding, max_embedding], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.float().reshape(
            features.shape[0] * self.pairs_per_sample,
            self.pair_bits,
        )
        dbit_embeddings = self.encoder(pair_features)
        cell_embeddings = self._cell_embedding(pair_features)
        fused_pair_embeddings = torch.cat([dbit_embeddings, cell_embeddings], dim=1).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.fused_pair_embedding_bits,
        )
        attention_logits = self.attention(fused_pair_embeddings).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        self.last_attention_weights = attention_weights.detach()
        attention_embedding = torch.sum(
            fused_pair_embeddings * attention_weights.unsqueeze(-1),
            dim=1,
        )
        mean_embedding = fused_pair_embeddings.mean(dim=1)
        max_embedding = fused_pair_embeddings.max(dim=1).values
        pooled = torch.cat([attention_embedding, mean_embedding, max_embedding], dim=1)
        return self.classifier(pooled)



__all__ = ["SpnCellPairSetDBitNetDistinguisher"]
