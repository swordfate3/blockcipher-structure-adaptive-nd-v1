from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.baseline.autond_dbitnet2023 import (
    AutoNDDBitNet2023Distinguisher,
)
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    present_inverse_p_indices,
)


class PresentMappedDeltaDBitNet2023Distinguisher(nn.Module):
    """AutoND DBitNet over mapped per-pair PRESENT ciphertext differences."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        mapping_mode: str = "true",
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "PresentMappedDeltaDBitNet2023 expects raw 128-bit ciphertext pairs"
            )
        if input_bits <= 0:
            raise ValueError("input_bits must be positive")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if mapping_mode not in {"true", "shuffled", "raw"}:
            raise ValueError(f"unsupported mapping_mode: {mapping_mode}")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.mapped_input_bits = self.pairs_per_sample * 64
        self.mapping_mode = mapping_mode
        mapping_indices = (
            torch.arange(64, dtype=torch.long)
            if mapping_mode == "raw"
            else present_inverse_p_indices(mapping_mode)
        )
        self.register_buffer("mapping_indices", mapping_indices, persistent=False)

        self.dbitnet = AutoNDDBitNet2023Distinguisher(
            input_bits=self.mapped_input_bits
        )
        for field in (
            "dilations",
            "output_width",
            "output_channels",
            "flattened_width",
            "l2_coefficient",
        ):
            setattr(self, field, getattr(self.dbitnet, field))

    @property
    def last_auxiliary_loss(self) -> torch.Tensor:
        return self.dbitnet.last_auxiliary_loss

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def mapped_delta_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
        return difference.index_select(2, self.mapping_indices).reshape(
            features.shape[0], self.mapped_input_bits
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dbitnet(self.mapped_delta_view(features))


class PresentInvPDBitNet2023Distinguisher(
    PresentMappedDeltaDBitNet2023Distinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_mapping(kwargs, "true")
        super().__init__(*args, **kwargs)


class PresentShuffledPDBitNet2023Distinguisher(
    PresentMappedDeltaDBitNet2023Distinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_mapping(kwargs, "shuffled")
        super().__init__(*args, **kwargs)


class PresentRawDeltaDBitNet2023Distinguisher(
    PresentMappedDeltaDBitNet2023Distinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_mapping(kwargs, "raw")
        super().__init__(*args, **kwargs)


def _set_fixed_mapping(kwargs: dict, fixed_mapping: str) -> None:
    requested_mapping = kwargs.get("mapping_mode", fixed_mapping)
    if requested_mapping != fixed_mapping:
        raise ValueError(
            f"fixed mapping {fixed_mapping!r} received conflicting value "
            f"{requested_mapping!r}"
        )
    kwargs["mapping_mode"] = fixed_mapping


__all__ = [
    "PresentInvPDBitNet2023Distinguisher",
    "PresentMappedDeltaDBitNet2023Distinguisher",
    "PresentRawDeltaDBitNet2023Distinguisher",
    "PresentShuffledPDBitNet2023Distinguisher",
]
