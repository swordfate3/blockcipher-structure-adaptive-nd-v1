from __future__ import annotations

from blockcipher_nd.models.structure.adaptive_dbitnet import (
    StructureAdaptivePairSetDBitNetDistinguisher,
)


class ArxStructureAdaptivePairSetDBitNetDistinguisher(
    StructureAdaptivePairSetDBitNetDistinguisher
):
    """ARX-specialized entry point for the structure-adaptive pair-set DBitNet.

    The underlying architecture is shared with the generic structure-adaptive
    model, but this wrapper fixes the default structure prior to ARX so that
    ARX experiments and paper descriptions have an explicit module boundary.
    """

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 96,
        base_channels: int = 32,
        structure: str = "ARX",
        pooling: str = "attention_mean_max",
    ) -> None:
        super().__init__(
            input_bits=input_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            structure=structure,
            pooling=pooling,
        )


__all__ = ["ArxStructureAdaptivePairSetDBitNetDistinguisher"]
