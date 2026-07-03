from __future__ import annotations

import torch


def present_invp_active_mask_targets(features: torch.Tensor, *, pair_bits: int = 128) -> torch.Tensor:
    """Return InvP(DeltaC) active-cell targets for PRESENT ciphertext-pair bits."""

    if pair_bits != 128:
        raise ValueError("PRESENT active auxiliary targets expect 128-bit ciphertext pairs")
    if features.ndim != 2 or features.shape[1] % pair_bits != 0:
        raise ValueError("features must be a 2D tensor with a multiple of 128 bits")
    inverse_p = torch.tensor(
        [_present_inverse_p_index(index) for index in range(64)],
        dtype=torch.long,
        device=features.device,
    )
    pairs_per_sample = features.shape[1] // pair_bits
    raw_pairs = features.float().reshape(features.shape[0], pairs_per_sample, 2, 64)
    difference = (raw_pairs[:, :, 0, :] - raw_pairs[:, :, 1, :]).abs()
    aligned_difference = difference.index_select(dim=2, index=inverse_p)
    cells = aligned_difference.reshape(features.shape[0], pairs_per_sample, 16, 4)
    return (cells.sum(dim=-1) > 0).to(torch.float32)


def shuffled_active_mask_targets(targets: torch.Tensor, *, seed: int = 20260703) -> torch.Tensor:
    """Return a deterministic shuffled-target control preserving global marginals."""

    if targets.ndim != 3:
        raise ValueError("active mask targets must have shape (batch, pairs, cells)")
    flat = targets.reshape(-1)
    generator = torch.Generator(device=targets.device).manual_seed(seed)
    order = torch.randperm(flat.numel(), generator=generator, device=targets.device)
    return flat.index_select(0, order).reshape_as(targets)


def _present_inverse_p_index(target_bit_index: int) -> int:
    source_lsb_index = _present_inverse_p_lsb_index(63 - target_bit_index)
    return 63 - source_lsb_index


def _present_inverse_p_lsb_index(target_lsb_index: int) -> int:
    if target_lsb_index == 63:
        return 63
    return (16 * target_lsb_index) % 63


__all__ = ["present_invp_active_mask_targets", "shuffled_active_mask_targets"]
