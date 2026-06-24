from __future__ import annotations


def int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def pair_to_bits(left: int, right: int, width: int) -> list[int]:
    return int_to_bits(left, width) + int_to_bits(right, width)


def xor_bits(left: int, right: int, width: int) -> list[int]:
    return int_to_bits(left ^ right, width)


def pair_xor_bits(left: int, right: int, width: int) -> tuple[list[int], list[int], list[int]]:
    left_bits = int_to_bits(left, width)
    right_bits = int_to_bits(right, width)
    difference_bits = [left_bit ^ right_bit for left_bit, right_bit in zip(left_bits, right_bits)]
    return left_bits, right_bits, difference_bits
