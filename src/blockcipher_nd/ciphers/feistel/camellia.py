"""Camellia block cipher implementations from RFC 3713."""

from __future__ import annotations

from dataclasses import dataclass

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK128 = (1 << 128) - 1
_SIGMA = (
    0xA09E667F3BCC908B,
    0xB67AE8584CAA73B2,
    0xC6EF372FE94F82BE,
    0x54FF53A5F1D36F1C,
    0x10E527FADE682D1D,
    0xB05688C2B3E6C1FD,
)

SBOX1 = (
    112,130,44,236,179,39,192,229,228,133,87,53,234,12,174,65,
    35,239,107,147,69,25,165,33,237,14,79,78,29,101,146,189,
    134,184,175,143,124,235,31,206,62,48,220,95,94,197,11,26,
    166,225,57,202,213,71,93,61,217,1,90,214,81,86,108,77,
    139,13,154,102,251,204,176,45,116,18,43,32,240,177,132,153,
    223,76,203,194,52,126,118,5,109,183,169,49,209,23,4,215,
    20,88,58,97,222,27,17,28,50,15,156,22,83,24,242,34,
    254,68,207,178,195,181,122,145,36,8,232,168,96,252,105,80,
    170,208,160,125,161,137,98,151,84,91,30,149,224,255,100,210,
    16,196,0,72,163,247,117,219,138,3,230,218,9,63,221,148,
    135,92,131,2,205,74,144,51,115,103,246,243,157,127,191,226,
    82,155,216,38,200,55,198,59,129,150,111,75,19,190,99,46,
    233,121,167,140,159,110,188,142,41,245,249,182,47,253,180,89,
    120,152,6,106,231,70,113,186,212,37,171,66,136,162,141,250,
    114,7,185,85,248,238,172,10,54,73,42,104,60,56,241,164,
    64,40,211,123,187,201,67,193,21,227,173,244,119,199,128,158,
)


def _rol8(value: int, amount: int) -> int:
    amount &= 7
    return ((value << amount) | (value >> (8 - amount))) & 0xFF


SBOX2 = tuple(_rol8(x, 1) for x in SBOX1)
SBOX3 = tuple(_rol8(x, 7) for x in SBOX1)
SBOX4 = tuple(SBOX1[_rol8(x, 1)] for x in range(256))


def _rol32(value: int, amount: int) -> int:
    amount &= 31
    return ((value << amount) | (value >> (32 - amount))) & _MASK32


def _rol128(value: int, amount: int) -> int:
    amount %= 128
    return ((value << amount) | (value >> (128 - amount))) & _MASK128


def _left(rotated: int) -> int:
    return (rotated >> 64) & _MASK64


def _right(rotated: int) -> int:
    return rotated & _MASK64


def _f(value: int, key: int) -> int:
    x = value ^ key
    t1 = SBOX1[(x >> 56) & 0xFF]
    t2 = SBOX2[(x >> 48) & 0xFF]
    t3 = SBOX3[(x >> 40) & 0xFF]
    t4 = SBOX4[(x >> 32) & 0xFF]
    t5 = SBOX2[(x >> 24) & 0xFF]
    t6 = SBOX3[(x >> 16) & 0xFF]
    t7 = SBOX4[(x >> 8) & 0xFF]
    t8 = SBOX1[x & 0xFF]
    y1 = t1 ^ t3 ^ t4 ^ t6 ^ t7 ^ t8
    y2 = t1 ^ t2 ^ t4 ^ t5 ^ t7 ^ t8
    y3 = t1 ^ t2 ^ t3 ^ t5 ^ t6 ^ t8
    y4 = t2 ^ t3 ^ t4 ^ t5 ^ t6 ^ t7
    y5 = t1 ^ t2 ^ t6 ^ t7 ^ t8
    y6 = t2 ^ t3 ^ t5 ^ t7 ^ t8
    y7 = t3 ^ t4 ^ t5 ^ t6 ^ t8
    y8 = t1 ^ t4 ^ t5 ^ t6 ^ t7
    return (y1 << 56) | (y2 << 48) | (y3 << 40) | (y4 << 32) | (y5 << 24) | (y6 << 16) | (y7 << 8) | y8


def _fl(value: int, key: int) -> int:
    x1 = (value >> 32) & _MASK32
    x2 = value & _MASK32
    k1 = (key >> 32) & _MASK32
    k2 = key & _MASK32
    x2 ^= _rol32(x1 & k1, 1)
    x1 ^= x2 | k2
    return ((x1 & _MASK32) << 32) | (x2 & _MASK32)


def _flinv(value: int, key: int) -> int:
    y1 = (value >> 32) & _MASK32
    y2 = value & _MASK32
    k1 = (key >> 32) & _MASK32
    k2 = key & _MASK32
    y1 ^= y2 | k2
    y2 ^= _rol32(y1 & k1, 1)
    return ((y1 & _MASK32) << 32) | (y2 & _MASK32)


def _round_count_for_key_bits(key_bits: int) -> int:
    if key_bits == 128:
        return 18
    if key_bits in {192, 256}:
        return 24
    raise ValueError(f"unsupported Camellia key size: {key_bits}")


def _make_ka_kb(kl: int, kr: int) -> tuple[int, int]:
    d1 = ((kl ^ kr) >> 64) & _MASK64
    d2 = (kl ^ kr) & _MASK64
    d2 ^= _f(d1, _SIGMA[0])
    d1 ^= _f(d2, _SIGMA[1])
    d1 ^= (kl >> 64) & _MASK64
    d2 ^= kl & _MASK64
    d2 ^= _f(d1, _SIGMA[2])
    d1 ^= _f(d2, _SIGMA[3])
    ka = ((d1 & _MASK64) << 64) | (d2 & _MASK64)
    d1 = ((ka ^ kr) >> 64) & _MASK64
    d2 = (ka ^ kr) & _MASK64
    d2 ^= _f(d1, _SIGMA[4])
    d1 ^= _f(d2, _SIGMA[5])
    kb = ((d1 & _MASK64) << 64) | (d2 & _MASK64)
    return ka, kb


def _expand_key(key: int, key_bits: int) -> tuple[list[int], list[int], list[int]]:
    if key < 0 or key >= (1 << key_bits):
        raise ValueError(f"Camellia key must fit in {key_bits} bits")
    if key_bits == 128:
        kl = key
        kr = 0
    elif key_bits == 192:
        kl = (key >> 64) & _MASK128
        tail = key & _MASK64
        kr = ((tail << 64) | ((~tail) & _MASK64)) & _MASK128
    elif key_bits == 256:
        kl = (key >> 128) & _MASK128
        kr = key & _MASK128
    else:
        raise ValueError(f"unsupported Camellia key size: {key_bits}")
    ka, kb = _make_ka_kb(kl, kr)

    if key_bits == 128:
        kw = [_left(_rol128(kl, 0)), _right(_rol128(kl, 0)), _left(_rol128(ka, 111)), _right(_rol128(ka, 111))]
        k = [
            _left(_rol128(ka, 0)), _right(_rol128(ka, 0)),
            _left(_rol128(kl, 15)), _right(_rol128(kl, 15)),
            _left(_rol128(ka, 15)), _right(_rol128(ka, 15)),
            _left(_rol128(kl, 45)), _right(_rol128(kl, 45)),
            _left(_rol128(ka, 45)), _right(_rol128(kl, 60)),
            _left(_rol128(ka, 60)), _right(_rol128(ka, 60)),
            _left(_rol128(kl, 94)), _right(_rol128(kl, 94)),
            _left(_rol128(ka, 94)), _right(_rol128(ka, 94)),
            _left(_rol128(kl, 111)), _right(_rol128(kl, 111)),
        ]
        ke = [_left(_rol128(ka, 30)), _right(_rol128(ka, 30)), _left(_rol128(kl, 77)), _right(_rol128(kl, 77))]
        return kw, k, ke

    kw = [_left(_rol128(kl, 0)), _right(_rol128(kl, 0)), _left(_rol128(kb, 111)), _right(_rol128(kb, 111))]
    k = [
        _left(_rol128(kb, 0)), _right(_rol128(kb, 0)),
        _left(_rol128(kr, 15)), _right(_rol128(kr, 15)),
        _left(_rol128(ka, 15)), _right(_rol128(ka, 15)),
        _left(_rol128(kb, 30)), _right(_rol128(kb, 30)),
        _left(_rol128(kl, 45)), _right(_rol128(kl, 45)),
        _left(_rol128(ka, 45)), _right(_rol128(ka, 45)),
        _left(_rol128(kr, 60)), _right(_rol128(kr, 60)),
        _left(_rol128(kb, 60)), _right(_rol128(kb, 60)),
        _left(_rol128(kl, 77)), _right(_rol128(kl, 77)),
        _left(_rol128(kr, 94)), _right(_rol128(kr, 94)),
        _left(_rol128(ka, 94)), _right(_rol128(ka, 94)),
        _left(_rol128(kl, 111)), _right(_rol128(kl, 111)),
    ]
    ke = [
        _left(_rol128(kr, 30)), _right(_rol128(kr, 30)),
        _left(_rol128(kl, 60)), _right(_rol128(kl, 60)),
        _left(_rol128(ka, 77)), _right(_rol128(ka, 77)),
    ]
    return kw, k, ke


@dataclass(frozen=True)
class Camellia:
    rounds: int
    key: int
    key_bits: int = 128
    name: str = "Camellia"
    structure: str = "Feistel-like"
    block_bits: int = 128

    def __post_init__(self) -> None:
        full_rounds = _round_count_for_key_bits(self.key_bits)
        if self.rounds < 1 or self.rounds > full_rounds:
            raise ValueError(f"Camellia-{self.key_bits} supports 1..{full_rounds} rounds")
        if self.key < 0 or self.key >= (1 << self.key_bits):
            raise ValueError(f"Camellia key must fit in {self.key_bits} bits")

    def encrypt(self, plaintext: int) -> int:
        if plaintext < 0 or plaintext >= (1 << 128):
            raise ValueError("Camellia plaintext must fit in 128 bits")
        kw, k, ke = _expand_key(self.key, self.key_bits)
        d1 = ((plaintext >> 64) & _MASK64) ^ kw[0]
        d2 = (plaintext & _MASK64) ^ kw[1]
        for index in range(self.rounds):
            if index % 2 == 0:
                d2 ^= _f(d1, k[index])
            else:
                d1 ^= _f(d2, k[index])
            if index == 5 and self.rounds > 6:
                d1 = _fl(d1, ke[0])
                d2 = _flinv(d2, ke[1])
            elif index == 11 and self.rounds > 12:
                d1 = _fl(d1, ke[2])
                d2 = _flinv(d2, ke[3])
            elif index == 17 and self.key_bits != 128 and self.rounds > 18:
                d1 = _fl(d1, ke[4])
                d2 = _flinv(d2, ke[5])
        if self.rounds == _round_count_for_key_bits(self.key_bits):
            d2 ^= kw[2]
            d1 ^= kw[3]
        return ((d2 & _MASK64) << 64) | (d1 & _MASK64)


class Camellia128(Camellia):
    def __init__(self, rounds: int = 18, key: int = 0) -> None:
        super().__init__(rounds=rounds, key=key, key_bits=128, name="Camellia-128")


class Camellia192(Camellia):
    def __init__(self, rounds: int = 24, key: int = 0) -> None:
        super().__init__(rounds=rounds, key=key, key_bits=192, name="Camellia-192")


class Camellia256(Camellia):
    def __init__(self, rounds: int = 24, key: int = 0) -> None:
        super().__init__(rounds=rounds, key=key, key_bits=256, name="Camellia-256")
