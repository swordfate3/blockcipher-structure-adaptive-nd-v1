from __future__ import annotations

from blockcipher_nd.ciphers import (
    Aes128,
    Aes192,
    Aes256,
    Aria128,
    Aria192,
    Aria256,
    Camellia128,
    Camellia192,
    Camellia256,
    Cham64_128,
    Des,
    Gift64,
    Lea128,
    Lea192,
    Lea256,
    Present80,
    Rectangle80,
    ReducedRoundCipher,
    Simeck64_128,
    Simon64_128,
    Skinny64,
    Sm4Reduced,
    Speck32_64,
    TripleDes,
    UknitBc,
)


def build_cipher(name: str, rounds: int, key: int | None = None) -> ReducedRoundCipher:
    if name == "aes128":
        return Aes128(rounds=rounds, key=0x000102030405060708090A0B0C0D0E0F if key is None else key)
    if name == "aes192":
        return Aes192(
            rounds=rounds,
            key=0x000102030405060708090A0B0C0D0E0F1011121314151617 if key is None else key,
        )
    if name == "aes256":
        return Aes256(
            rounds=rounds,
            key=0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F
            if key is None
            else key,
        )
    if name == "aria128":
        return Aria128(rounds=rounds, key=0x000102030405060708090A0B0C0D0E0F if key is None else key)
    if name == "aria192":
        return Aria192(
            rounds=rounds,
            key=0x000102030405060708090A0B0C0D0E0F1011121314151617 if key is None else key,
        )
    if name == "aria256":
        return Aria256(
            rounds=rounds,
            key=0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F
            if key is None
            else key,
        )
    if name == "camellia128":
        return Camellia128(rounds=rounds, key=0x0123456789ABCDEFFEDCBA9876543210 if key is None else key)
    if name == "camellia192":
        return Camellia192(
            rounds=rounds,
            key=0x0123456789ABCDEFFEDCBA98765432100011223344556677 if key is None else key,
        )
    if name == "camellia256":
        return Camellia256(
            rounds=rounds,
            key=0x0123456789ABCDEFFEDCBA987654321000112233445566778899AABBCCDDEEFF
            if key is None
            else key,
        )
    if name == "des":
        return Des(rounds=rounds, key=0x133457799BBCDFF1 if key is None else key)
    if name == "3des":
        return TripleDes(
            rounds=rounds,
            key1=0x0123456789ABCDEF,
            key2=0x23456789ABCDEF01,
            key3=0x456789ABCDEF0123,
        )
    if name == "speck32":
        return Speck32_64(rounds=rounds, key=0x1918111009080100 if key is None else key)
    if name == "cham64":
        return Cham64_128(rounds=rounds, key=0x010003020504070609080B0A0D0C0F0E if key is None else key)
    if name == "lea128":
        default_key = int.from_bytes(bytes.fromhex("0f1e2d3c4b5a69788796a5b4c3d2e1f0"), "little")
        return Lea128(rounds=rounds, key=default_key if key is None else key)
    if name == "lea192":
        default_key = int.from_bytes(
            bytes.fromhex("0f1e2d3c4b5a69788796a5b4c3d2e1f0f0e1d2c3b4a59687"),
            "little",
        )
        return Lea192(rounds=rounds, key=default_key if key is None else key)
    if name == "lea256":
        default_key = int.from_bytes(
            bytes.fromhex(
                "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
                "f0e1d2c3b4a5968778695a4b3c2d1e0f"
            ),
            "little",
        )
        return Lea256(rounds=rounds, key=default_key if key is None else key)
    if name == "simon64":
        return Simon64_128(rounds=rounds, key=0x1B1A1918131211100B0A090803020100 if key is None else key)
    if name == "simeck64":
        return Simeck64_128(rounds=rounds, key=0x1B1A1918131211100B0A090803020100 if key is None else key)
    if name == "present80":
        return Present80(rounds=rounds, key=0x00000000000000000000 if key is None else key)
    if name == "gift64":
        return Gift64(rounds=rounds, key=0x00000000000000000000000000000000 if key is None else key)
    if name == "rectangle80":
        return Rectangle80(rounds=rounds, key=0x00000000000000000000 if key is None else key)
    if name == "skinny64":
        return Skinny64(rounds=rounds, key=0x0000000000000000 if key is None else key)
    if name == "uknit64":
        return UknitBc(
            rounds=rounds,
            key=0x00000000000000000000000000000000 if key is None else key,
        )
    if name == "sm4":
        return Sm4Reduced(rounds=rounds, key=0x0123456789ABCDEFFEDCBA9876543210 if key is None else key)
    raise ValueError(f"unsupported cipher: {name}")


def default_difference(name: str) -> int:
    if name in {
        "aes128",
        "aes192",
        "aes256",
        "aria128",
        "aria192",
        "aria256",
        "camellia128",
        "camellia192",
        "camellia256",
        "lea128",
        "lea192",
        "lea256",
    }:
        return 0x00000000000000000000000000000040
    if name in {"des", "3des"}:
        return 0x0000000000000040
    if name == "speck32":
        return 0x0040
    if name == "cham64":
        return 0x0000000000000040
    if name in {"simon64", "simeck64"}:
        return 0x0000000000000040
    if name in {"present80", "gift64", "rectangle80", "skinny64", "uknit64"}:
        return 0x0000000000000040
    if name == "sm4":
        return 0x00000000000000000000000000000040
    raise ValueError(f"unsupported cipher: {name}")
