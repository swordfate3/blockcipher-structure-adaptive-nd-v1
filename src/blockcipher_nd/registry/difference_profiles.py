from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifferenceProfile:
    name: str
    cipher: str
    kind: str
    differences: tuple[int, ...]
    source: str
    word_difference: tuple[str, ...] = ()
    note: str = ""
    difference_kind: str = "xor"
    pairs_per_sample: int = 1
    related_key_difference: int | None = None
    polytope_size: int = 2

    @property
    def difference(self) -> int:
        if self.kind != "fixed":
            raise ValueError(f"profile {self.name} is not a fixed input difference")
        if self.difference_kind != "xor":
            raise ValueError(
                f"profile {self.name} uses {self.difference_kind}, not a fixed xor input difference"
            )
        return self.differences[0]


def literature_difference_profiles() -> dict[str, DifferenceProfile]:
    return {
        "speck32_gohr2019": DifferenceProfile(
            name="speck32_gohr2019",
            cipher="speck32",
            kind="fixed",
            differences=(0x00400000,),
            word_difference=("0x0040", "0x0000"),
            source="Gohr 2019 SPECK32/64 neural distinguisher",
            note="Input difference 0x0040/0000 for Gohr N5-N8 distinguishers.",
        ),
        "present_wang_jain2021": DifferenceProfile(
            name="present_wang_jain2021",
            cipher="present80",
            kind="multi_fixed",
            differences=(
                0x0700000000000700,
                0x7000000000007000,
                0x0070000000000070,
                0x0007000000000007,
            ),
            source="Wang differentials via Jain/Kohli/Mishra 2020/2021",
            note="Four high-probability PRESENT input differential classes.",
        ),
        "present_wang_jain2021_1": DifferenceProfile(
            name="present_wang_jain2021_1",
            cipher="present80",
            kind="fixed",
            differences=(0x0700000000000700,),
            source="Wang differentials via Jain/Kohli/Mishra 2020/2021",
        ),
        "present_wang_jain2021_2": DifferenceProfile(
            name="present_wang_jain2021_2",
            cipher="present80",
            kind="fixed",
            differences=(0x7000000000007000,),
            source="Wang differentials via Jain/Kohli/Mishra 2020/2021",
        ),
        "present_wang_jain2021_3": DifferenceProfile(
            name="present_wang_jain2021_3",
            cipher="present80",
            kind="fixed",
            differences=(0x0070000000000070,),
            source="Wang differentials via Jain/Kohli/Mishra 2020/2021",
        ),
        "present_wang_jain2021_4": DifferenceProfile(
            name="present_wang_jain2021_4",
            cipher="present80",
            kind="fixed",
            differences=(0x0007000000000007,),
            source="Wang differentials via Jain/Kohli/Mishra 2020/2021",
        ),
        "present_zhang_wang2022_mcnd": DifferenceProfile(
            name="present_zhang_wang2022_mcnd",
            cipher="present80",
            kind="fixed",
            differences=(0x0000000000000009,),
            source="Zhang/Wang 2022 PRESENT Inception-MCND differential-neural distinguisher",
            word_difference=("0x0000", "0x0000", "0x0000", "0x0009"),
            note="Input difference (0,0,0,0x9) for PRESENT 6-7 round MCND experiments in arXiv:2204.06341.",
        ),
        "present_autond_dbitnet2023_highround": DifferenceProfile(
            name="present_autond_dbitnet2023_highround",
            cipher="present80",
            kind="fixed",
            differences=(0x000000000D000000,),
            source="AutoND/DBitNet 2023 cipher-agnostic neural training pipeline PRESENT high-round screen",
            word_difference=("0x0000", "0x0000", "0x0d00", "0x0000"),
            note="Input difference 0xd000000 reported for weak 8-9 round PRESENT DBitNet distinguishers; treat as a high-round search seed, not a strong MCND baseline.",
        ),
        "present_entropy2026_gohr": DifferenceProfile(
            name="present_entropy2026_gohr",
            cipher="present80",
            kind="fixed",
            differences=(0x0000000000D00000,),
            source="Gauthier-Umana/Martinez/Obando/Perez 2026 entropy-based PRESENT neural distinguisher",
            word_difference=("0x0000", "0x0000", "0x00d0", "0x0000"),
            note="Input difference used by the 2026 entropy-based bit-reduced PRESENT distinguisher; pair feature selection should be supplied separately.",
        ),
        "gift64_shen2024_spn_screen": DifferenceProfile(
            name="gift64_shen2024_spn_screen",
            cipher="gift64",
            kind="fixed",
            differences=(0x0000000000000040,),
            source="Shen/Song/Lu/Long/Tian 2024 GIFT neural distinguisher screening profile",
            note="Fixed xor input difference for first GIFT-64 SPN structure-alignment screening; refine after literature-specific reproduction.",
        ),
        "sm4_yu2023_conv_resnet": DifferenceProfile(
            name="sm4_yu2023_conv_resnet",
            cipher="sm4",
            kind="fixed",
            differences=(0x00000000000000000000000000000001,),
            word_difference=(
                "0x00000000",
                "0x00000000",
                "0x00000000",
                "0x00000001",
            ),
            source="Yu/Wu/Zhang 2023 SM4 convolutional residual network",
            note="Plaintext difference used for SM4 3-8 round neural distinguishers.",
        ),
        "sm4_li_sun_2025_19r_family": DifferenceProfile(
            name="sm4_li_sun_2025_19r_family",
            cipher="sm4",
            kind="difference_family",
            differences=(),
            word_difference=(
                "a''0 in {x xor e79393d6 | PrT(0000003c, x) > 0}",
                "0xe793932d",
                "0x6fb9b98f",
                "0x882a2a9e",
            ),
            source="Li/Sun 2025 key-recovery-friendly SM4 differential family",
            note="Constrained family; not directly usable as a fixed neural input difference.",
        ),
        "simon32_rx_neural2025_schema": DifferenceProfile(
            name="simon32_rx_neural2025_schema",
            cipher="simon32",
            kind="schema_only",
            differences=(),
            source="Liu/Chen/Xiang/Zhang/Zeng 2025 RX-neural Simon/Simeck",
            note="Schema placeholder for RX-neural distinguishers; requires SIMON/SIMECK generators.",
            difference_kind="rx",
            pairs_per_sample=8,
        ),
        "speck32_polytopic2026_schema": DifferenceProfile(
            name="speck32_polytopic2026_schema",
            cipher="speck32",
            kind="schema_only",
            differences=(),
            source="Mirzaali/Sadeghi/Bagheri 2026 polytopic neural distinguishers",
            note="Schema placeholder for ciphertext quadruple/polytope inputs.",
            difference_kind="polytope",
            polytope_size=4,
        ),
    }


def difference_for_profile(name: str, member_index: int = 0) -> int:
    profile = literature_difference_profiles()[name]
    if profile.kind in {"difference_family", "schema_only"}:
        raise ValueError(f"profile {name} is not a fixed input difference")
    try:
        return profile.differences[member_index]
    except IndexError as exc:
        raise ValueError(f"profile {name} has no member {member_index}") from exc
