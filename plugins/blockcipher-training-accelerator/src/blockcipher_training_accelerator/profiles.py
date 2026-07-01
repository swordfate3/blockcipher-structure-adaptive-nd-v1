from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedProfile:
    name: str
    dataloader_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    non_blocking_transfer: bool = False
    amp_dtype: str | None = None
    compile_model: bool = False

    def to_json_dict(self) -> dict[str, object]:
        return {
            "profile": self.name,
            "dataloader_workers": self.dataloader_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "non_blocking_transfer": self.non_blocking_transfer,
            "amp_dtype": self.amp_dtype,
            "compile_model": self.compile_model,
        }


PROFILES: dict[str, SpeedProfile] = {
    "baseline": SpeedProfile(name="baseline"),
    "dataloader": SpeedProfile(
        name="dataloader",
        dataloader_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        non_blocking_transfer=True,
    ),
    "amp-bf16": SpeedProfile(
        name="amp-bf16",
        pin_memory=True,
        non_blocking_transfer=True,
        amp_dtype="bf16",
    ),
    "compile": SpeedProfile(
        name="compile",
        non_blocking_transfer=True,
        compile_model=True,
    ),
    "amp-bf16-compile": SpeedProfile(
        name="amp-bf16-compile",
        pin_memory=True,
        non_blocking_transfer=True,
        amp_dtype="bf16",
        compile_model=True,
    ),
    "full": SpeedProfile(
        name="full",
        dataloader_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        non_blocking_transfer=True,
        amp_dtype="bf16",
        compile_model=True,
    ),
}


def resolve_profile(name: str) -> SpeedProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown speed profile: {name}. Known profiles: {known}") from exc
