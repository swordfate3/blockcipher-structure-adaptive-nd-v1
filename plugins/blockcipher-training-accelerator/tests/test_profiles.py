from __future__ import annotations

import pytest

from blockcipher_training_accelerator.profiles import SpeedProfile, resolve_profile


def test_resolve_baseline_profile_keeps_acceleration_disabled():
    profile = resolve_profile("baseline")

    assert profile == SpeedProfile(
        name="baseline",
        dataloader_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        non_blocking_transfer=False,
        amp_dtype=None,
        compile_model=False,
    )


def test_resolve_amp_compile_profile_enables_safe_cuda_knobs():
    profile = resolve_profile("amp-bf16-compile")

    assert profile.name == "amp-bf16-compile"
    assert profile.amp_dtype == "bf16"
    assert profile.compile_model is True
    assert profile.non_blocking_transfer is True


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown speed profile"):
        resolve_profile("turbo")
