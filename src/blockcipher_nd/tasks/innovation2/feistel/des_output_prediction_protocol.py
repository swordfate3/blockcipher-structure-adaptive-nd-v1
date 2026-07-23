from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DesOutputPredictionDataConfig:
    run_id: str = "i2_output_prediction_feistel1_des_r2_readiness_seed31"
    mode: str = "readiness"
    rounds: int = 2
    seed: int = 31
    key_seed: int = 31
    train_rows: int = 64
    test_rows: int = 64
    chunk_rows: int = 32

    def __post_init__(self) -> None:
        allowed_modes = {
            "readiness",
            "f1_a",
            "f1_b",
            "f1_c",
            "f1_r_a",
            "f1_r_b",
        }
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in allowed_modes:
            raise ValueError(f"mode must be one of {sorted(allowed_modes)}")
        if self.rounds < 1 or self.rounds > 16:
            raise ValueError("DES output prediction rounds must be in 1..16")
        if min(self.train_rows, self.test_rows, self.chunk_rows) <= 0:
            raise ValueError("row and chunk values must be positive")
        if min(self.seed, self.key_seed) < 0:
            raise ValueError("seed and key_seed must be non-negative")
        if self.train_rows + self.test_rows > 1 << 64:
            raise ValueError("unique DES plaintext rows cannot exceed 2^64")
        expected_formal = {
            "f1_a": (2, 31, 1 << 20, 1 << 15),
            "f1_b": (2, 32, 1 << 20, 1 << 15),
            "f1_c": (2, 31, 1 << 22, 1 << 15),
            "f1_r_a": (3, 31, 1 << 20, 1 << 15),
            "f1_r_b": (3, 32, 1 << 20, 1 << 15),
        }
        if self.mode in expected_formal:
            rounds, key_seed, train_rows, test_rows = expected_formal[self.mode]
            if (
                self.rounds != rounds
                or self.seed != 31
                or self.key_seed != key_seed
                or self.train_rows != train_rows
                or self.test_rows != test_rows
                or self.chunk_rows != 4096
            ):
                raise ValueError(f"formal {self.mode} DES data protocol is frozen")

    @classmethod
    def f1_a(
        cls,
        *,
        run_id: str = "i2_output_prediction_feistel1a_des_r2_key31",
    ) -> DesOutputPredictionDataConfig:
        return cls(
            run_id=run_id,
            mode="f1_a",
            train_rows=1 << 20,
            test_rows=1 << 15,
            chunk_rows=4096,
        )

    @classmethod
    def f1_b(
        cls,
        *,
        run_id: str = "i2_output_prediction_feistel1b_des_r2_key32",
    ) -> DesOutputPredictionDataConfig:
        return cls(
            run_id=run_id,
            mode="f1_b",
            key_seed=32,
            train_rows=1 << 20,
            test_rows=1 << 15,
            chunk_rows=4096,
        )

    @classmethod
    def f1_c(
        cls,
        *,
        run_id: str = "i2_output_prediction_feistel1c_des_r2_2p22_key31",
    ) -> DesOutputPredictionDataConfig:
        return cls(
            run_id=run_id,
            mode="f1_c",
            train_rows=1 << 22,
            test_rows=1 << 15,
            chunk_rows=4096,
        )

    @classmethod
    def f1_r(
        cls,
        *,
        key_seed: int,
        run_id: str | None = None,
    ) -> DesOutputPredictionDataConfig:
        if key_seed not in {31, 32}:
            raise ValueError("formal DES r3 key_seed must be 31 or 32")
        mode = "f1_r_a" if key_seed == 31 else "f1_r_b"
        return cls(
            run_id=run_id or f"i2_output_prediction_feistel1r_des_r3_key{key_seed}",
            mode=mode,
            rounds=3,
            key_seed=key_seed,
            train_rows=1 << 20,
            test_rows=1 << 15,
            chunk_rows=4096,
        )


def secret_key_for_seed(key_seed: int) -> int:
    return random.Random(1_310_000 + key_seed).getrandbits(64)


def serializable_config(config: DesOutputPredictionDataConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "DesOutputPredictionDataConfig",
    "secret_key_for_seed",
    "serializable_config",
]
