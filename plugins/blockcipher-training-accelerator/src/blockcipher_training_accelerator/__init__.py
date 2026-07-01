"""Opt-in training speed utilities for blockcipher experiments."""

from blockcipher_training_accelerator.benchmark import BenchmarkReport, run_benchmark
from blockcipher_training_accelerator.launcher import LaunchPlan, ShardCommand, build_shard_commands
from blockcipher_training_accelerator.matrix import MatrixSplitResult, split_matrix
from blockcipher_training_accelerator.profiles import SpeedProfile, resolve_profile
from blockcipher_training_accelerator.quality_gate import QualityGateReport, compare_result_files

__all__ = [
    "BenchmarkReport",
    "LaunchPlan",
    "MatrixSplitResult",
    "QualityGateReport",
    "ShardCommand",
    "SpeedProfile",
    "build_shard_commands",
    "compare_result_files",
    "resolve_profile",
    "run_benchmark",
    "split_matrix",
]
