"""Opt-in training speed utilities for blockcipher experiments."""

from blockcipher_training_accelerator.benchmark import BenchmarkReport, run_benchmark
from blockcipher_training_accelerator.matrix import MatrixSplitResult, split_matrix

__all__ = [
    "BenchmarkReport",
    "MatrixSplitResult",
    "run_benchmark",
    "split_matrix",
]
