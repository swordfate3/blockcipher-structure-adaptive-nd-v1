# Training Speed Optimization Plan

**Date:** 2026-06-29

**Status:** implementation in progress

**Scope:** Training-loop speed improvements for medium and large block-cipher neural distinguisher experiments.

## Plain-Language Summary

Current training is slow partly because the loop behaves like this:

```text
train one epoch
scan the full validation set
scan the full training set
scan some datasets twice while evaluating
repeat
```

For `262144/class`, the training set has `524288` rows and validation has `262144` rows. Full train metrics are useful diagnostics, but they are too expensive to compute after every epoch in every medium/large remote run.

The first optimization should be "lossless speedup":

```text
do not change data
do not change labels
do not change negative samples
do not change validation key
do not change exact AUC computation
do not change model architecture
only remove repeated evaluation work
```

## Current Bottlenecks

1. `evaluate_binary_classifier()` currently performs two full forward passes over the same dataset: one for loss/labels and another for probabilities.
2. `train_binary_classifier()` evaluates both validation and full training datasets after every epoch.
3. Full train AUC and calibrated accuracy are diagnostics; checkpointing only needs the configured validation metric, usually `val_auc`.
4. Disk-backed dataset loading may also be slow, but it should be optimized after the evaluation waste is removed.

## Strategy

### Phase 1: Evaluation Fast Path

Implement these changes first:

1. Make `evaluate_binary_classifier()` collect loss, labels, and probabilities in one dataset pass.
2. Add training evaluation controls:
   - `train_eval_interval`
   - `train_eval_max_rows`
3. Preserve validation evaluation every epoch by default.
4. Preserve final full validation evaluation after checkpoint restore.
5. Record whether train metrics are full, sampled, or skipped.

Default behavior remains compatible for local/small runs:

```text
train_eval_interval = 1
train_eval_max_rows = 0
```

Fast remote behavior should use:

```text
train_eval_interval = 0
train_eval_max_rows = 0
```

where `0` interval means skip per-epoch train-set metrics and keep per-epoch validation metrics.

### Phase 2: DataLoader Throughput

After Phase 1 is verified, evaluate:

```text
num_workers
pin_memory
persistent_workers
prefetch_factor
non_blocking device transfer
```

This needs a Windows remote smoke/benchmark because the dataset uses mmap-backed `.npy` arrays.

### Phase 3: Experiment-Level Parallelism

Prefer map/reduce at the experiment level before DDP:

```text
map: run different models or seeds on different GPUs
reduce: merge JSONL, gate, plots, and docs
```

This fits the project better than immediate gradient all-reduce because current models are not large enough to justify DDP complexity before evaluation and loading waste is removed.

## Success Criteria

For Phase 1:

1. Metrics from `evaluate_binary_classifier()` remain numerically equivalent to the previous two-pass behavior.
2. Full-metrics mode still records train and validation history fields.
3. Fast mode skips per-epoch full train metrics without breaking checkpoint selection, result JSONL, plots, or docs.
4. A small local smoke still passes.
5. A later remote 2-epoch benchmark should show lower epoch wall time before using the mode for formal medium diagnostics.

## Claim Scope

These changes optimize training execution. They do not create new cryptanalytic evidence by themselves. Any speed benchmark should be reported separately from model accuracy results.
