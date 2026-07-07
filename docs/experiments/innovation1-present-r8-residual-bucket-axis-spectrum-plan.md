# Innovation 1 PRESENT r8 Residual Bucket Axis Spectrum Plan

**Date:** 2026-07-07

**Status:** tool-ready local diagnostic plan; no training launched

## Question

The current best local third-family candidate is:

```text
trail-position neural expert
+ matched raw117 compressed SPN structural expert
+ bucket-conditioned residual feature expert
```

The open question is not whether to add more similar neural networks. The open
question is:

```text
Inside each trail+raw117 residual reliability bucket, which SPN depth/word/cell
feature groups explain the remaining threshold errors or soft residual loss?
```

If those groups are stable, they can become a better next candidate source than
generic near-neighbor score averaging. The current 262144/class trail-position
result is strong but mixed: the candidate is nearly perfect, while the same-input
global control is also nearly perfect. That means the next useful question is
not "can another similar model average help?", but "which SPN axes still carry
independent residual signal after the current strong scores are frozen?"

## Diagnostic Tool

New local CLI:

```text
scripts/analyze-residual-bucket-axis-spectrum
```

Inputs:

```text
feature_dir = existing bit-sensitivity / compressed SPN feature artifact
bucket_artifacts = exactly two aligned frozen score artifacts
bucket_feature = logit_gap_abs by default
bucket_count = train/local diagnostic bucket count
target = residual_error_at_0_5 by default
```

Output:

```text
outputs/local_audits/i1_present_r8_residual_bucket_axis_spectrum.json
```

The tool:

```text
1. validates labels and sample_ids against the frozen score artifacts;
2. computes the same residual bucket values used by V16/V17;
3. groups feature columns by semantic axis names such as
   primary_depth_mean or aux_word_mean;
4. reports per-bucket label AUC, hard residual-error AUC, and selected target
   AUC for each group;
5. ranks groups by AUC distance from 0.5;
6. also reports `global_top_groups` so sparse buckets do not hide useful soft
   residual axes.
```

Supported targets:

```text
residual_error_at_0_5 = hard mistakes from mean frozen probability at threshold 0.5
residual_loss = abs(label - mean frozen probability)
signed_margin = signed distance from the correct side of threshold
global_candidate_gap = absolute probability gap between the two frozen artifacts
```

Continuous targets are binarized at the median for AUC ranking. This is a
diagnostic ranking device only; it is not a replacement for held-out validation
or a publication-style metric.

## Why Continuous Targets Matter

On very strong local validation artifacts, hard 0.5-threshold errors can be too
sparse to rank feature axes reliably. A bucket may contain zero hard mistakes or
only one residual class, which makes per-bucket residual-error AUC undefined.

`residual_loss` keeps the same frozen labels and probabilities but asks a softer
question:

```text
Which SPN feature groups correlate with samples that the current frozen scores
still find less comfortable, even when they remain on the correct side of 0.5?
```

This is useful for source selection. A group that consistently explains soft
residual loss is a better candidate for a non-neighbor residual expert than a
new model that only re-learns the same trail-position/global-stat signal.

## Guardrails

This diagnostic must not change:

```text
labels
negative_mode
sample_structure
validation keys
metric computation for existing gates
```

It is not a new trained expert yet. It is a source-selection diagnostic for a
future residual structural expert.

## Intended Use After 262k Retrieval

After the 262144/class trail-position score artifacts are complete and the
V16/V17 bucket residual planner is ready:

```text
1. export train/validation compressed SPN span features;
2. run trail+raw117 frozen score exports;
3. run `scripts/analyze-residual-bucket-axis-spectrum` on train artifacts with
   both `residual_error_at_0_5` and `residual_loss`;
4. freeze any candidate group selection before validation scoring;
5. require shuffle-label, train-bucket-shuffle, validation-bucket-shuffle, and
   no-bucket controls before any remote-scale claim.
```

## Claim Scope

This is a local diagnostic plan and tooling record only. It does not report a
new PRESENT r8 result, does not claim a breakthrough, does not replace the
pending 262144/class trail-position retrieval, and does not provide formal
SPN/PRESENT evidence.
