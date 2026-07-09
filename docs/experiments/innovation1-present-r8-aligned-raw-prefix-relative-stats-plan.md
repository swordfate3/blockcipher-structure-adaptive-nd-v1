# Innovation 1 PRESENT r8 Aligned Raw-Prefix Relative-Stats Gate

## Status

status = completed local diagnostic gate
decision = raw-prefix relative summary statistics are weak; move to per-cell/per-edge dynamic representation
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1/history.csv

## Question

The raw-prefix P-layer mixer stayed near chance, even after adding active-nibble
metadata. That suggests a simple token bias is not enough.

This gate tests a cheaper relative-coordinate representation before building a
larger dynamic graph model: compute position statistics after reindexing cells
relative to the active nibble, with no DDT trail-value block.

## Method

Use:

- `feature_encoding = present_pair_xor_paligned_sinv_cell_matrix_bits`
- `sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata`
- `trail_depth = 0`
- `trail_words_per_depth = 0`
- `metadata_bits = 16`

With `trail_depth = 0`, `present_trail_position_stats_pairset` only summarizes
raw prefix words: ciphertext left, ciphertext right, XOR, P-aligned XOR, and
structural inverse prefix. It does not consume any DDT beam trail values.

Compare:

- `p_layer_relative_stats`: put the active source cell and its PRESENT P-layer
  targets at the head of the coordinate frame;
- `relative_stats`: simple cyclic active-nibble rotation;
- `none`: metadata-only control, no active-relative reindexing.

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1

## Gate

Desired pattern:

```text
p_layer_relative_stats > relative_stats on both seeds
p_layer_relative_stats > metadata-only on both seeds
```

If this fails, then relative summary statistics over raw prefix are still too
weak. The next architecture should use per-cell/per-edge tokens or a dynamic
active-conditioned P-layer graph instead of summary statistics.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| p-layer-relative | 0.523223877 | 0.560836792 | weak; does not beat metadata-only |
| simple-relative | 0.510566711 | 0.543212891 | weak; lower than p-layer-relative but near chance |
| metadata-only | 0.549835205 | 0.576095581 | highest on both seeds |

The desired gate did not pass. P-layer-relative statistics are slightly above
simple cyclic relative statistics on both seeds, but the metadata-only control
is higher on both seeds. This means the small gain is not evidence that the
model is using a useful PRESENT P-layer relative coordinate system.

## Decision

Do not scale this summary-stat route. The result is diagnostic only.

The next architecture should stop relying on hand-compressed summary statistics
and should instead keep local cell/edge evidence as tokens. A better next local
gate is a dynamic active-conditioned SPN cell graph:

- each cell token keeps raw prefix channels;
- the active nibble selects the per-sample relative frame;
- P-layer/S-box neighborhood edges are encoded as message paths or edge tokens;
- controls include shuffled topology and metadata-only/no-relative variants.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1/progress.jsonl
```
