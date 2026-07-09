# Innovation 1 PRESENT r8 Aligned Active Difference Screen

## Status

status = completed local diagnostic
claim_scope = not formal PRESENT r8 evidence
scale = 256/class seed0
plan = configs/experiment/innovation1/innovation1_spn_present_r8_aligned_active_difference_screen_256_seed0.csv

## Question

The previous single-active sweep showed that only active nibble 0 stayed high.
Source inspection found the likely reason: `integral_active_nibble` moved, but
the Zhang/Wang input difference stayed fixed at low nibble 0:

```text
input_difference = 0x0000000000000009
```

This screen tests the repaired protocol:

```text
active nibble k -> input difference 0x9 << (4*k)
```

The goal is to ask a cleaner question: if the same one-nibble difference
structure is translated to the active coordinate, does the signal also follow?

## Matrix

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 256
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
- model = present_trail_position_stats_pairset
- seed = 0

Rows:

| Row | Active set | Metadata/conditioning | Purpose |
| --- | --- | --- | --- |
| active0 | {0} | p_layer_relative_stats | Anchor comparable to previous high nibble0 row. |
| active1 | {1} | p_layer_relative_stats | Checks immediate moved coordinate. |
| active5 | {5} | p_layer_relative_stats | Checks mid-state moved coordinate. |
| active15 | {15} | p_layer_relative_stats | Checks high-nibble moved coordinate. |
| unconditioned random16 | 0..15 | none | Same-budget random-active baseline under aligned protocol. |
| p-layer relative random16 | 0..15 | p_layer_relative_stats | Tests conditioned random-active under aligned protocol. |
| p-layer relative active4 | 0..3 | p_layer_relative_stats | Tests a small active-set curriculum. |
| p-layer relative heldout4to4 | train 0..3, validation 4..7 | p_layer_relative_stats | Tests transfer to unseen aligned active coordinates. |

## Gate

This is a local protocol screen only.

- If active1/active5/active15 rise near active0, the earlier active-only sweep
  failure was largely a protocol-alignment problem.
- If only active0 remains high, the issue is deeper than the fixed input
  difference anchor and we should audit feature/cell ordering.
- If single active rows are high but random16 or heldout remain low, the next
  architecture work should target active-conditioned or equivariant aggregation.
- No remote scale follows from this screen alone.

## Commands

Training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_active_difference_screen_256_seed0.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_active_difference_screen_256_seed0 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_active_difference_screen_256_seed0.csv \
  --results outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/results.jsonl \
  --expected-rows 8
```

Plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/history.csv \
  --title i1_present_r8_aligned_active_difference_screen_256_seed0
```

## Results

Artifacts:

- results = outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/results.jsonl
- progress = outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/progress.jsonl
- curves = outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/curves.svg
- history = outputs/local_smoke/i1_present_r8_aligned_active_difference_screen_256_seed0/history.csv
- dataset_cache = outputs/local_cache/i1_present_r8_aligned_active_difference_screen_256_seed0

Validation:

```text
validate-results status = pass
expected_rows = 8
field_mismatches = []
duplicate_plan_keys = []
duplicate_result_keys = []
missing_result_keys = []
unexpected_result_keys = []
```

Metrics:

| Row | Active train | Active validation | AUC | Best accuracy |
| --- | --- | --- | ---: | ---: |
| aligned active0 p-layer-relative | [0] | [] | 0.971069336 | 0.906250 |
| aligned active1 p-layer-relative | [1] | [] | 0.951477051 | 0.882812 |
| aligned active5 p-layer-relative | [5] | [] | 0.983154297 | 0.941406 |
| aligned active15 p-layer-relative | [15] | [] | 0.987121582 | 0.953125 |
| aligned random16 unconditioned | [0..15] | [] | 0.958923340 | 0.906250 |
| aligned random16 p-layer-relative | [0..15] | [] | 0.657165527 | 0.652344 |
| aligned active4 p-layer-relative | [0..3] | [] | 0.712707520 | 0.683594 |
| aligned heldout4to4 p-layer-relative | [0..3] | [4..7] | 0.758972168 | 0.691406 |

## Decision

The aligned protocol resolves the strongest concern from the previous
single-active sweep. Once the input difference follows the active nibble,
representative single active positions are all high:

```text
active0 = 0.971
active1 = 0.951
active5 = 0.983
active15 = 0.987
```

This means the earlier "only active0 is high" result was largely a protocol
alignment problem, not evidence that PRESENT only has a low-nibble signal.

The more surprising result is that the unconditioned aligned random16 baseline
is also high:

```text
aligned random16 unconditioned AUC = 0.958923340
```

But the p-layer-relative stats route is weaker on mixed active sets:

```text
aligned random16 p-layer-relative AUC = 0.657165527
aligned active4 p-layer-relative AUC = 0.712707520
aligned heldout4to4 p-layer-relative AUC = 0.758972168
```

Interpretation:

- The aligned data protocol is much cleaner than the previous random-active
  protocol.
- The current strong feature route can already distinguish aligned random
  active without explicit active metadata at this small scale.
- The `p_layer_relative_stats` conditioning is not currently the best route for
  mixed active sets; it may discard useful absolute-coordinate or feature-order
  information.
- This is still only 256/class seed0. It is a local diagnostic, not formal r8
  evidence.

## Next Action

Do not remote-scale from this screen alone. The next local gate should be:

1. full aligned single-active sweep over all 16 nibbles, seed0+seed1;
2. aligned random16 unconditioned versus p-layer-relative at 512/class,
   seed0+seed1;
3. feature ablation under aligned random16 to see whether the high score comes
   from ciphertext xor, paligned, sinv, DDT/beamstats, or the full
   trail-position stack;
4. only after those controls, revisit active-token-bias or other graph/token
   architecture changes.
