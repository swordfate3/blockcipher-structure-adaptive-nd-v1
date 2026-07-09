# Innovation 1 PRESENT r8 Single Active Sweep

## Status

status = completed local diagnostic
claim_scope = not formal PRESENT r8 evidence
scale = 256/class seed0
plan = configs/experiment/innovation1/innovation1_spn_present_r8_single_active_sweep_256_seed0.csv

## Question

The active-conditioned curriculum showed a sharp pattern:

- fixed or one-active rows are high;
- two-active rows keep some signal;
- four, eight, sixteen active rows and heldout-active rows collapse.

This sweep asks a narrower question before changing the architecture again:

Is the high fixed-active result mostly a property of active nibble 0, or does
every single fixed active nibble remain easy when tested alone?

## Matrix

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 256
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_difference_matched_negative_random_active_metadata
- feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
- model = present_trail_position_stats_pairset
- active_conditioning = p_layer_relative_stats
- seed = 0

Rows:

```text
active nibble {0}, {1}, ..., {15}
```

## Gate

Interpretation:

- If all or most single active nibbles are high, fixed-active signal is not
  unique to nibble 0. The failure is likely that the model cannot merge many
  coordinate templates into one random-active rule.
- If only a small subset is high, especially only nibble 0, audit the protocol
  and feature route for position-specific bias before spending more model work.
- This is still a local diagnostic only. It does not justify remote scale by
  itself.

## Commands

Training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_single_active_sweep_256_seed0.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_single_active_sweep_256_seed0 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_single_active_sweep_256_seed0.csv \
  --results outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/results.jsonl \
  --expected-rows 16
```

Plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/history.csv \
  --title i1_present_r8_single_active_sweep_256_seed0
```

## Results

Artifacts:

- results = outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/results.jsonl
- progress = outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/progress.jsonl
- curves = outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/curves.svg
- history = outputs/local_smoke/i1_present_r8_single_active_sweep_256_seed0/history.csv
- dataset_cache = outputs/local_cache/i1_present_r8_single_active_sweep_256_seed0

Validation:

```text
validate-results status = pass
expected_rows = 16
field_mismatches = []
duplicate_plan_keys = []
duplicate_result_keys = []
missing_result_keys = []
unexpected_result_keys = []
```

Metrics:

| Active nibble | seed0 AUC | Best accuracy |
| ---: | ---: | ---: |
| 0 | 0.949279785 | 0.894531 |
| 1 | 0.487915039 | 0.519531 |
| 2 | 0.475769043 | 0.519531 |
| 3 | 0.547973633 | 0.558594 |
| 4 | 0.478271484 | 0.523438 |
| 5 | 0.530273438 | 0.562500 |
| 6 | 0.525299072 | 0.570312 |
| 7 | 0.531494141 | 0.550781 |
| 8 | 0.477416992 | 0.507812 |
| 9 | 0.488281250 | 0.531250 |
| 10 | 0.477539062 | 0.535156 |
| 11 | 0.492858887 | 0.546875 |
| 12 | 0.502624512 | 0.542969 |
| 13 | 0.491882324 | 0.550781 |
| 14 | 0.515106201 | 0.546875 |
| 15 | 0.501525879 | 0.546875 |

Summary:

```text
min_auc = 0.475769043
max_auc = 0.949279785
mean_auc = 0.529594421
```

## Decision

This result does not support the idea that every fixed active nibble is easy.
Only active nibble 0 is high at this local scale; active nibbles 1..15 are
near chance.

Source inspection explains the likely cause: the current random-active
integral protocol moves `integral_active_nibble`, but the Zhang/Wang input
difference remains fixed at `0x0000000000000009`, i.e. low nibble 0. Therefore
the existing active-nibble generalization protocol is not yet a clean test of
"the same differential structure translated to different coordinates." It is
mostly testing what happens when the integral active coordinate moves while the
differential anchor stays at nibble 0.

This makes the previous active1/fixed-active high result narrower:

```text
active nibble 0 plus low-nibble input difference is learnable.
```

It does not show:

```text
single fixed active nibble at arbitrary coordinates is learnable.
```

## Next Action

Defer the active-token-bias smoke until the protocol is repaired or explicitly
reframed. The next experiment should align the input difference with the sampled
active nibble, e.g. active nibble `k` uses `0x9 << (4*k)`, then rerun:

- aligned single-active sweep;
- aligned random-active baseline;
- aligned random-active plus conditioning;
- aligned heldout-active transfer.
