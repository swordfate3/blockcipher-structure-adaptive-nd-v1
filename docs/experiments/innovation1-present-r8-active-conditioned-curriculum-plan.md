# Innovation 1 PRESENT r8 Active-Conditioned Trail Curriculum

## Status

status = completed local diagnostic
claim_scope = not formal PRESENT r8 evidence
scale = 512/class seed0+seed1
plan = configs/experiment/innovation1/innovation1_spn_present_r8_active_conditioned_curriculum_512_seed0_seed1.csv

## Question

The previous active-nibble generalization diagnostic showed that fixed-active
trail-position remains high, while random-active, heldout-active, shallow
metadata, and first-pass relative-coordinate rows are near chance. This follow-up
tests a narrower architecture fix:

Can the trail-position statistics model use active-nibble metadata inside the
statistics layer, so every word/cell/depth statistic is computed in an
active-relative coordinate system?

## Change

Add `active_conditioning = relative_stats` to
`present_trail_position_stats_pairset`.

The older shallow metadata route did this:

```text
raw pair features -> fixed-coordinate statistics -> concatenate active one-hot -> classifier
```

The new route does this:

```text
raw pair features + active one-hot
  -> rotate cell statistics so active nibble is relative cell 0
  -> word/cell/depth/trail statistics in active-relative coordinates
  -> concatenate active one-hot
  -> classifier
```

This is still a small local step. It is not yet a full PRESENT P-layer graph
equivariant model.

## Matrix

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
- model = present_trail_position_stats_pairset
- checkpoint_metric = val_auc
- seeds = 0, 1

Rows per seed:

| Row | Active set | Conditioning | Purpose |
| --- | --- | --- | --- |
| unconditioned random16 | 0..15 | none | Same-budget random-active baseline. |
| shallow metadata random16 | 0..15 | final-vector metadata only | Controls whether simple metadata concatenation helps. |
| relative_stats active1 | 0 | stats-level active conditioning | Checks fixed-active ceiling under the new path. |
| relative_stats active2 | 0..1 | stats-level active conditioning | First curriculum widening step. |
| relative_stats active4 | 0..3 | stats-level active conditioning | Four-coordinate curriculum. |
| relative_stats active8 | 0..7 | stats-level active conditioning | Half-state curriculum. |
| relative_stats active16 | 0..15 | stats-level active conditioning | Full random-active conditioned route. |
| relative_stats heldout4to4 | train 0..3, validation 4..7 | stats-level active conditioning | Tests transfer to unseen active coordinates. |

## Commands

Training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_conditioned_curriculum_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_conditioned_curriculum_512_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/results.jsonl \
  --expected-rows 16
```

Plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/history.csv \
  --title i1_present_r8_active_conditioned_curriculum_512_seed0_seed1
```

## Gate

This is a local diagnostic gate only.

Remote scale remains blocked unless the local result satisfies both:

1. active-conditioned random16, heldout4to4, or another conditioned route is
   meaningfully above the unconditioned random-active baseline on both seeds;
2. the result is not only active1/fixed-active high while active2/4/8/16 collapse.

If the curriculum shows a sudden collapse when moving from active1 to active2
or active4, the next step is a more faithful PRESENT P-layer/S-box relative
coordinate model, not remote scale-up.

## Results

Artifacts:

- results = outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/results.jsonl
- progress = outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/progress.jsonl
- curves = outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/curves.svg
- history = outputs/local_smoke/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1/history.csv
- dataset_cache = outputs/local_cache/i1_present_r8_active_conditioned_curriculum_512_seed0_seed1

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

| Row | seed0 AUC | seed1 AUC | Interpretation |
| --- | ---: | ---: | --- |
| unconditioned random16 | 0.498641968 | 0.525375366 | random-active baseline is near chance. |
| shallow metadata random16 | 0.473663330 | 0.471611023 | final-vector active one-hot does not help. |
| relative_stats active1 | 0.979125977 | 0.994842529 | fixed-active remains high. |
| relative_stats active2 | 0.691108704 | 0.768592834 | two active coordinates retain some local signal. |
| relative_stats active4 | 0.556152344 | 0.514373779 | four active coordinates mostly collapse. |
| relative_stats active8 | 0.544342041 | 0.521804810 | half-state active set remains near chance. |
| relative_stats active16 | 0.526580811 | 0.481582642 | full random-active conditioned route does not beat baseline on both seeds. |
| relative_stats heldout4to4 | 0.496734619 | 0.460891724 | train 0..3 does not transfer to validation 4..7. |

## Decision

Remote scale remains blocked.

The useful signal is still concentrated in fixed or very narrow active-coordinate
sets. The stats-level shift is better than shallow metadata for active2, but it
does not solve random-active all16 or heldout-active transfer. Under the local
gate, this is a diagnostic failure for remote readiness, not a formal failure
of the broader SPN-adaptive route.

Next action: implement a stricter PRESENT-aware relative representation where
coordinates are reorganized through the S-box/P-layer graph, instead of only
rotating the 16 cell columns inside each word. Then rerun the same local gate
against the unconditioned random-active baseline.
