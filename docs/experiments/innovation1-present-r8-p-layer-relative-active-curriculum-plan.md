# Innovation 1 PRESENT r8 P-Layer Relative Active Curriculum

## Status

status = completed local diagnostic
claim_scope = not formal PRESENT r8 evidence
scale = 512/class seed0+seed1
plan = configs/experiment/innovation1/innovation1_spn_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1.csv

## Question

The previous `active_conditioning = relative_stats` gate showed that simple
cell rotation is not enough: active1 stayed high, active2 retained some signal,
but active4/8/16 and heldout-active collapsed.

This follow-up tests one stricter local step:

Can the trail-position statistics model use active metadata to reorder cells by
PRESENT S-box/P-layer coordinates, instead of treating active coordinates as a
simple circular shift?

## Change

Add `active_conditioning = p_layer_relative_stats` to
`present_trail_position_stats_pairset`.

The older `relative_stats` route does:

```text
active nibble -> simple circular cell shift -> statistics
```

The new route does:

```text
active nibble metadata
  -> convert active nibble to PRESENT feature-cell coordinate
  -> place the active S-box cell first
  -> place its direct P-layer target cells next
  -> append the remaining cells in a stable local order
  -> compute word/cell/depth/trail statistics
```

This is still not a full graph neural network. It is a small, inspectable
PRESENT-aware relative-coordinate diagnostic inside the existing statistics
model.

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
| relative_stats random16 | 0..15 | simple active-relative cell shift | Previous all16 route control. |
| p_layer_relative active1 | 0 | PRESENT-aware relative stats | Checks fixed-active ceiling under the new coordinate path. |
| p_layer_relative active2 | 0..1 | PRESENT-aware relative stats | First curriculum widening step. |
| p_layer_relative active4 | 0..3 | PRESENT-aware relative stats | Four-coordinate curriculum. |
| p_layer_relative active8 | 0..7 | PRESENT-aware relative stats | Half-state curriculum. |
| p_layer_relative active16 | 0..15 | PRESENT-aware relative stats | Full random-active conditioned route. |
| p_layer_relative heldout4to4 | train 0..3, validation 4..7 | PRESENT-aware relative stats | Tests transfer to unseen active coordinates. |

## Commands

Training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/results.jsonl \
  --expected-rows 16
```

Plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/history.csv \
  --title i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1
```

## Gate

Remote scale remains blocked unless this local result satisfies both:

1. p-layer-relative active16, heldout4to4, or another conditioned route is
   meaningfully above the unconditioned random-active baseline on both seeds;
2. the result is not only active1/fixed-active high while active4/8/16 collapse.

If P-layer-relative statistics still collapse on active4/8/16 or heldout, the
next step should be a real graph/token model over active-conditioned PRESENT
coordinates, not 65k/262k scale-up.

## Results

Artifacts:

- results = outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/results.jsonl
- progress = outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/progress.jsonl
- curves = outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/curves.svg
- history = outputs/local_smoke/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1/history.csv
- dataset_cache = outputs/local_cache/i1_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1

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
| unconditioned random16 | 0.500534058 | 0.525375366 | random-active baseline remains near chance. |
| relative_stats random16 | 0.526580811 | 0.481582642 | previous simple relative all16 route remains unstable. |
| p_layer_relative active1 | 0.975738525 | 0.993362427 | fixed-active remains high. |
| p_layer_relative active2 | 0.754165649 | 0.761306763 | two-coordinate curriculum improves and is stable. |
| p_layer_relative active4 | 0.537399292 | 0.520980835 | four active coordinates still mostly collapse. |
| p_layer_relative active8 | 0.540145874 | 0.531463623 | half-state active set stays near chance. |
| p_layer_relative active16 | 0.526840210 | 0.519485474 | full random-active route does not beat baseline on both seeds. |
| p_layer_relative heldout4to4 | 0.492782593 | 0.492889404 | train 0..3 still does not transfer to validation 4..7. |

## Decision

Remote scale remains blocked.

The P-layer-relative coordinate path is a useful local refinement: active2 is
more stable than the previous simple `relative_stats` route. However, the
active4/8/16 and heldout rows still do not show broad active-coordinate
generalization. This supports keeping the route local and moving to a stronger
active-conditioned graph/token model rather than scaling this statistics-only
variant.

Next action: build a real active-conditioned PRESENT graph/token model where
active metadata modulates token embeddings or message passing directly, then
rerun the same local random-active/heldout gate.
