# Innovation 1 PRESENT r8 Aligned Trail Normalization Gate

## Status

status = completed local diagnostic gate
decision = discard simple normalization as remote-scale route
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_normalization_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_trail_normalization_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_trail_normalization_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_trail_normalization_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_trail_normalization_512_seed0_seed1/history.csv

## Question

The aligned trail-value mismatch gate showed that the current
beamstats8deep4 trail block is too permissive: masked wrong-source trail stays
as high as the true full route, and fixed-source trail remains much higher than
prefix-only.

This gate tests the first redesign direction: normalize the trail-stat block so
fixed nonzero trail sources cannot act as a scaffold for prefix separation.

## Method

Add model-side trail normalization inside
`present_trail_position_stats_pairset`, after bit cells are converted to
activity values and before trail-position statistics are computed:

- `trail_center`: subtract each sample-pair trail block mean.
- `trail_zscore`: subtract each sample-pair trail block mean and divide by its
  standard deviation.

Prefix words are not normalized. Only the trail region is changed.

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active
- integral_active_nibbles = `{0..15}`
- model = present_trail_position_stats_pairset
- seeds = 0, 1

Routes per seed:

| normalization | feature route | purpose |
| --- | --- | --- |
| trail_center | full | check whether true trail remains high after centering |
| trail_center | maskedsource | check whether wrong-source shortcut drops after centering |
| trail_center | constantsource | check whether fixed-source scaffold drops after centering |
| trail_zscore | full | check whether true trail remains high after z-score normalization |
| trail_zscore | maskedsource | check whether wrong-source shortcut drops after z-score normalization |
| trail_zscore | constantsource | check whether fixed-source scaffold drops after z-score normalization |

## Gate

The desired pattern is:

```text
normalized full remains clearly above prefix-only
normalized maskedsource drops substantially
normalized constantsource drops substantially
```

If normalized mismatch controls remain high, simple trail-stat normalization is
not enough and the next redesign should move to split prefix/trail branches or
gated auxiliary trail.

## Result

Plan validation passed for all 12 expected rows.

| normalization | route | seed0 AUC | seed1 AUC | interpretation |
| --- | --- | ---: | ---: | --- |
| trail_center | full | 0.973510742 | 0.979187012 | true route remains high |
| trail_center | maskedsource | 0.992431641 | 0.991027832 | wrong-source trail remains too high |
| trail_center | constantsource | 0.905258179 | 0.946670532 | fixed-source trail remains too high |
| trail_zscore | full | 0.987579346 | 0.984100342 | true route remains high |
| trail_zscore | maskedsource | 0.991653442 | 0.990798950 | wrong-source trail remains too high |
| trail_zscore | constantsource | 0.921203613 | 0.948593140 | fixed-source trail remains too high |

The gate failed. Centering or z-score normalization does not remove the
beamstats8deep4 scaffold/shortcut at this small diagnostic scale. The strongest
evidence is that `maskedsource`, whose trail values are computed from the wrong
source while preserving shape, remains about as strong as the true full route.
The `constantsource` control also remains far above the earlier prefix-only
anchor around 0.60 AUC.

## Decision

Do not remote-scale `full beamstats8deep4` with simple trail normalization.
Keep the code option only as a diagnostic control. The next redesign should
change how trail information enters the model, not only rescale it. Prefer a
split prefix/trail or gated auxiliary-trail architecture where:

- the real ciphertext/prefix branch can be measured independently;
- the trail branch has a controlled contribution path;
- mismatch controls must drop before any medium or remote scale-up.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_normalization_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_trail_normalization_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_trail_normalization_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_trail_normalization_512_seed0_seed1/progress.jsonl
```
