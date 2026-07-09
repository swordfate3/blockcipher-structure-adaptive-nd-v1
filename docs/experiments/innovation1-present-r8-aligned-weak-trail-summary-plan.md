# Innovation 1 PRESENT r8 Aligned Weak Trail Summary Gate

## Status

status = completed local diagnostic gate
decision = discard tested weak trail summaries as remote-scale route
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_weak_trail_summary_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_weak_trail_summary_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_weak_trail_summary_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_weak_trail_summary_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_weak_trail_summary_512_seed0_seed1/history.csv

## Question

The full `beamstats8deep4` trail block gives very high AUC, but mismatch gates
show that wrong-source and fixed-source trail can also be strong. Normalization
and gated auxiliary fusion reduced some fixed-source scaffold behavior but did
not solve the wrong-source `maskedsource` problem.

This gate tests a different redesign: weaken the trail feature itself. Instead
of asking whether a more careful model can control full `beamstats8deep4`, it
asks whether shallower/local trail summaries keep true signal while reducing
mismatch shortcuts.

## Method

Use the existing parameterized PRESENT S-box-DDT trail encoder with strict
source controls:

- `beamstats4deep2`: two trail depths, beam width 4, statistics output.
- `beamstats2deep1`: one local trail depth, beam width 2, statistics output.

For each feature family, compare:

- `full`: trail statistics computed from the real structural inverse
  difference.
- `maskedsource`: trail statistics computed from a wrong-source transformed
  difference while preserving the same prefix and shape.
- `constantsource`: trail statistics computed from a fixed source while
  preserving the same prefix and shape.

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

Routes:

| family | route | feature encoding | model trail depth |
| --- | --- | --- | ---: |
| beamstats4deep2 | full | `present_delta_paligned_sinv_sboxddt_beamstats4deep2_cell_matrix_bits` | 2 |
| beamstats4deep2 | maskedsource | `present_delta_paligned_sinv_sboxddt_beamstats4deep2_maskedsource_cell_matrix_bits` | 2 |
| beamstats4deep2 | constantsource | `present_delta_paligned_sinv_sboxddt_beamstats4deep2_constantsource_cell_matrix_bits` | 2 |
| beamstats2deep1 | full | `present_delta_paligned_sinv_sboxddt_beamstats2deep1_cell_matrix_bits` | 1 |
| beamstats2deep1 | maskedsource | `present_delta_paligned_sinv_sboxddt_beamstats2deep1_maskedsource_cell_matrix_bits` | 1 |
| beamstats2deep1 | constantsource | `present_delta_paligned_sinv_sboxddt_beamstats2deep1_constantsource_cell_matrix_bits` | 1 |

## Gate

The desired pattern is:

```text
full remains clearly above prefix-only anchor
maskedsource drops below full on both seeds
constantsource drops close to prefix-only
```

If both weak families collapse near prefix-only, the full trail route may depend
on the large hand-crafted scaffold. If maskedsource remains close to full even
with weak/local summaries, the mismatch issue is not mainly beam width/depth.

## Result

Plan validation passed for all 12 expected rows.

| family | route | seed0 AUC | seed1 AUC | interpretation |
| --- | --- | ---: | ---: | --- |
| beamstats4deep2 | full | 0.959899902 | 0.952911377 | true route remains high |
| beamstats4deep2 | maskedsource | 0.959884644 | 0.974182129 | wrong-source remains as high or higher |
| beamstats4deep2 | constantsource | 0.929473877 | 0.934555054 | fixed-source remains high |
| beamstats2deep1 | full | 0.904388428 | 0.901809692 | local route remains above prefix-only |
| beamstats2deep1 | maskedsource | 0.945175171 | 0.916732788 | wrong-source remains higher than full |
| beamstats2deep1 | constantsource | 0.909851074 | 0.942443848 | fixed-source remains high |

The gate failed. Reducing beam width and depth did not produce a clean
sample-specific trail signal. Both shallower/local feature families still allow
wrong-source and fixed-source controls to stay close to, or above, the true full
route.

This differs from the gated auxiliary result, where constantsource dropped
substantially. Here, because the concat trail-position statistics model still
receives a nonzero DDT-derived trail block, even a one-depth/two-beam summary can
act as a strong scaffold.

## Decision

Do not remote-scale these weak trail summaries. The mismatch issue is not solved
by simply lowering beam width/depth under the same concatenated DDT-trail-value
representation.

The next local direction should stop feeding DDT trail values as a dense input
block. Prefer a more native SPN cell/coordinate model that uses raw prefix
signals plus PRESENT topology/relative coordinates, or a training objective that
explicitly penalizes full and wrong-source trail embeddings being equally useful.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_weak_trail_summary_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_weak_trail_summary_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_weak_trail_summary_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_weak_trail_summary_512_seed0_seed1/progress.jsonl
```
