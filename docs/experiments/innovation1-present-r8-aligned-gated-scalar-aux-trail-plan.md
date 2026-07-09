# Innovation 1 PRESENT r8 Aligned Gated Scalar Auxiliary Trail Gate

## Status

status = completed local diagnostic gate
decision = discard scalar-gated full beamstats8deep4 route as remote-scale candidate
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1/history.csv

## Question

The vector-gated auxiliary trail gate reduced the fixed-source trail scaffold
but did not cleanly reduce wrong-source trail on both seeds. This follow-up
tests a stricter gate: trail gets one scalar gate per sample instead of one gate
per hidden channel.

The question is whether a scalar gate can preserve true full-route signal while
making wrong-source and fixed-source trail less useful.

## Method

Use `present_trail_position_stats_pairset` with:

- `trail_fusion = gated_auxiliary`
- `trail_gate = scalar`
- `trail_auxiliary_scale = 0.25`

The scalar gate computes one value from the prefix embedding:

```text
combined = prefix_embedding + 0.25 * scalar_gate(prefix_embedding) * trail_embedding
```

This is stricter than the vector gate because the model cannot independently
open many trail hidden channels.

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

| route | feature encoding | purpose |
| --- | --- | --- |
| full | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits` | true route |
| maskedsource | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_maskedsource_cell_matrix_bits` | wrong-source trail control |
| constantsource | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_constantsource_cell_matrix_bits` | fixed-source trail control |

## Gate

Pass pattern:

```text
full remains clearly above prefix-only anchor
maskedsource drops below full on both seeds
constantsource stays much lower than full on both seeds
```

If maskedsource remains close to full, scalar gating is still not sufficient and
the next redesign should move away from full beamstats8deep4 trail values,
either by weakening trail summaries or by using a stronger mismatch-invariant
regularization/control objective.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| full | 0.896041870 | 0.959915161 | true route remains useful but less stable |
| maskedsource | 0.927520752 | 0.977416992 | wrong-source trail exceeds full on both seeds |
| constantsource | 0.787994385 | 0.757537842 | fixed-source trail remains reduced |

Scalar gating did not solve the key mismatch problem. It kept the fixed-source
control much lower than the old concatenated route, but wrong-source
`maskedsource` stayed higher than the true full route on both seeds.

## Decision

Do not remote-scale scalar-gated full beamstats8deep4. The result suggests that
fusion control alone is insufficient: the trail values themselves are still too
easy for the model to use even when they are sourced from the wrong per-sample
signal.

Next local direction: weaken the trail feature family rather than only changing
the branch fusion. Prefer a small, same-budget matrix over shallower/local
trail summaries, such as lower beam width/depth or aggregated per-depth summary,
with the same full/maskedsource/constantsource gate.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_gated_scalar_aux_trail_512_seed0_seed1/progress.jsonl
```
