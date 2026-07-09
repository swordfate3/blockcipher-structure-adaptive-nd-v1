# Innovation 1 PRESENT r8 Aligned Gated Auxiliary Trail Gate

## Status

status = completed local diagnostic gate
decision = diagnostic partial improvement, not remote-scale ready
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_gated_aux_trail_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_gated_aux_trail_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_gated_aux_trail_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_gated_aux_trail_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_gated_aux_trail_512_seed0_seed1/history.csv

## Question

The previous trail-value mismatch and trail-normalization gates showed that the
full beamstats8deep4 trail block is too permissive. Wrong-source and fixed-source
trail blocks remain high, so the current concatenated statistics can use trail
as a scaffold even when the trail values are not sample-specific.

This gate tests a smaller architecture redesign: make trail an auxiliary branch
with a controlled contribution path instead of concatenating all prefix and trail
statistics into one dense vector.

## Method

Add `trail_fusion = gated_auxiliary` to
`present_trail_position_stats_pairset`.

The model computes:

- `prefix_stats` from the real prefix words only;
- `trail_stats` from the beamstats trail words only;
- `prefix_embedding = prefix_encoder(prefix_stats)`;
- `trail_embedding = trail_encoder(trail_stats)`;
- `gate = sigmoid(gate_network(prefix_embedding))`;
- `combined = prefix_embedding + trail_auxiliary_scale * gate * trail_embedding`.

The local gate uses `trail_auxiliary_scale = 0.25`. This does not prove the
right scale, but it prevents trail from entering as an unconstrained full-width
statistics block.

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
- model option = `trail_fusion: gated_auxiliary`
- model option = `trail_auxiliary_scale: 0.25`
- seeds = 0, 1

Routes per seed:

| route | feature encoding | purpose |
| --- | --- | --- |
| full | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits` | true route should remain above prefix-only if trail is useful |
| maskedsource | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_maskedsource_cell_matrix_bits` | wrong-source trail should drop if the model needs sample-specific trail |
| constantsource | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_constantsource_cell_matrix_bits` | fixed-source trail should drop if scaffold use is reduced |

## Gate

The desired pattern is:

```text
gated_aux full > prefix-only anchor
gated_aux maskedsource substantially lower than gated_aux full
gated_aux constantsource substantially lower than gated_aux full
```

If full, maskedsource, and constantsource all stay high together, this
architecture is still too permissive. The next step should lower the trail scale,
make the gate scalar instead of vector-valued, or move to weaker/local trail
summaries.

If full collapses close to prefix-only, the auxiliary trail path is too weak or
the signal depends on the large concatenated trail scaffold.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| full | 0.959564209 | 0.958099365 | true route remains high |
| maskedsource | 0.899963379 | 0.956344604 | wrong-source trail drops on seed0 but not seed1 |
| constantsource | 0.737449646 | 0.787750244 | fixed-source trail drops substantially |

The vector-gated auxiliary trail path is a partial improvement over the
concatenated beamstats8deep4 route. It strongly reduces the fixed-source
scaffold compared with the prior constantsource AUC range around 0.90-0.95.
However, it does not cleanly solve the wrong-source problem: maskedsource remains
close to full on seed1.

## Decision

Do not remote-scale the vector-gated auxiliary route. It is useful as evidence
that controlling the trail branch can reduce fixed nonzero scaffold behavior,
but it is not enough for a positive route decision because wrong-source trail is
still too strong on one seed.

The immediate follow-up was a scalar-gate variant. If that also fails, the next
local route should weaken or localize the trail summary itself instead of only
changing the fusion path.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_gated_aux_trail_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_gated_aux_trail_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_gated_aux_trail_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_gated_aux_trail_512_seed0_seed1/progress.jsonl
```
