# Innovation 1 PRESENT r8 Active-Relative Contrast Pair4 4096/Class Gate

## Status

status = completed local diagnostic gate
decision = control-clean but fragile; do not remote-scale from this gate
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_4096_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/history.csv

## Question

The active-relative contrast pair4 2048/class gate was the first local variant
in this family to pass both shuffled and metadata-only controls on both seeds.
This gate asks whether that ordering survives a slightly larger local budget
before considering any remote scale-up.

Only one protocol value changes relative to the completed pair4 2048/class
gate:

- `samples_per_class = 4096`

The pair count, feature encoding, strict encrypted-random-plaintext negative
protocol, active-nibble metadata, topology contrast branch, active-relative
contrast fusion, auxiliary scale, and training settings stay unchanged. The
input size remains `4 * 320 + 16 = 1296` bits.

## Method

Use no DDT trail-value block. Each pair is encoded with:

- ciphertext left word;
- ciphertext right word;
- ciphertext xor;
- P-aligned xor;
- structural inverse prefix.

The model uses:

- `edge_mode = persistent`
- `cross_pair_consistency = edge_mean_absdev`
- `active_metadata_fusion = coordinate_only`
- `topology_auxiliary_scale = 0.3`
- `topology_contrast_fusion = true_minus_shuffled`
- `active_relative_contrast_fusion = true_minus_shuffled_slots`

The active-relative contrast branch computes both true and shuffled
source/target-slot summaries from the same pair hidden tokens, then projects
`[true_summary, shuffled_summary, true - shuffled, abs(true - shuffled)]` into
one sample-level classifier token.

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 4096
- pairs_per_sample = 4
- feature_encoding = `present_pair_xor_paligned_sinv_cell_matrix_bits`
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1
- model_key = `present_active_cell_graph_pairset`

Routes:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | true graph-mode message passing plus active-relative contrast |
| shuffled | `shuffled` | shuffled graph-mode message passing plus the same active-relative contrast branch |
| metadata-only | `metadata_only` | local-only graph path plus the same active-relative contrast branch |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

Compare against the completed 2048/class active-relative contrast gate:

```text
active_rel_contrast_pair4_2048 true AUC = 0.520668507 seed0, 0.520288944 seed1
active_rel_contrast_pair4_2048 true-shuffled = +0.042667866 seed0, +0.021866322 seed1
active_rel_contrast_pair4_2048 true-metadata = +0.041338444 seed0, +0.035569668 seed1
```

## Result

Completed locally with 6 plan-aligned rows. Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_4096_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/results.jsonl \
  --expected-rows 6

status = pass
```

Plot/history artifacts:

```text
outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/curves.svg
outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/history.csv
```

Metrics:

| route | seed | input_bits | val_auc | val_accuracy | val_best_accuracy | train_auc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true | 0 | 1296 | 0.502257228 | 0.510498047 | 0.516845703 | 0.530149132 |
| shuffled | 0 | 1296 | 0.497941256 | 0.500000000 | 0.507812500 | 0.533782035 |
| metadata-only | 0 | 1296 | 0.490000367 | 0.496826172 | 0.501708984 | 0.544323593 |
| true | 1 | 1296 | 0.515355825 | 0.500000000 | 0.517578125 | 0.515179366 |
| shuffled | 1 | 1296 | 0.514409304 | 0.500000000 | 0.518798828 | 0.522285134 |
| metadata-only | 1 | 1296 | 0.497988820 | 0.500000000 | 0.509521484 | 0.535691023 |

Control deltas:

| seed | true - shuffled AUC | true - metadata-only AUC |
| ---: | ---: | ---: |
| 0 | +0.004315972 | +0.012256861 |
| 1 | +0.000946522 | +0.017367005 |

Comparison to the completed 2048/class active-relative contrast gate:

| samples_per_class | seed0 true AUC | seed1 true AUC | seed0 true-shuffled | seed1 true-shuffled |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.520668507 | 0.520288944 | +0.042667866 | +0.021866322 |
| 4096 | 0.502257228 | 0.515355825 | +0.004315972 | +0.000946522 |

## Decision

Keep the route as control-clean, but downgrade its readiness. The 4096/class
gate still satisfies the desired ordering on both seeds:

```text
true > shuffled
true > metadata-only
```

However, the result is much less convincing than the 2048/class gate. The
true-vs-shuffled margins collapse from +0.042667866/+0.021866322 at 2048/class
to +0.004315972/+0.000946522 at 4096/class. Seed1 is technically ordered but
nearly tied with the shuffled control, and the absolute true AUC does not
increase with more local data.

This means the active-relative contrast idea is still plausible, but not
remote-ready. Do not launch a 65536/class remote scale-up from this evidence.
The next reasonable local step is either:

- run the same 6-row matrix at 8192/class as a fragility check; or
- redesign the contrast branch so true-vs-shuffled separation is larger before
  spending a remote slot.

The 4096/class result should be reported as diagnostic-only and control-clean
but fragile.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_4096_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_4096_seed0_seed1/progress.jsonl
```
