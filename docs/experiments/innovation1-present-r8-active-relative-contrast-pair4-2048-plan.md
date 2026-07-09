# Innovation 1 PRESENT r8 Active-Relative Contrast Pair4 2048/Class Gate

## Status

status = completed local diagnostic gate
decision = keep as strongest current local active-relative candidate; run 4096/class or 8192/class stability gate before remote scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_2048_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1/history.csv

## Question

The direct active-relative slot summary improved seed0 margins but failed
true-vs-shuffled on seed1. This gate asks whether using an explicit
true-minus-shuffled active-relative slot contrast is more stable than directly
feeding route-specific source/target slots into the pair embedding.

Only one model-side representation setting changes relative to pair4
2048/class:

- `active_relative_contrast_fusion = true_minus_shuffled_slots`

The sample count, pair count, feature encoding, strict
encrypted-random-plaintext negative protocol, active-nibble metadata,
topology contrast branch, and auxiliary scale stay unchanged. The input size
remains `4 * 320 + 16 = 1296` bits.

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
- samples_per_class = 2048
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

Also compare against the direct active-relative slot summary:

```text
active_rel_pair4_2048 true AUC = 0.501859188 seed0, 0.502976418 seed1
active_rel_pair4_2048 true-shuffled = +0.026293278 seed0, -0.009750366 seed1
active_rel_pair4_2048 true-metadata = +0.028957367 seed0, +0.004674911 seed1
```

## Result

Completed locally with 6 plan-aligned rows. Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_2048_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1/results.jsonl \
  --expected-rows 6

status = pass
```

Metrics:

| route | seed | input_bits | val_auc | val_accuracy | val_best_accuracy | train_auc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true | 0 | 1296 | 0.520668507 | 0.500000000 | 0.527343750 | 0.563698769 |
| shuffled | 0 | 1296 | 0.478000641 | 0.500000000 | 0.504394531 | 0.557123542 |
| metadata-only | 0 | 1296 | 0.479330063 | 0.500488281 | 0.502441406 | 0.555219650 |
| true | 1 | 1296 | 0.520288944 | 0.500000000 | 0.526855469 | 0.537480712 |
| shuffled | 1 | 1296 | 0.498422623 | 0.500000000 | 0.512207031 | 0.553471804 |
| metadata-only | 1 | 1296 | 0.484719276 | 0.500000000 | 0.507324219 | 0.551700115 |

Control deltas:

| seed | true - shuffled AUC | true - metadata-only AUC |
| ---: | ---: | ---: |
| 0 | +0.042667866 | +0.041338444 |
| 1 | +0.021866322 | +0.035569668 |

Comparison to prior pair4 2048/class local gates:

| route | seed0 true AUC | seed1 true AUC | seed0 true-shuffled | seed1 true-shuffled |
| --- | ---: | ---: | ---: | ---: |
| raw-prefix topology contrast | 0.487174988 | 0.509993553 | +0.004761219 | +0.005915165 |
| direct active-relative summary | 0.501859188 | 0.502976418 | +0.026293278 | -0.009750366 |
| active-relative contrast | 0.520668507 | 0.520288944 | +0.042667866 | +0.021866322 |

## Decision

Keep this candidate and advance it one local/medium step, but do not make a
formal PRESENT r8 claim from this gate. The active-relative contrast route is
the first active-relative pair4 2048/class variant that passes both controls
on both seeds:

```text
true > shuffled
true > metadata-only
```

It also improves absolute true AUC and true-vs-control margins over both the
raw-prefix pair4 2048/class route and the direct active-relative slot summary.
Most importantly, it fixes the seed1 shuffled-control reversal seen in the
direct active-relative route.

This is still only a 2048/class CPU local diagnostic with 3 epochs and 2
seeds. It is not formal PRESENT evidence, not a breakthrough claim, and not
enough to remote-scale directly to 65536/class by default. The next step should
be the same 6-row matrix at either 4096/class or 8192/class, keeping the
strict encrypted-random-plaintext negative protocol, the same pair count, the
same feature encoding, and both shuffled and metadata-only controls.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_2048_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_2048_seed0_seed1/progress.jsonl
```
