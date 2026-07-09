# Innovation 1 PRESENT r8 Active-Relative Topology Pair4 2048/Class Gate

## Status

status = completed local diagnostic gate
decision = active-relative pair4 2048/class improves seed0 margins but fails true-vs-shuffled on seed1; do not scale yet
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_topology_pair4_2048_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_active_relative_topology_pair4_2048_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_topology_pair4_2048_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_topology_pair4_2048_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_active_relative_topology_pair4_2048_seed0_seed1/history.csv

## Question

The raw-prefix pair-count ladder found that pair4 is the minimum
control-clean pair count, but pair4 2048/class is weak and pair8 reintroduces
seed0 control instability. This gate asks whether exposing topology before
global pooling as active-relative source/target slots improves the pair4
2048/class route.

Only one model-side representation setting changes relative to pair4
2048/class:

- `active_relative_summary = source_target_slots`

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
- `active_relative_summary = source_target_slots`

The active-relative summary gathers, per pair and per active nibble:

- source cell token;
- four P-layer target cell tokens in active-relative slot order;
- target-mean minus source delta.

It projects these six token slots into one extra pair embedding before
pair-set pooling.

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
| true | `true` | active-relative slots use true PRESENT P-layer targets |
| shuffled | `shuffled` | active-relative slots use the fixed shuffled target-cell permutation |
| metadata-only | `metadata_only` | active-relative slot summary is zeroed; controls active metadata plus added capacity |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

Also compare true AUC and margins against pair4 2048/class:

```text
pair4_2048 true AUC = 0.487174988 seed0, 0.509993553 seed1
pair4_2048 true-shuffled = +0.004761219 seed0, +0.005915165 seed1
pair4_2048 true-metadata = +0.009122372 seed0, +0.010695457 seed1
```

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.501859188 | 0.502976418 | improves seed0 versus pair4 2048/class, but loses true-vs-shuffled on seed1 |
| shuffled | 0.475565910 | 0.512726784 | below true on seed0; beats true on seed1 |
| metadata-only | 0.472901821 | 0.498301506 | below true on both seeds |

Gate deltas:

```text
active_rel_pair4_2048 seed0 true-shuffled = +0.026293278
active_rel_pair4_2048 seed0 true-metadata = +0.028957367
active_rel_pair4_2048 seed1 true-shuffled = -0.009750366
active_rel_pair4_2048 seed1 true-metadata = +0.004674911
```

The desired gate failed:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds yes
```

Compared with pair4 2048/class:

```text
seed0 true AUC: 0.487174988 -> 0.501859188
seed1 true AUC: 0.509993553 -> 0.502976418
seed0 true-shuffled: +0.004761219 -> +0.026293278
seed1 true-shuffled: +0.005915165 -> -0.009750366
```

The active-relative summary helped seed0 substantially but destabilized the
seed1 true-vs-shuffled control. This makes it a useful diagnostic direction,
not a keep/scale result.

## Decision

Do not scale the current active-relative summary. The next local design should
keep the idea of pre-pooling active-relative topology, but reduce the chance
that shuffled slots become a stronger learned shortcut on one seed. Candidate
follow-ups include a lower-weight active-relative branch, an auxiliary-only
pretext stage, or a contrast that compares true and shuffled active-relative
slot embeddings before adding them to the classifier.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_topology_pair4_2048_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_relative_topology_pair4_2048_seed0_seed1 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_relative_topology_pair4_2048_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_relative_topology_pair4_2048_seed0_seed1/progress.jsonl
```
