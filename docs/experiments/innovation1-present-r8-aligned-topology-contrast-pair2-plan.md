# Innovation 1 PRESENT r8 Aligned Topology-Contrast Pair2 Gate

## Status

status = completed local diagnostic gate
decision = pair2 fails the true-vs-shuffled and metadata-only ordering on seed0; keep pair4 as the current minimum useful pair count
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1/history.csv

## Question

The 4-pair topology-contrast diagnostic reduced the input from 5136 bits to
1296 bits and passed the true-vs-shuffled plus metadata-only ordering on both
seeds. This gate asks whether the same route still preserves topology signal
when reduced to 2 pairs per sample.

Only one data-budget setting changes relative to pair4:

- `pairs_per_sample` changes from `4` to `2`

The feature encoding, strict encrypted-random-plaintext negative protocol,
active-nibble metadata, model route, topology contrast branch, and auxiliary
scale stay unchanged. The input size changes from `4 * 320 + 16 = 1296` bits
to `2 * 320 + 16 = 656` bits.

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

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 2
- feature_encoding = `present_pair_xor_paligned_sinv_cell_matrix_bits`
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1
- model_key = `present_active_cell_graph_pairset`

Routes:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | candidate true-topology graph with 2-pair aligned integral samples |
| shuffled | `shuffled` | controls whether 2-pair topology contrast depends on true P-layer targets |
| metadata-only | `metadata_only` | controls whether active metadata plus the contrast branch explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate passes, the reduced-pair route is a stronger local diagnostic than
pair4 because it reaches the same ordering with fewer input bits. If it fails,
pair4 remains the current minimum useful pair count for this representation.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.479537964 | 0.522048950 | fails both controls on seed0; passes both controls on seed1 |
| shuffled | 0.507637024 | 0.513809204 | beats true on seed0 |
| metadata-only | 0.494918823 | 0.507316589 | beats true on seed0 |

Gate deltas:

```text
pair2 seed0 true-shuffled = -0.028099060
pair2 seed0 true-metadata = -0.015380859
pair2 seed1 true-shuffled = +0.008239746
pair2 seed1 true-metadata = +0.014732361
```

The desired gate failed:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds no
```

The input size is `656` bits, which confirms the intended reduction from
pair4's `1296` bits. However, the topology ordering does not survive on seed0.
This suggests that 2 aligned pairs are too sparse for the current raw-prefix
topology-contrast representation at this local diagnostic budget.

## Decision

Do not scale pair2. Treat pair4 as the current minimum useful pair count for
this representation. The next local step should be a narrow pair4-focused
follow-up, such as slightly increasing the local budget or checking a
pair4-vs-pair8 ladder under the same protocol, before considering remote scale.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair2_512_seed0_seed1/progress.jsonl
```
