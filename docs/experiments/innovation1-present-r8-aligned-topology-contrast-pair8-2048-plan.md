# Innovation 1 PRESENT r8 Aligned Topology-Contrast Pair8 2048/Class Gate

## Status

status = completed local diagnostic gate
decision = pair8 2048/class fails the true-vs-shuffled and metadata-only ordering on seed0; do not scale raw-prefix pair8
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1/history.csv

## Question

The 2048/class pair4 gate preserved the desired topology-control ordering, but
the absolute AUC and margins were weak. This gate asks whether increasing the
sample from 4 pairs to 8 pairs improves absolute AUC while still avoiding the
16-pair shuffled-topology instability.

Only one setting changes relative to pair4 2048/class:

- `pairs_per_sample` changes from `4` to `8`

The sample count, feature encoding, strict encrypted-random-plaintext negative
protocol, active-nibble metadata, model route, topology contrast branch, and
auxiliary scale stay unchanged. The input size changes from
`4 * 320 + 16 = 1296` bits to `8 * 320 + 16 = 2576` bits.

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
- samples_per_class = 2048
- pairs_per_sample = 8
- feature_encoding = `present_pair_xor_paligned_sinv_cell_matrix_bits`
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1
- model_key = `present_active_cell_graph_pairset`

Routes:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | candidate true-topology graph with 8-pair aligned integral samples |
| shuffled | `shuffled` | controls whether pair8 topology contrast depends on true P-layer targets |
| metadata-only | `metadata_only` | controls whether active metadata plus the contrast branch explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate passes and improves absolute AUC versus pair4 2048/class, pair8
becomes the better local candidate than pair4. If it fails or remains weak,
the raw-prefix topology-contrast route should not be remote-scaled yet.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.502781391 | 0.525839329 | fails both controls on seed0; passes both controls on seed1 |
| shuffled | 0.510715961 | 0.490370274 | beats true on seed0 |
| metadata-only | 0.504064560 | 0.482904911 | slightly beats true on seed0 |

Gate deltas:

```text
pair8_2048 seed0 true-shuffled = -0.007934570
pair8_2048 seed0 true-metadata = -0.001283169
pair8_2048 seed1 true-shuffled = +0.035469055
pair8_2048 seed1 true-metadata = +0.042934418
```

The desired gate failed:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds no
```

The input size is `2576` bits. Pair8 improves seed1's absolute AUC and control
margin versus pair4 2048/class, but seed0 reintroduces the true-vs-shuffled
reversal. This is the same kind of instability that motivated moving away from
the original 16-pair route, only at a smaller input width.

## Decision

Do not scale pair8. Keep pair4 as the current minimum control-clean pair count,
but treat the raw-prefix topology-contrast family as weak and not remote-ready.
The next meaningful local work should redesign the representation so topology
is expressed relative to the active coordinate before global pooling, rather
than continuing to sweep pair counts.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair8_2048_seed0_seed1/progress.jsonl
```
