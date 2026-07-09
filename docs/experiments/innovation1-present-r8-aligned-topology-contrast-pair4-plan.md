# Innovation 1 PRESENT r8 Aligned Topology-Contrast Pair4 Gate

## Status

status = completed local diagnostic gate
decision = pair4 passes true-vs-shuffled and metadata-only ordering on both seeds; test pair2 next before scale-up
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1/history.csv

## Question

The 16-pair topology-contrast route uses 5136 input bits per sample and remains
unstable against shuffled topology on seed1. This gate asks whether reducing
the same aligned random-active protocol to 4 pairs per sample reduces
high-dimensional shortcut pressure and stabilizes the true-vs-shuffled control.

Only one data-budget setting changes relative to the strongest aux03 local
diagnostic:

- `pairs_per_sample` changes from `16` to `4`

The feature encoding, strict encrypted-random-plaintext negative protocol,
active-nibble metadata, model route, topology contrast branch, and auxiliary
scale stay unchanged. The input size changes from `16 * 320 + 16 = 5136` bits
to `4 * 320 + 16 = 1296` bits.

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
| true | `true` | candidate true-topology graph with 4-pair aligned integral samples |
| shuffled | `shuffled` | controls whether 4-pair topology contrast depends on true P-layer targets |
| metadata-only | `metadata_only` | controls whether active metadata plus the contrast branch explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate passes, test 2-pair before any scale-up. If it fails, the next
step should be either a 2-pair diagnostic or a true single-pair
active-metadata protocol ablation, not a remote run.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.530555725 | 0.540649414 | above both controls on both seeds |
| shuffled | 0.500633240 | 0.473312378 | below true on both seeds |
| metadata-only | 0.489967346 | 0.490112305 | below true on both seeds |

Gate deltas:

```text
pair4 seed0 true-shuffled = +0.029922485
pair4 seed0 true-metadata = +0.040588379
pair4 seed1 true-shuffled = +0.067337036
pair4 seed1 true-metadata = +0.050537109
```

The desired gate passed:

```text
true > shuffled on both seeds      yes
true > metadata-only on both seeds yes
```

The absolute AUC values remain modest, so this is not scale-ready evidence by
itself. However, it is the cleanest true-vs-shuffled ordering observed in the
raw-prefix topology-contrast family so far. Reducing the sample from 16 pairs
to 4 pairs changed the input from 5136 bits to 1296 bits and removed the seed1
shuffled reversal seen in the 16-pair gates.

## Decision

Keep pair4 as a diagnostic improvement, but do not remote-scale yet. The next
local gate should test `pairs_per_sample = 2` under the same protocol. If pair2
also preserves `true > shuffled` and `true > metadata-only` on both seeds, then
the evidence supports the hypothesis that the 16-pair aggregation was
introducing unstable shortcut pressure. If pair2 fails, pair4 may be the useful
minimum for this representation.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_512_seed0_seed1/progress.jsonl
```
