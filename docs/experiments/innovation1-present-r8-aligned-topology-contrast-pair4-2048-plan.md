# Innovation 1 PRESENT r8 Aligned Topology-Contrast Pair4 2048/Class Gate

## Status

status = completed local diagnostic gate
decision = pair4 2048/class passes control ordering on both seeds, but the AUC margins are small and remain diagnostic-only
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1/history.csv

## Question

The 512/class pair-count gate found that `pairs_per_sample = 4` passed both
topology controls, while `pairs_per_sample = 2` failed on seed0. This gate asks
whether the pair4 route remains control-clean when the local budget is raised
from 512/class to 2048/class.

Only one budget setting changes relative to the completed pair4 local gate:

- `samples_per_class` changes from `512` to `2048`

The pair count, feature encoding, strict encrypted-random-plaintext negative
protocol, active-nibble metadata, model route, topology contrast branch, and
auxiliary scale stay unchanged. The input size remains
`4 * 320 + 16 = 1296` bits.

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
| shuffled | `shuffled` | controls whether pair4 topology contrast depends on true P-layer targets at 2048/class |
| metadata-only | `metadata_only` | controls whether active metadata plus the contrast branch explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate passes, pair4 earns a stronger local diagnostic status and can be
considered for a larger, disk-cached remote diagnostic after a pushable commit
is available. If it fails, treat the 512/class pair4 pass as fragile and
redesign the representation before scale-up.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.487174988 | 0.509993553 | above both controls on both seeds, but close to chance |
| shuffled | 0.482413769 | 0.504078388 | below true on both seeds |
| metadata-only | 0.478052616 | 0.499298096 | below true on both seeds |

Gate deltas:

```text
pair4_2048 seed0 true-shuffled = +0.004761219
pair4_2048 seed0 true-metadata = +0.009122372
pair4_2048 seed1 true-shuffled = +0.005915165
pair4_2048 seed1 true-metadata = +0.010695457
```

The desired gate passed:

```text
true > shuffled on both seeds      yes
true > metadata-only on both seeds yes
```

The input size remains `1296` bits. Compared with the 512/class pair4 gate,
the ordering survives, but the absolute AUC and margins shrink. Training AUC
also stays low, so this does not look like a high-capacity overfit case. The
result is best interpreted as a weak but control-clean local diagnostic.

## Decision

Keep pair4 as the current minimum useful pair count, but do not remote-scale
from this result alone. The next local diagnostic should check whether a modest
increase to `pairs_per_sample = 8` improves absolute AUC while avoiding the
16-pair shuffled-topology instability, or whether the current representation
needs stronger active-coordinate-relative topology features.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_pair4_2048_seed0_seed1/progress.jsonl
```
