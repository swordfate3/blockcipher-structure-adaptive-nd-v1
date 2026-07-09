# Innovation 1 PRESENT r8 Aligned Dynamic Active Cell Graph Gate

## Status

status = completed local diagnostic gate
decision = dynamic active cell graph is unstable; keep as diagnostic and redesign before scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1/history.csv

## Question

Prior local gates showed that dense DDT trail-value inputs are too permissive,
while raw-prefix topology mixers and summary statistics are weak or fail their
metadata/topology controls.

This gate tests the next representation step: keep raw cell-local evidence as
tokens and let the active nibble select per-sample PRESENT P-layer source-target
messages.

## Method

Use no DDT trail-value block. Each 64-bit PRESENT word is split into 16 nibbles,
and each cell token receives the raw prefix channels from:

- ciphertext left word;
- ciphertext right word;
- ciphertext xor;
- P-aligned xor;
- structural inverse prefix.

For each sample, the 16-bit active-nibble metadata selects the active source
cell. In the true graph, that source cell sends messages to the real PRESENT
P-layer target cells. In the shuffled control, target cells are deterministically
shuffled. In metadata-only mode, the model sees active metadata but uses no
dynamic source-target edge messages.

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- feature_encoding = `present_pair_xor_paligned_sinv_cell_matrix_bits`
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1

Routes:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | candidate dynamic active PRESENT cell graph |
| shuffled | `shuffled` | controls whether true P-layer target cells matter |
| metadata-only | `metadata_only` | controls whether active metadata alone explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If true beats both controls but remains only modestly above chance, keep the
route as a candidate for a slightly larger local screen. If true ties or loses
to either control, do not scale; the representation still is not using PRESENT
topology in a convincing way.

## Result

Plan validation passed for all 6 expected rows.

The final run used tightened dynamic messages plus an edge-local contrast
embedding. Source messages are sent only to active P-layer target cells,
target-aggregate messages are sent only back to the source cell, and the pair
embedding includes a learned contrast over source-target cell pairs.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.530868530 | 0.461181641 | wins seed0 but fails seed1 |
| shuffled | 0.511169434 | 0.486450195 | below true on seed0, above true on seed1 |
| metadata-only | 0.509994507 | 0.486816406 | below true on seed0, above true on seed1 |

The desired gate did not pass:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds no
```

The seed0 pattern is encouraging because true topology is about 0.02 AUC above
both controls. However, seed1 reverses the ordering and falls below both
controls. Therefore this is not stable evidence that the model is using true
PRESENT P-layer topology.

## Decision

Do not scale this route. Keep the implementation as a diagnostic baseline for
future graph routes, but do not launch remote runs from this result.

The next redesign should change the representation rather than enlarge this
route:

- encode persistent edge tokens across all P-layer cell relations instead of
  only the currently active source-to-target set;
- use a contrastive or auxiliary control objective that makes true and shuffled
  topology separable before classification;
- keep the same true/shuffled/metadata-only gate before any larger run.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_dynamic_active_cell_graph_512_seed0_seed1/progress.jsonl
```
