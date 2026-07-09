# Innovation 1 PRESENT r8 Aligned Cross-Pair Edge Consistency Gate

## Status

status = completed local diagnostic gate
decision = cross-pair edge consistency is unstable; do not scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1/history.csv

## Question

The persistent edge-token graph keeps public PRESENT P-layer edge tokens, but it
still summarizes each pair mostly independently before pair-set pooling. The
512/class gate was unstable: seed1 showed the desired true-over-controls order,
while seed0 reversed it.

This gate tests the next minimal representation change: keep persistent edge
tokens for each pair, then compare the same public edge across all 16 pairs in a
sample. The goal is to see whether repeated edge-local behavior across the
pair-set is more useful than independent pair summaries.

## Method

Use no DDT trail-value block. Each pair is encoded with:

- ciphertext left word;
- ciphertext right word;
- ciphertext xor;
- P-aligned xor;
- structural inverse prefix.

The model uses `edge_mode = persistent` and
`cross_pair_consistency = edge_mean_absdev`. For each sample it preserves the
edge token for each persistent P-layer source-target relation in every pair,
then computes, per edge across the 16 pairs:

- mean response;
- mean absolute deviation from that response;
- max response.

Those cross-pair edge summaries are pooled and appended to the final classifier.
Active metadata still acts as coordinate conditioning; it is not a DDT trail
label or candidate beam value.

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
- model_key = `present_active_cell_graph_pairset`
- model option `edge_mode = persistent`
- model option `cross_pair_consistency = edge_mean_absdev`

Routes:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | candidate true-topology cross-pair edge consistency |
| shuffled | `shuffled` | controls whether true P-layer targets matter |
| metadata-only | `metadata_only` | controls whether active metadata alone explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If the gate passes, run a slightly larger local/GPU screen such as 2048/class
or 8192/class before considering remote scale. If true ties or loses to either
control, do not scale; redesign locally.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.547439575 | 0.461982727 | wins seed0, loses seed1 |
| shuffled | 0.505828857 | 0.485153198 | below true on seed0, above true on seed1 |
| metadata-only | 0.507049561 | 0.484283447 | below true on seed0, above true on seed1 |

The desired gate did not pass:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds no
```

Compared with the persistent edge-token gate, the instability moved rather than
disappeared. Persistent edge tokens had seed1 in the desired order and seed0
reversed. Cross-pair edge consistency flips that pattern: seed0 now shows the
desired order, but seed1 reverses.

## Decision

Do not scale this route. Keep `cross_pair_consistency = edge_mean_absdev` as a
diagnostic baseline, but do not launch 2048/class, 8192/class, or remote runs
from this result.

The next local route should address the deeper instability rather than adding
sample count. Two better directions are:

- remove direct final active-metadata injection and keep active only inside
  coordinate/edge conditioning;
- add a topology-control auxiliary objective that forces true and shuffled
  topology embeddings to be separable without feeding DDT trail values.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_cross_pair_edge_consistency_512_seed0_seed1/progress.jsonl
```
