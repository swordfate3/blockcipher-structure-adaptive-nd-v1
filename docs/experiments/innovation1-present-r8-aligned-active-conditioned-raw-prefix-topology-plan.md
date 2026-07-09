# Innovation 1 PRESENT r8 Aligned Active-Conditioned Raw-Prefix Topology Gate

## Status

status = completed local diagnostic gate
decision = active-conditioned raw-prefix topology is still weak; redesign representation before scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1/history.csv

## Question

The unconditioned raw-prefix P-layer topology gate removed DDT trail values and
stayed near chance. It also failed the topology-control check because true
P-layer adjacency did not beat shuffled adjacency.

This gate asks a narrower follow-up question: does raw-prefix topology become
usable when each sample tells the model which PRESENT nibble is active?

In plain terms, the previous model saw the same board layout but the active
starting square moved around from sample to sample. This gate gives the model a
16-bit active-nibble marker and injects that marker into the P-layer token
stream, while still refusing to feed DDT trail-value blocks.

## Method

Use raw prefix feature encodings only:

- `present_pair_xor_paligned_sinv_cell_matrix_bits`
- `present_pair_xor_paligned_cell_matrix_bits`

Use:

- `sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata`
- `metadata_bits = 16`
- `active_conditioning = p_layer_active_token_bias`

Compare:

- true P-layer token mixer with active conditioning and `sinv`;
- shuffled P-layer topology control with active conditioning and `sinv`;
- true P-layer token mixer with active conditioning but without `sinv`.

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1

Routes:

| route | model | feature | purpose |
| --- | --- | --- | --- |
| true-sinv | present_p_layer_mixer_pairset | `present_pair_xor_paligned_sinv_cell_matrix_bits` | candidate active-conditioned raw-prefix topology model |
| shuffled-sinv | present_p_layer_mixer_pairset | same | controls whether true PRESENT P-layer topology matters |
| true-no-sinv | present_p_layer_mixer_pairset | `present_pair_xor_paligned_cell_matrix_bits` | controls whether structural inverse prefix matters |

## Gate

Desired pattern:

```text
true-sinv improves over the unconditioned true-sinv raw-prefix gate
true-sinv > shuffled-sinv on both seeds
true-sinv > true-no-sinv if structural inverse prefix is useful
```

If true-sinv improves but shuffled-sinv improves similarly, the active marker is
helping but the current topology mixer still is not using true PRESENT adjacency.
If true-sinv remains near chance, this simple active-token bias is not enough
and the next representation should use stronger relative-coordinate or
cell-equivariant structure.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true-sinv | 0.550659180 | 0.510643005 | weak; only seed0 slightly improves over unconditioned |
| shuffled-sinv | 0.506835938 | 0.530029297 | topology control wins on seed1 |
| true-no-sinv | 0.502883911 | 0.512893677 | near chance |

The desired pattern did not hold. Active metadata did not produce a stable
raw-prefix signal, and true PRESENT P-layer topology did not consistently beat
the shuffled-topology control.

Compared with the unconditioned raw-prefix gate:

| route | unconditioned seed0 | active-conditioned seed0 | unconditioned seed1 | active-conditioned seed1 |
| --- | ---: | ---: | ---: | ---: |
| true-sinv | 0.540695190 | 0.550659180 | 0.520507812 | 0.510643005 |
| shuffled-sinv | 0.567588806 | 0.506835938 | 0.537811279 | 0.530029297 |
| true-no-sinv | 0.535354614 | 0.502883911 | 0.524757385 | 0.512893677 |

So the active-token bias is not a clear rescue. It slightly raises true-sinv on
seed0, lowers it on seed1, and does not establish that the model is using true
PRESENT topology.

## Decision

Do not remote-scale this active-conditioned raw-prefix topology route.

This gate reinforces the current direction: the high AUC from dense DDT trail
features is not recovered by simply removing trail values, adding P-layer
adjacency, and telling the model the active nibble. The next architecture should
change the representation more fundamentally, for example:

- build a relative-coordinate/cell-equivariant view that reindexes cells by
  PRESENT S-box/P-layer propagation from the active nibble;
- encode local transition evidence as small per-cell/per-edge tokens instead of
  dense DDT trail-value blocks;
- use mismatch controls as a required local gate before any medium or remote
  PRESENT r8 run.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1/progress.jsonl
```
