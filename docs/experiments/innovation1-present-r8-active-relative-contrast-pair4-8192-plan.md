# Innovation 1 PRESENT r8 Active-Relative Contrast Pair4 8192/Class Gate

## Status

status = completed historical implementation-misaligned diagnostic
decision = topology verdict superseded; repair and re-adjudicate with E1-R1 before E2
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_8192_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/history.csv

## 2026-07-10 Semantic-Layout Correction

Post-run source inspection found that the cell-matrix encoder emits global
bit-plane order while the active-cell graph reshaped the 320 pair bits as
word-major cell groups. The model's graph tokens therefore mixed semantic
PRESENT cells. The six AUC values below remain valid historical measurements
of commit `8c75600`, but they do not cleanly adjudicate true PRESENT-cell
topology.

```text
historical E1 topology verdict = superseded / not adjudicated
remote_scale = no
next_adjudication = E1-R1 active-cell layout repair at 2048/class
E2 = deferred until E1-R1 resolves
```

Repair and re-adjudication plan:

```text
docs/experiments/innovation1-present-r8-active-cell-layout-repair-readjudication-plan.md
```

## Verdict Context

This is E1 from the Innovation 1 route verdict:

```text
docs/experiments/innovation1-route-verdict-2026-07-09.md
```

It is an adjudication experiment, not another broad exploration variant. It
decides whether the active-relative topology-architecture branch is still alive
after the 4096/class near-tie.

## Question

The active-relative contrast route passed controls at 2048/class, but the
4096/class follow-up became fragile:

```text
4096/class seed0 true-shuffled = +0.004315972
4096/class seed1 true-shuffled = +0.000946522
```

This gate asks whether the same protocol at 8192/class reopens the
true-vs-shuffled margin or confirms that the topology-architecture route is
only weak/fragile.

Only one protocol value changes relative to the completed 4096/class gate:

- `samples_per_class = 8192`

The pair count, feature encoding, strict encrypted-random-plaintext negative
protocol, active-nibble metadata, topology contrast branch, active-relative
contrast fusion, auxiliary scale, graph modes, and training settings stay
unchanged. The input size remains `4 * 320 + 16 = 1296` bits.

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
- samples_per_class = 8192
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

Required ordering:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

Adjudication threshold:

```text
true-shuffled margins should be materially larger than the 4096/class near-tie:
seed0 +0.004315972
seed1 +0.000946522
```

Decision rules:

| Outcome | Decision |
| --- | --- |
| true loses to shuffled or metadata-only on either seed | stop active-relative topology-architecture route; no remote scale |
| true stays ordered but remains near-tied with shuffled | classify as control-clean but fragile; stop scale and redesign |
| true is ordered and margins reopen on both seeds | prepare one medium diagnostic plan; still not formal evidence |

## Result

Completed locally with 6 plan-aligned rows. Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_8192_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/results.jsonl \
  --expected-rows 6

status = pass
```

Plot/history artifacts:

```text
outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/curves.svg
outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/history.csv
```

Metrics:

| route | seed | input_bits | val_auc | val_accuracy | val_best_accuracy | train_auc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true | 0 | 1296 | 0.503246903 | 0.500000000 | 0.507812500 | 0.511724174 |
| shuffled | 0 | 1296 | 0.499265939 | 0.500000000 | 0.506225586 | 0.513318032 |
| metadata-only | 0 | 1296 | 0.493802458 | 0.500122070 | 0.503784180 | 0.523837022 |
| true | 1 | 1296 | 0.517796367 | 0.500000000 | 0.515502930 | 0.508807257 |
| shuffled | 1 | 1296 | 0.509485662 | 0.500000000 | 0.513061523 | 0.511076964 |
| metadata-only | 1 | 1296 | 0.497134238 | 0.499877930 | 0.505004883 | 0.513066925 |

Control deltas:

| seed | true - shuffled AUC | true - metadata-only AUC |
| ---: | ---: | ---: |
| 0 | +0.003980964 | +0.009444445 |
| 1 | +0.008310705 | +0.020662129 |

Comparison to previous active-relative contrast gates:

| samples_per_class | seed0 true AUC | seed1 true AUC | seed0 true-shuffled | seed1 true-shuffled |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.520668507 | 0.520288944 | +0.042667866 | +0.021866322 |
| 4096 | 0.502257228 | 0.515355825 | +0.004315972 | +0.000946522 |
| 8192 | 0.503246903 | 0.517796367 | +0.003980964 | +0.008310705 |

## Decision

E1 passes the required ordering but does not pass the margin-reopen gate.

The route remains control-clean:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

However, only seed1 clearly improves over the 4096/class near-tie. Seed0 stays
near-tied and is slightly below its 4096/class true-vs-shuffled margin:

```text
seed0: +0.004315972 -> +0.003980964
seed1: +0.000946522 -> +0.008310705
```

This mixed result means the active-relative contrast idea is not dead, but it
has not earned remote scale-up or a 16k/32k local ladder. Treat it as
diagnostic-only, control-clean, and still fragile. Do not launch a 65536/class
remote run from this evidence.

Route verdict impact:

```text
active-relative topology-architecture branch = implementation-blocked / verdict superseded
next Innovation 1 adjudication = E1-R1 cell-aligned pair4 2048/class
E2 trail-position neural residual = deferred until E1-R1 resolves
```

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_8192_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/progress.jsonl
```
