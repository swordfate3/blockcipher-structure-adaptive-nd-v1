# Innovation 1 PRESENT r8 Aligned Trail-Value Mismatch Gate

## Status

status = completed local diagnostic gate
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_value_mismatch_512_seed0_seed1.csv

## Question

The aligned trail-position attribution gate showed that the r8 random16 signal
is carried by trail-derived values. Prefix-only drops to about 0.60 AUC, while
trail-only stays about 0.974 AUC. Reversing or permuting trail-word order also
stays high.

This gate asks the next and stricter question:

Do the trail-derived values need to correspond to the current ciphertext pair,
or are any strong-looking trail values enough?

## Baseline Context

Use these completed 512/class seed0+seed1 rows as anchors:

| route | seed0 AUC | seed1 AUC |
| --- | ---: | ---: |
| full | 0.971763611 | 0.972396851 |
| prefix_only | 0.600173950 | 0.601318359 |
| trail_only | 0.973968506 | 0.973541260 |
| per_sample_key_full | 0.970565796 | 0.980911255 |

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active
- integral_active_nibbles = `{0..15}`
- model = present_trail_position_stats_pairset
- seeds = 0, 1

Routes per seed:

| route | feature encoding | purpose |
| --- | --- | --- |
| maskedsource | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_maskedsource_cell_matrix_bits` | keep real prefix but compute beamstats trail from `sinv_delta xor 0x111...` |
| constantsource | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_constantsource_cell_matrix_bits` | keep real prefix but compute beamstats trail from fixed source `0x9` |

Both encodings preserve the full 2496-bit per-pair shape:

```text
[delta, p_aligned_delta, sinv_delta, 36 trail-stat words]
```

Only the trail-stat source is changed.

## Gate

- If both mismatch controls drop close to prefix-only, then sample-specific
  trail-derived values are necessary. This supports moving to a small remote
  ladder for the aligned trail route.
- If either mismatch control remains near full/trail-only, then the current
  trail feature construction may carry a protocol-level or value-distribution
  shortcut. Do not remote-scale; redesign the trail representation locally.
- If controls land in the middle, treat the route as mixed: useful SPN
  signal plus some trail-value distribution bias.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_value_mismatch_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_trail_value_mismatch_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_trail_value_mismatch_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_trail_value_mismatch_512_seed0_seed1/progress.jsonl
```

## Decision Rule

This is the intended final local gate before deciding whether remote 65k/class
is scientifically worth spending on the aligned trail route. Do not add another
local control by default unless this gate is ambiguous or exposes a concrete
failure mode.

## Results

Artifacts:

- Results JSONL: `outputs/local_smoke/i1_present_r8_aligned_trail_value_mismatch_512_seed0_seed1/results.jsonl`
- Plot: `outputs/local_smoke/i1_present_r8_aligned_trail_value_mismatch_512_seed0_seed1/curves.svg`
- History CSV: `outputs/local_smoke/i1_present_r8_aligned_trail_value_mismatch_512_seed0_seed1/history.csv`

Validation:

- `validate-results --expected-rows 4` passed.

Metrics:

| route | seed0 AUC | seed1 AUC | best interpretation |
| --- | ---: | ---: | --- |
| full anchor | 0.971763611 | 0.972396851 | prior aligned random16 full route |
| trail_only anchor | 0.973968506 | 0.973541260 | prior trail-only route |
| prefix_only anchor | 0.600173950 | 0.601318359 | prior prefix-only route |
| maskedsource | 0.981689453 | 0.988342285 | wrong-source trail remains as high as full |
| constantsource | 0.902770996 | 0.938705444 | fixed-source trail still much higher than prefix-only |

## Interpretation

This gate blocks remote scale for the current aligned trail-value route.

The intended pass pattern was:

```text
full/trail_only high
mismatched trail low
constant trail low
```

The observed pattern is:

```text
full/trail_only high
maskedsource also high
constantsource still high-ish
```

This means the current high AUC is not strict evidence that the trail values
must be correctly matched to the current ciphertext pair. A masked wrong-source
trail gives AUC 0.98+, and a fixed-source trail still gives AUC 0.90-0.94.

The important nuance is that `constantsource` is not identical to the earlier
`prefix_only` control. `prefix_only` zeroed trail activity inside the model.
`constantsource` keeps a full nonzero trail-stat block with fixed values. The
large gap between prefix-only and constantsource suggests that the model can
use the presence, scale, and distribution of the full trail-stat block together
with the real prefix, even when the trail values are not sample-specific.

Therefore the best current conclusion is:

The aligned trail representation is a very strong diagnostic feature, but the
current beamstats8deep4 trail-value construction is too permissive for a remote
scale claim. It may contain a representation-level shortcut or normalization
effect that makes prefix differences much more separable once any full trail
block is present.

## Gate Outcome

Status: hold remote scale; redesign locally.

Do not launch 65k/class or 262k/class for the current full/trail-only route as
a formal readiness run. The next local direction should reduce this shortcut by
changing the representation, not by adding more scale:

- normalize trail-stat blocks so fixed nonzero trail sources cannot act as a
  strong scaffold for prefix separation;
- separate prefix and trail branches with explicit same-budget controls;
- test weaker or more local trail features, for example lower beam width,
  shallower depth, or per-depth normalized summaries;
- consider a graph/token SPN model where trail information is auxiliary and
  gated, rather than a large dense hand-crafted trail-stat block.
