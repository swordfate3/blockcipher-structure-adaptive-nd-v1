# Innovation 1 uKNIT-Family CT-SPN Virtual-Slot Pair Count K1-AB

**Date:** 2026-07-29
**Status:** completed / pass / sixteen-pair added value retained
**Execution:** local CPU; sub-medium pair-count diagnostic, not formal training

## Research question

K1-AA selected the fixed sixteen-virtual-slot compact architecture at four
pairs/sample. It reached uKNIT r5 AUC `0.570870/0.590954` and retained correct
S-box margins `+0.067605/+0.079269` on seed3/4. Earlier K1-V showed that sixteen
pairs strongly help the old K1-T exact branch and also raised its invariant
branch to `0.591490/0.697591`.

K1-AB asks:

> With K1-AA and every data/training field held fixed, does changing only four
> to sixteen independent ciphertext pairs provide reproducible added value?

## Single variable

```text
K1-AA anchor:  4 pairs/sample ->  512 input bits/sample
K1-AB test:   16 pairs/sample -> 2048 input bits/sample
```

The model names, parameter shapes, virtual slots, runtime descriptor, S-box
controls, difference, keys, sample rows, optimizer, epochs and metric remain
unchanged. Parameter count remains `214316`.

## Frozen matrix

Train exactly four rows:

```text
uKNIT-BC r5 seed3,4 x {K1-AA exact, K1-AA wrong-Sbox}
```

| Field | Frozen value |
|---|---|
| Train / validation | `2048/class` / `1024/class` |
| Pairs/sample | `16` |
| Input width | `2048` bits |
| Difference | cell11 role1, `0x0000400000000000` |
| Negative definition | encrypted random plaintexts |
| Sample structure | independent pairs |
| Train / validation keys | same seed3/4 cross-key protocol |
| Runtime window | rounds 3/4 |
| Virtual projection slots | `16` fixed optimizer slots |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| LR / weight decay | `1e-4` / `1e-5` |
| Checkpoint | restored best validation AUC |
| Device | local CPU |

Reuse the four completed K1-V sixteen-pair disk caches through a read-only
directory link. Dataset generation is forbidden.

## Readiness gate

Before optimization require:

1. exact digests and valid completed-pass gates for K1-AA and K1-V;
2. exactly four frozen K1-AB tasks;
3. all four K1-V source-cache payload digests match;
4. each model accepts `[B,2048]` and decodes exactly sixteen 128-bit pairs;
5. exact and wrong-S-box models share one state geometry and exactly `214316`
   trainable parameters;
6. the virtual tensor remains `[16,128,40]`, no LR override exists, and Adam
   has one group at `1e-4`;
7. shared-state exact and wrong-S-box logits are finite and observable;
8. no cipher ID, absolute runtime position parameter or protocol change exists.

Any failure blocks optimizer steps. Repair only the plan, source binding, cache
link or existing K1-AA model binding and rerun unchanged.

## Frozen result gate

For each seed independently require:

```text
K1-AB exact16 - K1-AA exact4              >= +0.010
K1-AB exact16 - K1-AB wrong-Sbox16        >= +0.010
K1-AB exact16 >= K1-V invariant16 - 0.020
```

The K1-V invariant anchors are `0.591490269/0.697590828`. No seed averaging may
hide a failed gate.

## Decisions

- **All six checks pass:** retain sixteen pairs as the selected local query
  budget for K1-AA. Next perform a Dialga r4 retention check at sixteen pairs
  or prepare a separately gated uKNIT medium-scale run; do not combine them.
- **Semantic attribution fails:** reject sixteen pairs for K1-AA and audit pair
  aggregation before changing capacity.
- **Added value or K1-V retention fails:** retain four pairs and treat K1-V's
  larger gain as dependent on its native-position branch or training variance.
- **Protocol invalid:** repair only the failed binding and rerun unchanged.

Blocked: architecture changes, pairs beyond sixteen, more samples, epochs,
seeds, differences, Dialga in this matrix, remote launch and post-result gate
changes.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_virtual_slot_pair_count_k1ab_16pair_2048_seed3_seed4_20260729
```

Produce preflight, source-cache manifest, progress JSONL, four checkpoints,
four result rows, validation, gate, summary, comparison/history CSV and Chinese
SVG. Apply `visual-qa-redraw`, refresh both recent-result indexes and append the
evidence-backed next action here.

## Completed readiness

Every readiness check passed before optimization:

```text
K1-AA and K1-V source gates/digests = exact
K1-V 16-pair cache payloads         = 4/4 exact
input width / decoded pairs         = 2048 bits / 16 pairs
trainable parameters                = 214316
virtual projection tensor           = [16,128,40]
optimizer groups                    = one ordinary Adam group at 1e-4
shared-state S-box controls          = finite and observable
```

No dataset generation occurred. The four existing K1-V caches were linked
read-only and reused exactly eight times.

## Completed result

All four rows completed ten epochs and restored the best validation-AUC
checkpoint:

| Seed | K1-AB correct16 | K1-AB wrong16 | K1-AA correct4 | K1-V invariant16 | 16 - 4 | Correct - wrong | K1-AB - K1-V invariant |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | `0.613891125` | `0.514521122` | `0.570869923` | `0.591490269` | `+0.043021202` | `+0.099370003` | `+0.022400856` |
| 4 | `0.688437462` | `0.497613430` | `0.590953827` | `0.697590828` | `+0.097483635` | `+0.190824032` | `-0.009153366` |

Both seeds passed the `+0.010` pair-added-value gate, the `+0.010`
correct-S-box attribution gate and the K1-V invariant retention tolerance. The
frozen decision is:

```text
status       = pass
decision     = innovation1_uknit_family_ctspn_k1ab_16pair_supported
remote_scale = no
```

This supports sixteen pairs as the selected local query budget for K1-AA under
the frozen uKNIT r5 protocol. It remains a `2048/class` local diagnostic, not
formal scale, an attack, SOTA, cross-cipher transfer or proof that sixteen
pairs are globally optimal.

The valid output root is:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_virtual_slot_pair_count_k1ab_16pair_2048_seed3_seed4_20260729/
```

`validate-results` passed `4/4`. The Chinese SVG was rendered to
`1944x1056` pixels and passed `visual-qa-redraw` with no overlap, clipping,
missing glyph, ambiguous title or unreadable close value.

## Evidence-backed next action

Retain the K1-AA virtual-slot architecture and sixteen-pair uKNIT setting.
Before any remote uKNIT scale, run a separate Dialga-128 r4 retention check
using the same K1-AA parameterization and Dialga's already calibrated
difference. Change only Dialga's pair count from its existing four-pair K1-W
anchor to sixteen pairs, retain exact and wrong-S-box controls, and use the
existing local `2048/class`, two-seed, ten-epoch protocol.

This next gate asks whether sixteen-pair value and semantic attribution survive
on another heterogeneous SPN. Do not combine Dialga with uKNIT remote scale,
new differences, more data, MoE or architecture changes. If both Dialga seeds
retain their calibrated anchor and correct-operator attribution, prepare a
separate remote uKNIT `65536/class` readiness plan with disk-backed cache; if
not, keep pair count cipher-specific and audit the Dialga aggregation path.
