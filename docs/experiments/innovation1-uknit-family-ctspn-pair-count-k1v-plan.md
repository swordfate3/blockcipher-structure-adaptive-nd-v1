# Innovation 1 uKNIT-Family CT-SPN Pair Count K1-V

**Date:** 2026-07-28
**Status:** completed / pass / 16-pair added value supported
**Execution:** local CPU; sub-medium diagnostic, not formal training

## Research question

K1-T used four independent ciphertext pairs per sample and obtained cross-key
AUC `0.713162` (seed3) and `0.748229` (seed4). K1-U later showed at
`65536/class` that correct S-box semantics remained necessary, but native-cell
position preservation did not. The user requested a bounded pair-count test:

> With the K1-T network and protocol held fixed, does increasing only the
> number of ciphertext pairs per sample from 4 to 16 provide reproducible
> cross-key value?

This is a local `2048/class` pair-count diagnostic. It is not formal training,
an attack, a SOTA comparison, a family-transfer result, or a ceiling claim.

## Single variable

```text
K1-T anchor: 4 pairs/sample  ->  512 input bits/sample
K1-V test:  16 pairs/sample -> 2048 input bits/sample
```

The model factory already treats pair count as input geometry, so the same
trainable model names and parameter shapes are reused. No architecture option,
difference, key, loss, optimizer, epoch count, sample count or label protocol
may change.

## Frozen matrix

Train six independent rows:

```text
seed3/4 x {
  exact_position_histogram_residual,
  wrong_sbox_position_histogram_residual,
  invariant_histogram_residual
}
```

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Difference | cell11 role1, `0x0000400000000000` |
| Train | `2048/class`, `4096` total rows |
| Cross-key validation | `1024/class`, `2048` total rows |
| Seeds | `3`, `4` |
| Pairs/sample | `16` |
| Input width | `2048` bits/sample |
| Negative definition | encrypted random plaintexts |
| Sample structure | independent pairs |
| Runtime window | `round_start=3`, `rounds=2` |
| Hidden / pair embedding | `32` / `128` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC, restored |
| Device | local CPU |

The same seed uses identical generated training and validation data across all
three controls. Disk-backed cache/progress/reuse is required even though this
run is below the remote-only scale.

## Readiness gate

Before optimization require:

1. exactly six plan rows and the frozen protocol above;
2. each model accepts `[B, 2048]` and interprets it as sixteen 128-bit pairs;
3. all three controls have identical trainable parameter names, shapes and
   count `214316`;
4. exact versus wrong-S-box and exact versus invariant logits differ under a
   shared state dictionary;
5. outputs and one MSE backward pass are finite;
6. the K1-T gate is a valid two-seed completed local anchor.

Any failure prohibits training. Repair only the failed invariant and rerun the
same readiness gate.

## Frozen result gate

For each seed separately require semantic attribution:

```text
16-pair exact - 16-pair wrong-S-box >= +0.010
```

Then require at least one source of added value on that same seed:

```text
16-pair exact - 4-pair exact        >= +0.010
or
16-pair exact - 16-pair invariant   >= +0.010
```

Both seeds must pass both clauses to retain 16 pairs. Averages cannot hide a
failed seed.

## Decisions and next action

- **Both seeds pass:** retain 16 pairs as a promising query-budget setting, but
  do not mechanically scale it. First complete K1-W as a separate compact
  invariant-network experiment, then confirm pair count within the selected
  compact architecture.
- **Correct semantics pass but added value fails:** keep four pairs and proceed
  to K1-W compact invariant; more pairs merely add query/input cost here.
- **Correct semantics fail:** reject this 16-pair route and audit whether pair
  aggregation diluted the semantic residual before changing capacity.
- **Protocol invalid:** repair only the failed plan, cache, shape, checkpoint or
  artifact binding and rerun unchanged.

Blocked inside K1-V: more samples, pairs beyond 16, positions, epochs, seeds or
keys; a new model family; MoE; DDT/trails; another cipher; remote launch; and
post-result threshold changes.

## Run and artifacts

```text
run_id = i1_uknit_family_ctspn_pair_count_k1v_16pair_2048_seed3_seed4_20260728
```

Required artifacts: preflight/readiness, disk caches, progress JSONL, six
checkpoints, six result rows, comparison CSV, gate, validation, summary,
history CSV, Chinese SVG, plot report and `visual_qa_passed.marker`. After
completion refresh both recent-result indexes and update this document with the
observed metrics, decision and executable next action.

## Completed result

The frozen six-row run completed on 2026-07-28. The valid artifact root is:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_pair_count_k1v_16pair_2048_seed3_seed4_20260728_clean/
```

The `_clean` suffix identifies the uninterrupted evidence root. Earlier
interrupted output is not valid evidence and is excluded from the result index.

| Seed | 16-pair exact AUC | 4-pair exact anchor | 16-pair wrong-S-box | 16-pair invariant |
|---:|---:|---:|---:|---:|
| 3 | `0.902422905` | `0.713162422` | `0.506145477` | `0.591490269` |
| 4 | `0.932538986` | `0.748229027` | `0.499612808` | `0.697590828` |

The preregistered per-seed margins were:

| Seed | Exact16 - exact4 | Exact16 - wrong-Sbox16 | Exact16 - invariant16 |
|---:|---:|---:|---:|
| 3 | `+0.189260483` | `+0.396277428` | `+0.310932636` |
| 4 | `+0.184309959` | `+0.432926178` | `+0.234948158` |

All protocol checks and all four research checks passed. In particular, six
training rows completed, each sample had `2048` input bits interpreted as
sixteen independent ciphertext pairs, all controls retained the same `214316`
trainable parameters, best checkpoints were restored, and the disk cache
recorded four creations plus eight parameter-matched reuses.

```text
gate status   = pass
decision      = innovation1_uknit_family_ctspn_k1v_16pair_added_value_supported
remote_scale  = no
```

The result supports a narrow claim: at fixed `2048/class` uKNIT r5 budget,
increasing only the query count from four to sixteen pairs gave reproducible
cross-key value to the exact K1-T structure branch. The wrong-Sbox controls
remaining near chance show that the gain is not explained by input width
alone. The lower invariant controls also show that the exact branch retained
more useful signal at this local scale.

This remains a sub-medium local diagnostic. It is not formal training, a
remote-scale result, an attack, SOTA evidence, family transfer, or evidence
that sixteen pairs are optimal.

## Executable next action

Run K1-W as a separately frozen compact-invariant experiment before revisiting
pair count. Its research question is whether removing the redundant native-cell
projection preserves the existing uKNIT and Dialga signals while reducing the
model from `214316` to `137516` trainable parameters.

```text
same-budget anchors = K1-T uKNIT invariant + K1-N Dialga exact
one variable        = original histogram projection -> compact invariant projection
ciphers / rounds    = uKNIT-BC r5 and Dialga-128 r4
train / validation  = 2048/class / 1024/class
pairs               = 4
seeds                = uKNIT 3,4; Dialga 0,1
models               = compact exact, compact wrong-Sbox
epochs / batch       = 10 / 64
execution            = local CPU diagnostic
```

Advance only if every uKNIT seed retains its frozen invariant anchor within the
declared tolerance and beats compact wrong-Sbox by `+0.010`, while both Dialga
seeds retain the K1-N anchor within `0.005`. Hold on any failed seed. Only after
K1-W selects the compact architecture should a separate experiment compare
four versus sixteen pairs inside that same compact architecture. Do not combine
the architecture change with pair-count change, and do not mechanically launch
a remote sixteen-pair run from K1-V alone.
