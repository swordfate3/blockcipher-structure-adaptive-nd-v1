# Innovation 1 uKNIT r6 Last-Round Key-Hypothesis K1-BP Plan

**Date:** 2026-07-30
**Status:** completed / hold; weak discovery-only signal, not confirmed
**Execution:** local CPU; source-cache and frozen-checkpoint replay; zero neural updates

## 1. Research Question

K1-BO found no supported r6 signal from extending the public inverse-operator
window from two to three transitions. Its strongest three-transition row was
only `AUC=0.528496`, so direct r6 neural training and remote scale remain
unauthorized.

K1-BP changes the cryptanalytic protocol rather than adding another public
view:

> Can a bounded hypothesis for the last added round key map uKNIT r6
> ciphertext pairs onto a useful part of the strong frozen r5 protocol, and
> does the correct hypothesis rank first among every wrong bounded guess?

This is a last-round key-hypothesis feasibility audit. It is not new neural
training, formal scale, full-key recovery or a six-round attack claim.

## 2. Why A Key Hypothesis Is Required

For the added sixth transition,

```text
C6 = L5(S5(C5 XOR K5))
Z  = S5^-1(L5^-1(C6)) = C5 XOR K5
```

The same-key pair difference `Delta Z` equals `Delta C5`, which explains why
the first public difference stages are key-free. However, the frozen r5 model
also consumes nonlinear views of the two individual values. Recovering its
complete 64-bit input requires all 64 bits of `K5`, hence `2^64` candidates.
The true full key is therefore only an oracle upper bound.

For one target cell after `L4^-1`, every one of its four bits depends on three
`C5` bits. Under the published uKNIT transition-4 matrix, their union is
exactly twelve `C5` bits for every target cell. Those twelve source-key bits
enter the target cell through a rank-four GF(2) map, so they induce only four
independent effective subkey bits:

```text
12 source K5 bits -> 4 effective linear combinations -> 16 hypotheses
kernel size = 2^(12 - 4) = 256 equivalent source-key assignments per hypothesis
```

K1-BP tests whether that one-cell nonlinear statistic is sufficient for key
ranking.

## 3. Frozen Evidence And Data

The experiment reuses, without regeneration:

| Source | Use |
|---|---|
| K1-Q r5 cell11 caches | seed2 target-cell discovery; seed3/4 sparse r5 anchors |
| K1-BL r6 cell11 caches | unchanged r6 rows for key hypotheses |
| K1-BO completed hold gate | binds the failed public-window route and paired r5/r6 keys |
| K1-U invariant checkpoints | strongest frozen r5 full-model oracle |
| `configs/runtime/spn/uknit64.json` | exact public S boxes and GF(2) matrices |

Frozen protocol:

```text
difference           = cell11 role1 = 0x0000400000000000
pairs/sample         = 4
negative definition  = encrypted random plaintexts
train_seen           = 2048/class = 4096 total rows
same_key_fresh       = 1024/class = 2048 total rows
cross_key_validation = 1024/class = 2048 total rows
discovery seed       = 2 only
confirmation seeds   = 3, 4 untouched by target-cell selection
```

K1-U itself was trained remotely at `65536/class`, with `32768/class`
cross-key validation, four pairs and ten epochs. Its invariant r5 validation
AUC was `0.977200513` for seed3 and `0.974682369` for seed4.

## 4. Phase A: Exact Dependency Audit

The audit must prove and record:

1. Correct full `K5` stripping reproduces five-round encryption exactly on
   independently generated plaintext fixtures.
2. The complete K1-U adapter structurally reads all 64 recovered `C5` bits.
3. Each one-cell `L4^-1 -> S4^-1` readout has exactly twelve distinct source
   bits, GF(2) rank four, sixteen effective hypotheses and a 256-member source
   key equivalence class per hypothesis.
4. No true key bit is appended to a feature or passed to a model. It is used
   only to identify the correct hypothesis after all candidate scores exist.

## 5. Phase B: Discovery-Only Sparse Readout Selection

For each of the sixteen target cells, fit one 16-bin diagonal Fisher scorer on
the seed2 r5 `train_seen` cache. The feature is only the four-pair histogram of
that cell's difference after `L4^-1 -> S4^-1`.

Rank target cells by the minimum of seed2 `same_key_fresh` and
`cross_key_validation` AUC. Break ties by the smaller cell index. Freeze the
winner before reading any seed3/4 key ranks.

Discovery readiness requires the selected cell's minimum fresh AUC to be at
least `0.55`. If it fails, no bounded key enumeration is authorized because
the restricted r5 statistic itself is not a valid anchor.

## 6. Phase C: Full-Model Oracle Upper Bound

For each seed3/4 fresh split:

1. derive the actual sixth-round `K5` from the source master key;
2. strip the sixth round with that full key;
3. evaluate the matching frozen K1-U invariant checkpoint;
4. repeat with zero key, bit-shuffled key and eight deterministic
   same-Hamming-weight wrong keys.

The correct full-key oracle must reach `AUC >= 0.90` and beat the strongest
wrong full-key control by at least `+0.01` on all four panels. Even if it
passes, `2^64` remains infeasible and the row must never be reported as a
six-round attack.

## 7. Phase D: Exhaustive 4-Bit Effective-Key Ranking

For the frozen target cell and each confirmation seed, fit the same 16-bin
Fisher scorer on r5 `train_seen`. Then enumerate all `16` effective cell-key
hypotheses on both r6 fresh splits. The output is an effective subkey after
`L4^-1`, not twelve individually recovered `K5` source bits.

Candidate ranking uses only rows with the known chosen input difference
(`label=1`): rank by mean frozen r5 score over those queries. Labels are used
afterward only for diagnostic AUC. With 1024 positive rows and four pairs per
row, each panel uses 4096 chosen plaintext pairs.

Required controls:

- all 16 exact-S-box effective hypotheses, including zero;
- matching wrong-S-box lookup with the same cone and candidate count;
- a scorer fitted to deterministically shuffled r5 labels;
- untouched same-key and cross-key panels for seeds 3 and 4;
- exact guessed-bit count, candidate count, query count and true-key rank.

The bounded route passes only if every confirmation panel satisfies:

```text
r5 sparse fresh AUC       >= 0.55
r6 correct-key sparse AUC >= 0.55
correct effective-key rank = 1 of 16
exact rank better than wrong-Sbox and shuffled-scorer true-key ranks
```

Reporting uses a second, non-advancing evidence threshold requested after the
run:

```text
AUC < 0.51         = no supported positive signal
0.51 <= AUC < 0.55 = weak signal; continue local confirmation only
AUC >= 0.55        = strong candidate floor
```

This reporting tier does not retroactively alter the frozen `0.55` pass gate
and does not override the key-rank or wrong-S-box/label-shuffle controls.

## 8. Decisions

- **Oracle and bounded route pass:** authorize K1-BQ, a separate larger-query,
  multi-key six-round key-ranking confirmation. Keep the 4-bit effective
  hypothesis, target cell, difference, four pairs and scorer fixed; report
  query and guess complexity separately from AUC.
- **Oracle passes, sparse r5 anchor passes, but key rank fails:** do not scale.
  Train at most one small sparse r5 specialist against the same 4-bit-effective
  feature and retry key ranking locally; do not return to the full 64-bit model.
- **Oracle passes but sparse r5 anchor fails:** keep the full-key oracle only as
  a mechanism result, then preregister a multi-cell sparse audit. Two target
  cells expose at most 8 effective bits (`256` candidates), and three expose at
  most 12 (`4096` candidates); measure matrix-rank overlap before enumeration.
- **Oracle fails:** mark K1-BP invalid and repair the inverse-round mapping,
  checkpoint binding or bit order. Do not interpret it as cryptanalytic
  evidence.

Blocked actions include remote r6 training, more public-window depth, more
pairs/samples, post-result target-cell changes, hiding `2^b` complexity, and
calling a full-key oracle a six-round attack.

## 9. Required Artifacts

```text
preflight.json
source_cache_manifest.jsonl
dependency_cones.json
discovery_results.jsonl
selection.json
full_oracle_results.jsonl
sparse_rank_results.jsonl
comparison.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
plot_report.json
curves.rendered.png
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, run focused tests and result validation, perform
`visual-qa-redraw`, refresh `outputs/00_RECENT_RESULTS.md` and
`outputs/00_RECENT_RESULTS.json`, append the observed result and recommended
next action here, then commit and push only K1-BP-scoped files.

## 10. Observed Result

The complete correct 64-bit `K5` oracle restored the frozen r5 signal on every
confirmation panel:

| Seed / split | Correct full-key AUC | Best wrong full-key AUC |
|---|---:|---:|
| seed3 / same-key | `0.978914` | `0.513849` |
| seed3 / cross-key | `0.980791` | `0.513922` |
| seed4 / same-key | `0.974061` | `0.518814` |
| seed4 / cross-key | `0.973081` | `0.517076` |

This is mechanism evidence only because exhaustive full-key enumeration costs
`2^64` candidates.

The seed2 discovery winner was cell 0 with minimum fresh AUC `0.514677`. Under
the reporting taxonomy above, this is a **weak discovery-only signal**. It did
not reproduce on frozen seed3/4 confirmation: r6 sparse correct-key AUC ranged
from `0.484064` to `0.501928`, while true-key ranks were `3/16`, `3/16`, `1/16`
and `12/16`. Therefore:

```text
status                  = hold
evidence_tier           = weak_discovery_only_unconfirmed
weak_signal_observed    = true
weak_signal_confirmed   = false
bounded_route_pass      = false
neural_training         = not authorized
remote_scale            = no
```

## 11. Recommended Next Action

Run K1-BQ as a local, zero-training two- versus three-cell effective-key audit.
Use seed2 only to select cell subsets; freeze the subsets before reading seed3/4
same-key and cross-key panels. Compute joint GF(2) rank before enumeration and
cap the search at 12 effective bits (`4096` candidates). Report `0.51` as weak
and `0.55` as strong, but require frozen-seed replication plus better rank than
wrong-S-box and label-shuffled controls before authorizing any neural training.
Do not scale queries, pairs, samples or remote compute on the single-cell route.
