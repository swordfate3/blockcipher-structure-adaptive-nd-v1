# Innovation 1 Dialga-128 Runtime-E4 D5 Input-Difference Screen Plan

Date: 2026-07-25

## Status

```text
phase = preregistered cipher-statistical shortlist
source = completed D4 data-depth diagnosis
neural_training = prohibited during shortlist
remote_scale = prohibited
```

## Evidence And Research Question

D4 fixed each D1 checkpoint and crossed prefix-r4/r5 data with runtime windows
2/3 and 3/4. Changing only the window retained more than `93%` of the D1 AUC
excess over random, while changing only the data to prefix-r5 retained less than
`6.4%`. The first question is therefore not whether to enlarge Runtime-E4, but
whether the current single-bit difference `0x40` is a poor five-round data
source.

The Dialga paper reports that maximum-DCP differential distinguishers for the
16-round family can extend to five rounds. It does not publish a fixed-tweak,
fixed-key Dialga-128 input-difference list aligned with this neural protocol;
the appendix clustering examples concern Dialga-256. D5 is consequently a
bounded empirical shortlist, not a reproduction of the paper's SAT trails.

## Frozen Candidate Space

Screen every 128-bit Hamming-weight-one input difference:

```text
candidate(bit) = 1 << bit
bit = 0..127, counted from the integer least-significant bit
reference = 0x40 = bit 6
```

Using all 128 positions avoids choosing only locations that look favorable
under the known runtime topology. The exact Dialga cell id and bit role are
recorded for every candidate but do not affect ranking.

## Shortlist Protocol

For each candidate, evaluate two independently generated panels:

| Panel | Key | Purpose |
| --- | --- | --- |
| `train_key` | `0` | match D1/D3 training-key semantics |
| `validation_key` | `0x11` repeated over 32 bytes | match D1/D3 held-out-key semantics |

Each panel uses:

```text
cipher = Dialga-128 prefix-r5
tweak = 0
calibration rows/class = 512
evaluation rows/class = 512
pairs/row = 4
positive pair = (P, P xor candidate)
negative pair = two independently sampled plaintexts, both encrypted
feature = 128-bit ciphertext XOR for each pair
screen model = bit-marginal Bernoulli naive Bayes fitted on calibration rows
primary metric = evaluation AUC after summing four pair log-likelihood scores
```

The negative plaintexts and base positive plaintexts are fixed within each key
panel and shared across all candidates. Calibration and evaluation plaintexts
are disjoint. The two key panels use different deterministic RNG seeds. This is
a small local cipher-statistical diagnostic; it trains no neural weights and
does not use DDT, trail, partial-decryption or guessed-key features.

## Gate

Protocol validity requires:

1. Exactly 256 rows: 128 candidates by two key panels.
2. Every candidate is a unique Hamming-weight-one 128-bit difference and bit 6
   is the exact `0x40` anchor.
3. All candidates within a key panel share the same plaintext and strict
   encrypted-negative fingerprints.
4. Panel keys, row counts, pair counts, prefix rounds, metric definition,
   candidate-to-cell mapping and finite metrics match this plan.
5. No neural training occurs and all score/artifact hashes are present.

Candidates are ranked by worst-key AUC, then mean AUC, then bit index. A
candidate is eligible only if:

```text
train-key AUC >= 0.520
validation-key AUC >= 0.520
worst-key AUC >= anchor worst-key AUC + 0.010
```

At most the top two eligible candidates are reported. The neural stage initially
trains only the top candidate; the second candidate remains a reserve, keeping
the experiment matrix lean.

## Decision Routes

- Eligible candidate found: preregister a D5 neural matrix for the top candidate
  using the exact D3 `2048/class`, `1024/class`, four-pair, ten-epoch, two-seed
  Runtime-E4 correct/corrupted/no-topology protocol. Reuse D3 `0x40` as the
  same-budget anchor rather than retraining it.
- No eligible candidate: stop mechanical Dialga difference search and implement
  an independent representation plus residual/gated topology messages as the
  next single-variable model experiment.
- Protocol failure: repair only shortlist generation or source binding; keep
  candidate space, seeds, keys, rows, metric and thresholds unchanged.

## Explicitly Blocked

- No neural training during the shortlist.
- No multi-bit, DDT-ranked or hand-picked trail candidates in D5.
- No change to cipher rounds, tweak, negative definition, pair count or keys.
- No remote GPU, sample increase or formal/breakthrough claim.
- No simultaneous input-difference and Runtime-E4 architecture change.

## Execution

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/screen-runtime-spn-dialga-d5 \
  --run-id i1_dialga128_runtime_e4_d5_difference_screen_20260725 \
  --output-root outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725
```

After the shortlist completes, generate the gate and Chinese SVG, inspect the
rendered pixels with `visual-qa-redraw`, refresh the result index, and record the
eligible candidates and executable next action here before any neural run.

## Completed Result

```text
run_id = i1_dialga128_runtime_e4_d5_difference_screen_20260725
status = hold
decision = innovation1_dialga_runtime_e4_d5_no_difference_candidate
result_rows = 256
shortlist = []
neural_training = not launched by gate
```

All 11 protocol checks passed. The run evaluated every bit `0..127` under both
fixed keys, preserved the exact shared plaintext and strict encrypted-negative
panels within each key, recorded the real runtime cell mapping, and produced
finite hashed scores without neural training.

The `0x40` reference produced:

| Candidate | Train-key AUC | Validation-key AUC | Worst-key AUC | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| bit 6 / `0x40` | 0.474739 | 0.500477 | 0.474739 | 0.487608 |

The strongest conservative candidates were:

| Rank | Candidate | Cell / role | Train-key AUC | Validation-key AUC | Worst-key AUC |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | bit 76 | cell 12 / role 1 | 0.510098 | 0.516060 | 0.510098 |
| 2 | bit 4 | cell 31 / role 0 | 0.512466 | 0.508224 | 0.508224 |
| 3 | bit 19 | cell 27 / role 3 | 0.506390 | 0.520584 | 0.506390 |

Bit 76 exceeds the weak `0x40` worst-key anchor by more than `0.01`, but still
misses the absolute `0.520` gate on both keys. Bit 19 reaches `0.520584` only on
the validation key and remains near random on the training key. No candidate
therefore satisfies both preregistered requirements, and the conditional neural
matrix was correctly not launched.

The result does not prove that every possible differential data representation
is exhausted. It rejects the narrow mechanical route of replacing `0x40` with
another single-bit input difference under a bit-marginal four-pair screen. In
combination with D4, there is no evidence-backed reason to spend another local
training slot on a single-bit difference.

The final SVG was rendered at 1800 pixels and passed `visual-qa-redraw` after
the decision line was clarified to distinguish the absolute `0.52` gate from
the `0x40 + 0.01` relative gate. Titles, Chinese glyphs, legends, threshold
lines, 24 bar labels and candidate labels have no overlap, clipping, missing
content or structural ambiguity.

Artifacts:

```text
outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725/results.jsonl
outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725/progress.jsonl
outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725/validation.json
outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725/gate.json
outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725/summary.json
outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725/curves.svg
outputs/local_diagnostic/i1_dialga128_runtime_e4_d5_difference_screen_20260725/visual_qa_passed.marker
```

## Recommended Next Action

Implement D6 as a single architecture hypothesis: an independently useful
state/pair representation plus residual, learnably gated runtime-topology
messages. The topology path must be unable to erase or replace the independent
base representation. Keep Dialga prefix-r5, input difference `0x40`, four
pairs, `2048/class` training, `1024/class` validation, two seeds, ten epochs,
keys, loss, optimizer, disk cache and strict encrypted-random-plaintext
negatives identical to D3.

Compare only the new correct-topology candidate and its required corrupted and
no-topology controls; reuse completed D3 Runtime-E4 rows as the old same-budget
anchor. Advance only if each seed independently reaches:

```text
correct AUC >= 0.520
correct - corrupted >= +0.005
correct - no topology >= +0.005
correct - old Runtime-E4 correct >= +0.010
```

Before training, add a no-topology functional test proving that the independent
base produces nonconstant logits and a zero-gate equivalence test proving that
disabling topology leaves exactly that base prediction. D6 remains a local
diagnostic. Do not change the difference, data budget or negative definition,
do not train more than the three required roles, and do not launch remote GPU
scale.
