# Innovation 1 uKNIT r6 Public-Window Depth K1-BO Plan

**Date:** 2026-07-30
**Status:** completed / hold / three-transition signal not supported
**Execution:** local CPU; source-cache replay; zero neural updates

## 1. Research Question

K1-Q established a strong uKNIT r5 cell11 role1 deterministic signal. K1-U
then trained the corresponding r5 specialist at remote `65536/class`, reaching
cross-key AUC `0.974540495/0.967867357` with wrong-S-box controls at
`0.503900695/0.505827348`. At r6, K1-BL/K1-BM/K1-BN found no confirmed
candidate under the existing two-transition, five-stage statistic.

K1-BO tests one narrower explanation for that r5-to-r6 drop:

> Does the unchanged uKNIT r6 cell11 dataset retain a reproducible signal when
> the public exact-operator view is extended from the last two transitions to
> the last three transitions?

This is a representation-depth audit, not a new difference search and not
neural training. `Exact` means exact public S-box/linear-operator semantics; it
does not mean real partial decryption or recovery of keyed internal states.

## 2. Frozen Source Evidence

K1-BO reuses existing disk-backed caches and does not generate ciphertexts:

| Source | Use |
|---|---|
| K1-Q r5 cell11 confirmation, seeds 3/4 | two-transition positive anchor and replay check |
| K1-BL r6 cell11 confirmation, seeds 3/4 | exact same r6 rows for every depth/control view |
| K1-BN completed hold gate | binds the prior searched-family boundary |
| `configs/runtime/spn/uknit64.json` | public per-transition S-box and GF(2) operators |

For each round count and seed, the frozen splits are:

```text
train_seen           = 2048/class = 4096 total rows
same_key_fresh       = 1024/class = 2048 total rows
cross_key_validation = 1024/class = 2048 total rows
pairs/sample         = 4
negative definition  = encrypted random plaintexts
difference           = cell11 role1, 0x0000400000000000
seeds                 = 3, 4
```

The r5 and r6 ciphertexts necessarily differ because the encryption prefix
differs. Within r6, every view must bind to the identical feature/label cache
for each seed and split.

## 3. One Variable And Public Windows

The strict one-variable comparison is:

```text
r6 exact-2-position:
  round_start=4, rounds=2
  ciphertext + 2 x (inverse linear, inverse S-box) = 5 stages

r6 exact-3-position:
  round_start=3, rounds=3
  ciphertext + 3 x (inverse linear, inverse S-box) = 7 stages
```

Cipher, six-round encryption, ciphertext rows, labels, keys, difference,
pairs, seeds, splits, scorer, variance floor and metric remain fixed. The only
changed hypothesis is the number of public transition descriptors composed
from the ciphertext end.

The r5 exact-2 row uses `round_start=3, rounds=2` only as a pipeline/source
anchor. It is not part of the strict depth comparison because its ciphertexts
come from a five-round prefix.

## 4. Frozen Views And Controls

Fit one diagonal Fisher scorer per seed/view on `train_seen`, then evaluate the
same frozen scorer on all three splits.

| View | Shape before flattening | Purpose |
|---|---:|---|
| `r5_exact2_position` | `5 x 16 x 16` | strong source anchor and two-round replay |
| `r6_exact2_position` | `5 x 16 x 16` | existing r6 result replay |
| `r6_exact3_position` | `7 x 16 x 16` | primary depth candidate |
| `r6_wrong3_position` | `7 x 16 x 16` | equal-shape wrong-S-box semantic control |
| `r6_shuffle3_position` | `7 x 16 x 16` | exact features fitted to shuffled labels |
| `r6_exact3_invariant` | `7 x 16` | position-erasure candidate/control |
| `r6_wrong3_invariant` | `7 x 16` | matching invariant wrong-S-box control |
| `r6_shuffle3_invariant` | `7 x 16` | matching invariant shuffled-label control |
| `r6_raw` | `1 x 16 x 16` | raw ciphertext-XOR histogram anchor |

The wrong-S-box structures preserve cell membership and every linear matrix;
only deterministic S-box assignment semantics change. Label-shuffled rows
reuse byte-identical exact features and change only the fitting labels.

## 5. Readiness And Protocol Gates

Execution is valid only if all of the following hold:

1. source gate decisions and run ids match the frozen configuration;
2. exactly twelve source caches bind two rounds, two seeds and three splits;
3. every cache contains `metadata.json`, `features.npy` and `labels.npy`;
4. r5/r6 paired seed-split rows retain the same keys, counts, difference,
   pair count and strict negative protocol;
5. r6 views for a seed/split share one dataset SHA256;
6. the generalized two-transition implementation reproduces all K1-Q/K1-BL
   source AUC rows within `1e-9`;
7. exact and label-shuffled features are byte-identical;
8. exact and wrong-S-box features differ while retaining equal dimensions;
9. every histogram is finite, nonnegative and normalized;
10. training rows, neural parameters, optimizer steps and epochs all remain zero.

## 6. Advance And Stop Gates

Evaluate position-preserving and invariant routes independently on both seeds
and both fresh splits. A route passes only if every one of its four rows meets:

```text
r6 exact-3 AUC                         >= 0.550
r6 exact-3 - r6 exact-2 AUC           >= +0.010
r6 exact-3 - r6 raw AUC               >= +0.010
r6 exact-3 - matching wrong-Sbox AUC  >= +0.005
r6 exact-3 - matching label-shuffle   >= +0.010
```

- **At least one route passes:** authorize a separate local r6 neural matrix
  using the passed aggregation only. Keep K1-U data, four pairs, seeds 3/4,
  ten epochs, exact/wrong-S-box/raw-or-no-structure controls and all keys fixed.
- **Exact-3 improves but misses a semantic control:** record generic extra-view
  signal; do not attribute it to correct uKNIT structure and do not train.
- **Neither route passes:** reject insufficient public-window depth as the
  explanation under this frozen cell11 protocol. Retain the K1-BN observed
  r5-to-r6 boundary and do not scale r6.
- **Any protocol/replay check fails:** mark the run invalid and repair only the
  failed binding, geometry or generalized-composition implementation.

No post-result threshold, difference, pair, seed, sample, key, model-width or
candidate change is allowed inside K1-BO.

## 7. Required Artifacts

```text
preflight.json
source_cache_manifest.jsonl
feature_manifest.jsonl
scorer_manifest.jsonl
results.jsonl
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
`outputs/00_RECENT_RESULTS.json`, append the result and evidence-backed next
action here, then commit and push only K1-BO-scoped files.

## 8. Completed Result

K1-BO reused all twelve frozen K1-Q/K1-BL caches and completed without data
generation or neural training:

```text
source cache rows                 = 12
feature/result rows               = 54 / 54
scorer rows                       = 18
training rows / optimizer steps   = 0 / 0
failed protocol checks            = []
two-transition source AUC deltas  = 0.0 for all 12 replay rows
```

Every three-transition view's first five stages matched the corresponding
two-transition view exactly. Therefore the generalized implementation changed
only the added oldest transition and did not perturb the established r5/r6
two-transition evidence.

The position-preserving route produced:

| Seed / fresh split | exact-2 | exact-3 | wrong-Sbox-3 | shuffled | raw |
|---|---:|---:|---:|---:|---:|
| 3 / same-key | `0.513282` | `0.505440` | `0.492264` | `0.481627` | `0.506370` |
| 3 / cross-key | `0.520939` | `0.528496` | `0.524715` | `0.476456` | `0.508342` |
| 4 / same-key | `0.477784` | `0.485719` | `0.475278` | `0.488698` | `0.496267` |
| 4 / cross-key | `0.507025` | `0.510899` | `0.504083` | `0.492809` | `0.485698` |

Its best AUC was only `0.528496`; none of the four fresh rows reached `0.550`
or improved over exact-2 by the required `+0.010`. The invariant route was
weaker:

| Seed / fresh split | exact-3 invariant | wrong-Sbox-3 | shuffled | raw |
|---|---:|---:|---:|---:|
| 3 / same-key | `0.503830` | `0.498762` | `0.503938` | `0.506370` |
| 3 / cross-key | `0.502844` | `0.499009` | `0.498846` | `0.508342` |
| 4 / same-key | `0.447320` | `0.480348` | `0.509159` | `0.496267` |
| 4 / cross-key | `0.493879` | `0.506927` | `0.483951` | `0.485698` |

Final decision:

```text
status                     = hold
decision                   = innovation1_uknit_r6_k1bo_three_round_window_signal_not_supported
passed_routes              = []
neural_training_authorized = false
remote_scale               = no
```

The Chinese SVG was rendered to `2040 x 1320` pixels and passed
`visual-qa-redraw`. The title, subtitle, four panel labels, grouped bars,
numeric annotations, legend, `0.50/0.55` references and bottom decision line
have no overlap, clipping, missing glyphs or ambiguous association.

## 9. Evidence-Backed Next Action

Reject insufficient public-window depth as the explanation for the current r6
boundary. Do not add a fourth window transition, train the r6 histogram model,
increase samples/pairs, or launch remote r6 scale on this failed data surface.

To continue a six-round cryptanalytic objective, preregister a separate K1-BP
**last-round key-hypothesis feasibility audit** instead of another direct-r6
classifier. Reuse the strong frozen r5 K1-U checkpoint and ask whether the
public inverse linear layer plus a bounded guess of only the dependency-cone
last-round subkey bits can map r6 ciphertext pairs onto the r5 specialist's
input protocol. The readiness audit must first compute the exact required key
bit count and prove correct-key versus wrong-key score separation on untouched
cross-key data. It must not feed the true key to the network, call a full-key
oracle an attack, or hide the number of guesses. If the dependency cone
requires an infeasible full-round-key guess or correct-key scores do not beat
wrong guesses, stop this extension route; if a bounded key subset separates,
then implement one explicit six-round key-ranking experiment with query and
guess complexity reported separately from raw neural AUC.
