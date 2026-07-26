# Innovation 1 Runtime-SPN S2 Cell-Local S-box ANF Operator Plan

Date: 2026-07-26

```text
status = completed / hold
execution = local CPU sub-medium mechanism gate
remote_scale = no
```

## Research Question

Can a runtime-supplied 4-bit S-box become semantically identifiable when it is
executed as an exact, sample-conditioned Boolean operator inside cell message
passing, rather than encoded as an unconstrained 64-bit descriptor gate?

S1 proved that the A8 truth-table MLP is responsive but not identifiable. All
ten seed/cipher pairs changed numerically under valid counterfactual S-boxes,
yet correct S-boxes did not consistently dominate them. On unseen Dialga, a
RECTANGLE S-box exceeded the correct S-box by `0.062308/0.021152` AUC. S2
therefore disables that learned S-box gate in the candidate and tests one exact
Boolean-operator replacement.

## Why This Does Not Repeat U2-D Through U2-H

The completed uKNIT chain already closed these representations:

- U2-D replaced a previous-state token with four-bit inverse-S-box endpoint
  values. Correct ownership beat shuffled on both seeds but did not beat the
  state-triplet anchor.
- U2-E averaged state and inverse-S-box triplets and regressed on both seeds.
- U2-F exposed only `deltaU` as a third query token and passed on prefix-r4.
- U2-G confirmed that narrow checkpoint mechanism.
- U2-H changed only to prefix-r5 and failed every two-seed cross-window gate.
- U3 recurrently consumed the heterogeneous r5 window but passed only seed1.

S2 does not replace an input token, average two views, expose `deltaU`, or add a
final-transition query. It preserves the Runtime-E4 state-triplet input and
constructs the complete per-output-bit ANF contribution pattern of the exact
runtime inverse S-box for both ciphertext endpoints. Those contribution terms
form a sample-conditioned local gate on the GF(2) graph message at every loaded
round. The model parameter shape is independent of the supplied S-box tables.

The existing PRESENT monomial-support model is not a duplicate: it is a
PRESENT-fixed Innovation 2 property-prediction network over structural query
indices. S2 is a runtime multi-cipher ciphertext-pair neural distinguisher and
receives arbitrary per-round, per-cell 4-bit S-boxes.

## Exact Operator

For each cell and inverse S-box output bit `j`, derive the exact ANF
coefficients `a[j,m]` for all sixteen monomial masks `m`:

```text
S_inverse(v)[j] = XOR_m a[j,m] * product_i v[i]^(m[i])
```

For each observed endpoint after the exact inverse linear layer, retain all
`4 x 16` individual ANF contributions rather than reducing them immediately to
four output bits. Pair-swap-invariant mean and XOR summaries of the two
endpoints give `128` local Boolean features. A shared `128 -> token_dim`
projection produces a gate which modulates the ordinary exact-GF(2) graph
message. The same projection weights are used for every cell, round and cipher.

The candidate keeps `sbox_context_mode=edge_gate` only for state-dict
compatibility but freezes `sbox_context_scale=0.0`; the old free truth-table
gate is therefore functionally absent. The new operator residual scale is
frozen at `0.25` before training.

## Frozen Evidence And Budget

Reuse the exact A8 protocol, source caches and completed anchor:

```text
source ciphers = GIFT-64 r6, SKINNY-64/64 r7,
                 RECTANGLE-80 r6, uKNIT-BC prefix-r5
whole-cipher holdout = Dialga-128 prefix-r4
train = 2048/class/source, 4096 total rows/source
validation = 1024/class/source, 2048 total rows/source
seeds = 0,1
epochs = 10
pairs/sample = 4 independent ciphertext pairs
negative = encrypted random plaintexts
loss = MSE
optimizer = Adam, lr 1e-4, weight decay 1e-5
checkpoint = best four-source validation macro AUC
target training rows = 0
target optimizer steps = 0
execution = local CPU
```

`2048/class/source` is a local mechanism diagnostic, not formal training,
paper scale, an attack, SOTA, universality or breakthrough evidence.

## Lean Matrix And Controls

Only one new candidate is trained per seed. The unchanged Runtime-E4 anchor is
loaded from A8 rather than retrained. After candidate training, each candidate
checkpoint is evaluated without any update under:

| Condition | Main Runtime-E4 structure | Boolean operator structure |
| --- | --- | --- |
| `exact` | exact | exact inverse-S-box ANF |
| `input_permuted` | exact | deterministic input-permuted inverse-S-box ANF |
| `identity` | exact | identity/no-nonlinearity ANF |

Only the operator S-box changes. Data, labels, exact GF(2) matrices, cell
membership, bit roles, checkpoint, classifier and old disabled S-box context
remain bit-exact. Candidate/control parameter count is frozen at `459234`; the
completed A8 anchor remains `442466` parameters.

The result contract contains:

```text
2 seeds x ((4 sources x 3 operator conditions) +
           (1 holdout x 3 operator conditions) +
           (4 source + 1 holdout A8 anchor rows))
= 40 rows
```

## Readiness Gate

Before training require all of:

1. every runtime S-box and inverse table is a valid 4-bit permutation;
2. GF(2)-reducing the retained ANF contributions reconstructs the exact
   runtime inverse S-box for every table, input and output bit;
3. input-permuted and identity controls change only operator S-box truth bits;
4. exact, permuted and identity operator wrappers have identical state-dict
   geometry and `459234` parameters;
5. the old S-box edge-gate residual is exactly zero in the candidate;
6. exact/permuted/identity operators produce distinct fixed-weight logits;
7. pair swap and joint cell relabeling preserve exact-operator logits within
   `1e-6`;
8. existing Runtime-E4 default state dicts, parameter counts and fixed-output
   regressions remain unchanged;
9. source tasks are exactly the four A8 sources and Dialga has no train cache;
10. the A8/S1 frozen artifacts and decisions match their preregistered hashes;
11. all forward/backward values and gradients are finite;
12. the 40-row result and 20-row history contracts are internally complete.

Any failed readiness invariant blocks training and permits only repair of that
invariant.

## Completed Readiness

The implementation readiness gate completed at:

```text
outputs/local_readiness/i1_runtime_spn_sbox_anf_operator_s2_readiness_20260726/
checks = 15/15 passed
candidate parameters = 459234
A8 baseline parameters = 442466
target training rows = 0
target optimizer steps = 0
decision = innovation1_runtime_spn_sbox_anf_operator_s2_readiness_passed
```

Every runtime inverse S-box in GIFT, SKINNY, RECTANGLE, uKNIT and Dialga was
exactly reconstructed by GF(2)-reducing its retained ANF contributions. The
fixed-weight exact/permuted/identity operators produced distinct logits for all
five ciphers, the operator projection received a finite nonzero gradient, pair
swap error was exactly zero, and joint cell-relabel errors on uKNIT and Dialga
were `1.19e-7`. All 54 required cache files were present and no Dialga training
cache was referenced. The frozen A8 and S1 decisions and evidence hashes also
matched. The preregistered two-seed local diagnostic is therefore authorized;
this readiness result contains no trained performance evidence.

## Research Gate

Let `candidate_exact`, `candidate_permuted` and `candidate_identity` denote
same-checkpoint evaluations, and `A8_anchor` the completed unchanged
Runtime-E4 result. For each seed require:

```text
four-source exact macro - permuted macro >= +0.005
four-source exact macro - identity macro >= +0.005
four-source exact macro - A8 anchor macro >= -0.005

Dialga exact AUC >= 0.55
Dialga exact - permuted >= +0.005
Dialga exact - identity >= +0.005
Dialga exact - A8 anchor >= -0.005

max probability delta(exact, permuted) > 1e-6
max probability delta(exact, identity) > 1e-6
```

Both seeds must pass all checks for `sbox_anf_operator_supported`. If the
operator is responsive but any semantic or retention check misses, return
`sbox_anf_operator_not_supported` and close this S-box route. Protocol failure
is separate and cannot be interpreted as a research result.

## Required Artifacts

Readiness:

```text
outputs/local_readiness/i1_runtime_spn_sbox_anf_operator_s2_readiness_20260726/
```

Diagnostic:

```text
outputs/local_diagnostic/i1_runtime_spn_sbox_anf_operator_s2_2048_seed0_seed1_20260726/
  results.jsonl
  history.csv
  progress.jsonl
  validation.json
  gate.json
  summary.json
  checkpoints/
  role-results/
  curves.svg
  visual_qa_passed.marker
```

After a completed run, invoke `visual-qa-redraw`, refresh both recent-result
indexes, update this record with metrics and an evidence-backed next action,
then make a scoped commit and push.

## Completed Diagnostic

The preregistered two-seed diagnostic completed at:

```text
run_id = i1_runtime_spn_sbox_anf_operator_s2_2048_seed0_seed1_20260726
result rows = 40/40
history rows = 20/20
protocol_valid = true
target training rows = 0
target optimizer steps = 0
status = hold
decision = innovation1_runtime_spn_sbox_anf_operator_not_supported
```

The candidate remained numerically responsive to the supplied Boolean
operator, but the correct operator did not achieve semantic identifiability.
The four-source macro AUCs and same-checkpoint margins were:

| Seed | Correct ANF | Input-permuted | Identity | A8 anchor | Correct - permuted | Correct - identity | Correct - A8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.561917305 | 0.560489297 | 0.560117602 | 0.575356722 | +0.001428008 | +0.001799703 | -0.013439417 |
| 1 | 0.548926353 | 0.549696326 | 0.548798919 | 0.548085093 | -0.000769973 | +0.000127435 | +0.000841260 |

The whole-cipher Dialga holdout results were:

| Seed | Correct ANF | Input-permuted | Identity | A8 anchor | Correct - permuted | Correct - identity | Correct - A8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.865126133 | 0.865277767 | 0.863101006 | 0.834955215 | -0.000151634 | +0.002025127 | +0.030170918 |
| 1 | 0.818263531 | 0.819079876 | 0.817661285 | 0.848253727 | -0.000816346 | +0.000602245 | -0.029990196 |

Dialga maximum probability changes under the input-permuted and identity
operators were respectively `0.010865092/0.032540739` for seed 0 and
`0.011311352/0.015547752` for seed 1. The operator is therefore active rather
than a dead branch. Nevertheless, both seeds placed the input-permuted Dialga
operator slightly above the correct operator, and neither seed met the frozen
`+0.005` semantic margins on sources or holdout. Seed 0 also missed source
anchor retention, while seed 1 missed Dialga anchor retention.

This is a local `2048/class/source` mechanism diagnostic. It is not formal
scale, a general SPN ceiling, an attack result, SOTA evidence or a breakthrough
claim. The evidence rejects only this exact cell-local ANF residual hypothesis.
Increasing samples, epochs or ciphertext pairs would not resolve the observed
same-checkpoint semantic-control failure.

The rendered SVG was inspected at 1800 px and 1280 px widths through the
`visual-qa-redraw` workflow. The final rendering has no overlapping or clipped
text, ambiguous labels, hidden legends or insufficient separation of the
near-tied controls; the lower panels plot the control margins directly.

## Blocked Routes

Regardless of result, do not revive dense DDT input, U2 inverse-triplet,
dual-view or final-transition delta-U query variants. Do not tune the ANF
scale, add S-box-specific experts, increase samples/epochs/pairs, launch a
remote run or relax the gate inside S2. A local miss closes this exact operator
hypothesis. A pass authorizes only a separately preregistered medium remote
confirmation against the same Runtime-E4 anchor and controls.

## Evidence-Backed Next Action

Close S-box conditioning and retain the already supported exact-GF(2) topology
contribution. Do not remotely scale or tune the ANF operator, and do not revive
dense DDT, inverse-triplet, dual-view, delta-U query, S-box-specific expert,
target-supervision or MoE rescue variants. The next Innovation 1 experiment
must change a different, preregistered structure hypothesis while keeping this
negative S-box result and its same-checkpoint controls intact.
