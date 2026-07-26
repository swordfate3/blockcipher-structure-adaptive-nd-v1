# Innovation 1 Runtime-SPN S1 Frozen-Checkpoint S-box Identifiability Plan

Date: 2026-07-26

```text
status = completed / hold
execution = local CPU frozen-checkpoint audit
training = none
remote_scale = no
decision = innovation1_runtime_spn_sbox_responsive_but_not_identifiable
```

## Research Question

Does the completed A8 Runtime-E4 candidate use the externally supplied S-box
truth-table descriptor in an identifiable way, or does it merely respond to
the descriptor while assigning equal or better scores to incorrect S-boxes?

A8 showed strong zero-training Dialga sensitivity to the exact GF(2) topology,
but its correct Dialga S-box did not beat a broadcast GIFT S-box. S1 resolves
whether that result was an unusually favorable GIFT-to-Dialga counterfactual,
whether the S-box path is functionally ignored, or whether the path is
responsive but semantically aliased.

## Frozen Evidence And One Variable

S1 reuses the exact two A8 `correct_candidate` checkpoints, the exact cached
validation rows and the exact Runtime-E4 parameter geometry. It performs no
training, optimizer update, checkpoint selection or target-head adaptation.

```text
checkpoints = A8 seed0/seed1 correct_candidate
ciphers = GIFT-64 r6, SKINNY-64/64 r7, RECTANGLE-80 r6,
          uKNIT-BC prefix-r5, Dialga-128 prefix-r4
validation = 1024/class/cipher/seed
pairs/sample = 4 independent ciphertext pairs
negative = encrypted random plaintexts
model parameters = 442466
device = local CPU
```

For every seed and target cipher, only `sbox_truth_bits` changes. Cell
membership, bit roles, exact GF(2) matrices, data rows, labels, checkpoint and
relation mode remain bit-exact.

## Counterfactual Panel

Each target is evaluated with:

1. its exact runtime S-box tensor;
2. homogeneous GIFT, SKINNY, RECTANGLE and Dialga S-boxes;
3. the deterministic first uKNIT S-box in the loaded runtime window;
4. an identity 4-bit S-box;
5. a deterministic valid input-permuted version of the target S-box tensor;
6. an all-zero descriptor ablation, explicitly marked as a non-S-box control.

A broadcast control that is bit-exactly equal to a homogeneous target's exact
descriptor is recorded as an equivalence check and excluded from foreign-S-box
margins. uKNIT's exact heterogeneous tensor remains the only correct uKNIT
condition; its representative homogeneous table is a counterfactual.

## Readiness And Validity Gate

Before evaluation require all of:

1. the A8 config, result, gate and validation hashes match the frozen values;
2. A8 validation passed and its final decision is the recorded hold decision;
3. both candidate checkpoint and role-result hashes match;
4. checkpoint metadata proves four-source selection and Dialga exclusion;
5. the five validation cache manifests and arrays exist for both seeds;
6. every counterfactual changes only `sbox_truth_bits`;
7. all valid S-box controls are 4-bit permutations;
8. equivalent controls produce bit-exact probabilities;
9. no training or optimizer path is called;
10. all metrics and probability deltas are finite.

Any validity failure makes S1 invalid and permits only repair of that failed
invariant.

## Frozen Research Classification

Use AUC as the primary metric and best validation accuracy as supporting
evidence. The S-box descriptor is considered numerically responsive when a
non-equivalent valid S-box changes at least one probability by more than
`1e-6` for every seed/cipher pair.

Full S-box identifiability requires on both seeds:

```text
four-source correct macro AUC - every foreign valid-S-box macro AUC >= +0.005
Dialga correct AUC - every foreign valid-S-box AUC >= +0.005
correct AUC - identity/input-permuted/zero controls >= +0.005
```

The decision vocabulary is:

```text
sbox_identifiable
    all responsiveness and correctness margins pass

sbox_responsive_but_not_identifiable
    probabilities change, but correct S-boxes do not dominate controls

sbox_descriptor_functionally_ignored
    at least one seed/cipher is invariant to every non-equivalent valid S-box

protocol_invalid
    a frozen artifact, topology-only-change or no-training invariant fails
```

This is a mechanism audit, not formal-scale, universal-SPN, attack, SOTA or
breakthrough evidence.

## Required Artifacts

```text
results.jsonl
sensitivity.csv
validation.json
gate.json
summary.json
progress.jsonl
curves.svg
```

The SVG must show per-cipher correct-versus-counterfactual AUC and probability
sensitivity for both seeds, use Chinese explanatory titles, and pass rendered
pixel inspection through `visual-qa-redraw`. The completed audit must refresh
the recent-results index.

## Evidence-Backed Next Action

## Completed Audit

The complete frozen inference matrix ran at:

```text
outputs/local_audit/i1_runtime_spn_sbox_identifiability_s1_20260726/
result rows = 90
protocol checks = 12/12
training performed = false
optimizer steps = 0
A8 exact AUC reproduction = bit-exact
```

All five ciphers and both seeds were numerically responsive to a changed valid
S-box. The largest per-example probability change ranged from `0.1124` to
`0.2919`, far above the frozen `1e-6` responsiveness floor. The S-box path is
therefore not functionally ignored.

Correct S-boxes did not, however, consistently dominate the counterfactuals:

| Seed | Cipher | Exact AUC | Strongest non-exact control | Control AUC | Exact minus control | Max probability change |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 0 | GIFT-64 r6 | `0.538085` | Dialga S-box | `0.537181` | `+0.000905` | `0.1981` |
| 0 | SKINNY-64/64 r7 | `0.516874` | input-permuted | `0.498305` | `+0.018569` | `0.2581` |
| 0 | RECTANGLE-80 r6 | `0.725778` | Dialga S-box | `0.723262` | `+0.002516` | `0.2919` |
| 0 | uKNIT prefix-r5 | `0.520689` | SKINNY S-box | `0.526992` | `-0.006303` | `0.1612` |
| 0 | Dialga prefix-r4 | `0.834955` | RECTANGLE S-box | `0.897264` | `-0.062308` | `0.1778` |
| 1 | GIFT-64 r6 | `0.511875` | zero descriptor | `0.519396` | `-0.007522` | `0.1124` |
| 1 | SKINNY-64/64 r7 | `0.485603` | identity S-box | `0.479246` | `+0.006357` | `0.1335` |
| 1 | RECTANGLE-80 r6 | `0.683919` | uKNIT reference S-box | `0.680905` | `+0.003014` | `0.1715` |
| 1 | uKNIT prefix-r5 | `0.510944` | identity S-box | `0.514509` | `-0.003565` | `0.1184` |
| 1 | Dialga prefix-r4 | `0.848254` | RECTANGLE S-box | `0.869406` | `-0.021152` | `0.1291` |

On the four trained sources, every seed0 macro margin was positive and at
least `+0.007616`. Seed1 was not stable: foreign/control macro margins ranged
from `+0.001962` to `+0.005815`, with several below the required `+0.005`.
For unseen Dialga, the correct S-box lost to the RECTANGLE S-box by
`0.062308/0.021152` AUC and lost to the zero descriptor by
`0.040953/0.011719` AUC. The earlier GIFT-to-Dialga failure was therefore not
an isolated favorable swap.

```text
descriptor responsiveness = supported on 10/10 seed/cipher pairs
source S-box identifiability = not supported
Dialga unseen-S-box identifiability = not supported
decision = innovation1_runtime_spn_sbox_responsive_but_not_identifiable
```

The rendered SVG passed `visual-qa-redraw` at `1800x1000` and `1280x711`.
Titles explain that positive heatmap values favor the exact S-box while the
lower panel measures response only, not semantic correctness.

## Evidence-Backed Next Action

Close the current truth-table conditioning path. S1 supports exactly one new
method-level hypothesis: a cell-local Boolean S-box operator whose output is
used inside message computation, rather than an unconstrained descriptor gate.
Preregister it against unchanged Runtime-E4, a parameter-matched no-S-box
operator and a wrong-operator control at the same local budget. Do not reopen
A8, and do not treat numerical response to a descriptor as S-box understanding.

Blocked regardless of outcome:

```text
more A8 samples or epochs
remote A8 scale-up
Dialga supervision or target-head fitting
Adapter, FiLM, MoE or typed-relation rescue
dense DDT input revival
relaxing the A8 gate
```
