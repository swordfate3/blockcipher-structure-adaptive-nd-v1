# Innovation 1 uKNIT-Family CT-SPN Native Endpoint K1-B

**Date:** 2026-07-28
**Run ID:** `i1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1_20260728`
**Status:** completed / hold
**Prerequisites:** K1 protocol-clean `hold`; K1-A endpoint-alignment-loss gate `pass`

## 1. Question

K1-A proved that Dialga's correct and rotated schedules change 97.9% of native
edge endpoint identities and change the raw edge values, while the current
edge-permutation-invariant transition summary changes by only about `7e-7` and
the final logit by only about `3e-6`. The current 12-value edge token contains
endpoint values but no endpoint identity.

K1-B tests one learned hypothesis:

> Does retaining directed native cell position and bit-role identity in every
> canonical edge token recover uKNIT signal and make the correct ordered
> transition schedule identifiable, without changing the rest of CT-SPN?

## 2. One Variable

The K1 edge token is:

```text
current left/right/xor
previous left/right/xor
endpoint products
endpoint xors
= 12 values
```

K1-B appends exactly ten deterministic values:

```text
normalized native target cell position      1
normalized native source cell position      1
native target bit role, four-way one-hot     4
native source bit role, four-way one-hot     4
                                              --
total additional values                     10
```

Target and source channels are separate, so edge direction is retained. Cell
positions are normalized to `[-1, 1]`, giving the same tensor and parameter
shape for 64-bit uKNIT and 128-bit Dialga. The model receives no cipher name,
cipher id, round count or global descriptor fingerprint.

Everything after the first edge encoder remains identical to K1: shared
equivariant edge mixer, mean/max/RMS edge pooling, kernel-three depthwise
temporal convolution, pair projection, pair attention and classifier. The new
candidate must remain no larger than the Runtime-E4 anchor.

## 3. Frozen Protocol

| Field | uKNIT-BC | Dialga-128 |
|---|---:|---:|
| Prefix rounds | 5 | 4 |
| Runtime window | transitions 3-4 | transitions 2-3 |
| Training | `2048/class` = 4096 total | `2048/class` = 4096 total |
| Validation | `1024/class` = 2048 total | `1024/class` = 2048 total |
| Seeds | 0, 1 | 0, 1 |
| Pairs/sample | 4 | 4 |
| Epochs | 10 | 10 |
| Batch size | 64 | 64 |
| Loss/optimizer | MSE / Adam | MSE / Adam |
| Learning rate/weight decay | `1e-4` / `1e-5` | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC | best validation AUC |
| Negative mode | encrypted random plaintexts | encrypted random plaintexts |
| Sample structure | independent pairs | independent pairs |
| Input difference | `0x40` | `0x40` |

The four K1-B training rows are compared against the already completed K1
Runtime-E4 and edge-identity-free CT-SPN rows. Dataset hashes must match the K1
validation datasets for every cipher and seed. K1-B must not retrain or select a
new anchor after seeing its result.

## 4. Frozen-Checkpoint Controls

Every trained K1-B checkpoint is evaluated without further optimizer steps on
the same validation rows under:

```text
correct_ordered
repeat_last
rotated
corrupted
no_topology
```

All five models must have identical parameter geometry and strict-load the
same selected state dictionary. Control construction must change only the
runtime structure/schedule intervention.

## 5. Readiness Gate

Training is authorized only if:

- the K1 and K1-A gates match their exact run ids and decisions;
- all four plan rows match the frozen protocol;
- 64- and 128-bit candidates have identical state-dictionary geometry;
- the edge encoder input width is exactly 22;
- parameter count does not exceed the 442466-parameter Runtime-E4 anchor;
- native endpoint channels change under repeated/rotated controls;
- a same-state deterministic Dialga probe produces nonzero transition-summary
  and logit differences for correct versus rotated/repeated schedules;
- all controls strict-load one common state dictionary;
- no cipher identity tensor or learned cipher router exists;
- readiness uses zero training rows and zero optimizer steps.

## 6. Advance Gate

No macro average may hide a cipher or seed failure.

For both uKNIT seeds:

```text
candidate AUC >= 0.520
candidate - max(K1 Runtime-E4, K1 CT-SPN) >= +0.005
candidate - repeat_last >= +0.005
candidate - rotated >= +0.005
candidate - corrupted >= +0.005
candidate - no_topology >= +0.005
```

For both Dialga seeds:

```text
candidate >= K1 CT-SPN - 0.005
candidate - repeat_last >= +0.005
candidate - rotated >= +0.005
candidate - corrupted >= +0.005
candidate - no_topology >= +0.005
```

Protocol checks additionally require four training rows, twenty frozen-control
rows, selected best-AUC checkpoints, matching K1 validation dataset hashes and
zero control optimizer steps.

## 7. Decisions

- **All cipher/seed gates pass:** retain native endpoint identity and plan K2 as
  a separate canonical S-box composition experiment at the same local budget.
- **uKNIT improves but a control or Dialga retention fails:** hold the family
  claim; inspect only the failed endpoint/temporal interaction without scaling.
- **uKNIT remains below its K1 Runtime-E4 anchor:** discard K1-B. Native endpoint
  identity alone is insufficient; do not increase samples or model width.
- **Protocol failure:** invalidate the run and repair only the protocol mismatch.

## 8. Blocked Routes

- No remote launch or mechanical sample/epoch/pair increase from K1-B.
- No K2, MoE, DDT, trail, partial decryption, guessed keys or cipher-id routing
  unless every K1-B seed-level gate passes.
- No benchmark changes: labels, validation rows, key sampling, negative mode,
  input difference and metric computation remain frozen.
- No generalized-Feistel MSX claim.

## 9. Artifacts

```text
configs/experiment/innovation1/
  innovation1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1.csv

outputs/local_readiness/
  i1_uknit_family_ctspn_native_endpoint_k1b_readiness_20260728/

outputs/local_diagnostic/
  i1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1_20260728/
    results.jsonl
    controls.jsonl
    history.csv
    progress.jsonl
    validation.json
    gate.json
    summary.json
    curves.svg
```

Every completed result must refresh the recent-result index. The final chart
must pass `visual-qa-redraw` before the experiment is reported complete.

## 10. Completed Readiness

The zero-training readiness gate passed on 2026-07-28:

```text
decision = innovation1_uknit_family_ctspn_k1b_native_endpoint_execution_authorized
training rows = 0
optimizer steps = 0
candidate parameters = 439982 <= Runtime-E4 442466
edge input width = 22
64/128-bit state-dictionary geometry = identical
```

Under one shared randomly initialized state dictionary, the new endpoint
channels made the controlled schedules observable before training:

```text
uKNIT repeat/rotated transition-summary delta = 0.557 / 0.811
uKNIT repeat/rotated logit delta              = 0.0227 / 0.0224
Dialga repeat/rotated transition-summary delta = 0.580 / 0.541
Dialga repeat/rotated logit delta              = 0.0153 / 0.0161
```

All readiness protocol and evidence checks passed. This only proves that the
new representation no longer mathematically erases endpoint/schedule identity;
the four-row K1-B training and twenty-row frozen control panel must still decide
whether that information is useful.

## 11. Completed Result

K1-B completed locally on 2026-07-28 with four training rows, twenty
same-checkpoint control rows and all eleven protocol checks passing:

```text
decision = innovation1_uknit_family_ctspn_k1b_native_endpoint_not_supported
status   = hold

uKNIT seed0:
  K1-B candidate                    = 0.510782
  strongest K1 prior               = 0.526651
  candidate - strongest prior      = -0.015869
  candidate - repeat/rotated       = -0.004974 / +0.000113
  candidate - corrupted/no-topology = -0.011662 / +0.003957

uKNIT seed1:
  K1-B candidate                    = 0.508568
  strongest K1 prior               = 0.528809
  candidate - strongest prior      = -0.020240
  candidate - repeat/rotated       = +0.001424 / -0.015071
  candidate - corrupted/no-topology = +0.000560 / +0.010372

Dialga seed0:
  K1-B candidate                    = 0.960106
  prior CT-SPN                      = 0.963836
  candidate - repeat/rotated       = +0.002678 / +0.007235
  candidate - corrupted/no-topology = +0.008107 / +0.440607

Dialga seed1:
  K1-B candidate                    = 0.961374
  prior CT-SPN                      = 0.963462
  candidate - repeat/rotated       = +0.005142 / +0.010541
  candidate - corrupted/no-topology = +0.008962 / +0.438411
```

The native endpoint channels repaired part of K1's order blindness: Dialga now
separates rotated and corrupted schedules on both seeds while retaining the
prior absolute AUC within `0.005`. Seed0 still misses the repeated-last margin,
and uKNIT remains below the Runtime-E4 anchor on both seeds. On uKNIT, a wrong
corrupted topology wins seed0 and a rotated schedule wins seed1. Native endpoint
identity is therefore observable but insufficient to produce a generalizing,
correctly attributed uKNIT distinguisher.

This is a local `2048/class` mechanism diagnostic, not a formal-scale failure or
a uKNIT ceiling. Mechanical scale-up, extra width, K2 S-box composition, MoE,
DDT/trail and remote training remain blocked.

The evidence-backed next action is K1-C: replay these four frozen checkpoints
under the same five topology conditions on both the exact training cache and
the unchanged validation data. This zero-training split-attribution audit asks
whether K1-B learned correct topology only on its training rows or failed to use
correct topology even while fitting. It changes no model, data, checkpoint,
metric or optimizer state. The result decides between a relative/cross-transition
path representation and closing this endpoint route.

The final Chinese `curves.svg` was rendered at `1600 x 1020` pixels and passed
`visual-qa-redraw`: no text overlap, clipping, missing glyphs, ambiguous title or
unreadable near-zero margin was observed.
