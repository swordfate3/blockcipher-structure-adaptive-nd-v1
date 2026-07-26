# Innovation 1 Runtime-SPN A7 Holdout Qualification Audit

Date: 2026-07-26

```text
status = completed / pass / Dialga selected
execution = local CPU, zero training and zero new data
remote_scale = no
```

## Research Question

Which completed Runtime-E4 target is a valid second independent whole-cipher
holdout for testing compositional SPN structure transfer, after separating
target learnability from cross-cipher transfer and requiring complete source
coverage of the target's atomic GF(2) relation types?

A6 was protocol-valid but could not support its relation-mass pooling
primitive on uKNIT. The candidate target AUCs were only `0.510880/0.518289`,
and correct pooling did not consistently beat no-topology, uniform or shuffled
same-checkpoint controls. Because target-trained uKNIT evidence is also weak,
retraining the same holdout would confound target learnability with transfer.

## One Audited Variable

Only the proposed whole-cipher holdout/source split changes. A7 does not train
weights, generate data, change labels, select a new input difference, or alter
the Runtime-E4 architecture. It recomputes qualification from frozen A3, A6,
uKNIT U3 and Dialga D1/D2 artifacts whose file hashes are fixed in the JSON
configuration.

## Frozen Candidate Panel

| Candidate | Frozen evidence | Other four ciphers as sources | Prior holdout |
| --- | --- | --- | --- |
| RECTANGLE-80 r6 | A3 zero-target-step whole-cipher result | GIFT, SKINNY, uKNIT, Dialga | yes |
| uKNIT prefix-r5 | U3 target-trained attribution result | GIFT, SKINNY, RECTANGLE, Dialga | yes |
| Dialga prefix-r4 | D1 target-trained plus D2 same-checkpoint audit | GIFT, SKINNY, RECTANGLE, uKNIT | no |

Every evidence row must use `2048/class` training where training occurred,
`1024/class` validation, pair4, ten epochs, seeds 0 and 1, MSE, fixed keys and
encrypted-random-plaintext negatives. The A3 target and D2 counterfactual rows
perform zero target optimization steps.

## Atomic Structure Definition

An atomic GF(2) relation type is the ordered pair

```text
(source bit role inside its 4-bit cell, target bit role inside its 4-bit cell)
```

for each nonzero coefficient in the inverse linear matrix. This creates at
most 16 role-to-role types and deliberately ignores global cipher identity,
absolute cell number and the complete local signature. A target has complete
source support only when every one of its atomic types appears in at least one
of the other four ciphers' two-transition runtime windows.

Exact S-box truth-table overlap is reported but is not a qualification gate.
An unseen S-box is expected in the final transfer experiment and will receive
a same-checkpoint wrong-S-box counterfactual. RECTANGLE already showed that an
unseen exact S-box does not by itself prevent transfer.

## Frozen Qualification Gate

For both seeds, a candidate must satisfy:

```text
correct AUC >= 0.55
correct - corrupted topology AUC >= +0.005
correct - no-topology AUC >= +0.005
atomic GF(2) source coverage = 100%
cell-relabeling max logit error <= 1e-6
all evidence hashes and protocol checks valid
```

To be selected, the candidate must additionally not have been used as an
earlier whole-cipher holdout. Eligible candidates are ranked by their minimum
two-seed correct AUC, with frozen candidate order as the deterministic tie
break. If none qualifies, no new training route opens.

## Required Outputs

```text
results.jsonl   one normalized row per candidate and seed
validation.json frozen hashes, protocol and zero-work checks
gate.json       per-candidate qualification and deterministic selection
summary.json    concise status and next action
progress.jsonl  bounded audit events
curves.svg      Chinese candidate qualification figure
```

The SVG must pass rendered-pixel `visual-qa-redraw` inspection before A7 is
complete. The completed result must also refresh `outputs/00_RECENT_RESULTS.md`
and `outputs/00_RECENT_RESULTS.json`.

## Advance And Stop Decisions

If exactly one unused candidate qualifies, preregister A8 at the same local
budget. A8 must train one cipher-name-free shared Runtime-E4 state on the other
four ciphers and evaluate the chosen target with zero target rows and zero
target optimizer steps. It must include a parameter-matched independently
trained no-topology anchor and same-checkpoint correct, corrupted,
no-topology and wrong-S-box target controls. The target-trained result is only
an oracle upper bound.

Do not revive A6 relation-mass pooling, retry uKNIT r5, add MoE, Adapter, FiLM,
typed residuals or target heads, increase epochs/samples, or launch remote
scale. A7 qualifies the split; it does not claim cross-cipher success.

## Completed Audit

The frozen evidence audit completed without training or new data generation:

```text
run_id = i1_runtime_spn_holdout_qualification_a7_20260726
status = pass
decision = innovation1_runtime_spn_holdout_qualification_dialga128_selected
validation checks = 10/10
selected holdout = Dialga prefix-r4
```

Normalized qualification evidence was:

| Candidate | Seed | Correct | Corrupted | No topology | Correct-corrupted | Correct-no topology |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RECTANGLE r6 | 0 | `0.690377` | `0.630342` | `0.609710` | `+0.060036` | `+0.080667` |
| RECTANGLE r6 | 1 | `0.660227` | `0.610850` | `0.616998` | `+0.049376` | `+0.043229` |
| uKNIT prefix-r5 | 0 | `0.501017` | `0.513706` | `0.514815` | `-0.012689` | `-0.013798` |
| uKNIT prefix-r5 | 1 | `0.527083` | `0.521579` | `0.497512` | `+0.005505` | `+0.029571` |
| Dialga prefix-r4 | 0 | `0.958417` | `0.925256` | `0.517403` | `+0.033161` | `+0.441014` |
| Dialga prefix-r4 | 1 | `0.958679` | `0.918380` | `0.526351` | `+0.040299` | `+0.432328` |

Every candidate had complete leave-one-out atomic GF(2) coverage:

```text
RECTANGLE = 4/4
uKNIT     = 16/16
Dialga    = 16/16
```

Every target had zero exact S-box truth-table overlap with its four-source
panel. This remains an informational novelty property, not a failure gate.
Cell-relabeling maximum logit errors were `1.64e-7`, `2.09e-7` and `8.94e-8`,
all below `1e-6`.

RECTANGLE is technically qualified but already served as the first
whole-cipher holdout. uKNIT is not qualified because both seeds miss the
`0.55` floor and seed0 misses both topology margins. Dialga is the only unused
qualified candidate.

Artifacts:

```text
outputs/local_audit/i1_runtime_spn_holdout_qualification_a7_20260726/
```

The final SVG passed rendered-pixel `visual-qa-redraw` at `1600 x 740` and
`1280 x 592` after moving the axis/threshold note away from the Dialga GF(2)
coverage annotation. No overlap, clipping, unreadable labels, ambiguous title,
misleading axis, incomplete legend or missing glyph remained.

## Evidence-Backed Next Action

Preregister A8 with GIFT, SKINNY, RECTANGLE and uKNIT as the only training and
checkpoint-selection tasks, and Dialga prefix-r4 as a zero-training-row,
zero-optimizer-step target. Use the already supported base Runtime-E4 exact
GF(2) path; do not reopen the completed unsupported typed-relation residual.
Compare the correct candidate against a same-initialization independently
trained no-topology anchor, then evaluate the candidate checkpoint under
correct, corrupted, no-topology and wrong-S-box Dialga structures. Preserve
`2048/class/source`, pair4, ten epochs and seeds 0/1 locally. The completed D1
result is an oracle upper bound only.
