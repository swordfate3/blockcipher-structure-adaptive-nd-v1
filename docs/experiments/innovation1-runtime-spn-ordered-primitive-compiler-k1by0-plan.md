# Innovation 1 K1-BY0 Ordered Primitive Compiler Audit

**Date:** 2026-08-01
**Status:** completed / pass / local deterministic compiler audit
**Run ID:** `i1_runtime_spn_ordered_primitive_compiler_k1by0_20260801`

## Research Question

K1-BX0 made unseen Dialga S-box, linear topology and target-cell binding
observable, but its pooled structure vector still failed the stage-order gate.
K1-BY0 therefore changes the representation boundary instead of training a
wider encoder:

> Can a deterministic compiler preserve the exact ordered primitive schedule,
> route each cell to shared primitive expert types, and replay all seven runtime
> structures without using cipher identity?

This audit contains no ciphertext rows, labels, optimizer steps, differential
backbone or remote execution.

## One Representation Change

K1-BX0 anchor:

```text
ordered cell updates -> pool -> one 64-dimensional structure vector
```

K1-BY0 candidate:

```text
runtime descriptor
  -> ordered stage list
  -> per-cell 4-bit S-box primitive + 64-bit truth-table payload
  -> per-target-cell inverse-linear edge set
  -> route to shared permutation or general-GF(2) primitive expert type
  -> retain stage order and endpoint binding in an executable IR
```

Cell numbers are routing addresses rather than learned cipher identifiers.
Expert contracts have fixed local payload widths independent of block width,
cipher name and cipher ID.

## Frozen Protocol

The source gate, validation, results and config are SHA256-bound to K1-BX0.
Use exactly the same two-transition windows for GIFT, PRESENT, RECTANGLE,
SKINNY, Midori, uKNIT and Dialga-128. Controls use deterministic seeds
`{11,23,37,53}`.

Required checks:

1. Compile and exactly replay every S-box truth tensor, forward linear matrix
   and inverse linear matrix for all seven structures.
2. Jointly relabel cells, transport their semantic IDs, recompile, and require
   the same semantic program digest plus exact replay of the relabeled runtime
   structure.
3. Rotate the two stages when their contents differ and require a different
   semantic digest and a replay matching the rotated structure. Identical
   homogeneous stages are explicitly not applicable rather than false
   positives.
4. Keep every edge payload but send each target-cell bundle to the wrong cell;
   require a different semantic digest and a different replay for every
   structure and control seed.
5. Require exactly the same cipher-name-free expert contract for 64- and
   128-bit programs and record zero training steps and zero ciphertext rows.

## Gates And Decisions

```text
source authority and protocol errors              = 0
exact source replay                               = all seven
transported cell-relabel semantic equality        = all rows
applicable wrong-stage-order rejection             = all rows
wrong-target-binding rejection                     = all rows
expert contract identical across structures        = true
cipher name / cipher ID input                       = false
```

- **Pass:** authorize a separate K1-BY1 readiness design for a small
  same-budget Runtime-E4 comparison using ordered compiler routing, wrong-order
  routing, wrong-binding routing and no-structure controls.
- **Invalid:** repair only the exact replay, source-binding or control defect
  and rerun unchanged.
- **Hold:** stop compiler-to-expert integration and identify the failed
  primitive coverage; do not substitute a learned global vector.

No outcome authorizes remote scale, formal SPN claims, unseen-cipher neural
transfer, differential AUC, attack rounds, MoE growth, more epochs or more
training ciphers.

## Required Artifacts

Write `preflight.json`, `programs.json`, `results.jsonl`, `results.csv`,
`gate.json`, `validation.json`, `summary.json`, `progress.jsonl` and a Chinese
`curves.svg` under the local audit output root. The final SVG must pass a
`2700 x 1800` rendered-pixel `visual-qa-redraw` inspection before indexing and
reporting.

## Completed Result

The deterministic audit completed with no protocol errors:

```text
validation status = pass
result rows        = 70 / 70
compiled programs = 7
training steps     = 0
ciphertext rows    = 0
gate status        = pass
decision           = innovation1_runtime_spn_k1by0_ordered_compiler_ready
```

Control results were:

| Control | Applicable rows | Passed rows | Result |
|---|---:|---:|---|
| exact executable replay | 7 | 7 | pass |
| transported joint cell relabel | 28 | 28 | pass |
| wrong target-cell binding | 28 | 28 | pass |
| wrong order with distinct stages | 2 | 2 | pass |

All seven source programs exactly reconstructed their cell layout, bit roles,
S-box truth tensors, forward linear matrices and inverse linear matrices. The
four transported relabelings per cipher retained the same semantic program
digest while exactly replaying the relabeled native structure. Every wrong
target-cell binding changed the semantic digest and failed source replay.

Only uKNIT and Dialga have two distinct stage contents in the frozen windows;
their rotations changed the semantic program digest and replayed exactly as the
rotated structures. GIFT, PRESENT, RECTANGLE, SKINNY and Midori repeat identical
stage content, so swapping the two copies correctly remained semantically
identical rather than creating a false order signal.

The compiler selected shared local expert contracts without cipher identity:

```text
4-bit S-box expert calls      = 256
permutation expert calls      = 104
general GF(2) expert calls    = 152
```

The same fixed contracts cover 64- and 128-bit structures: a 64-bit S-box
truth descriptor for the 4-bit S-box expert and a fixed 10-field edge token for
both linear expert types. State width changes the number of routed cell calls,
not the learned expert parameter shape.

The final SVG passed `visual-qa-redraw` at `2700 x 1800`. The first rendering's
stage-order count was clarified to state that only two ciphers require order
rejection; the final image has no overlap, clipping, missing glyphs or semantic
count ambiguity.

## Verdict And Recommended Next Action

K1-BY0 establishes the missing deterministic front end. A new SPN runtime
descriptor can now be compiled into an ordered, executable program of shared
S-box, permutation and general-GF(2) primitive types without supplying its
cipher name. This is compiler readiness, not evidence that learned experts
improve a differential neural distinguisher.

The next bounded question is K1-BY1:

> At one small, fixed differential budget, does executing learned local
> primitive adapters in the compiler's exact stage and endpoint order improve
> Runtime-E4 over the same backbone without structure, while correct routing
> beats wrong-order and wrong-binding controls?

Use a recent plan-aligned low-round signal surface rather than tuning a new
difference position. Keep cipher, rounds, difference, pair count, negative
definition, key sampling, train/validation rows, seeds, epochs, optimizer and
metric identical. Change only the conditioner:

```text
same-budget anchor     = Runtime-E4 without compiler conditioner
candidate              = ordered compiler-routed shared primitive adapters
required controls      = wrong-order routing, wrong-binding routing
optional sanity        = label-shuffled only if not already bound by the source protocol
scale                  = sub-medium local diagnostic only
seeds                  = two frozen seeds
advance gate           = candidate beats anchor and both routing controls on every seed
stop gate              = any routing control matches/beats candidate, or local AUC is unstable
```

Before optimization, bind the exact source signal artifact and create a
separate K1-BY1 experiment plan. Do not launch remote training, enlarge expert
count, tune the difference position per model, mix benchmark changes, or call
K1-BY0 neural transfer evidence.
