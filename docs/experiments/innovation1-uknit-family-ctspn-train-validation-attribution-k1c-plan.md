# Innovation 1 uKNIT-Family CT-SPN K1-C Train/Validation Attribution Audit

**Date:** 2026-07-28
**Run ID:** `i1_uknit_family_ctspn_train_validation_attribution_k1c_20260728`
**Status:** completed / pass / split-specific topology overfit confirmed

## 1. Question

K1-B made native endpoint and schedule differences observable, partially restored
Dialga order attribution, but left uKNIT near chance and below Runtime-E4. The next
question is narrower:

> Did K1-B learn a correct-topology shortcut on its training rows that failed to
> generalize, or did the fitted model fail to prefer correct topology even on the
> training split?

This audit does not test a new architecture and cannot improve the reported AUC.
It determines which single architecture hypothesis is justified next.

## 2. Frozen Evidence

```text
source run = i1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1_20260728
ciphers = uKNIT-BC prefix-r5, Dialga-128 prefix-r4
seeds = 0, 1
pairs/sample = 4
training split = exact K1-B cached 2048/class rows
validation split = exact K1/K1-B 1024/class rows
checkpoints = four selected best-val-AUC K1-B checkpoints
optimizer steps = 0
```

The source K1-B gate must be protocol-clean `hold`, and all checkpoint, state-dict,
dataset and cache metadata must match the completed source run.

## 3. One Variable

The only changed variable is the evaluated split:

```text
training cache versus unchanged validation data
```

For each cipher and seed, evaluate the same checkpoint under the same five
conditions on both splits:

```text
correct_ordered
repeat_last
rotated
corrupted
no_topology
```

No model parameter, topology intervention, label, key, negative definition,
input difference, sample order, metric or threshold may change between splits.

## 4. Outputs

The audit must produce forty rows: `2 ciphers x 2 seeds x 2 splits x 5 controls`.
Every row records AUC, correct-minus-control AUC, probability deltas, dataset SHA,
checkpoint SHA, state-dict SHA, strict-load status, training flag and optimizer steps.

## 5. Decision Gate

For each cipher and seed define correct-topology attribution as:

```text
correct - repeat_last >= +0.005
correct - rotated     >= +0.005
correct - corrupted   >= +0.005
correct - no_topology >= +0.005
```

No average may hide a seed or control failure.

- **Training attribution passes, validation attribution fails:** confirm
  split-specific structural overfitting. The only eligible learned successor is a
  same-budget relative cross-transition path representation; absolute native cell
  positions must not be mechanically widened or scaled.
- **Training and validation attribution both pass:** the prior K1-B gate or replay
  binding is inconsistent; audit protocol/checkpoint selection before any training.
- **Training attribution also fails:** close the learned per-transition endpoint
  summary route. Before further training, require a zero-training exact
  cross-transition endpoint-composition feasibility audit.
- **Protocol failure:** repair only the evidence binding and rerun unchanged.

Dialga serves as the positive mechanism calibration. Its result cannot hide a failed
uKNIT seed.

## 6. Blocked Routes

- No optimizer step, new checkpoint or retraining.
- No remote execution, sample increase, extra pairs, epochs or model width.
- No K2, S-box conditioning, MoE, DDT, trail, partial decryption or cipher identity.
- No claim of attack, formal scale, SOTA, arbitrary-SPN transfer or uKNIT ceiling.

## 7. Next Action

Implement a fail-closed K1-C replay runner that reads only the completed K1-B
artifacts and cached datasets. Run it locally, validate all forty rows, document the
decision and refresh the recent-result index before selecting any learned successor.

## 8. Completed Result

K1-C completed locally on 2026-07-28. It reused all eight source caches, loaded the
four selected K1-B checkpoints strictly, performed no training and produced all
forty planned rows. All twenty-three final protocol checks passed:

```text
status   = pass
decision = innovation1_uknit_family_ctspn_k1c_split_specific_topology_overfit_confirmed
training_rows = 0
optimizer_steps = 0

uKNIT seed0 train:
  correct AUC                    = 0.638102
  correct - repeat/rotated       = +0.101195 / +0.132710
  correct - corrupted/no-topology = +0.141390 / +0.158409

uKNIT seed0 validation:
  correct AUC                    = 0.510782
  correct - repeat/rotated       = -0.004976 / +0.000111
  correct - corrupted/no-topology = -0.011663 / +0.003957

uKNIT seed1 train:
  correct AUC                    = 0.608175
  correct - repeat/rotated       = +0.063330 / +0.097402
  correct - corrupted/no-topology = +0.112075 / +0.110766

uKNIT seed1 validation:
  correct AUC                    = 0.508569
  correct - repeat/rotated       = +0.001423 / -0.015068
  correct - corrupted/no-topology = +0.000561 / +0.010373
```

Dialga supplies the positive calibration. Both training splits passed every
attribution margin. Validation retained the high correct AUC
`0.960106/0.961374`; seed1 passed every control, while seed0 remained below the
`+0.005` repeated-last margin at `+0.002678`. Dialga cannot hide the uKNIT split
failure.

The first gate used an over-strict `1e-7` exact-AUC replay tolerance and therefore
returned protocol-invalid even though all twenty validation dataset SHA values and
all checkpoint/state SHA values matched. The observed maximum CPU replay AUC delta
was only `2.861e-6`. The protocol repair set an explicit `5e-6` tolerance, added
the measured delta to the gate, retained exact dataset/checkpoint/state binding and
re-adjudicated the existing forty rows without another inference or optimizer step.

The result confirms that K1-B can fit the correct topology on its training rows but
that its absolute native endpoint summaries do not generalize. The next eligible
candidate is K1-D: one same-budget relative cross-transition path representation.
It must remove absolute cell positions, compose adjacent transitions before
invariant pooling and retain the same five topology controls. No remote scale-up,
extra data, width, MoE or K2 nonlinear conditioning is justified by K1-C.

Artifacts:

```text
outputs/local_audit/i1_uknit_family_ctspn_train_validation_attribution_k1c_20260728/
  results.jsonl
  attribution.csv
  gate.json
  validation.json
  summary.json
  progress.jsonl
  curves.svg
  visual_qa_passed.marker
```

The Chinese `curves.svg` was rendered at `1600 x 1020` and passed
`visual-qa-redraw`: no overlap, clipping, missing glyphs, ambiguous scales or
unreadable near-zero margins were observed. The run is entry `001` in
`outputs/00_RECENT_RESULTS.md` immediately after indexing.
