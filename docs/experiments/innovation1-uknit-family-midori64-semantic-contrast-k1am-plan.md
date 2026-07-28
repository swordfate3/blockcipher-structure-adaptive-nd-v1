# Innovation 1 uKNIT-Family Midori64 Paired Semantic Contrast K1-AM

**Date:** 2026-07-29
**Status:** completed / hold / substitute unresolved
**Execution:** local CPU only; no remote scale

## 1. Research Question

K1-AL strict-loaded each K1-AK correct checkpoint and showed that both the
correct S-box and the new S-box-transition branch are causally used: replacing
the S-box reduced fresh AUC by `0.024103-0.062850`, while removing the branch
reduced it by `0.075548-0.120970`. K1-AK nevertheless showed that a wrong-S-box
model trained independently can learn a substitute and match the correct
model.

K1-AM asks:

> Can a bounded correct-versus-wrong runtime contrast during training preserve
> the K1-AK signal and make the learned solution prefer the S-box orientation
> that matches the real Midori64 data, rather than whichever orientation the
> optimizer is told to prefer?

## 2. Frozen Sources

K1-AK source:

```text
run_id  = i1_uknit_family_midori64_sbox_transition_k1ak_2048_seed6_seed7_20260729
status  = hold
decision = innovation1_uknit_family_midori64_k1ak_sbox_transition_discrimination_failed
```

K1-AL source:

```text
run_id  = i1_uknit_family_midori64_transition_causal_k1al_20260729
status  = pass
decision = innovation1_uknit_family_midori64_k1al_transition_and_sbox_causal_use_supported
```

Required source digests:

```text
k1ak_gate              a8cd9de68a7b4e43a4c8f0793e31cbf8ce87f090c35be6f6821cab282e927f8f
k1ak_validation        2d64a4e27b39a65fda5b44b217226fabb78a954d843573b47abbe34e0070e419
k1ak_controls          3b667435eb6c91dfb1c828953e834e9556dedf16c5054b4e70ded1d598e6e04e
k1ak_dataset_manifest  5525a28f099a21bcca09aafbe05498f0f7951e22e171eaac6db055c174ff35bc
k1al_gate              481cf6c90c281766e891c9a04de28d82cdf2b5051abbb570e83e81b0d5a433c2
k1al_validation        f58f38e72b8b27bcd2ac75502265526bf049c79ee7756ed70501db9670a15a65
k1al_results           a7a28643e76a36456143c7e112641fe8c2890eda9b892241b51b8f60dc463ce5
```

Any source drift makes K1-AM invalid.

## 3. Single Experimental Variable

Retain the exact K1-AK architecture and standard MSE classification loss. Add
one bounded per-sample functional contrast:

```text
primary_error        = (sigmoid(primary_logit) - label)^2
counterfactual_error = (sigmoid(counterfactual_logit) - label)^2

contrast = 0.25 * mean(
    relu(0.02 + primary_error - counterfactual_error)
)

total_loss = primary_MSE + contrast
```

`0.02` is a fixed modest margin on the bounded `[0,1]` squared-probability
error; `0.25` keeps the original classification loss dominant. These values are
preregistered once and must not be swept inside K1-AM.

The counterfactual uses the same trainable tensors and the same batch. Runtime
descriptors add no parameters. The general trainer, labels, data, validation,
checkpoint metric and metric computation remain unchanged.

## 4. Fair Four-Model Matrix

For each seed, create one parameter initialization and strict-load it into both
orientations before training:

| Orientation | Primary runtime | Counterfactual runtime |
|---|---|---|
| `correct_oriented` | correct Midori64 S-box | deterministic wrong S-box |
| `swapped_orientation` | deterministic wrong S-box | correct Midori64 S-box |

The complete training matrix is `2 seeds x 2 orientations = 4 models`.
Initial state hashes must be identical between orientations within a seed.
Both orientations use the same contrast scale, margin, model geometry,
optimizer steps and checkpoint rule. The swapped row is a required placebo: if
it performs equally well, the auxiliary objective can impose an arbitrary
orientation and has not identified the real S-box.

## 5. Frozen Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | Midori64 r4 |
| Difference | cell8 role1, `0x0000000400000000` |
| Seeds | `6`, `7` |
| Pairs per sample | `4` |
| Train | `2048/class`, `4096` total rows per seed |
| Same-key fresh | `1024/class`, `2048` total rows per seed |
| Cross-key fresh | `1024/class`, `2048` total rows per seed |
| Negative definition | encrypted random plaintexts |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE + fixed contrast / Adam `1e-4`, weight decay `1e-5` |
| Scheduler | none |
| Checkpoint | best primary cross-key validation AUC |
| Expected optimizer steps | `640` per model |
| Execution | local CPU |

Reuse all six bound K1-AK/K1-AH cache payloads. Generate no new dataset.

## 6. Evaluation Panel

Strict-load each of the four best checkpoints under three inference conditions:

```text
correct_runtime
wrong_sbox_same_checkpoint
transition_branch_off_same_checkpoint
```

Evaluate train-seen, same-key fresh and cross-key fresh. The complete panel is
`2 seeds x 2 orientations x 3 splits x 3 conditions = 36 rows`. Evaluation
performs zero training and must preserve checkpoint/state/dataset hashes within
each seed/orientation/split.

## 7. Protocol Gate

Require exact source digests and decisions; exact four plan rows; six source
dataset digests; equal initialization within each seed; four `219320`-parameter
models; ten complete epochs and exactly `640` optimizer steps each; nonzero
finite auxiliary losses; fixed contrast scale/margin; four best checkpoints;
`36/36` strict-load evaluation rows; exact primary cross-key AUC replay; fixed
row counts, input geometry, difference and strict negative definition; correct
runtime fingerprints for all interventions; and zero training during the final
panel.

Any failed protocol check makes K1-AM invalid.

## 8. Research Gates

Apply every gate independently to seed6/7 and both fresh splits:

```text
correct-oriented correct-runtime AUC - K1-AK correct anchor       >= -0.010
correct-oriented correct-runtime AUC - K1-AK independent wrong    >= +0.005
correct-oriented primary AUC - swapped-orientation primary AUC    >= +0.005
correct-oriented correct - wrong S-box same checkpoint             >= +0.005
correct-oriented correct - transition branch off                   >= +0.005
max probability delta for both interventions                        > 1e-6
```

The K1-AK independent wrong-S-box row is the decisive substitute-resolution
anchor. Train-seen rows are diagnostic only. Do not average seeds or splits to
hide a failed panel.

## 9. Decisions And Required Next Action

- **All gates pass:** retain the paired semantic objective and the K1-AK
  representation; run one same-protocol uKNIT-BC or Dialga family-transfer
  attribution diagnostic before any scale.
- **Same-checkpoint causality passes but independent-wrong or orientation gate
  fails:** the objective can enforce a preferred runtime but does not resolve
  the independently trained substitute. Discard this objective and audit a
  shared-normalization or representation-level identifiability constraint;
  do not tune its weight or margin mechanically.
- **Anchor retention fails:** discard the objective as destructive to usable
  Midori64 signal and return to the unmodified K1-AK optimizer path.
- **Protocol invalid:** repair only the failed source, initialization, data,
  training, checkpoint or intervention binding and rerun unchanged.

Do not add samples, pairs, epochs, seeds, positions, rounds, capacity,
hyperparameter sweeps, DDT/trail inputs, MoE, family transfer or remote
execution inside K1-AM.

## 10. Required Artifacts

```text
run_id = i1_uknit_family_midori64_semantic_contrast_k1am_2048_seed6_seed7_20260729

results.jsonl
controls.jsonl
history.csv
checkpoint_manifest.json
dataset_manifest.jsonl
preflight.json
progress.jsonl
gate.json
validation.json
summary.json
comparison.csv
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, append observed metrics, decision and an executable next
action here, then refresh `outputs/00_RECENT_RESULTS.md` and JSON.

## 11. Completed Result

The run completed locally with all four training rows, four best checkpoints
and all `36/36` strict-load evaluation rows. Every protocol check passed. The
first process stopped after writing the `seed6/correct_oriented` checkpoint
because the progress callback forwarded a reserved `path` field. The recovery
path verified the bound preflight, source hashes, dataset manifest, checkpoint
state, ten-epoch history, best-checkpoint metadata and `640` optimizer steps,
then reconstructed the missing result row without retraining it. The remaining
three models trained normally.

Fresh-split results:

| Seed / split | Candidate correct AUC | vs K1-AK correct | vs independent wrong | vs swapped primary | vs wrong, same checkpoint | vs branch off |
|---|---:|---:|---:|---:|---:|---:|
| seed6 same-key | 0.672109 | +0.003771 | +0.000073 | +0.007081 | +0.046444 | +0.092107 |
| seed6 cross-key | 0.658066 | +0.001935 | +0.001367 | +0.004746 | +0.042604 | +0.080859 |
| seed7 same-key | 0.663667 | +0.000640 | +0.005182 | +0.007456 | +0.085898 | +0.116735 |
| seed7 cross-key | 0.653212 | -0.000651 | -0.016104 | -0.010551 | +0.084621 | +0.105190 |

Observed decision:

```text
status   = hold
decision = innovation1_uknit_family_midori64_k1am_
           semantic_preference_imposed_substitute_unresolved
remote_scale = no
```

The paired objective retained the K1-AK correct signal and made the correct
runtime beat both same-checkpoint interventions on every fresh seed/split.
However, it missed the decisive independently trained wrong-S-box margin on
three of four fresh panels and missed the swapped-orientation margin on both
cross-key panels. The objective can impose a preferred runtime on one weight
set, but it does not make the true Midori64 S-box identifiable against a model
allowed to adapt its own representation. K1-AM is therefore discarded rather
than tuned.

## 12. Executable Next Action

Run K1-AN as one same-budget representation-identifiability diagnostic.

- **Question:** is the independent wrong-S-box substitute enabled by the
  trainable `256 -> 20` S-box-transition encoder and transition projection?
- **Anchor:** K1-AK correct and independently trained wrong-S-box rows on the
  same six bound datasets; use K1-AM only as causal context, not as a retained
  training objective.
- **Single variable:** replace the trainable transition re-encoding with one
  deterministic, orientation-shared canonical transition representation. Keep
  the base path, classifier, data, labels, negative definition and metric
  unchanged.
- **Matrix:** seed6/7 times correct S-box, wrong S-box and transition-branch-off
  control; Midori64 r4 cell8, `4` pairs, `2048/class`, ten epochs, batch 64,
  Adam `1e-4`, weight decay `1e-5`, best cross-key validation AUC.
- **Advance gate:** on every same-key and cross-key fresh panel, correct AUC
  retains K1-AK within `-0.010`, correct minus independently trained wrong is
  at least `+0.005`, and correct minus branch-off is at least `+0.005`.
- **Execution:** local CPU diagnostic. If it passes, test the unchanged
  canonical representation on one uKNIT-BC/Dialga family member before any
  remote scale. If it fails, stop single-cipher semantic regularization and
  move to a preregistered multi-cipher shared-weight identifiability test.
- **Do not pursue:** 16 pairs, larger samples, remote scale, contrast
  weight/margin scans, more epochs, MoE, DDT/trail inputs or a new difference
  before K1-AN resolves the representation hypothesis.

Artifacts:

```text
outputs/local_diagnostic/
i1_uknit_family_midori64_semantic_contrast_k1am_2048_seed6_seed7_20260729/
  results.jsonl
  controls.jsonl
  comparison.csv
  checkpoint_manifest.json
  gate.json
  validation.json
  summary.json
  curves.svg
  visual_qa_render_report.json
  visual_qa_passed.marker
```

The final Chinese two-heatmap SVG was rendered at `2040 x 1104` pixels and
passed `visual-qa-redraw`: no text overlap, clipping, missing glyphs, ambiguous
scale or legend/annotation collision was found. Close AUC values and decision
margins are shown in separate panels, and every failed threshold is marked
explicitly.
