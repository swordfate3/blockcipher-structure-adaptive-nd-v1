# Innovation 1 K1-AW Dual-Path Structure Modulation Training

**Status:** completed / hold / channel-orientation audit required
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_dual_path_structure_modulation_k1aw_2048_replica0_replica1_20260729`

## 1. Research Question

K1-AV proved that one shared `34 -> 12 -> 2` projection can separately drive
the existing GF(2) edge and S-box-transition residual paths, exactly replay
K1-AT when the new edge channel is disabled, and respond to all required
descriptor controls. It did not train the new channel or measure AUC.

K1-AW asks:

> Under K1-AT's exact data and optimizer budget, does training the dual-path
> projection improve or retain cross-key macro AUC in both replicas, avoid
> meaningful harm to any cipher/split panel, and learn a preference for the
> correct runtime descriptor over full, S-box-only and linear-only mismatches?

This is a local `2048/class/cipher` diagnostic, not formal scale, an attack,
unseen-cipher transfer or SOTA evidence.

## 2. Frozen Authority And Same-Budget Anchor

Bind K1-AV's passed gate, validation, structure summaries and checkpoint
manifest by SHA-256. Bind K1-AT's gate, validation, two training rows, sixty
same-checkpoint controls and checkpoint manifest. Reuse K1-AT's exact K1-AO
data authority and all eighteen disk-backed datasets.

```text
ciphers = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
replicas = 0/1
initialization seeds = 30/31
dataset seeds:
  replica0 = uKNIT 3, Midori 6, Dialga 0
  replica1 = uKNIT 4, Midori 7, Dialga 1
negative mode = encrypted random plaintexts
```

K1-AT correct-descriptor AUC on each replica/cipher/fresh-split panel is the
same-budget anchor. No K1-AO value may be substituted for this comparison.

## 3. One Changed Variable

For each replica, construct K1-AT's single-output model and K1-AV's two-output
model from the same initialization seed. Copy every K1-AT initial tensor into
K1-AV. Expand only the final output weight:

```text
K1-AT learned path at initialization -> K1-AV output row 1
new deterministic GF(2) edge row     -> K1-AV output row 0
```

Then train the K1-AV model from this migrated initialization. This keeps the
base encoder, GF(2) residual, S-box-transition residual, classifier, shared
hidden layer and original transition row exactly aligned with K1-AT before the
first update. The extra output row and its connection to the edge gate are the
only architectural change.

## 4. Fixed Training Protocol

```text
train rows                 = 2048/class/cipher = 4096 total/cipher
fresh rows                 = 1024/class/cipher = 2048 total/cipher/split
pairs per sample           = 4
epochs                     = 10
batch size                 = 64
equal batches/cipher/epoch = 64
steps/epoch                = 192
total Adam steps/replica   = 1920
loss                       = MSE(sigmoid(logit), label)
learning rate              = 1e-4
weight decay               = 1e-5
checkpoint metric          = minimum cross-key AUC across three ciphers
device                     = local CPU
```

Each epoch uses the same deterministic cipher order and permutation formula as
K1-AT. A single model and optimizer are shared across all three ciphers.

## 5. Same-Checkpoint Controls

Restore each selected checkpoint and evaluate both fresh splits under:

```text
correct descriptor
full mismatch
S-box-only mismatch
linear-only mismatch
dual path disabled (supporting compatibility control)
```

The encryption runtime remains correct in every row. Both effective gates,
dataset/probability/checkpoint hashes, state immutability and zero evaluation
steps must be recorded.

## 6. Frozen Gates

Protocol passes only with two training rows, ten epochs, exactly 1920 Adam
steps per replica, two valid checkpoints, sixty complete controls, strict
encrypted-random-plaintext negatives, the exact `219764`-parameter geometry,
the migrated initialization contract, correct runtime binding and immutable
same-checkpoint evaluation.

The research route advances only if:

```text
cross-key macro AUC improvement vs K1-AT >= 0.0 in both replicas;
correct AUC minus K1-AT AUC >= -0.005 on all 12 panels;
correct minus each mismatch AUC >= +0.001 on at least 10/12 panels;
each mismatch's passing panels cover all ciphers, replicas and splits.
```

The dual-path-disabled row is supporting evidence and does not replace the
three semantic mismatch gates. Thresholds are frozen before training.

## 7. Decisions

- **All gates pass:** retain dual-path modulation as the stronger local family
  candidate. Before any `65536/class/cipher` remote job, separately audit
  route-specific disk cache/resume, pushed source and remote launch readiness.
- **Macro or no-harm fails:** hold dual-path training and freeze the selected
  checkpoints for an epoch/channel-orientation audit. Do not increase scale.
- **Semantic mismatch fails:** the two channels remain observable but have not
  learned correct descriptor preference. Audit correct-versus-mismatch channel
  orientation at the same checkpoint; do not add pairs or experts.
- **Protocol invalid:** repair only the failed source, migration, step,
  checkpoint, row or runtime binding and replay unchanged.

No branch authorizes 16 pairs, larger data, extra epochs/seeds/width, loss
balancing, PCGrad, experts/MoE or remote execution.

## 8. Required Artifacts

Write under `outputs/local_diagnostic/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
structure_summaries.json
results.jsonl
controls.jsonl
checkpoint_manifest.json
history.csv
comparison.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, record metrics, claim scope and an executable next action
here; refresh both recent-result indexes; run focused regression tests; and
commit/push only K1-AW files.

## 9. Completed Result

The frozen protocol completed with two training rows, sixty same-checkpoint
control rows, two selected checkpoints and no failed protocol checks. Both
replicas selected epoch 10 after exactly `1920` Adam updates.

Cross-key AUC at the selected checkpoints was:

```text
replica0:
  uKNIT-BC = 0.674100876
  Midori64 = 0.621796131
  Dialga-128 = 0.966804504
  macro = 0.754233837
  K1-AT macro = 0.741832415
  delta = +0.012401422

replica1:
  uKNIT-BC = 0.683011055
  Midori64 = 0.611639977
  Dialga-128 = 0.972631454
  macro = 0.755760829
  K1-AT macro = 0.750893593
  delta = +0.004867236
```

Both replicas retained or improved the K1-AT macro AUC. The gain was not
uniform, however. Replica 0 harmed both Dialga panels beyond the frozen
`-0.005` line:

```text
Dialga replica0 same-key  correct - K1-AT = -0.005478859
Dialga replica0 cross-key correct - K1-AT = -0.005101204
```

The correct runtime descriptor also failed to establish a stable advantage
over semantic mismatches:

```text
correct - full mismatch   >= +0.001 in 3/12 panels
correct - S-box mismatch  >= +0.001 in 2/12 panels
correct - linear mismatch >= +0.001 in 1/12 panels
```

The experiment therefore has status `hold` and decision:

```text
innovation1_uknit_family_k1aw_dual_path_training_not_supported
```

This does not discard the dual-path architecture. It shows a real and
replicated macro-AUC improvement, especially on uKNIT, but the learned gates
do not yet prefer the correct structure descriptor reliably and one
cipher/replica combination crosses the no-harm line. The result remains a
local `2048/class/cipher`, four-pair diagnostic only.

## 10. Evidence Artifacts

The complete run is stored at:

```text
outputs/local_diagnostic/
i1_uknit_family_dual_path_structure_modulation_k1aw_2048_replica0_replica1_20260729/
```

Protocol validation passed with `2/2` training rows, `60/60` control rows and
`1920/1920` optimizer steps per replica. The Chinese four-panel chart was
rendered to `2160 x 1440` pixels and passed the two-iteration
`visual-qa-redraw` inspection after both legends were moved outside their data
regions.

## 11. Executable Next Action

Open K1-AX as a zero-update channel-orientation audit. Freeze the two K1-AW
epoch-10 checkpoints, the same twelve cipher/replica/fresh-split panels, all
datasets, the correct encryption runtime and the four-pair protocol. Change no
training variable and perform no optimizer update.

For each correct, full-mismatch, S-box-only-mismatch and
linear-only-mismatch descriptor, record separately:

```text
effective GF(2) edge gate
effective S-box transition gate
edge-residual logit contribution
transition-residual logit contribution
total logit delta and AUC delta relative to the correct descriptor
```

The audit must determine whether descriptor replacement changes the intended
channel, whether either learned channel has the wrong sign, and whether the
two path contributions cancel. That result will choose exactly one next
variable: correct a channel orientation if one path is systematically wrong,
or hold this gate formulation if the paths are individually aligned but the
descriptor remains non-identifying. Do not add 16 pairs, samples, epochs,
seeds, per-cipher heads, experts/MoE or remote execution before this audit.
