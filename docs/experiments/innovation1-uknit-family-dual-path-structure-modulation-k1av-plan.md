# Innovation 1 K1-AV Dual-Path Structure Modulation Readiness

**Status:** completed / pass / K1-AW same-budget training authorized
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_dual_path_structure_modulation_k1av_readiness_20260729`

## 1. Research Question

K1-AU established that K1-AT's 34-value runtime summary and shared 12-value
hidden embedding preserve descriptor differences. The bottleneck appeared only
after the hidden embedding was projected to one scalar: the two replicas gave
the three correct cipher descriptors a Spearman gate-order correlation of
`-0.5` instead of the frozen `1.0` requirement.

K1-AV asks one implementation question:

> Can the same preserved hidden representation produce two bounded outputs,
> with one connected to the existing GF(2) edge residual gate and one connected
> to the existing S-box-transition residual gate, while exactly replaying the
> frozen K1-AT model when the newly added edge modulation is disabled?

This is a zero-training readiness gate. It cannot improve AUC and is not an
attack, transfer, scale or SOTA experiment.

## 2. Frozen Authority

Bind K1-AU's gate, validation, results, controls, checkpoint manifest and
summary by SHA-256. Through K1-AU, rebind the two K1-AT epoch-9 checkpoints,
the exact K1-AT datasets, the three runtime descriptors and all inherited
K1-AS/K1-AO authority checks.

```text
replicas                 = 0/1
ciphers                  = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
fresh splits             = same-key and cross-key
deterministic rows/split = first 32
result panels            = 12
descriptor controls      = 36
training / optimizer     = 0 epochs / 0 steps
device                   = local CPU
```

Encryption structures, keys, negatives, samples, pair count and checkpoints
remain unchanged in every control. Only the structure summary supplied to the
modulation network may change.

## 3. One Changed Variable

Retain K1-AT's full backbone, classifier, runtime summary and learned
`34 -> 12` hidden layer. Replace only its final `12 -> 1` projection with a
shared `12 -> 2` projection:

```text
output[0] -> tanh(global GF(2) edge bias + output[0])
output[1] -> tanh(global S-box-transition bias + output[1])
```

Copy K1-AT's learned one-row output weight exactly into `output[1]`. Initialize
only the new `output[0]` row from one frozen seed. In compatibility mode,
ignore `output[0]` and use the original global edge gate while continuing to
use `output[1]`; this must reproduce K1-AT logits bit-for-bit. In dual-path
mode, both outputs are enabled.

The candidate remains one `34 -> 12 -> 2` network with one parameter geometry.
It receives no cipher name, block-width lookup, per-cipher head, adapter,
router, expert or MoE component.

## 4. Required Controls

For each replica and cipher, retain K1-AT's cyclic descriptor mismatch order:

```text
correct descriptor
full mismatch
S-box-only mismatch
linear-only mismatch
```

The correct encryption runtime is held fixed. The linear-only mismatch must
observably alter the GF(2) edge gate. The S-box-only mismatch must observably
alter the S-box-transition gate. Full mismatch remains a joint observability
control, not a claim that either wrong descriptor must have lower AUC before
training.

Reverse every runtime cell label and require the 34-value summary to remain
exactly unchanged. Verify path wiring through gradients: the edge gate must
have a finite nonzero gradient through output row 0 and exactly zero gradient
through output row 1; the transition gate must have the converse pattern.

## 5. Frozen Gates

K1-AV advances only if all of the following pass:

```text
all K1-AU and inherited source hashes/checks are exact;
two K1-AT checkpoints migrate by expanding only the final output row;
parameter count = 219764 and state entries = 55 for all three ciphers;
parameter/state geometry and initialization are identical across ciphers;
compatibility mode replays K1-AT logits with maximum delta = 0;
all effective gates are finite and strictly inside (-1, 1);
GF(2) edge gate linear-summary Jacobian L2 >= 1e-6;
S-box-transition gate S-box-summary Jacobian L2 >= 1e-6;
cross-channel output-row Jacobian L2 = 0 exactly;
linear-only mismatch changes edge gate by >= 1e-6;
S-box-only mismatch changes transition gate by >= 1e-6;
enabled dual-path logits differ observably by >= 1e-8 on every panel;
cell relabeling invariance, state immutability and zero-update contract pass;
no cipher identity, per-cipher parameters, adapter, router, expert or MoE exists.
```

A source, checkpoint, migration, replay, state or row failure is protocol
invalid. A component-sensitivity, wiring, mismatch, invariance or output
observability failure is a readiness hold. Thresholds are not tuned after the
result.

## 6. Decision And Executable Next Action

If readiness passes, open one separately preregistered K1-AW same-budget
training comparison:

```text
candidate = K1-AV dual-path structure modulation
anchor    = frozen K1-AT single-scalar gate
ciphers   = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
budget    = 2048/class/cipher train, 1024/class/cipher fresh
pairs     = 4
epochs    = 10
replicas  = 0/1
batch     = 64, exactly 1920 Adam steps/replica
negative  = encrypted random plaintexts
device    = local diagnostic
```

K1-AW must change only the output dimension and path wiring. It must compare
cross-key macro AUC and per-cipher AUC against K1-AT at the same budget, retain
the correct/full/S-box/linear descriptor controls from the same checkpoint,
and enforce per-cipher no-harm plus correct-descriptor semantic preference.

If readiness holds, repair only the failed K1-AV mechanism and replay the same
zero-update matrix. Do not add 16 pairs, samples, epochs, width, seeds, loss
balancing, PCGrad, adapters, experts/MoE or remote GPU execution in either
branch.

## 7. Required Artifacts

Write under `outputs/local_readiness/<run_id>/`:

```text
preflight.json
results.jsonl
controls.jsonl
checkpoint_manifest.json
structure_summaries.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, record exact metrics, claim scope and the evidence-backed
next action here; refresh both recent-result indexes; run focused regression
tests; and commit/push only the scoped K1-AV files.

## 8. Completed Result

K1-AV completed all twelve frozen result panels and thirty-six descriptor
controls with zero training and zero optimizer updates. Every K1-AU artifact,
K1-AT checkpoint, dataset, runtime descriptor and inherited source binding
passed. The candidate retained one shared parameter geometry across uKNIT-BC,
Midori64 and Dialga-128.

Protocol result:

```text
candidate parameter count           = 219764 on all three ciphers
candidate state entries             = 55 on all three ciphers
shared network                      = 34 -> 12 -> 2
checkpoint migrations               = 2 / 2
only expanded tensor                = final 12 -> 1 weight to 12 -> 2
legacy transition row copied exact  = 2 / 2
compatibility replay panels         = 12 / 12
maximum K1-AT replay logit delta    = 0.0
state immutable                     = 12 / 12 result panels and 36 / 36 controls
failed protocol checks              = []
```

Research result:

```text
minimum GF(2)-edge linear-summary Jacobian L2 = 0.016188014
minimum S-box-transition S-box-summary Jacobian L2 = 0.035951752
maximum cross-channel output-row Jacobian L2 = 0.0
minimum linear-mismatch edge-gate delta = 0.000105083
minimum S-box-mismatch transition-gate delta = 0.000105485
minimum enabled-vs-compatible logit delta = 0.011970937
cell relabeling invariance = 3 / 3 exact
failed research checks = []
```

The relevant descriptor deltas exceed the frozen `1e-6` gate in every replica
and cipher, and enabling the new GF(2) channel exceeds the frozen `1e-8` logit
observability threshold on all twelve data panels. The output-row gradients
show exact wiring separation: the GF(2) edge gate has zero gradient through
row 1, and the S-box-transition gate has zero gradient through row 0.

Final adjudication:

```text
status                   = pass
decision                 = innovation1_uknit_family_k1av_dual_path_modulation_runtime_ready
next_training_authorized = true
remote_scale             = no
```

The `2160 x 1440` Chinese figure passed `visual-qa-redraw` after two rendered
pixel inspections. The first render had legends covering lower-panel bars; the
second added panel headroom and moved both legends into unused space. The final
figure has no text overlap, clipping, missing glyph, misleading axis,
incomplete legend or ambiguous title.

## 9. Evidence-Backed Next Action

K1-AV proves that the exact architectural change requested by K1-AU is
implementable, backward compatible, structure sensitive and wired to distinct
residual paths. It still provides no AUC evidence. The next question is K1-AW:

> Under K1-AT's exact equal-budget protocol, does training the two-channel
> projection improve cross-key macro AUC without harming any cipher and while
> preferring the correct runtime descriptor over full, S-box-only and
> linear-only mismatches?

Use K1-AT as the frozen historical same-budget anchor and train only the K1-AV
candidate at `2048/class/cipher`, `1024/class/cipher` fresh validation, four
pairs per sample, ten epochs, replicas 0/1, batch size 64 and exactly 1920 Adam
steps per replica. Preserve MSE, Adam `1e-4`, weight decay `1e-5`, strict
encrypted-random-plaintext negatives, dataset seeds, task order and checkpoint
selection. Evaluate the same correct/full/S-box/linear descriptor conditions
from each selected checkpoint on both fresh splits.

Advance requires both replicas to improve or retain K1-AT cross-key macro AUC,
all three ciphers to meet a preregistered no-harm tolerance, and correct
descriptors to beat every mismatch control at a frozen positive margin. If the
candidate fails, hold the dual-path training route and audit learned channel
orientation before changing scale. Do not add 16 pairs, samples, epochs,
width, seeds, loss balancing, PCGrad, experts/MoE or remote execution.
