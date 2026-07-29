# Innovation 1 K1-AZ Component-Separated Structure-Gate Training

**Status:** completed / hold
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_component_separated_structure_gate_k1az_2048_replica0_replica1_20260729`

## 1. Research Question

K1-AY proved that K1-AW's exact `34 -> 12 -> 2` parameter tensors can be
evaluated with component-separated connectivity: S-box features affect only
the S-box-transition gate, linear features affect only the GF(2)-edge gate,
and compatibility mode replays K1-AW bitwise. It did not train the isolated
connectivity or measure whether the change improves AUC.

K1-AZ asks:

> Under K1-AW's exact data, initialization and optimizer budget, does training
> component-separated connectivity improve or retain cross-key macro AUC in
> both replicas, avoid meaningful harm to every cipher panel, and learn a
> stable preference for the correct descriptor?

This is a local `2048/class/cipher` diagnostic, not formal scale, an attack,
unseen-cipher transfer, arbitrary-SPN generalization or SOTA evidence.

## 2. Frozen Authority And Same-Budget Anchor

Bind K1-AY's passed gate, validation, primary/control rows, checkpoint manifest,
geometry and structure summaries by SHA-256. Bind K1-AW's complete training
result and use its correct-descriptor AUC on each of twelve fresh panels as the
same-budget anchor. Reuse K1-AW's exact eighteen disk-backed datasets.

```text
ciphers = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
replicas = 0/1
initialization seeds = 30/31
dataset seeds:
  replica0 = uKNIT 3, Midori 6, Dialga 0
  replica1 = uKNIT 4, Midori 7, Dialga 1
negative mode = encrypted random plaintexts
```

## 3. One Changed Variable

For each replica, reconstruct K1-AW's pretraining state exactly:

1. build the K1-AT source and K1-AW model from the frozen initialization seed;
2. perform K1-AW's frozen final-output expansion;
3. capture the complete resulting K1-AW state dictionary;
4. build K1-AY and load that state dictionary with `strict=True`.

The K1-AW and K1-AZ initial state hashes must match exactly. Parameter names,
shapes, values, base encoder, residual branches, classifier, global gate
biases and optimizer settings remain identical. The only changed variable is:

```text
K1-AW = both output gates read one hidden state from all 34 summary values
K1-AZ = edge gate reads linear[16:34], transition gate reads S-box[0:16]
```

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

Use K1-AW's exact deterministic cipher order and permutation formula. One model
and one optimizer are shared across all three ciphers.

## 5. Same-Checkpoint Controls

Restore each selected K1-AZ checkpoint and evaluate both fresh splits under:

```text
correct descriptor
full mismatch
S-box-only mismatch
linear-only mismatch
dual path disabled (supporting control)
```

The encryption runtime remains correct for every row. Record both gates,
dataset/probability/checkpoint hashes, state immutability and zero evaluation
steps. The component-separated forward remains enabled for all non-disabled
conditions.

## 6. Frozen Gates

Protocol passes only with exact K1-AY/K1-AW bindings, exact initial-state
equality, `219764` trainable parameters, `55` state entries, two training rows,
ten epochs, exactly 1920 Adam steps per replica, two valid checkpoints, sixty
complete controls, strict negatives, correct runtime binding and immutable
same-checkpoint evaluation.

The research route advances only if:

```text
cross-key macro AUC improvement vs K1-AW >= 0.0 in both replicas;
correct AUC minus K1-AW AUC >= -0.005 on all 12 panels;
correct minus each mismatch AUC >= +0.001 on at least 10/12 panels;
each mismatch's passing panels cover all ciphers, replicas and splits.
```

The disabled row remains supporting evidence only.

## 7. Executable Decisions

- **All gates pass:** retain component separation as the stronger local family
  candidate. Before remote scale, open a separate disk-cache/resume and exact
  same-budget remote readiness audit.
- **Macro or no-harm fails:** hold component separation; compare epoch-wise
  trajectories with K1-AW before changing another model variable.
- **Semantic mismatch fails:** the hard connectivity fixed routing leakage but
  did not make the descriptor identifying. Freeze checkpoints and audit which
  structure-summary dimensions or residual response cause the remaining
  mismatch invariance; do not add scale.
- **Protocol invalid:** repair only source, initialization, data, step,
  checkpoint, row or runtime binding and rerun unchanged.

No branch authorizes 16 pairs, larger samples/epochs/seeds/width, loss
balancing, PCGrad, per-cipher modules, experts/MoE or remote execution.

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

After completion, record exact metrics, claim scope and the selected next
action here; refresh both recent-result indexes; run focused regressions; and
commit/push only K1-AZ files.

## 9. Completed Result

The local CPU run completed both replicas, all ten epochs and all frozen
same-checkpoint controls. Protocol validation passed without errors:

```text
training rows                     = 2/2
control evaluation rows           = 60/60
optimizer steps                   = 1920/1920 per replica
checkpoints                       = 2/2
failed protocol checks            = []
validation status                 = pass
```

Checkpoint selection used the frozen minimum-cross-key-AUC rule. Both replicas
selected epoch 9. Cross-key macro AUC relative to K1-AW was:

| Replica | K1-AZ | K1-AW | Delta | Gate |
|---|---:|---:|---:|---|
| 0 | 0.739590327 | 0.754233837 | -0.014643510 | fail |
| 1 | 0.763059139 | 0.755760829 | +0.007298311 | pass |

The two independent initializations therefore moved in opposite directions.
Replica 0's failure was concentrated in uKNIT-BC: its cross-key AUC fell from
`0.674100876` to `0.627626419` (`-0.046474457`), and its same-key fresh AUC
fell from `0.695286751` to `0.631846428` (`-0.063440323`). Midori64 and
Dialga-128 stayed within the frozen `-0.005` per-panel tolerance in both
replicas. Overall, `2/12` panels failed the no-harm gate.

Correct-descriptor preference also remained sparse:

| Mismatch control | Panels with correct-minus-mismatch >= +0.001 | Required | Gate |
|---|---:|---:|---|
| Full descriptor mismatch | 4/12 | 10/12 plus full axis coverage | fail |
| S-box-only mismatch | 4/12 | 10/12 plus full axis coverage | fail |
| Linear-only mismatch | 0/12 | 10/12 plus full axis coverage | fail |

The four passing full/S-box panels were all uKNIT-BC panels. Neither count nor
cipher coverage passed. Most importantly, hard separation did not make the
correct GF(2) linear descriptor identifiable: the linear-only mismatch passed
zero panels.

Final adjudication:

```text
status   = hold
decision = innovation1_uknit_family_k1az_component_separated_training_not_supported
```

This result rejects component-separated connectivity as a supported training
change under this fixed local budget. It does not prove a model ceiling or a
formal-scale failure. The evidence remains a `2048/class/cipher`, four-pair,
two-replica local diagnostic, not an attack, arbitrary-SPN transfer or SOTA
result.

## 10. Recommended Next Action

Run a zero-training K1-BA frozen-checkpoint response audit before changing the
model again.

```text
research question = which of the 18 linear-summary dimensions are ignored,
                    or have their response cancelled downstream, when the
                    runtime linear descriptor is wrong?
same-budget anchor = the two selected K1-AZ checkpoints and the exact 12 fresh
                     panels used above
required controls  = correct, full mismatch, S-box-only mismatch,
                     linear-only mismatch and dual-path-disabled replay
one variable       = substitute or mask one frozen linear-summary dimension at
                     a time; no weight update
scale              = existing 1024/class fresh datasets, replicas 0/1,
                     zero epochs and zero optimizer steps
execution path     = local CPU audit only
readiness gate     = exact checkpoint/data hashes, immutable state, exact base
                     replay, complete dimension-by-panel response matrix
decision gate      = distinguish ignored edge-gate inputs from a changed edge
                     gate whose logit/AUC contribution is cancelled downstream
```

If the edge gate itself does not react to the dimensions that differ under the
linear mismatch, the next trainable hypothesis may target the linear-summary
encoding or initialization. If the edge gate reacts but the final logits stay
invariant, the next hypothesis must target downstream residual fusion. Do not
combine these repairs in one experiment. Do not add 16 pairs, samples, epochs,
seeds, width, loss balancing, per-cipher modules, experts/MoE or remote scale.

## 11. Artifacts And Visual QA

The completed result is under:

```text
outputs/local_diagnostic/
  i1_uknit_family_component_separated_structure_gate_k1az_2048_replica0_replica1_20260729/
```

`gate.json`, `validation.json`, `results.jsonl`, `controls.jsonl`, both
checkpoints, history and comparison tables are complete. The Chinese
`curves.svg` was rendered to `2700 x 1800` pixels and inspected through the
`visual-qa-redraw` workflow. The title, four panels, local axis ranges,
legends, threshold lines and next-action caption have no overlap, clipping,
missing glyphs or ambiguous scale claims. The inspection is recorded in
`visual_qa_render_report.json` and `visual_qa_passed.marker`.
