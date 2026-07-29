# Innovation 1 K1-AY Component-Separated Structure-Gate Readiness

**Status:** completed / component separation runtime ready
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_component_separated_structure_gate_k1ay_readiness_20260729`

## 1. Research Question

K1-AX found that both K1-AW residual paths were label-aligned, while the shared
`34 -> 12 -> 2` structure gate routed the dominant S-box or linear response to
the wrong output in four of twelve panels for each component. K1-AY asks the
narrow implementation question required before another training slot:

> Can the exact K1-AW parameter tensors be reused with component-separated
> connectivity so that S-box features affect only the S-box-transition gate
> and linear features affect only the GF(2)-edge gate?

This is a zero-training migration/readiness audit. It is not a new accuracy
result, a scale experiment, an attack, arbitrary-SPN generalization, unseen
cipher transfer or SOTA evidence.

## 2. Frozen Authority

Bind K1-AX's passed attribution audit by exact SHA-256 and load the same two
K1-AW epoch-10 checkpoints through the K1-AX authority chain. Reuse the exact
K1-AW disk-backed datasets and descriptor controls:

```text
ciphers = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
replicas = 0/1
splits = same-key fresh / cross-key validation
fresh data = 1024/class/cipher/split
rows inspected = first 32 rows per panel
pairs per sample = 4
negative mode = encrypted random plaintexts
training = 0 epochs / 0 optimizer steps
```

The encryption runtime remains correct for every row. Only the gate-summary
connectivity changes during the enabled counterfactual.

## 3. Single Model Change

Keep the K1-AW parameter names, shapes and values unchanged:

```text
first layer weight = [12, 34]
first layer bias   = [12]
output weight     = [2, 12]
parameter count   = 219764
state entries     = 55
```

Compatibility mode retains the original computation exactly:

```text
hidden = tanh(linear(summary[0:34]))
edge projection       = output row 0(hidden)
transition projection = output row 1(hidden)
```

Component-separated mode reuses slices of those same tensors:

```text
edge hidden = tanh(
    linear(summary[16:34], first_weight[:, 16:34], first_bias)
)
edge projection = output_weight[0](edge hidden)

transition hidden = tanh(
    linear(summary[0:16], first_weight[:, 0:16], first_bias)
)
transition projection = output_weight[1](transition hidden)
```

No parameter is added, removed, reshaped or copied into a cipher-specific
module. The repeated use of the existing first-layer bias is part of the
enabled counterfactual; compatibility mode must still replay K1-AW bitwise.

## 4. Frozen Audit Matrix

For each of `2 replicas x 3 ciphers x 2 fresh splits`, inspect 32 fixed rows and
write one primary row. For each primary panel, replay exactly three summary
controls:

```text
full mismatch
S-box-only mismatch
linear-only mismatch
```

Expected artifacts contain twelve primary rows and thirty-six control rows.
All candidate states must remain immutable.

## 5. Readiness Gates

Protocol gates:

- exact K1-AX source decision and artifact digests;
- strict loading of both K1-AW state dictionaries without key or shape edits;
- exactly `219764` trainable parameters and `55` state entries;
- component separation disabled replays K1-AW logits with maximum delta `0.0`;
- twelve primary and thirty-six control rows are complete;
- every row records zero training and zero optimizer steps;
- candidate state is immutable.

Research gates, required on every applicable panel:

- edge gate linear-summary Jacobian L2 is at least `1e-6`;
- edge gate S-box-summary Jacobian L2 is exactly `0.0`;
- transition gate S-box-summary Jacobian L2 is at least `1e-6`;
- transition gate linear-summary Jacobian L2 is exactly `0.0`;
- S-box-only mismatch changes transition gate by at least `1e-6` and edge gate
  by exactly `0.0`;
- linear-only mismatch changes edge gate by at least `1e-6` and transition gate
  by exactly `0.0`;
- full mismatch changes both gates by at least `1e-6`;
- enabled component separation changes logits by at least `1e-8`;
- every observed gate is finite and strictly bounded by `(-1, 1)`.

## 6. Executable Decisions

- **Pass:** preregister K1-AZ as one local same-budget comparison against
  K1-AW. Change only component-separated connectivity. Keep 4 pairs,
  `2048/class/cipher`, 10 epochs, replicas 0/1 and all strict controls.
- **Research hold:** repair only the failed relevance, isolation or observable
  response mechanism. Do not train.
- **Protocol invalid:** repair only source, strict-load, geometry, replay, row
  completeness or immutability binding and rerun unchanged.

No outcome authorizes 16 pairs, larger samples/epochs/seeds/width, remote GPU,
loss balancing, per-cipher modules, routers, adapters, experts or MoE.

## 7. Required Artifacts

Write under `outputs/local_readiness/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
structure_summaries.json
geometry.json
results.jsonl
controls.jsonl
checkpoint_manifest.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After execution, append exact gate values, claim scope and the selected next
action here; refresh both recent-result indexes; run focused regressions; and
commit/push only the K1-AY files.

## 8. Completed Result

K1-AY completed all twelve primary panels and thirty-six descriptor controls.
Both K1-AW epoch-10 state dictionaries loaded with `strict=True` without any
missing, unexpected, copied or reshaped state entry. The candidate retained
exactly `219764` trainable parameters and `55` state entries.

Compatibility and isolation results were:

```text
maximum disabled K1-AW logit replay delta       = 0.0
maximum edge Jacobian from S-box features       = 0.0
maximum transition Jacobian from linear features = 0.0
minimum edge Jacobian from linear features      = 0.039820596576
minimum transition Jacobian from S-box features = 0.042949583381
```

Descriptor counterfactuals also isolated exactly:

```text
S-box-only mismatch:
  maximum GF(2)-edge gate delta = 0.0
  minimum S-box-transition gate delta = 0.000063568354

linear-only mismatch:
  minimum GF(2)-edge gate delta = 0.000446066260
  maximum S-box-transition gate delta = 0.0

minimum enabled component-separation logit delta = 0.352983057499
```

All protocol and research checks passed. Model state remained immutable and
the run performed zero optimizer steps. The decision is:

```text
innovation1_uknit_family_k1ay_component_separation_runtime_ready
```

This proves only that the proposed connectivity is compatible, isolated and
observable on the frozen local K1-AW evidence. It does not show that retraining
will improve AUC or correct semantic descriptor margins.

## 9. Evidence And Visual QA

Artifacts are stored under:

```text
outputs/local_readiness/
i1_uknit_family_component_separated_structure_gate_k1ay_readiness_20260729/
```

The Chinese four-panel chart was rendered from the final SVG to a
`2700 x 1800` pixel image. A two-iteration `visual-qa-redraw` inspection moved
isolation notes out of the data region, shortened the twelve-panel labels and
confirmed no text overlap, clipping, ambiguous title, missing glyph, misleading
zero on a logarithmic axis or incomplete legend.

## 10. Selected Next Action

Preregister K1-AZ as the single authorized local same-budget training test:

```text
question = does component-separated connectivity improve K1-AW training?
anchor = K1-AW shared-summary dual-path gate
candidate = K1-AY component-separated dual-path gate
ciphers = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
train = 2048/class/cipher
fresh = 1024/class/cipher/split
pairs = 4
epochs = 10
replicas = 0/1
optimizer steps = 1920/replica
negative mode = encrypted random plaintexts
execution = local diagnostic
```

The primary gate must require both replicas to improve cross-key macro AUC over
K1-AW and must retain per-cipher non-regression plus correct-versus-mismatched
descriptor margins. Do not add 16 pairs, data, epochs, seeds, width, remote GPU,
loss balancing, per-cipher modules, experts or MoE.
