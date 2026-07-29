# Innovation 1 K1-AX Dual-Path Channel-Orientation Audit

**Status:** completed / component routing misalignment supported
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_dual_path_channel_orientation_k1ax_20260729`

## 1. Research Question

K1-AW improved cross-key macro AUC over K1-AT in both replicas, but harmed
both Dialga replica-0 panels beyond `-0.005` and failed all three semantic
mismatch gates. K1-AX asks which mechanism caused that contradiction:

1. S-box and linear summary components route more strongly into the wrong
   output gate;
2. one learned residual path is harmful even under the correct descriptor;
3. the two individually useful path effects oppose and cancel; or
4. none of these mechanisms is strong enough to resolve the failure.

This is a deterministic attribution audit of frozen checkpoints. It performs
no training and cannot establish formal-scale accuracy, an attack, unseen
cipher transfer, arbitrary-SPN generalization or SOTA.

## 2. Frozen Source And Data

Bind K1-AW's hold gate, passed validation, training rows, sixty control rows,
checkpoint manifest and structure summaries by SHA-256. Restore the exact two
epoch-10 checkpoints and reuse K1-AW's eighteen disk-backed datasets through
its source loader.

```text
ciphers = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
replicas = 0/1
splits = same-key fresh / cross-key validation
rows = 1024/class/cipher/split
pairs per sample = 4
negative mode = encrypted random plaintexts
training = 0 epochs / 0 optimizer steps
```

The encryption runtime stays correct for every row. Only the 34-dimensional
gate summary changes across descriptor conditions.

## 3. Path Decomposition

For each batch, compute the raw base embedding, GF(2) edge residual and S-box
transition residual once under the correct cipher runtime. For each descriptor
condition, apply its two learned gates and evaluate the same nonlinear
classifier at four states:

```text
pure_base       = classifier(base)
edge_only       = classifier(base + gated_edge)
transition_only = classifier(base + gated_transition)
full_dual_path  = classifier(base + gated_edge + gated_transition)
```

Record each AUC, full-forward replay error, effective gates, standalone and
full-context label-signed probability changes, helpful fractions, mean
absolute logit changes, path-opposition fraction and cancellation fraction.
The full state must replay `model.logits_with_runtime` within `1e-7`.

## 4. Descriptor Conditions

Evaluate exactly:

```text
correct descriptor
full mismatch
S-box-only mismatch
linear-only mismatch
```

The K1-AT/K1-AW deterministic hybrid-summary construction and mismatch order
remain unchanged. Expected output is `2 replicas x 3 ciphers x 2 splits x 4
conditions = 48` rows.

## 5. Frozen Mechanism Gates

For every cipher/replica/split panel:

- S-box routing is aligned when replacing only the S-box summary changes the
  transition gate more than the edge gate.
- Linear routing is aligned when replacing only the linear summary changes the
  edge gate more than the transition gate.
- A path is harmful under the correct descriptor when its full-context mean
  label-signed probability contribution is negative.
- A panel is cancellation-heavy when edge-only and transition-only effects
  oppose and the cancellation fraction is at least `0.5`.

Mechanism ordering is fixed before execution:

1. If either component has fewer than `10/12` routing-aligned panels, decide
   `component_routing_misalignment`.
2. Otherwise, if either correct-descriptor path is harmful in at least `3/12`
   panels, decide `learned_path_harm`.
3. Otherwise, if at least `3/12` panels are cancellation-heavy, decide
   `path_cancellation`.
4. Otherwise decide `mechanism_unresolved`.

An audit status of `pass` means the frozen attribution protocol completed; it
does not mean the K1-AW architecture passed its training gate.

## 6. Executable Decisions

- **Component routing misalignment:** replace the shared `34 -> 12` summary
  encoder with component-separated inputs: the edge gate may read only the
  18 linear features and the transition gate only the 16 S-box features.
  First build a zero-update migration/readiness gate; do not train yet.
- **Learned path harm:** keep descriptor routing fixed and audit the harmful
  residual representation/sign using the same checkpoint before proposing a
  gate change.
- **Path cancellation:** keep both encoders and add only a bounded interaction
  or contribution-normalization candidate after a zero-update readiness gate.
- **Unresolved:** hold the dual-path gate route and return to representation
  design; do not scale an unexplained mechanism.
- **Protocol invalid:** repair only source, checkpoint, data, row, replay or
  immutability binding and rerun unchanged.

No branch authorizes 16 pairs, more samples/epochs/seeds/width, loss balancing,
per-cipher heads, adapters, experts/MoE or remote execution.

## 7. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
structure_summaries.json
results.jsonl
panel_summary.csv
checkpoint_manifest.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, append exact mechanism counts, claim scope and the selected
next action here; refresh both recent-result indexes; run focused regressions;
and commit/push only K1-AX files.

## 8. Completed Result

The audit completed all `48/48` rows over twelve frozen panels, with two exact
epoch-10 checkpoints, zero training steps, immutable model state and maximum
full-forward replay error `0.0`. All protocol checks passed.

The component-routing counts were:

```text
S-box-only mismatch changes transition gate more than edge gate = 8/12
linear-only mismatch changes edge gate more than transition gate = 8/12
required for either component = 10/12
```

The failures were replica-consistent by family rather than random numerical
noise:

```text
replica0:
  S-box routing reversed on uKNIT and Dialga, both fresh splits

replica1:
  linear routing reversed on uKNIT and Midori, both fresh splits
```

The two alternative explanations did not reach their frozen mechanism gates:

```text
correct-descriptor GF(2) edge path harmful = 0/12
correct-descriptor S-box transition path harmful = 0/12
cancellation fraction >= 0.5 = 1/12
mechanism threshold = 3/12
```

The decision is therefore:

```text
innovation1_uknit_family_k1ax_component_routing_misalignment_supported
```

This sharpens the K1-AW interpretation. Both residual paths make positive
label-aligned contributions under the correct descriptor; the dominant defect
is the shared `34 -> 12` encoder, which permits the 16 S-box features and 18
linear features to influence both output gates and swaps the dominant response
for different replica/cipher combinations.

## 9. Evidence And Visual QA

Artifacts are stored under:

```text
outputs/local_audit/
i1_uknit_family_dual_path_channel_orientation_k1ax_20260729/
```

The Chinese four-panel chart was rendered at `2160 x 1416` pixels and passed a
two-iteration `visual-qa-redraw` check after both component-delta legends were
moved outside their data regions. The result remains a zero-update attribution
of the local `2048/class/cipher`, four-pair K1-AW evidence, not a new accuracy
experiment or formal-scale claim.

## 10. Selected Next Action

Open K1-AY as a zero-update migration/readiness gate for component-separated
structure encoders:

```text
GF(2) edge gate input       = linear summary[16:34] only (18 dimensions)
S-box transition gate input = S-box summary[0:16] only (16 dimensions)
```

Keep the K1-AW base encoder, edge residual, transition residual, classifier,
global gate biases, checkpoints, datasets and encryption runtime fixed. Define
an explicit migration from each trained K1-AW output row into its
component-specific encoder, then require:

```text
strict checkpoint migration and finite bounded gates;
S-box-only mismatch changes transition gate but exactly not edge gate;
linear-only mismatch changes edge gate but exactly not transition gate;
disabled compatibility mode replays K1-AW exactly;
both new component paths change logits observably when enabled;
0 epochs and 0 optimizer steps.
```

Only after K1-AY passes may one same-budget local training comparison be
planned against K1-AW. Do not add 16 pairs, data, epochs, seeds, loss weighting,
per-cipher modules, experts/MoE or remote execution.
