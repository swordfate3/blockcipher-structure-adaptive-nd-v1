# Innovation 1 K1-AS Structure-Derived Transition-Gate Readiness

**Status:** completed replay fix / pass / K1-AT local training authorized
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_structure_derived_gate_k1as_readiness_replay_fix_20260729`

The initial run without `replay_fix` is protocol-invalid and remains preserved.
The first implementation added optional gate-control arguments directly to the
frozen K1-AK adapter's `logits_with_runtime` method. Its old logits replayed
exactly, but K1-AO correctly rejected the public runtime-call signature drift.
The repair moves all new arguments and parameters into a K1-AS subclass. It
restores the K1-AK signature and replay gate without changing the preregistered
summary, mismatch controls, thresholds or research decision.

## 1. Research Question

K1-AR showed that inverse-norm training increased the S-box-transition path's
utility for every Midori64 panel while decreasing it for every uKNIT-BC and
Dialga-128 panel. A single global transition scalar cannot express that stable
heterogeneity. K1-AS asks the narrower implementation question:

> Can one shared, bounded gate derive a different transition-path strength from
> runtime S-box and GF(2) diffusion statistics without reading a cipher name or
> introducing per-cipher parameters?

K1-AS is a zero-training readiness gate. It does not test whether the new gate
improves AUC and cannot support an attack, transfer, scale or SOTA claim.

## 2. Frozen Authority

Bind the completed K1-AR replay-fix gate, validation and checkpoint manifest.
Reuse K1-AR's exact K1-AO and K1-AQ checkpoint loader, its 18 disk-backed fresh
datasets, its three runtime descriptors and all inherited source checks.

The audit covers:

```text
2 checkpoint families x 2 replicas x 3 ciphers x 2 fresh splits = 24 panels
rows inspected per panel = 32 deterministic leading rows
training rows = 0
optimizer steps = 0
data generation = false
device = local CPU
```

## 3. One Changed Variable

Keep the K1-AK base encoder, GF(2) edge residual, compact S-box-transition
residual and classifier unchanged. Replace only the effective transition gate:

```text
K1-AK:
  tanh(global_bias)

K1-AS:
  tanh(global_bias + shared_gate_network(runtime_structure_summary))
```

The shared gate network is one fixed `34 -> 12 -> 1` MLP for all block widths,
cell counts and ciphers. It is not a MoE router: there is one output, one set of
weights and no expert, adapter, lookup table, independent head or cipher ID.

## 4. Fixed-Width Structure Summary

The 34 values are deterministic, bounded and invariant to cell relabeling:

```text
S-box component (16 values):
  seven normalized distribution statistics over nontrivial DDT entries;
  seven normalized distribution statistics over nontrivial absolute LAT entries;
  normalized unique-S-box ratio;
  normalized per-round S-box-signature diversity.

GF(2) component (18 values):
  seven normalized row-weight distribution statistics;
  seven normalized column-weight distribution statistics;
  matrix density;
  normalized mean GF(2) rank;
  normalized minimum GF(2) rank;
  normalized unique-linear-transition ratio.
```

The function accepts only a `RuntimeSpnStructure`. Cipher name, key, label,
input difference and absolute cell identity are not arguments or features.

## 5. Required Controls

For each cipher, bind a cyclically selected other-cipher descriptor and derive:

```text
correct descriptor
full mismatched descriptor
S-box-only mismatch = other S-box component + correct linear component
linear-only mismatch = correct S-box component + other linear component
descriptor disabled = exact global-scalar K1-AK path
```

Also relabel every cell in reverse order and require the 34-value summary to be
exactly unchanged.

## 6. Readiness Gates

The result passes only if all of the following hold:

```text
K1-AR authority and all inherited source bindings pass;
candidate parameter/state geometry is identical across all three ciphers;
parameter count is exactly 219752 and state entries exactly 55;
all 24 source checkpoints load with only the three new gate-network tensors missing;
disabled candidate logits exactly replay the K1-AK source logits;
all gate values are finite and strictly inside (-1, 1);
the shared gate network has a finite nonzero gradient path;
cell relabeling leaves every summary value exactly unchanged;
correct vs full, S-box-only and linear-only mismatch gate deltas exceed 1e-6;
the gate changes logits observably on every panel;
model state is immutable and optimizer steps remain zero;
no cipher ID, per-cipher head, adapter, MoE or expert is present.
```

Any source drift or replay failure is protocol-invalid. A descriptor collision,
zero gradient, relabeling failure or unobservable mismatch is a readiness hold,
not permission to tune thresholds after seeing the result.

## 7. Decision And Next Action

If K1-AS passes, open exactly one local same-budget K1-AT comparison:

```text
candidate = K1-AS structure-derived transition gate
anchor = K1-AO equal-loss shared training
ciphers = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
2048/class/cipher, 4 pairs/sample, 10 epochs
replicas = 0/1, batch size = 64, 1920 Adam steps/replica
strict encrypted-random-plaintext negatives
same fresh same-key and cross-key controls
```

K1-AT must additionally compare correct, full-mismatched, S-box-mismatched,
linear-mismatched and descriptor-disabled inference from the same checkpoint.
Advance requires improved cross-key macro AUC over K1-AO, per-cipher no-harm,
correct descriptors beating mismatch controls, and K1-AO anchor retention.

If K1-AS holds, stop this gate design and audit summary identifiability or gate
initialization without training. In either case do not tune K1-AQ loss scales,
use PCGrad, increase pairs/samples/epochs/width, add experts/MoE, or launch a
remote job.

## 8. Required Artifacts

Write under `outputs/local_readiness/<run_id>/`:

```text
preflight.json
results.jsonl
structure_summaries.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, refresh `outputs/00_RECENT_RESULTS.md` and
`outputs/00_RECENT_RESULTS.json`, record the evidence-backed next action here,
run focused tests and commit/push only the scoped files.

## 9. Completed Result

The repaired run completed all `24/24` frozen panels. Every K1-AR source,
K1-AO/K1-AQ checkpoint, disk-backed dataset and inherited authority check
passed. Moving the new API into a K1-AS subclass restored the frozen K1-AK
runtime-call signature and its historical readiness replay.

Protocol result:

```text
candidate parameter count             = 219752 on all three ciphers
candidate state entries               = 55 on all three ciphers
shared parameter geometry             = exact across all three ciphers
old checkpoint load                   = only 3 new gate tensors missing
descriptor-disabled source replay     = exact on 24/24 panels
state immutable                       = 24/24 panels
training / optimizer steps            = 0 / 0
failed protocol checks                = []
```

Research result:

```text
34-value summaries finite/bounded     = pass
cell-relabel summaries exact          = 3/3
shared finite nonzero gradient path   = 24/24
full mismatch gate delta minimum      = 7.548779e-4
S-box-only gate delta minimum         = 1.756847e-5
linear-only gate delta minimum        = 4.675165e-4
full mismatch logit delta minimum     = 7.719994e-3
S-box-only logit delta minimum        = 9.894371e-5
linear-only logit delta minimum       = 4.834652e-3
failed research checks                = []
```

The minimum gate deltas exceed the frozen `1e-6` threshold and the minimum
logit deltas exceed the frozen `1e-8` threshold in every panel. These are
observability checks, not evidence that any mismatch has worse AUC.

Final adjudication:

```text
status                 = pass
decision               = innovation1_uknit_family_k1as_structure_gate_runtime_ready
next_training_authorized = true
remote_scale           = no
```

The Chinese SVG was rendered at `2160 x 1320` and inspected through
`visual-qa-redraw`. The first render exposed crowded heatmap ticks, a legend
covering a bar, an overlapping threshold annotation and internal English
control keys. The second render fixed all four defects and passed overlap,
clipping, glyph, scale, legend and claim-scope checks. The result directory
contains `visual_qa_render_report.json` and `visual_qa_passed.marker`.

## 10. Evidence-Backed Next Action

K1-AS proves only that the proposed shared gate is implementable, bounded,
structure-sensitive and backward-compatible. It does not show that the learned
gate chooses useful strengths. The next question is therefore K1-AT:

> Under the exact K1-AO equal-batch protocol, does the structure-derived gate
> improve shared cross-key performance without sacrificing uKNIT, Midori or
> Dialga and while preferring correct descriptors over all mismatch controls?

Use the preregistered local `2048/class/cipher`, four-pair, ten-epoch,
replica0/1 matrix with K1-AO as the only same-budget anchor. Do not change loss
weights, datasets, negatives, checkpoint selection or task order. Do not add a
second candidate, increase pairs/data/epochs/width, introduce experts or launch
remotely. A K1-AT hold closes this gate formulation rather than reopening
mechanical scale-up.
