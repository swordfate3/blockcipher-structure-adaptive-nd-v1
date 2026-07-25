# Innovation 1 Runtime-SPN Primitive Gated Modulation Five-Cipher Plan

Date: 2026-07-25

## Status

```text
phase = completed single-variable redesign
implementation = complete
readiness = pass, 10/10 checks
training = complete
execution = local sub-medium diagnostic only
remote_scale = prohibited
status = hold
decision = innovation1_runtime_spn_primitive_gated_modulation_not_supported
```

## Research Question

Can parameter-matched, structure-routed multiplicative feature modulation make
the primitive-conditioned contribution stable across all five SPNs where the
completed additive low-rank residual was functionally weak on seed1?

The one architecture variable is the Adapter effect:

```text
old additive       h_next = h + 0.1 * adapter(h)
new gated          h_next = h * (1 + 0.1 * tanh(adapter(h)))
```

The low-rank modules, parameter count, deterministic router, Runtime-E4
backbone, exact GF(2) views, data, labels, negatives, task weights, optimizer,
loss, epochs and checkpoint rule remain unchanged. `tanh` bounds the
multiplicative modulation and avoids the failed no-training `0.5` scale-up.

## Frozen Five-Cipher Protocol

```text
tasks = GIFT-64 r6, SKINNY-64/64 r7, RECTANGLE-80 r6,
        uKNIT-BC prefix-r5, Dialga-128 prefix-r4
train = 2048/class/cipher
validation = 1024/class/cipher
pairs/sample = 4
seeds = 0,1
epochs = 10
batch = 256
loss = MSE
optimizer = Adam, lr 1e-4, weight decay 1e-5
checkpoint = best validation five-cipher macro AUC, restored
negative = encrypted random plaintext pairs
task weight = exactly 0.2 each before one shared optimizer step
```

The existing immutable caches must be reused; no data or benchmark change is
allowed.

## Four Gated Roles

| Role | Gated module | Purpose |
| --- | --- | --- |
| dense | one rank-16 unconditioned gate | strongest same-budget non-conditional anchor |
| correct | two rank-8 gates with real local fan-in routing | candidate |
| uniform | same two rank-8 gates mixed 50/50 | selection-disabled control |
| shuffled | same two rank-8 gates with flipped assignments | wrong-routing control |

All roles must retain exactly `446562` trainable parameters. Correct, uniform
and shuffled must have identical state keys/shapes and active compute. No
cipher-specific trainable state is allowed.

## Readiness Gate

Before the real diagnostic:

1. additive mode remains the default and all historical regressions pass;
2. gated dense/correct/uniform/shuffled roles are parameter matched;
3. one gated state dictionary strictly loads across all five 64/128-bit tasks;
4. correct/uniform/shuffled weights differ only by routing assignment;
5. both gates receive finite nonzero gradients and traffic;
6. five-task weights and optimizer step counts remain equal;
7. cell relabeling invariance holds for all five descriptors;
8. a `32/class/cipher`, one-epoch CPU joint smoke completes for all roles.

The readiness artifact must have its own gated-modulation decision and may not
reuse the additive readiness marker by name alone.

## Local Diagnostic Gate

First apply the existing correct-versus-dense/uniform/shuffled core and stress
gate unchanged. In addition, compare the gated candidate to the completed
additive correct-routing source under the exact same data and budget.

Both seeds must satisfy:

```text
gated correct - gated dense     >= +0.005 core and stress macro
gated correct - gated uniform   >= +0.005 core and stress macro
gated correct - gated shuffled  >= +0.005 core and stress macro

gated correct - additive correct >= +0.005 core and stress macro
gated correct - additive correct >= -0.005 on every individual cipher
```

The existing per-cipher floor against the gated dense anchor also remains in
force. Dialga's high absolute AUC may not compensate for uKNIT or a failed core
task.

## Decision Routes

- Full pass: keep gated modulation and preregister whole-cipher holdouts for
  RECTANGLE, Dialga and uKNIT. Learned routing remains closed until a holdout
  passes.
- Core pass, stress hold: preserve only the core result and refine one local
  heterogeneous transition descriptor; no five-cipher claim.
- Valid hold: discard this gated effect. Do not increase scale, rank, experts,
  epochs or samples. Rank a parameter-matched dense conditional basis/FiLM
  candidate against stopping the differentiated route.
- Protocol failure: repair readiness or source-anchor alignment only.

## Required Artifacts

```text
results.jsonl
history.csv
per_cipher_metrics.json
router_utilization.json
gradient_diagnostics.json
validation.json
gate.json
summary.json
progress.jsonl
checkpoints/
curves.svg
visual_qa_passed.marker
```

Refresh the recent-results index after readiness and diagnostic completion.
The diagnostic remains local `2048/class/cipher` evidence, not formal scale,
unseen-cipher transfer or a universal SPN result.

## Blocked Actions

- No remote launch, larger data, extra epochs, increased rank or added experts.
- No learned router, cipher ID, block width or global fingerprint conditioning.
- No changes to differences, keys, negatives, labels, validation data or
  checkpoint selection.

## Completed Result

The independent gated-modulation readiness passed all ten checks. Dense,
correct, uniform and shuffled roles each retained exactly `446562` trainable
parameters; shared 64/128-bit loading, both-gate traffic/gradients, equal task
weights, cell relabeling and the `32/class/cipher` smoke all passed.

The real two-seed diagnostic then completed with 40 result rows and eight
shared checkpoints while reusing the exact additive source caches. Validation
confirmed parameter matching, strict encrypted-random-plaintext negatives,
no task-specific trainable state and the frozen source anchor.

Correct gated routing relative to its gated controls:

| Seed | Panel | vs dense | vs uniform | vs shuffled |
| ---: | --- | ---: | ---: | ---: |
| 0 | core macro | +0.001181 | +0.000241 | -0.003069 |
| 0 | stress macro | +0.000405 | -0.006063 | -0.000094 |
| 1 | core macro | -0.001855 | +0.000105 | -0.001562 |
| 1 | stress macro | +0.002956 | -0.000828 | +0.001923 |

Relative to the completed additive correct-routing source:

| Seed | core macro | stress macro | largest relevant regressions |
| ---: | ---: | ---: | --- |
| 0 | +0.001772 | +0.000729 | GIFT -0.003531 |
| 1 | -0.007200 | +0.003793 | SKINNY -0.015060; GIFT -0.005826 |

Neither seed reached the required `+0.005` core/stress margins against all
controls or the additive source. Both adapters had traffic and finite nonzero
gradients, so this is a valid local architecture hold rather than a protocol
failure.

Evidence:

```text
readiness = outputs/local_readiness/i1_runtime_spn_primitive_gated_modulation_five_cipher_readiness_20260725/
diagnostic = outputs/local_diagnostic/i1_runtime_spn_primitive_gated_modulation_five_cipher_joint_2048_seed0_seed1_20260725/
validation = pass, 40/40 rows, 8/8 checkpoints, source anchor valid
visual QA = pass
```

### Evidence-Backed Next Action

Discard multiplicative gating. Do not increase scale, rank, experts, samples,
epochs or remote compute. Before another training slot, rank one
parameter-matched dense conditional basis/FiLM candidate against stopping the
differentiated-Adapter route. A new candidate must alter one conditional
computation only, preserve the five-cipher protocol and controls, and pass a
new local readiness gate.
