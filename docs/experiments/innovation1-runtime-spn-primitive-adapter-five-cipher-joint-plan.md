# Innovation 1 Runtime-SPN Primitive Adapter Five-Cipher Joint Plan

Date: 2026-07-25

## Status

```text
phase = preregistered design
implementation = not started
training = not started
execution = local sub-medium diagnostic only
remote_scale = prohibited before the local gate passes
```

## Research Question

Can one jointly optimized, cipher-name-free Runtime-E4 parameter state use
deterministic structure-primitive adapters to distinguish five structurally
different SPNs more reliably than parameter-matched non-conditional controls?

The one architecture variable is primitive-conditioned low-rank residual
adaptation inside the runtime transition update. Data labels, negative
semantics, exact runtime inverse operators, metric calculation and checkpoint
selection remain unchanged.

## Five-Cipher Scope

| Group | Cipher task | Structural purpose | Frozen source protocol |
| --- | --- | --- | --- |
| core | GIFT-64 r6 | 64-bit, contiguous cells, fan-in-1 permutation | GIFT R2-F data protocol |
| core | SKINNY-64/64 r7 | 64-bit, multi-source general GF(2) | SKINNY T2-C data protocol |
| core | RECTANGLE-80 r6 | non-contiguous column cells, fan-in-1 row rotations | RECTANGLE RCT1 data protocol |
| new-cipher stress | uKNIT-BC prefix-r5 | heterogeneous S boxes and two distinct GF(2) transitions | uKNIT U3 data protocol |
| new-cipher stress | Dialga-128 prefix-r4 | 128-bit state, non-contiguous cells and heterogeneous transitions | Dialga D1 data protocol, fixed zero tweak |

The source protocol references freeze each cipher's rounds, input difference,
keys, validation split, pair structure and runtime descriptor. The new joint
trainer changes neither the cipher implementations nor the label definition.

MSX is explicitly outside this plan. It is a generalized Feistel design with
word arithmetic and 32-bit integer multiplication, not an SPN assembled from
4-bit S boxes and reversible GF(2) diffusion. It requires a later cross-family
primitive interface and separate controls.

## Common Model Contract

Every role uses:

```text
shared backbone        = RuntimeE4EquivariantSpnDistinguisher
cell width             = 4 bits
cell input             = state_triplet
S-box context          = edge_gate
round processing       = recurrent_window
runtime rounds         = 2
processor steps        = 2
pair embedding         = 128
dropout                = 0.0
pairs/sample           = 4
loss                   = MSE
optimizer              = Adam, lr 1e-4, weight decay 1e-5
epochs                 = 10
checkpoint             = best validation macro AUC, restored
seeds                  = 0,1
negative definition    = encrypted random plaintext pairs
train scale            = 2048/class/cipher
validation scale       = 1024/class/cipher
```

GIFT, SKINNY and RECTANGLE have homogeneous two-transition windows, so `full`
and `repeat_last` are identical by construction. uKNIT and Dialga retain their
real heterogeneous two-transition windows. This experiment attributes the
primitive adapter, not an earlier-round-use claim; the prior U3 repeat-last
control remains the authority for that separate question.

## One Shared Checkpoint

The five ciphers have different block widths and cannot be stacked into one
rectangular feature tensor. Joint training must therefore use task-wise
microbatches without creating task-wise backbone parameters:

```text
for each optimizer step:
    read one balanced batch from each of five disk-backed caches
    bind each batch to its own immutable runtime descriptor
    run the same shared backbone parameters five times
    compute one loss per cipher
    joint_loss = arithmetic mean of the five losses
    backward joint_loss once
    optimizer.step once
```

Each cipher contributes exactly one fifth of the optimization weight. Iterator
wraparound must be deterministic and recorded. No cipher-specific trainable
head, normalization statistics, embedding table or adapter is allowed. The
protocol bindings may differ only in non-parameter runtime tensors, input bit
width and frozen data metadata.

## Primitive Routing Candidate

Keep the exact inverse GF(2) state view outside the learned adapters. For hidden
token `h` and local runtime primitive descriptor `z`:

```text
h_next = shared_update(h, z) + sum_e alpha_e(z) * up_e(down_e(h))
```

First-version experts:

```text
fan_in_1_adapter      = rows with exactly one source bit
multi_source_adapter = rows with more than one source bit
```

`alpha_e` is deterministic. It may read local GF(2) row fan-in and relation
type. It must not read cipher name, cipher ID, key width, block width, round
count or a global profile fingerprint. Mixed transitions may route different
target cells to different adapters within the same sample.

## Four Model Roles

| Role | Parameters and compute | Purpose |
| --- | --- | --- |
| dense anchor | one width-matched low-rank residual with no primitive selection | strongest same-budget non-conditional anchor |
| correct routing | two low-rank adapters with deterministic real primitive routing | candidate |
| uniform mixture | same two adapters, fixed equal mixture at every location | selection-disabled control |
| shuffled routing | same two adapters, deterministic permutation of primitive assignments | wrong-routing control |

The candidate, uniform and shuffled roles must have identical state-dict keys
and active compute. The dense anchor's trainable parameter count must differ
from the candidate by no more than one percent. Width matching must be frozen
before any real AUC is read.

## Readiness Gate

Training is blocked until all checks pass:

1. Official/runtime equivalence tests pass for all five cipher descriptors.
2. One shared state dictionary strictly loads across all five protocol
   bindings, including the 64-bit and 128-bit tasks.
3. A multi-task optimizer step produces finite losses and gradients for all
   roles without task-specific trainable state.
4. Each cipher contributes exactly `0.2` of the joint loss and equal optimizer
   step counts.
5. Correct, uniform and shuffled roles have identical parameter keys, shapes
   and active compute; dense-to-candidate trainable parameters differ by at
   most one percent.
6. Correct routing changes when only the primitive assignment is shuffled;
   uniform routing does not.
7. Both primitive adapters receive traffic and nonzero finite gradients in a
   synthetic mixed-fan-in structure and in the five-cipher smoke batch.
8. Joint cell relabeling preserves logits for all five descriptors.
9. Existing Runtime-E4, uKNIT U3 and Dialga D1 regression tests remain green.
10. A `32/class/cipher`, one-epoch CPU smoke produces all expected per-cipher
    and aggregate metrics without creating a result claim.

## Local Diagnostic Gate

For each seed, report:

```text
core_macro       = mean AUC over GIFT, SKINNY, RECTANGLE
new_cipher_macro = mean AUC over uKNIT, Dialga
five_cipher_macro= mean AUC over all five
```

The five-cipher macro is descriptive only. Advance requires both seeds to
satisfy every condition:

```text
candidate core_macro - dense core_macro       >= +0.005
candidate core_macro - uniform core_macro     >= +0.005
candidate core_macro - shuffled core_macro    >= +0.005

candidate new_cipher_macro - dense            >= +0.005
candidate new_cipher_macro - uniform          >= +0.005
candidate new_cipher_macro - shuffled         >= +0.005
candidate uKNIT - dense uKNIT                  >= -0.005
candidate Dialga - dense Dialga                >= -0.005

candidate - dense on each core cipher          >= -0.005
both primitive adapters have traffic and gradients
no protocol, cache, parameter or checkpoint mismatch
```

The candidate may not use a high Dialga AUC to compensate for a weak uKNIT
result. A core pass plus a stress hold is recorded as `core_supported / new-
cipher_hold`, not as a universal pass.

## Decision Routes

- Full pass: keep deterministic adapters and run three separately trained
  whole-cipher holdouts: RECTANGLE, Dialga and uKNIT. The held-out cipher must
  be absent from training, validation, checkpoint selection and router
  statistics.
- Core pass, stress hold: preserve the core primitive result and inspect the
  missing heterogeneous S-box/round primitive locally. Do not call it five-
  cipher support and do not start learned MoE.
- Core hold: stop the differentiated-adapter route. Audit parameter matching,
  task balancing, descriptor fan-in classification and gradient flow. Do not
  add experts, epochs, samples or remote compute.
- Protocol failure: fix and rerun readiness only; no metric interpretation.

Only a full joint pass plus at least one whole-cipher holdout pass authorizes a
learned soft-routing comparison. Sparse Top-2 MoE requires both Dialga and
uKNIT holdout support and a separate router-collapse/load-balance protocol.

## Required Artifacts

A completed result-producing run must write:

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

The SVG must show per-cipher candidate/control margins and separate core,
new-cipher and five-cipher aggregates. It must pass `visual-qa-redraw` before
the result is complete. Refresh `outputs/00_RECENT_RESULTS.md` and JSON in the
same result-handling turn.

## Next Implementation Action

Implement only the shared primitive-adapter block, deterministic router and a
minimal task-wise joint optimizer harness. Start with synthetic 64/128-bit
mixed-fan-in readiness tests. Do not create the full training CSV or run real
five-cipher smoke until all ten readiness checks are executable and pass.
