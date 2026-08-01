# Innovation 1 Runtime SPN Trainable Post-Expert Adapter K1-BY13

**Date:** 2026-08-01
**Status:** zero-training readiness passed / remote diagnostic pending exact
published commit
**Execution:** zero-training readiness locally; local CUDA unavailable, so a
successful readiness is handed to the remote A6000 as a device-availability
exception at local diagnostic scale

## Research question

K1-BY8 located stable correct-runtime access after the learned linear primitive
expert. K1-BY9 through K1-BY12 then exhausted three deterministic input/post-
expert interventions: each preserved some runtime signal but failed to use the
correct source-cell bindings through every seed and control.

K1-BY13 changes the hypothesis class rather than another formula:

> Can one small, shared, exactly zero-initialized adapter learn to turn the
> already accessible post-expert state into a stable correct-edge advantage,
> without reducing the original correct-runtime anchor?

## Single learned change

For expert output `X[b,t,h]`, actual incoming source-cell mean `M[b,t,h]` and
the existing role/expert embedding `E[t,h]`, define:

```text
Z[b,t] = concat(X[b,t], tanh(M[b,t]-X[b,t]), tanh(E[t]))
H[b,t] = GELU(W_in Z[b,t] + b_in)
A[b,t] = W_out H[b,t] + b_out
candidate[b,t] = X[b,t] + A[b,t]
```

`W_out` and `b_out` initialize exactly to zero. Therefore the adapter is an
exact identity before optimization, while `W_in` may receive gradients after
the zero output layer starts moving. With hidden size `32` and bottleneck `16`,
the adapter adds exactly `2096` shared parameters, below one percent of the
`235780`-parameter anchor. The same adapter weights are reused at every stage
and cell. Source-cell numbers are gather indices only; cipher, absolute cell
and absolute bit identifiers are prohibited as features.

## Frozen matrix

Exactly four conditions per seed:

| Condition | Runtime state/program | Adapter gather edges | Adapter |
|---|---|---|---|
| anchor correct | correct | none | absent |
| adapter correct | correct | correct | trainable |
| adapter affine | affine endpoint control | affine | trainable |
| adapter shuffled | correct | fixed shuffled source cells | trainable |

The shuffled mapping is frozen before training:

```text
source_cell -> (7 * source_cell + 3) mod 16
mapping = [3,10,1,8,15,6,13,4,11,2,9,0,7,14,5,12]
```

It preserves the correct inverse state, target/source roles, expert types,
edge counts, data and all downstream modules. Only the adapter gather endpoint
changes. The affine control remains the independently identifiable K1-BY6
wrong-runtime program.

## Frozen protocol

```text
cipher / rounds        = PRESENT-80 / r7
difference             = present_zhang_wang2022_mcnd:0 / 0x9
sample organization   = Zhang/Wang Case2 official MCND
key sampling           = fresh random PRESENT-80 key per pair
train                  = 2048/class / 4096 total rows
validation             = 1024/class / 2048 total rows
seeds                  = 2,3
pairs / input          = 16 / 2048 bits
negative               = encrypted random plaintexts
epochs / batch         = 10 / 64
loss / optimizer       = MSE / Adam
learning rate / decay  = 1e-4 / 1e-5
checkpoint             = restored best validation AUC
data cache             = parameter-matched disk cache with progress/reuse
```

No difference, round, pair, seed, key, sample, epoch, optimizer, loss, pooler,
classifier or negative-definition change is allowed.

## Readiness gate

Before any optimizer step require:

1. the exact K1-BY3 and K1-BY12 source decisions and SHA-256 bindings;
2. exactly eight frozen plan rows and four conditions per seed;
3. identical common named parameters under equal initialization;
4. candidate controls share exactly `237876` trainable parameters and state
   geometry; the anchor remains exactly `235780`;
5. adapter output projection weight and bias are exactly zero;
6. the correct candidate exactly replays the anchor output at initialization;
7. correct and shuffled candidates exactly agree at initialization, proving
   the zero adapter does not leak its edge binding;
8. one backward pass gives a finite nonzero gradient to the zero output layer;
9. correct, affine and shuffled edge bindings are pairwise distinct, while the
   shuffled program semantic digest remains equal to the correct program;
10. all models exclude cipher and absolute cell/bit identity.

Local readiness performs no optimizer step. The current environment reports
`torch.cuda.is_available() == false`, so long CPU training is prohibited. A
readiness pass authorizes a remote A6000 diagnostic only after the exact source
commit is published and verified, generated scripts pass the `G:\\lxy` path
audit, and the remote run has durable cache/progress output.

## Research gate

For each seed independently require:

```text
adapter correct AUC                         >= 0.550
adapter correct - anchor correct AUC        >= -0.005
adapter correct - adapter affine AUC        >= +0.005
adapter correct - adapter shuffled AUC      >= +0.005
```

The three adapter conditions must also have nonzero learned output-projection
norms in their restored checkpoints. Both seeds must pass every clause; means
cannot hide a failed seed.

## Decisions

- **Complete pass:** retain the post-expert trainable adapter and run one fresh-
  seed local-scale confirmation before any `65536/class` remote scale gate.
- **Correct signal/retention passes but a structural margin fails:** the adapter
  adds capacity without stable correct-edge use. Stop this adapter; do not tune
  its bottleneck or add depth.
- **Signal or retention fails:** discard the adapter and return to the K1-BY8
  anchor while revisiting the training objective, not the deterministic input.
- **Protocol invalid:** repair only the failed source, initialization, control,
  cache, checkpoint or device invariant and rerun unchanged.

Blocked: CPU training, bottleneck tuning, another adapter architecture, label
or benchmark changes, extra seeds/pairs/epochs, new ciphers, `65536/class`
scale, transfer and publication claims.

## Required artifacts

```text
run_id = i1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_16pair_2048_seed2_seed3_20260801
```

The run must produce preflight, four durable caches with twelve exact reuses,
eight restored checkpoints, eight results, checkpoint adapter norms, gate,
validation, summary, comparison CSV, history, progress and a Chinese SVG. The
SVG must pass rendered-pixel `visual-qa-redraw`; both recent-result indexes and
this document must be updated with the measured decision and executable next
action before reporting completion.

## Readiness result

The local zero-training implementation gate completed on 2026-08-01:

```text
output = outputs/local_readiness/
         i1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_
         16pair_2048_seed2_seed3_20260801
status = pass
decision = innovation1_runtime_spn_k1by13_readiness_authorized
training_performed = false
optimizer_steps = 0
plan_sha256 = 29bbcce189c4229e71b12e8568b2624f632f4178319edc3cb969d4d40bdf72a5
```

All source SHA-256 bindings and all readiness invariants passed. The frozen
matrix contains eight rows and the anchor/candidate parameter counts are
`235780/237876`; the adapter adds exactly `2096` parameters (`0.889%`). The
correct and shuffled candidates exactly replay the anchor before optimization,
all candidate output projections are exactly zero, and the zero output layer
receives a finite nonzero gradient (`L1 = 0.0337089468`). Correct, affine and
shuffled edge bindings have distinct fingerprints, while the shuffled control
preserves the correct compiled-program semantic digest.

No AUC or research result exists yet. Local CUDA is unavailable, and full CPU
training is fail-closed by the runner. The next action is to publish and verify
the exact scoped source commit, audit the generated Windows launch for
`G:\\lxy` paths and durable cache/progress output, then execute the frozen
eight-row matrix on the remote A6000. Do not change the adapter, controls,
budget or gate before that run.

## First remote launch repair

The first launcher invocation on 2026-08-01 cloned and checked out the pinned
source commit successfully, but stopped before readiness or training. Staging
the six frozen K1-BY3/K1-BY12 evidence files under
`<long run root>\\source\\outputs\\...` exceeded the Windows path-length limit.
No optimizer step ran and this is not model or data evidence.

The repair changes only the remote source layout: the run-owned clean clone is
placed at `G:\\lxy\\bcnd-k1by13-src`, while caches, checkpoints, logs, results
and staged source evidence remain under the original `G:\\lxy` run root. The
launcher still checks out one exact pushed commit, rejects a dirty clone and
verifies all frozen source evidence. The plan, eight training rows, models,
keys, samples, pairs, seeds, epochs, controls and research gates are unchanged.
