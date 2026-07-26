# Innovation 1 H1-A6 uKNIT Heterogeneous-GF(2) Whole-Cipher Holdout Plan

Date: 2026-07-26

```text
status = completed / hold / relation-mass pooling not supported
execution = local readiness and 2048/class/source diagnostic completed
remote_scale = no
```

## Research Question

Can one shared Runtime-E4 state trained without any uKNIT rows use externally
supplied cell, S-box and general-GF(2) structure to distinguish uKNIT better
with its correct relation-activity pool than with corrupted, independent,
uniform or wrong-signature controls?

A5 could not answer this question. Its RECTANGLE holdout uses a one-to-one
linear layer, so correct, uniform and wrong-signature activity pooling were
bit-exact and its positive target pooling margin was structurally impossible.
A6 changes only the whole-cipher split needed to make the primitive
identifiable; it does not reinterpret A5 or reuse its invalid gate.

## One Experimental Variable

Train two parameter-matched models from the same initial state for each seed:

```text
candidate = relation_activity_pooling_mode=correct
anchor    = relation_activity_pooling_mode=uniform
```

Both use identical source examples, batch order, optimizer, loss, epochs and
checkpoint metric. No cipher ID, name, width embedding, target head or
task-specific trainable state is added. The candidate is also evaluated under
uniform and wrong-signature pooling without changing its checkpoint.

## Frozen Whole-Cipher Split

```text
sources = GIFT-64 r6, SKINNY-64/64 r7,
          RECTANGLE-80 r6, Dialga-128 prefix-r4
holdout = uKNIT-BC prefix-r5
uKNIT training rows = 0
uKNIT checkpoint-selection rows = 0
```

The uKNIT runtime window has 14 local relation-signature types. Correct,
uniform and wrong-signature pooling are therefore identifiable on the target.
The existing source cache may contain historical uKNIT training files, but A6
must never open or reference them; their mere existence is not target leakage.

## Frozen Model And Budget

```text
model = Runtime-E4, 442466 parameters for both roles
runtime descriptors = external cell membership, bit role, S-box truth table,
                      two-round GF(2) inverse-linear window
train/validation/target = 2048/1024/1024 per class
pairs/sample = 4 independent ciphertext pairs
negative = encrypted random plaintexts
seeds = 0, 1
epochs = 10
batch = 256
optimizer = A3 representation-L2 equalization + fixed-order PCGrad
loss = MSE
checkpoint = four-source validation macro AUC
device = local CPU diagnostic
```

This is a sub-medium local mechanism diagnostic, not formal scale,
universality, attack, SOTA or breakthrough evidence.

## Target Controls

For every seed, evaluate after both source-only roles finish:

| Evaluation | Checkpoint | uKNIT structure | Pooling | Target optimizer steps |
|---|---|---|---|---:|
| candidate correct | correct-trained | correct | correct | 0 |
| corrupted target | same candidate | corrupted | correct | 0 |
| no topology | same candidate | independent | forced uniform | 0 |
| uniform same checkpoint | same candidate | correct | uniform | 0 |
| wrong signature same checkpoint | same candidate | correct | shuffled | 0 |
| trained uniform anchor | uniform-trained | correct | uniform | 0 |

The first five rows must share one exact checkpoint SHA per seed. The sixth is
the independently trained same-budget architecture anchor.

## Readiness Gate

Before training, require all of:

1. the exact source panel excludes uKNIT and contains four ciphers;
2. the A5 source gate is protocol-invalid for the recorded one-to-one control
   reason, preventing accidental reinterpretation;
3. uKNIT exposes more than one local relation signature and correct differs
   from both uniform and wrong-signature logits under one state;
4. correct/uniform/wrong roles have identical `442466` parameters and state
   keys, and candidate/anchor initial states are bit-exact per seed;
5. independent relation mode forces uniform pooling;
6. correct and wrong-signature uKNIT logits preserve cell-relabeling
   invariance;
7. a synthetic source-only training smoke records exactly the four sources,
   then evaluates uKNIT with zero optimizer steps;
8. the cache manifest references source train/validation and uKNIT validation,
   but never uKNIT train;
9. all outputs are finite and the existing Runtime-E4/A3/A5 tests remain
   green.

Any failure stops A6 and permits only repair of the failed invariant. The
target control-identifiability check is logically prior to training.

## Frozen Advance Gate

For both seeds require:

```text
uKNIT correct AUC >= 0.55
uKNIT correct - corrupted/no-topology >= +0.005
uKNIT correct - same-checkpoint uniform/wrong >= +0.005
uKNIT correct >= trained uniform anchor - 0.01
candidate four-source macro >= uniform-anchor four-source macro - 0.005
conflict projections for both training roles >= 1
all initialization, checkpoint, cache and zero-target-step checks pass
```

A full pass supports relation-conditioned pooling on one unseen heterogeneous
SPN and opens a second independent heterogeneous holdout design. It does not
establish universal SPN adaptation.

If both seeds pass the target floor and all four same-checkpoint structural
controls but miss only trained-anchor or source-retention checks, retain a
partial unseen-structure attribution result and audit source calibration
before changing the architecture. Otherwise close this relation-mass pooling
primitive. An invalid protocol permits no metric interpretation.

Do not change the gate after results, load uKNIT training rows, select a
checkpoint on uKNIT, increase samples or epochs, launch remote scale, add a
target head, or revive MoE/Adapter/FiLM/typed residuals as a rescue.

## Readiness Result

The real implementation gate completed before training:

```text
run_id = i1_runtime_spn_uknit_heterogeneous_holdout_a6_readiness_20260726
status = pass
decision = innovation1_runtime_spn_uknit_heterogeneous_holdout_readiness_passed
checks = 15/15
uKNIT local relation signatures = 14
required cache files = 54/54
uKNIT train cache referenced = false
```

Correct, uniform and shuffled pooling had identical parameter geometry and
bit-exact initial states. Correct uKNIT weights and logits differed from both
controls, independent topology forced uniform pooling, and correct/shuffled
logits preserved cell relabeling up to `1.23e-7`. The target was evaluated
only after both synthetic source roles and with zero optimizer steps.

Artifacts:

```text
outputs/local_readiness/i1_runtime_spn_uknit_heterogeneous_holdout_a6_readiness_20260726/
```

## Completed Diagnostic

All four source-only roles completed, producing four checkpoints and 28 result
rows. Validation passed every initialization, cache, checkpoint, source-panel,
finite-metric and zero-target-training check.

uKNIT target AUCs were:

| Seed | Correct | Corrupted | No topology | Same-checkpoint uniform | Same-checkpoint shuffled | Trained uniform |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.510880` | `0.479408` | `0.510611` | `0.510861` | `0.511372` | `0.513043` |
| 1 | `0.518289` | `0.502498` | `0.522491` | `0.518338` | `0.518163` | `0.523032` |

The corresponding `correct - control` margins were:

| Seed | Corrupted | No topology | Uniform same checkpoint | Shuffled same checkpoint | Trained uniform |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `+0.031472` | `+0.000269` | `+0.000019` | `-0.000492` | `-0.002163` |
| 1 | `+0.015791` | `-0.004203` | `-0.000049` | `+0.000126` | `-0.004743` |

Both candidate roles retained the four-source macro AUC relative to their
uniform anchors (`+0.006584/+0.002292`) and both training roles recorded many
PCGrad conflict projections. The model therefore trained and retained source
performance, but the correct relation-mass pool did not produce a transferable
uKNIT readout. Neither seed passed the `0.55` floor, the no-topology margin, or
both same-checkpoint pooling margins.

```text
status = hold
decision = innovation1_runtime_spn_uknit_heterogeneous_holdout_not_supported
protocol_valid = true
target training rows = 0
target optimizer steps = 0
remote_scale = no
```

This closes only the tested scalar relation-mass activity pool. It does not
close Runtime-E4, general GF(2) structure input, or the method-level goal of a
runtime-parameterized SPN distinguisher. It also does not reinterpret A5: A5
remains protocol-invalid, while A6 is a valid negative mechanism result.

Artifacts:

```text
outputs/local_diagnostic/i1_runtime_spn_uknit_heterogeneous_holdout_a6_2048_seed0_seed1_20260726/
```

## Evidence-Backed Next Action

Do not train another architecture immediately. Preregister a zero-training
holdout-qualification audit with the following executable decision:

```text
research question = does a proposed unseen-cipher target have a demonstrated
                    same-budget signal, and are its local structure primitives
                    represented in the source panel?
same-budget anchors = completed per-cipher Runtime-E4 correct checkpoints at
                      2048/class, pair4, 10 epochs, seeds 0 and 1
required controls = per-cipher correct/corrupted/independent AUC; exact local
                    S-box and GF(2) primitive-support overlap; cipher-ID-free
                    cell-relabel invariance
one variable = holdout/source split qualification only; no neural weights change
scale = zero training, existing validation caches and frozen checkpoints
execution = local CPU audit
advance gate = candidate holdout reaches AUC >= 0.55 and correct-minus-both
               topology controls >= +0.005 on both seeds, while every target
               atomic primitive has source support
stop gate = no candidate satisfies both learnability and primitive-support gates
unlock = only a qualified split may receive one new compositional structure
         representation at the same 2048/class/source budget
blocked = uKNIT r5 retry, relation-mass pooling, typed residual, Adapter, FiLM,
          MoE, more epochs/samples, target supervision and remote scale
```

The audit must distinguish whole-signature novelty from atomic primitive
coverage. The next representation, if a split qualifies, must compose atomic
S-box/GF(2) relations rather than memorize a complete local signature or a
global cipher fingerprint.
