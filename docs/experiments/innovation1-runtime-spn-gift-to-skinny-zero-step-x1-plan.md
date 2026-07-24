# Innovation 1 Runtime SPN GIFT-to-SKINNY Zero-Step X1 Plan

Date: 2026-07-24

## Status

```text
stage       = completed and adjudicated
run_id      = i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724
execution   = local inference-only audit
training    = none
remote GPU  = no
claim scope = same-checkpoint cross-cipher runtime-topology diagnostic only
status      = hold
decision    = innovation1_runtime_spn_topology_sensitive_not_zero_step_discriminative
```

RTG2-B remains the active remote scale experiment. X1 is a deterministic,
small local audit that reuses completed checkpoints and validation caches. It
must not alter, compete with, supervise, or provide a rescue for RTG2-B.

## Research Question

When a runtime-E4 checkpoint trained on GIFT-64 is moved to SKINNY-64/64
without any target training, do the learned weights retain discriminative use
of the new runtime topology, or do they merely react numerically to changing
structure tensors?

The older typed-cell E4 study used exactly one epoch of target adaptation and
found robust source/target topology attribution but conditional advantage over
scratch. X1 asks a narrower, previously unmeasured zero-step question using the
new cipher-name-free runtime model. It is not another adaptation or scale run.

## Frozen Sources

For each seed, use the completed GIFT RTG1-R2F best checkpoints:

```text
seed0 source root = outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0
seed1 source root = outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed1

correct source checkpoint   = row0001_gift64_runtime_e4_equivariant_true_seed{seed}.pt
corrupted source checkpoint = row0002_gift64_runtime_e4_equivariant_corrupted_seed{seed}.pt
```

Use only the completed SKINNY RTG1-T2-C validation caches:

```text
seed0 target root = outputs/local_diagnostic/i1_rtg1_skinny64_general_gf2_attribution_t2c_2048_seed0_20260724
seed1 target root = outputs/local_diagnostic/i1_rtg1_skinny64_general_gf2_attribution_t2c_2048_seed1_20260724
validation seed   = 10000 + seed
validation rows   = 2048 total = 1024/class
input             = 4 independent ciphertext pairs = 512 bits
negative          = encrypted random plaintexts
target key        = 0x1111111111111111
target difference = 0x2000
target rounds     = SKINNY-64/64 r7
```

Every input/checkpoint array must be SHA-256 identified before evaluation.

## Frozen Model Contract

```text
hidden bits        = 64
pair embedding     = 128
processor steps    = 2
S-box context      = late_pair
dropout            = 0.0
parameters         = 442466
checkpoint         = restored best validation AUC
target descriptors = exact SKINNY general-GF(2) correct/corrupted/independent
```

The GIFT state dictionary must load strictly into all target adapters. Runtime
structure tensors must remain outside the state dictionary. No optimizer,
gradient, checkpoint selection, calibration, threshold fitting, or target
training is allowed.

## Four-Condition Panel

For each seed evaluate the same SKINNY validation examples and labels:

| Condition | Source checkpoint | Target runtime structure | Purpose |
| --- | --- | --- | --- |
| `true_source_true_target` | GIFT correct topology | SKINNY correct GF(2) | zero-step candidate |
| `corrupted_source_true_target` | GIFT corrupted topology | SKINNY correct GF(2) | source-topology control |
| `true_source_corrupted_target` | GIFT correct topology | SKINNY deterministic corrupted GF(2) | target-topology control |
| `true_source_no_target` | GIFT correct topology | SKINNY no linear topology | no-topology control |

The first, third and fourth conditions must use the exact same checkpoint SHA.
All four must use the same feature and label SHA within each seed.

## Preregistered Gate

Protocol failure if any source, cache, SHA, model option, strict load,
parameter count, validation size, target topology identity, or no-training
check fails.

Full zero-step support requires both seeds to satisfy:

```text
candidate AUC >= 0.52
candidate - corrupted-source/correct-target AUC >= +0.005
candidate - correct-source/corrupted-target AUC >= +0.005
candidate - correct-source/no-target AUC >= +0.005
```

Additionally, every control must change candidate probabilities by more than
`1e-6`; otherwise the corresponding runtime or source-checkpoint intervention
is not demonstrably used.

Decisions:

```text
pass = zero-step discriminative topology use supported on two seeds;
       after RTG2-B is adjudicated, rank one new local adaptation hypothesis
hold = probabilities change but the complete AUC gate misses;
       retain topology sensitivity only and do not train or scale this route
fail = protocol invalid or an intervention is exactly inactive;
       repair the audit only, without changing sources or thresholds
```

Near-chance AUC with nonzero probability changes is not zero-shot transfer.
It is only numerical topology sensitivity. Do not report it as generalization.

## Outputs

```text
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/
  results.jsonl
  validation.json
  gate.json
  summary.json
  progress.jsonl
  curves.svg
```

After completion, run `visual-qa-redraw`, refresh both recent-result indexes,
record the exact decision here, and give an evidence-backed next action. Do not
launch remote work, add samples/seeds/epochs, or change GIFT/SKINNY task
protocols from X1.

## Completed Result

The frozen two-seed, four-condition panel completed with eight result rows.
All 18 protocol checks passed, including strict state-dictionary loading,
checkpoint/cache SHA-256 provenance, exact source and target protocols,
equal 442466-parameter geometry, and the absence of target training. All six
intervention-sensitivity checks also passed: changing the source checkpoint,
target topology, or no-topology relation mode changed the candidate
probabilities by more than `1e-6` on both seeds.

| Seed | Correct source + correct target | Corrupted source + correct target | Correct source + corrupted target | Correct source + no topology |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.460699558 | 0.499159336 | 0.487567425 | 0.512140274 |
| 1 | 0.407617092 | 0.448855877 | 0.479323864 | 0.508840084 |

Candidate-minus-control AUC margins were:

| Seed | Minus corrupted source | Minus corrupted target | Minus no topology |
| ---: | ---: | ---: | ---: |
| 0 | -0.038459778 | -0.026867867 | -0.051440716 |
| 1 | -0.041238785 | -0.071706772 | -0.101222992 |

Maximum absolute probability changes from the candidate were:

| Seed | Corrupted source | Corrupted target | No topology |
| ---: | ---: | ---: | ---: |
| 0 | 0.014542431 | 0.006862193 | 0.009039074 |
| 1 | 0.037813038 | 0.020182848 | 0.016476929 |

The candidate missed the `0.52` AUC floor on both seeds and lost to every
control. Therefore the correct interpretation is topology sensitivity without
zero-step discriminative transfer. The nonzero probability changes prove that
the source and target interventions are active; they do not rescue the failed
discriminative gate. The below-chance candidate AUC must not be post-hoc
inverted or recalibrated inside X1.

```text
status   = hold
decision = innovation1_runtime_spn_topology_sensitive_not_zero_step_discriminative
claim    = no zero-step GIFT-to-SKINNY discriminative support
```

Artifacts:

```text
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/results.jsonl
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/validation.json
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/gate.json
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/summary.json
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/progress.jsonl
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/curves.svg
outputs/local_audits/i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724/visual_qa_passed.marker
```

The final SVG passed the `visual-qa-redraw` pixel gate at 1600 x 864. The
first render exposed ambiguous condition colors; the final render uses green
for seed0 and blue for seed1 consistently across both panels and has no text
overlap, clipping, missing glyphs, ambiguous legend, or unreadable labels.

## Evidence-Backed Next Action

X1 is closed as `hold`. Do not train, scale, add seeds, flip probabilities,
or calibrate this zero-step route. RTG2-B remains the only active remote
Innovation 1 training experiment and must be adjudicated from its retrieved,
verified artifacts before another training slot is opened.

The executable branch after RTG2-B is:

```text
question        = does SKINNY r7 correct general-GF(2) topology survive 262144/class?
same-budget rows = correct / corrupted / no topology
one variable     = seed only if seed0 passes
train scale      = 262144/class = 524288 total
validation scale = 131072/class = 262144 total
epochs           = 5
execution        = remote A6000 with disk-backed cache and automatic retrieval
seed0 advance    = AUC >= 0.55 and both control margins >= +0.005
seed0 pass       = launch the identical seed1 protocol from a pushed commit
seed0 hold/fail  = do not launch seed1; rank one new local adaptation hypothesis
blocked          = X1 scale-up, target calibration inside X1, extra X1 seeds,
                   or any zero-shot / SOTA / breakthrough claim
```

If RTG2-B seed0 does not pass, the next local design question is whether a
strictly separated target head can absorb the cross-cipher output-orientation
mismatch while the shared runtime-topology backbone stays frozen. That would
be a new preregistered adaptation experiment, not a reinterpretation of X1,
and it must compare the same target budget against scratch and frozen-head
controls before any remote scale-up.
