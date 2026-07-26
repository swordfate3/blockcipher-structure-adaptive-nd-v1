# Innovation 1 Runtime-SPN C1 Topology-Only Dialga Holdout Plan

Date: 2026-07-26

```text
status = completed / hold
execution = local CPU sub-medium mechanism diagnostic
remote_scale = no
```

## Research Question

Does the shared Runtime-E4 backbone retain attributable cross-cipher GF(2)
diffusion-topology signal when every learned or explicit S-box conditioning
path is disabled?

A8 transferred a strong Dialga topology-dependent signal, but its correct
S-box did not beat a wrong S-box. S1 showed that the learned truth-table gate
was responsive but not semantically identifiable. S2 replaced it with an exact
inverse-S-box ANF contribution operator, but the correct operator still did not
beat input-permuted and identity controls. C1 therefore removes the rejected
S-box mechanism instead of adding another replacement.

This does not redefine the long-term method as topology-only. It asks whether
the already supported exact-GF(2) component is independently composable and
worth retaining while nonlinear primitive adaptation remains unresolved.

## One Variable And Same-Budget Anchor

The frozen A8 candidate is the same-budget anchor. C1 changes one functional
field:

```text
sbox_context_scale = 1.0 -> 0.0
```

The old edge-gate parameters remain in the state dictionary solely to preserve
the exact `442466`-parameter geometry, but multiplying their contribution by
zero makes the supplied S-box truth table functionally inert. No ANF operator,
Adapter, FiLM, typed relation residual, relation-mass pooling, target head or
cipher-specific expert is added.

Everything else remains frozen from A8:

```text
sources = GIFT-64 r6, SKINNY-64/64 r7,
          RECTANGLE-80 r6, uKNIT-BC prefix-r5
holdout = Dialga-128 prefix-r4
train = 2048/class/source
validation = 1024/class/source
pairs/sample = 4 independent ciphertext pairs
seeds = 0,1
epochs = 10
negative = encrypted random plaintexts
loss = MSE
optimizer = Adam, lr 1e-4, weight decay 1e-5
checkpoint = best four-source validation macro AUC
target training rows = 0
target optimizer steps = 0
```

This is a local mechanism diagnostic, not formal scale, paper scale, an attack,
SOTA evidence, a breakthrough or proof of arbitrary-SPN adaptation.

## Lean Matrix And Controls

Train exactly one new topology-only candidate per seed. Evaluate the restored
checkpoint on Dialga under:

| Evaluation | GF(2) structure | S-box truth table |
| --- | --- | --- |
| `candidate_correct` | correct | correct but functionally disabled |
| `candidate_corrupted_target` | deterministically corrupted | unchanged |
| `candidate_no_topology_target` | no topology | unchanged |
| `candidate_wrong_sbox_target` | correct | wrong but functionally disabled |

The wrong-S-box result must be numerically identical to the correct result;
this is an invariance check, not a competing performance control. Frozen A8
correct-candidate source/Dialga rows and its trained no-topology Dialga row are
copied as reference evidence without optimizer steps.

The artifact contract is:

```text
new source rows       = 2 seeds x 4 sources = 8
new Dialga rows       = 2 seeds x 4 evaluations = 8
frozen A8 source rows = 2 seeds x 4 sources = 8
frozen A8 target rows = 2 seeds x 2 references = 4
total result rows     = 28
history rows          = 2 seeds x 10 epochs = 20
```

## Readiness Gate

Before training require all of:

1. A8, S1 and S2 hashes and final decisions match the frozen evidence;
2. candidate and A8 use identical parameter/state geometry at `442466`;
3. all S-box and ANF functional scales in C1 are exactly zero;
4. changing only the S-box truth table produces zero logit change on all five
   ciphers;
5. correct, corrupted and no-topology relations produce distinct logits on
   all five ciphers;
6. pair swap and joint cell relabeling preserve logits within `1e-6`;
7. the four source caches and Dialga validation caches are complete and
   parameter-matched, while no Dialga training cache is referenced;
8. all outputs and representation gradients are finite;
9. the target is loaded only after source training and receives zero optimizer
   steps;
10. the 28-row result and 20-row history contracts are exact.

Any readiness failure permits only repair of the failed invariant.

## Completed Readiness

The real readiness gate completed before training:

```text
run_id = i1_runtime_spn_topology_only_c1_readiness_20260726
checks = 16/16 passed
candidate parameters = 442466
target training rows = 0
target optimizer steps = 0
decision = innovation1_runtime_spn_topology_only_c1_readiness_passed
```

Across GIFT, SKINNY, RECTANGLE, uKNIT and Dialga, replacing the runtime
S-box produced exactly zero logit change. Pair-swap error was exactly zero for
all five ciphers. uKNIT and Dialga cell-relabeling errors were respectively
`4.47e-8` and `2.24e-8`. Correct topology remained numerically distinct from
both corrupted and no-topology controls on every cipher. All 54 required cache
files were present, no Dialga training cache was referenced, exact-GF(2)
representation gradients were finite and nonzero, and the disabled S-box
encoder gradient was exactly zero.

## Research Gate

For each seed require:

```text
Dialga candidate correct AUC >= 0.55
correct - corrupted >= +0.005
correct - same-checkpoint no topology >= +0.005
correct - A8 trained no-topology anchor >= +0.005
candidate source macro - A8 correct source macro >= -0.005
candidate Dialga - A8 correct Dialga >= -0.005
max probability delta(correct, wrong S-box) <= 1e-7
at least one source-gradient conflict projection observed
```

Both seeds must pass every check for
`innovation1_runtime_spn_topology_only_dialga_supported`. A valid miss returns
`innovation1_runtime_spn_topology_only_dialga_not_supported` and stops this
topology-only cross-cipher branch. Protocol failure is separate and cannot be
interpreted as neural evidence.

## Required Artifacts

Readiness:

```text
outputs/local_readiness/i1_runtime_spn_topology_only_c1_readiness_20260726/
```

Diagnostic:

```text
outputs/local_diagnostic/i1_runtime_spn_topology_only_c1_2048_seed0_seed1_20260726/
  results.jsonl
  history.csv
  progress.jsonl
  validation.json
  gate.json
  summary.json
  checkpoints/
  role-results/
  curves.svg
  visual_qa_passed.marker
```

After a completed result, run `visual-qa-redraw`, refresh both recent-result
indexes, write exact metrics and the evidence-backed next action here, then
make a scoped commit and push.

## Completed Diagnostic

The frozen two-seed diagnostic completed with a valid protocol:

```text
run_id = i1_runtime_spn_topology_only_c1_2048_seed0_seed1_20260726
result rows = 28/28
history rows = 20/20
protocol_valid = true
target training rows = 0
target optimizer steps = 0
status = hold
decision = innovation1_runtime_spn_topology_only_dialga_not_supported
```

Source-macro retention was:

| Seed | C1 topology-only | A8 correct anchor | C1 - A8 | Gate |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.562074423 | 0.575356722 | -0.013282299 | fail |
| 1 | 0.554997683 | 0.548085093 | +0.006912589 | pass |

The zero-training Dialga results were:

| Seed | Correct | Corrupted GF(2) | Same-checkpoint no topology | Wrong S-box | A8 correct | A8 trained no topology |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.852658272 | 0.785620689 | 0.508895874 | 0.852658272 | 0.834955215 | 0.520171642 |
| 1 | 0.812206745 | 0.748097420 | 0.522912025 | 0.812206745 | 0.848253727 | 0.528065205 |

The corresponding margins were:

| Seed | Correct - corrupted | Correct - no topology | Correct - trained no topology | Correct - A8 correct | S-box probability delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.067037582 | +0.343762398 | +0.332486629 | +0.017703056 | 0.0 |
| 1 | +0.064109325 | +0.289294720 | +0.284141541 | -0.036046982 | 0.0 |

Both seeds therefore retain a large, clean GF(2)-topology-dependent Dialga
signal, and the S-box nuisance has been removed exactly. C1 nevertheless does
not pass the preregistered simplification gate: seed 0 loses `0.013282` source
macro AUC relative to A8, while seed 1 loses `0.036047` Dialga AUC relative to
A8. The two failures occur on different evidence strata, so the absolute
topology margins cannot override them.

The final SVG passed the `visual-qa-redraw` rendered-pixel gate at 1800 px and
1280 px widths. The first version omitted the source-retention failure from the
margin panels, and the second allowed threshold lines to cross small-difference
labels. The final rendering shows both failed checks explicitly and has no
overlap, clipping, ambiguous title, hidden legend or insufficient control
separation.

This is a local `2048/class/source` mechanism diagnostic. It neither negates
the completed two-seed `1000000/class` SKINNY Runtime-E4 topology result nor
supports a universal cross-cipher claim.

## Evidence-Backed Next Action

Close topology-only whole-cipher transfer at this budget and do not remotely
scale or tune C1. Retain the narrower evidence that Runtime-E4 uses externally
supplied exact GF(2) topology on SKINNY at the completed two-seed
`1000000/class` project-formal scale and produces strong but not anchor-stable
local Dialga topology sensitivity.

The next Innovation 1 task is a zero-training, requirement-by-requirement
method-boundary audit. It must synthesize the formal SKINNY gate, GIFT/PRESENT
and RECTANGLE evidence, uKNIT/Dialga holdouts, S1/S2 and C1 into an explicit
matrix of supported, contradicted and missing requirements. The exact decision
to unlock is whether the thesis method should be frozen as a runtime
GF(2)-topology-aware SPN distinguisher with unsupported nonlinear composability,
or whether a genuinely new representation hypothesis remains that addresses
both nonlinear semantic identifiability and whole-cipher anchor stability.

Do not run more samples, epochs or pairs; do not add S-box, ANF, DDT,
inverse-triplet, delta-U, Adapter, FiLM, typed residual, MoE, target head or
target-supervision rescue before that audit identifies a non-duplicative gap.
