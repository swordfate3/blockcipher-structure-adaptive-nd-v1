# Innovation 1 E5 Source-Objective Ranking

**Date:** 2026-07-15

**Status:** E5-R0 completed and rejected / source-objective ranking closed

## Decision Question

E4 established robust source- and target-topology attribution but did not
establish source-seed-robust one-epoch advantage over target scratch training.
The next question is:

> Does source-only PRESENT validation-AUC training optimize a representation
> that distinguishes the source task well but is not explicitly organized for
> rapid cross-SPN adaptation?

This is a source-objective question. It is not permission to rescue E4 by
adding target samples, epochs, seeds, or a larger crossed matrix.

## E4 Source Audit

The two typed-true PRESENT source runs used the same architecture, parameter
count, data protocol, optimizer, and checkpoint rule:

```text
architecture               = cross-SPN typed-cell pair-set
typed parameter count      = 187426
source train               = 8192/class
source validation          = 4096/class
pairs/sample               = 16
epochs                     = 10
checkpoint metric          = val_auc
checkpoint selection       = restored best
negative                   = encrypted random plaintexts
key schedule               = per-pair random
```

| Source seed | Typed-true source AUC | Best epoch | Mean GIFT true-scratch delta | Target seeds |
| ---: | ---: | ---: | ---: | --- |
| 0 | `0.743810147047` | `10/10` | `+0.008948423667` | 2, 3 |
| 1 | `0.755739122629` | `10/10` | `+0.002004237380` | 4, 5 |

The typed-true source AUC rose through epoch 10 for both seeds. Therefore E4
does not show that the source selector chose an earlier, obviously inferior
epoch. The mismatch hypothesis is broader: the source classification objective
and its `val_auc` selector may not measure post-adaptation utility.

The two source strata use different target seeds, and target seeds are not
fully crossed. The inverse descriptive ordering is therefore a hypothesis
signal only. It is not a correlation, source-seed effect, or causal result.

## Literature Boundary

### Neural-cryptanalysis transfer

Transfer learning, frozen feature extractors, layer freezing, and round-wise
retraining already exist in neural cryptanalysis. Gohr, Leander, and Neumann
also report a KATAN adaptation with a frozen pretrained representation and a
new linear classifier trained on `5 * 10^6` samples. Generic feature freezing
or pretraining is not an Innovation 1 novelty.

### Transferability estimation

LEEP, LogME, H-Score, TransRate, GBC, PACTran, and ETran estimate downstream
transferability from target representations, usually with target labels. They
are relevant audit methods but introduce target-selection leakage unless E5
creates a dedicated probe split that is disjoint from adaptation validation
and final evaluation. Comparative studies also show that their rankings change
with protocol details; no proxy can be trusted here without direct correlation
evidence.

### One-step meta-learning

MAML directly optimizes loss after one or more target gradient steps, and
Reptile/FOMAML provide first-order approximations. The mechanism matches E4's
one-epoch question, but the current task family has only PRESENT and GIFT. Two
ciphers are insufficient for a credible meta-training task distribution:
including GIFT leaks the target, while excluding it leaves no diverse SPN
meta-task family. MAML/Reptile is premature.

### Topology-counterfactual representation learning

GraphCL, TopoGCL, GCC, and related graph pretraining methods establish that
structural perturbations can define useful self-supervised or contrastive
views. Those general ideas are occupied. E5's admissible method contribution
is narrower:

```text
cipher-spec-generated typed SPN operators
+ deterministic true-versus-shuffled permutation counterfactual
+ strict differential-neural source task
+ held-out cross-SPN exactly-one-epoch adaptation gate
```

The prior E1 active-cell route already used a topology auxiliary head in a
different PRESENT r8 architecture and did not justify scale. E5 may reuse the
small, tested mechanism, but it must not claim that binary true/shuffled
auxiliary learning itself is new. The research question is whether applying
that objective to E4's robust shared typed representation improves cross-SPN
adaptation under equal-capacity controls.

## Ranked E5 Hypotheses

| Rank | Hypothesis | E4 fit | Main risk | Decision |
| ---: | --- | --- | --- | --- |
| 1 | Topology-counterfactual auxiliary source objective | Directly targets E4's 4/4 robust topology effect | Auxiliary task may be easy or repeat E1's weak route | Run E5-R0 local gate |
| 2 | Transferability-aware checkpoint audit | Directly tests source-AUC mismatch | Target-label leakage and unstable proxy rankings | Hold as E5-A1 audit |
| 3 | MAML/Reptile one-step source objective | Mechanistically aligned with rapid adaptation | Only two comparable SPN tasks; meta-overfitting/target leakage | Stop for current task family |
| 4 | Joint PRESENT+GIFT multitask pretraining | Could improve target performance | Uses the target during source training; not transfer evidence | Stop |
| 5 | More source seeds/checkpoints under unchanged objective | Reduces descriptive uncertainty | Mechanical E4 rescue with no new method | Stop |

## Selected Route

E5-R0 changes one factor: the source auxiliary topology-counterfactual
objective. The main classifier, data, source and target ciphers, optimizer,
budgets, checkpoint metric, and one-epoch target protocol remain fixed.

The candidate must beat both:

1. an auxiliary-off same-architecture anchor; and
2. an equal-capacity shuffled-versus-shuffled placebo objective.

Target scratch remains the adaptation-efficiency anchor. No labeled GIFT data
may influence source checkpoint selection.

## Sources

- `sources/research_e5_neural_cryptanalysis_transfer_20260715.md`
- `sources/research_e5_transferability_selection_20260715.md`
- `sources/research_e5_one_step_meta_learning_20260715.md`
- `sources/research_e5_topology_contrastive_pretraining_20260715.md`
- `sources/research_spn_adaptation_20260705.md`
- `sources/research_gift64_neural_comparison_20260714.md`
- `papers/innovation_one/text/2022_gohr_leander_neumann_assessment_differential_neural.txt`

## Recommended Next Action

E5-R0 completed and was rejected on both target seeds. The candidate did not
beat the auxiliary-off transfer anchor on either seed, and seed3 also failed
the shuffled-placebo control. Do not launch `65536/class`, train source seed1,
tune the auxiliary scale, or add target epochs.

The off role had both the highest PRESENT source AUC and the highest GIFT target
AUC on target seeds2 and 3. This small crossed result does not validate source
AUC as a universal transfer proxy, but it weakens the immediate motivation for
the held label-aware E5-A1 checkpoint audit. Keep E5-A1 held rather than using
GIFT labels to select source checkpoints.

The next admissible hypothesis must replace the easy topology-identity
auxiliary task with a label-preserving functional question: can the same
cryptanalytic classification loss make the true cipher-spec topology perform
better than a paired shuffled topology during source training? Rank and freeze
that E6-R0 objective against the existing E5 off anchor and a same-compute
shuffled-vs-shuffled placebo before implementation.
