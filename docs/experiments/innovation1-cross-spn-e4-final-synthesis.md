# Innovation 1 E4 Cross-SPN Final Evidence Synthesis

**Date:** 2026-07-15

**Status:** completed medium-scale synthesis / transfer scale path closed

## Question

Which effects of the cipher-spec-generated shared typed SPN operator survive
across the verified E4-R4 and E4-R5 one-epoch target-adaptation experiments?

This synthesis introduces no new training, samples, seeds, metric, or gate. It
compares the frozen paired-gate artifacts at the same GIFT-64 r6 target budget:

```text
train              = 65536/class = 131072 total rows/target seed
validation         = 32768/class = 65536 total rows/target seed
target epochs      = exactly 1
pairs/sample       = 4
negative           = encrypted random plaintexts
checkpoint         = restored best validation AUC
bootstrap          = 10000 paired label-stratified replicates
source seed0       = E4-R4 target seeds 2, 3
source seed1       = E4-R5 target seeds 4, 5
```

## Evidence

| Source seed | Target seed | True AUC | True - scratch | Scratch 95% CI | True - source-shuffled | True - target-shuffled | Frozen gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 2 | `0.579260635655` | `+0.011247730348` | `[+0.008470458165, +0.014073612948]` | `+0.012428269722` | `+0.077061415184` | pass |
| 0 | 3 | `0.583190341946` | `+0.006649116985` | `[+0.003760749672, +0.009526491968]` | `+0.009724145755` | `+0.081193963531` | pass |
| 1 | 4 | `0.576149932574` | `+0.000173664652` | `[-0.002284564741, +0.002628424205]` | `+0.015344345942` | `+0.069083069451` | fail scratch criterion |
| 1 | 5 | `0.573475843295` | `+0.003834810108` | `[+0.001243890217, +0.006373106083]` | `+0.013115312438` | `+0.070617836900` | fail `+0.004` point criterion |

## Result

Two effects separate cleanly:

1. **Robust typed-topology attribution.** True transfer beats the
   source-shuffled control by `+0.009724` to `+0.015344` and the
   target-shuffled control by `+0.069083` to `+0.081194` on all four target
   seeds. Correct source and target SPN topology therefore contribute beyond
   equal-capacity shuffled mappings across both independently trained source
   checkpoints.
2. **Conditional scratch efficiency.** True transfer beats scratch under the
   complete frozen gate only for the source-seed0 E4-R4 stratum. E4-R5 seed4
   is tied with scratch and seed5 misses the point threshold. Faster one-epoch
   adaptation is therefore conditional evidence, not a source-seed-robust
   result.

Source seed and target seeds are not fully crossed, so these four cells cannot
estimate a causal source-seed effect independently of target-seed variation.
That limitation does not invalidate the frozen confirmation decision: E4-R5
was the required independent source-plus-fresh-target audit, and it did not
meet the advance gate.

## Verdict

```text
keep  = shared typed SPN representation and source/target topology attribution
hold  = E4-R4 conditional one-epoch target-adaptation efficiency
stop  = source-seed-robust scratch-efficiency claim
stop  = E4-R6, 262144/class, 1000000/class, extra epochs/seeds, rescue scaling
scope = controlled remote medium diagnostic; not formal, paper-scale, SOTA,
        breakthrough, persistent superiority, or end-to-end compute evidence
```

## Artifacts

```text
outputs/remote_results/i1_gift64_cross_spn_target_adaptation_r4_65536_seed2/
outputs/remote_results/i1_gift64_cross_spn_target_adaptation_r4_65536_seed3/
outputs/remote_results/i1_gift64_cross_spn_source_seed_r5_65536_seed4/
outputs/remote_results/i1_gift64_cross_spn_source_seed_r5_65536_seed5/
outputs/remote_results/i1_gift64_cross_spn_source_seed_r5_65536_joint_seed4_seed5/
```

## Recommended Next Action

Do not allocate another training slot to this transfer claim. Use this frozen
table as the Innovation 1 E4 result and move to paper-ready method,
limitations, and variance reporting. A future experiment is admissible only
after a genuinely new source-objective hypothesis is ranked against this
same-budget anchor and passes a small local gate; adding samples, seeds,
epochs, or a crossed matrix solely to rescue E4-R4 is not a new hypothesis.
