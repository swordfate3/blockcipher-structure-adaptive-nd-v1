# Innovation 1 GIFT-64 Mainstream Performance 1M Plan

**Date:** 2026-07-15

**Status:** local readiness passed / remote launch assets ready

## Research Question

At a fixed GIFT-64 r6 task and a large `1000000/class` target budget, is the
cipher-spec-generated typed SPN model competitive with mainstream LSTM and
residual-CNN families, and does PRESENT initialization retain any one-epoch or
fully-trained advantage over the same typed architecture from scratch?

This is a new performance benchmark question. It is not a mechanical E5/E6
source-objective scale-up and it does not overwrite the completed E4 topology
attribution result.

## Literature And Current Evidence

The credible external GIFT-64 r6 reference currently held by the project is
Sun et al.'s LSTM result: accuracy `0.5754`, about `17M` training samples and
`4M` validation samples. Its `3-2-CT-R` data structure is not identical to the
project's four-pair task, so that number remains an external reference rather
than a direct gate.

E4 used `65536/class` and exactly one target epoch. Its four typed-transfer
cells reached AUC `0.5735` to `0.5832`, but scratch efficiency passed only two
cells. It did not include a same-protocol mainstream architecture. The new run
fills that missing comparison without claiming an exact Sun-paper reproduction.

## One Variable And Anchors

The benchmark keeps the E4 GIFT task fixed and changes the evidence scale plus
the comparison set:

```text
target scale = 1000000/class
target epochs = 10
target seeds = 6, 7
source checkpoints = frozen PRESENT seeds 0 and 1
same-architecture anchor = typed scratch
mainstream families = Sun-style bidirectional LSTM and Gohr-style ResNet
```

The LSTM and ResNet are pair-set adaptations on the exact project input. They
encode each raw 128-bit `C || C'` pair as 64 aligned two-bit sequence steps,
then aggregate four pair embeddings with permutation-invariant mean/max
pooling. Their parameter counts must stay within 25% of the typed model. They
are architecture-family baselines, not source-code-exact paper reproductions.

## Frozen Data Protocol

```text
cipher                         = GIFT-64
rounds                         = 6
input difference               = 0x0000000000000040
difference profile             = gift64_shen2024_spn_screen member0
pairs/sample                   = 4 independent pairs
feature                        = raw ciphertext_pair_bits, 512 bits/sample
negative                       = encrypted random plaintexts
train                          = 1000000/class = 2000000 total rows
validation                     = 500000/class = 1000000 total rows
fresh final test               = 5 repeats x 500000/class
train key                      = 0x00000000000000000000000000000000
validation key                 = 0x11111111111111111111111111111111
final-test key                 = 0x22222222222222222222222222222222
batch size                     = 1024
checkpoint                     = best validation AUC, restored
disk cache/progress/reuse      = required
```

All five rows within a target seed reuse the identical parameter-matched train,
validation, and fresh-test caches. Validation selects checkpoints; the five
fresh test repeats provide final performance means and variance.

## Lean Five-Role Matrix

| Role | Initialization | Purpose |
| --- | --- | --- |
| typed scratch | scratch | same-architecture anchor |
| typed transfer source0 | frozen PRESENT seed0 | conditional strong E4 stratum |
| typed transfer source1 | frozen PRESENT seed1 | source robustness stratum |
| Sun-style LSTM pair-set | scratch | credible GIFT literature-family baseline |
| Gohr-style ResNet pair-set | scratch | standard neural-differential residual baseline |

No E5/E6 objective, shuffled objective placebo, deterministic feature route,
ensemble, DDT branch, or extra architecture sweep is included.

## Training And Decision Axes

Typed rows retain the frozen E4 optimizer (`MSE`, Adam, `1e-4`, no scheduler).
The scratch mainstream baselines use the same data and epochs with a standard
cyclic `1e-4` to `2e-3` schedule. Training cost, parameter count, selected
epoch, and all ten validation histories must be reported.

Two claims are evaluated separately:

```text
adaptation efficiency = epoch1 validation AUC versus typed scratch
final performance     = mean AUC over five fresh final-test repeats
```

Per target seed, let the strongest mainstream baseline be the larger fresh-test
mean AUC of LSTM and ResNet.

```text
mainstream superiority:
  both source0 and source1 typed transfer >= strongest mainstream + 0.002 AUC

mainstream competitiveness without superiority:
  both typed transfers >= strongest mainstream - 0.001 AUC

persistent transfer value:
  both typed transfers >= typed scratch + 0.002 fresh-test AUC

one-epoch source-robust adaptation:
  both typed transfers >= typed scratch + 0.004 epoch1 validation AUC
```

The joint result requires both target seeds. A point-estimate pass is a
large-scale performance candidate, not yet a paper-exact or SOTA claim. The
primary fresh repeat also exports aligned labels, sample IDs, logits, and
probabilities so a paired interval can be added without retraining.

## Advance And Stop Rules

If both target seeds pass mainstream superiority and primary-score alignment,
the next action is a paired confidence-interval adjudication followed by an
exact Sun-protocol reproduction plan at the published sample totals.

If the typed model is competitive but not superior, retain the architecture as
a controlled structure-adaptive method and report no accuracy lead. Do not add
models or tune after seeing the result.

If either target seed is below both mainstream baselines beyond the
noninferiority tolerance, stop the performance-lead claim. Retain only the E4
topology-attribution result; do not rescue E5/E6 or mechanically increase to
`5M/class`.

Any cache, source-hash, key, score-alignment, checkpoint, plan-alignment, or
archive error makes the run invalid rather than negative.

## Execution Path

1. Run the five-role `64/class` CPU readiness matrix.
2. Validate five rows, checkpoint creation, strict source SHA-256 loading,
   final-test key separation, and pair-set model geometry.
3. Commit and push the frozen plan and implementation.
4. Launch target seed6 on A6000 GPU0 and target seed7 on GPU1 from the pushed
   commit in independent run-owned clones under `G:\lxy`.
5. Use disk-backed caches and progress logs from the first generated chunks.
6. Export primary fresh-test scores, build per-seed gates, push curated result
   branches, and let the local tmux watcher retrieve and index results.

Planned run IDs:

```text
i1_gift64_mainstream_performance_1m_seed6
i1_gift64_mainstream_performance_1m_seed7
```

## Recommended Next Action

Launch both remote target seeds only after local readiness and pushed-commit
gates pass. While they run, do not start another Innovation 1 scale job. The
result-driven next action is fixed by the advance/stop rules above.

## Readiness Completion

The local five-role CPU readiness run completed from the frozen plan:

```text
run_id          = i1_gift64_mainstream_performance_readiness_seed6
result rows     = 5/5
validation      = pass, errors=[]
source seed0    = strict checkpoint SHA-256 load passed
source seed1    = strict checkpoint SHA-256 load passed
final-test key  = distinct key and fresh split passed
plot/index      = generated and refreshed
```

Artifacts:

```text
outputs/local_smoke/i1_gift64_mainstream_performance_readiness_seed6/
```

Both remote configs also pass `scripts/check-remote-readiness` with five plan
rows, `1000000/class`, disk-backed cache requirements, and no readiness errors.
The next executable action is commit/push followed by the two-GPU launch from
the exact pushed commit.
