# Innovation 1 H1-A1 Source Gradient And Alignment Audit Plan

Date: 2026-07-26

```text
status = completed / pass
execution = local no-training audit
remote_scale = no
```

## Research Question

Did H1 seed 1 fail stable RECTANGLE topology attribution because one of the
four source tasks dominated or conflicted with the shared representation
gradient, or does the frozen checkpoint instead point to a representation
alignment limitation?

H1 trained the same Runtime-E4 classifier on GIFT, SKINNY, uKNIT and Dialga,
then evaluated RECTANGLE with zero target rows and zero fine-tuning. Seed 0
passed all target controls, while seed 1 retained `0.588227` absolute target AUC
but failed correct-topology attribution. The seed-1 source macro hid Dialga
`0.942334` beside GIFT `0.477778`, SKINNY `0.534819` and uKNIT `0.483970`.

## Frozen Evidence

```text
source = completed H1 output and exact config SHA256
checkpoints = seed0-correct.pt, seed1-correct.pt
model = unchanged conditioner-free Runtime-E4, 442466 parameters
tasks = GIFT-64 r6, SKINNY-64/64 r7, uKNIT prefix-r5, Dialga prefix-r4
data = exact cached train split, 4096 total rows per cipher
loss = MSE
batch = 256
optimizer steps = 0
target rows = 0
```

No checkpoint, model parameter, input row, label, topology, loss or source AUC
may be changed. RECTANGLE is not loaded by this audit.

## Measurements

For each seed and source task, compute the mean full-split loss gradient for:

1. `representation_backbone`: all shared Runtime-E4 parameters except the
   final classifier;
2. `classifier`: the shared final classifier only;
3. `all_parameters`: the complete shared model.

Record L2 norm, normalized four-task norm share, largest-task/other-median norm
ratio, every pairwise cosine and the already completed H1 per-cipher validation
AUC. The representation view is the preregistered primary gate.

## Decision Gate

For the failing H1 seed, call task-gradient imbalance only if the largest source
task has either:

```text
gradient share >= 0.50
or
norm / median(other three norms) >= 2.0
```

Call a failing-seed conflict only when a pairwise representation-gradient cosine
is at most `-0.10`. Call a stable conflict only when the same source pair has
negative cosine in both seeds. The source difficulty imbalance must retain an
AUC range of at least `0.30` before either mechanism can authorize a training
follow-up.

Outcomes:

- task-gradient imbalance: preregister one same-budget source-gradient
  normalization gate;
- reproducible gradient conflict: preregister one same-budget conflict treatment
  while keeping task weights and model fixed;
- neither: do not train; audit per-cipher representation geometry and classifier
  accessibility instead;
- protocol invalid: repair only the exact checkpoint/cache/source mismatch.

## Blocked Actions

Do not add MoE, Adapter, FiLM, typed GNN, cipher IDs, target heads, samples,
epochs or remote compute. Do not use this audit as unseen-cipher performance or
claim that a checkpoint gradient reconstructs the complete training trajectory.

## Required Artifacts

```text
results.jsonl
gradient_norms.csv
gradient_cosines.csv
source_auc.csv
validation.json
gate.json
summary.json
progress.jsonl
curves.svg
```

## Evidence-Backed Next Action

Execute the frozen audit locally. The observed primary-view gate selects exactly
one next route: source-gradient normalization, conflict treatment, or a
no-training representation-alignment audit. Mechanical scale-up and target
supervision remain prohibited under every outcome.

## Completed Result

The frozen audit completed at:

```text
outputs/local_audit/i1_runtime_spn_h1_source_gradient_alignment_a1_seed0_seed1_20260726/
status = pass
decision = innovation1_runtime_spn_h1_source_gradient_imbalance_supported
results rows = 68
gradient norm rows = 24
gradient cosine rows = 36
source AUC rows = 8
training or optimizer steps = 0
RECTANGLE rows loaded = 0
```

All eleven checkpoint, cache, source-evidence, finite-gradient, row-count and
zero-training checks passed. The two correct H1 checkpoints retained their
frozen SHA256 values, and every source cache supplied exactly 4096 rows.

Primary representation-gradient evidence:

| Seed | Largest task | Gradient share | Norm / other-task median | Source AUC range |
| --- | --- | ---: | ---: | ---: |
| 0 | Dialga | 0.699065 | 6.209037 | 0.432147 |
| 1 | Dialga | 0.858244 | 18.636338 | 0.464556 |

The seed-1 SKINNY/Dialga representation-gradient cosine was `-0.148227`. The
same pair was already slightly negative for seed 0 at `-0.039034`, satisfying
the preregistered stable-negative check. GIFT/Dialga and uKNIT/Dialga changed
from positive seed-0 alignment to approximately orthogonal or weakly negative
seed-1 alignment (`0.051` and `-0.097`).

The result supports a concrete optimization imbalance mechanism: Dialga owns
most of the shared representation-gradient norm even though every source task
has equal loss weight. It does not prove that endpoint gradients reconstruct
the complete training trajectory, and it is not new unseen-cipher performance.

## Final Next Action

Preregister one local H1-A2 candidate that changes only the per-step source
gradient combination. For the shared representation parameters, rescale each
task gradient to the four-task mean L2 norm before averaging; retain the raw
arithmetic mean for the shared classifier. Preserve the H1 model, initialization,
data, task sampling, validation, labels, negatives, batch, learning rate,
epochs, checkpoint selection and RECTANGLE zero-leakage evaluation.

Train only the two candidate seeds. Compare them with the completed H1 correct
checkpoints and evaluate the candidate with same-checkpoint correct,
corrupted-target and no-topology-target counterfactuals. Advance only if source
task balance improves without losing seed-0 holdout signal and both seeds pass
the original target AUC and topology margins. Do not add a second model change,
remote scale or target supervision.

## Visual QA Result

The first render exposed a misleading fixed `0.7` gradient-share axis that
clipped the seed-1 Dialga share of `0.858`. The plot was corrected to a `0--1`
axis and exact gradient-share/AUC bar labels were added. The final `curves.svg`
was rendered and inspected at `1800x1059` and `1280x753`; both passed overlap,
clipping, title, axis, heatmap, colorbar, numeric-label and readability checks
under `visual-qa-redraw`.
