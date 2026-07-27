# Innovation 1 uKNIT-Family CT-SPN Linear Schedule K1

Date: 2026-07-27

```text
status = implementation and execution path ready / PRESENT launch interlock pending
run_id = i1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1_20260727
execution = local sub-medium diagnostic after active PRESENT evidence closes
remote = no
```

## 1. Question

On the same uKNIT-r5 and Dialga-r4 differential datasets, does exact alignment of
each heterogeneous transition into its canonical linear-primitive coordinates make
the real transition schedule more useful than Runtime-E4's raw-coordinate recurrent
sum?

K0 passed all exact component-factorization checks. K1 tests one new learned
hypothesis only:

```text
raw-coordinate recurrent transition fusion
  -> canonical exact-state views + shallow order-sensitive temporal convolution
```

Learned S-box truth-table conditioning is disabled in both roles. K1 therefore does
not reopen the rejected truth-table MLP, ANF, Adapter, FiLM, typed-GNN or learned-MoE
routes.

## 2. Frozen Data Protocol

| Field | uKNIT-BC | Dialga-128 |
|---|---:|---:|
| Prefix rounds | 5 | 4 |
| Runtime window | start 3, length 2 | start 2, length 2 |
| Input difference | `0x40` | `0x40` |
| Training | `2048/class` = 4096 total | `2048/class` = 4096 total |
| Validation | `1024/class` = 2048 total | `1024/class` = 2048 total |
| Pairs per sample | 4 | 4 |
| Seeds | 0, 1 | 0, 1 |
| Epochs | 10 | 10 |
| Batch size | 64 | 64 |
| Loss / optimizer | MSE / Adam | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC | best validation AUC |
| Negative definition | encrypted random plaintexts | encrypted random plaintexts |

Training keys remain all-zero at each cipher's real key width. Validation keys remain
all-`1` at the same real width. Existing same-protocol disk caches must be reused or
regenerated with parameter-matched metadata; labels, splits, key semantics and metric
calculation may not change.

## 3. Lean Training Matrix

Train exactly two roles per cipher and seed, eight rows total:

1. `Runtime-E4 recurrent-window anchor`: existing raw-coordinate state-triplet
   backbone, S-box learned path disabled.
2. `CT-SPN K1 candidate`: exact canonical transition views, shared 12-value
   primitive-edge encoder, invariant edge pooling, kernel-three order-sensitive
   temporal convolution, pair pooling and a shared classifier.

The anchor is frozen by architecture, not selected after looking at each seed. The
candidate must have the same trainable parameter shape for 64- and 128-bit inputs;
its trainable parameter count must differ from the `442466`-parameter anchor by at
most one percent.

The CT-SPN processor receives no cipher name, cipher id, round count or global
fingerprint. Exact non-trainable factor buffers may vary with the supplied runtime
operators. State width and transition length are handled by shared operations and
invariant pooling. The implemented candidate has `438702` trainable parameters,
`0.851%` fewer than the `442466`-parameter anchor.

### 3.1 Readiness-Directed Representation Correction

The initial draft grouped each recovered canonical state into consecutive four-bit
cells. Zero-training cell-relabel probes showed that this was not well-defined:
the MIDORI-family linear graph has many valid input/output automorphisms, and the
first graph-isomorphism solution need not preserve the native S-box cell partition.
Adding cell constraints made the highly symmetric isomorphism search intractable and
would incorrectly assume that uKNIT's verified factorization preserves native bit
roles.

K1 therefore represents every nonzero edge of the canonical GF(2) primitive as one
token. Each token contains the left, right and XOR values at its canonical output
and input endpoints plus deterministic product/XOR interactions. An equivariant edge
mixer and mean/max/RMS pooling produce one embedding per real transition before the
temporal convolution. Any alternative valid factorization only permutes canonical
edges, so the pooled representation is invariant while transition order remains
observable. This remains a linear-schedule-only change; learned S-box semantics stay
disabled.

## 4. Frozen-Checkpoint Controls

After training, load each selected checkpoint without optimizer steps and evaluate
the exact same validation rows under:

```text
correct ordered transition factors
same-length repeated-last factors
shuffled transition order
deterministically corrupted topology (seed 20260727)
no topology / identity traversal
```

Controls must not retrain or select a new checkpoint. They change only non-trainable
runtime operators/factor buffers. Data, labels, classifier and learned parameters
remain identical. Both anchor and candidate checkpoints are evaluated so a generally
easy control cannot be mistaken for candidate-specific attribution.

## 5. Readiness Gate

The zero-training implementation readiness requires:

1. K0 validation and gate pass from result-index entry `001`;
2. all eight plan rows parse, build and reuse identical model geometry across the two
   ciphers;
3. candidate parameter difference from the anchor is at most one percent;
4. candidate canonical views reconstruct the native previous state for every unit
   bit of both loaded transitions;
5. cell relabeling leaves pooled logits invariant within `1e-6`;
6. correct, repeated, shuffled, corrupted and no-topology schedules have deterministic
   fingerprints and the required wrong controls are distinct;
7. a strict state-dict load succeeds between correct/control instances of the same
   candidate checkpoint while factor buffers stay control-specific;
8. no cipher-name or cipher-id tensor enters the model;
9. the anchor's learned S-box path has zero gradient contribution at scale zero.

Separately, the launch interlock requires the active PRESENT formal seed1 result to be
locally retrieved and adjudicated before the first K1 optimizer step. Implementation
readiness may pass while this external interlock remains false; that state authorizes
no training.

An implementation failure permits only implementation/protocol repair. A passed
implementation gate with a pending launch interlock preserves zero optimizer steps.

### 5.1 Completed Readiness Result

The zero-training readiness completed on 2026-07-27:

```text
run_id = i1_uknit_family_ctspn_linear_schedule_k1_readiness_20260727
status = pass
decision = innovation1_uknit_family_ctspn_k1_readiness_passed_waiting_present
implementation_ready = true
optimizer_step_authorized = false
training_rows / optimizer_steps = 0 / 0
candidate / anchor parameters = 438702 / 442466
```

Both 64- and 128-bit candidates have identical learned state-dict geometry. Unit-bit
inverse and canonical-edge reconstruction checks passed for both loaded transitions;
all five control fingerprints were deterministic and distinct; the same checkpoint
loaded strictly into every control without overwriting factor metadata. Reversing all
cell labels changed pooled logits by at most `7.0780516e-8`, below `1e-6`. The only
closed item is the existing PRESENT formal seed1 result, which is not yet locally
retrieved and adjudicated.

### 5.2 Completed Execution-Path Readiness

The guarded K1 runner and postprocessor are implemented without consuming the
launch authorization:

```text
scripts/run-uknit-family-ctspn-k1
scripts/plot-uknit-family-ctspn-k1
```

The runner recomputes K0/K1/PRESENT preflight evidence before creating a run root.
It returns without writing run artifacts when `optimizer_step_authorized=false`.
After authorization, it freezes `batch_size=64`, invokes the existing matrix trainer
for exactly eight rows, writes one best-AUC checkpoint per row, and reuses one
in-memory validation dataset per cipher and seed for all counterfactual evaluations.

Both the Runtime-E4 anchor and CT-SPN candidate are evaluated under all five
conditions. This produces `2 ciphers x 2 seeds x 2 trained roles x 5 controls = 40`
inference-only rows. Every row records the validation-data digest, checkpoint and
state-dict digests, control fingerprint, probability digest, AUC and zero optimizer
steps. The result gate fails closed unless the correct condition exactly replays the
source best-checkpoint AUC and every per-source control shares the same learned state
and validation rows.

Ten focused tests cover readiness, cross-width geometry, transition rotation,
strict five-control state reuse, fail-closed launch, per-cipher/per-seed adjudication
and complete Chinese chart labels. This is execution readiness only; no K1 training
has started.

## 6. Advance Gate

The candidate advances only if both seeds pass every per-cipher condition:

```text
uKNIT candidate AUC >= 0.520
Dialga candidate AUC >= 0.550
candidate - raw Runtime-E4 anchor >= +0.005
candidate - repeated-last >= +0.005
candidate - shuffled-order >= +0.005
candidate - corrupted topology >= +0.005
candidate - no topology >= +0.005
```

No macro average may hide a uKNIT or Dialga failure. A valid miss is `hold`, not a
definitive ceiling: this is `2048/class`, local diagnostic evidence only.

## 7. Artifacts

```text
outputs/local_readiness/i1_uknit_family_ctspn_linear_schedule_k1_readiness_20260727/
outputs/local_diagnostic/i1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1_20260727/
```

The completed diagnostic must emit:

```text
results.jsonl              = exactly 8 trained rows
controls.jsonl             = exactly 40 frozen-checkpoint inference rows
history.csv                = exactly 80 epoch rows
checkpoint_manifest.json   = exactly 8 selected best-AUC checkpoints
preflight.json             = source hashes and launch authorization
validation.json            = fail-closed protocol checks
gate.json                  = per-cipher, per-seed research adjudication
summary.json               = claim scope and evidence-backed next action
progress.jsonl             = training and frozen-control progress
curves.svg                 = Chinese AUC and attribution-margin comparison
plot_report.json           = rendered-pending-visual-QA state
```

The chart must pass the `visual-qa-redraw` workflow before completion. Both completed
readiness and training results must refresh the recent-results index.

### 7.1 Guarded Execution

After a real PRESENT seed1 adjudication is locally retrieved, run:

```bash
RUN_ROOT=outputs/local_diagnostic/i1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1_20260727

UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --no-sync python \
  scripts/run-uknit-family-ctspn-k1 \
  --plan configs/experiment/innovation1/innovation1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1.csv \
  --k0-gate outputs/local_audit/i1_uknit_family_canonical_component_factorization_k0_20260727/gate.json \
  --k0-validation outputs/local_audit/i1_uknit_family_canonical_component_factorization_k0_20260727/validation.json \
  --present-gate <retrieved-present-seed1-result-gate.json> \
  --output-root "${RUN_ROOT}" \
  --device cpu

UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --no-sync python \
  scripts/plot-uknit-family-ctspn-k1 \
  --gate "${RUN_ROOT}/gate.json" \
  --output "${RUN_ROOT}/curves.svg" \
  --report "${RUN_ROOT}/plot_report.json"
```

Then invoke `visual-qa-redraw`, mark the visual gate only after rendered-pixel
inspection passes, and refresh `outputs/00_RECENT_RESULTS.md` plus its JSON companion.
The run is incomplete if training, frozen replay, adjudication, plotting, visual QA,
or indexing fails.

## 8. Evidence-Dependent Next Action

- **Both seeds pass on both ciphers:** retain CT-SPN linear-schedule fusion and plan
  K2 as a separate exact MANTIS S-box composition experiment.
- **uKNIT fails but Dialga passes:** hold K1; inspect canonical edge/transition
  alignment only. Do not add capacity, samples or a learned expert.
- **Dialga fails but uKNIT passes:** hold the two-cipher family claim; audit whether
  the 128-bit primitive-edge summary loses necessary byte-local information.
- **Controls fail attribution:** discard the neural interpretation even if absolute
  AUC is high.
- **Protocol invalid:** repair and rerun unchanged.

Do not remotely scale K1, start K2, train a learned MoE, change the input difference,
or add another cipher from this diagnostic alone.
