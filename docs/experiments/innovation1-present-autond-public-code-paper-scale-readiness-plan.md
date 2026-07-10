# Innovation 1 PRESENT AutoND Public-Code Paper-Scale Readiness Plan

**Date:** 2026-07-10

**Status:** implementation planned; no paper-scale run launched

**Claim scope:** public-code mechanism reproduction readiness, not a completed
paper-scale reproduction and not an Innovation 1 novelty result

## Goal

Add an isolated AutoND public-code data and evaluation protocol, prove it with
a tiny end-to-end smoke, and only then prepare one PRESENT-80 r9 paper-scale
run. Existing strict encrypted-random-plaintext experiments must remain
unchanged.

## Source Contract

Locked source references:

```text
paper:
  A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good
  Input Differences

public source commit:
  8d66e0a43a9e6fbbb3624d68296c1b77aa2da779

public training rows per round:
  NUM_SAMPLES = 10000000 total rows

public validation rows per round:
  NUM_VAL_SAMPLES = 1000000 total rows

public labels:
  random binary labels, not exactly balanced classes

public key protocol:
  one random key per row

public negative protocol:
  negative-class ciphertext input bits replaced with random bits

checkpoint and transition:
  best val_loss weights + one Adam/AMSGrad optimizer carried across rounds

PRESENT r9 budget:
  r5 -> r6 -> r7 -> r8 -> r9, 40 epochs per round

final evaluation:
  five fresh 1000000-row test sets; report mean accuracy
```

The paper text describes encrypted random-plaintext negatives, while the public
code uses random ciphertext bits. This plan reproduces the public code and must
be labeled `public_code_aligned`; it is not an exact paper-and-code reproduction.

## Architecture

Preserve the current balanced `samples_per_class` generator as the default.
Add an opt-in dataset mode:

```text
dataset_label_mode = random_labels_total
samples_total      = exact requested row count
```

In this mode, the generator samples one random label per row, uses a random key
for every row through `key_rotation_interval=1`, generates a real differential
pair for label 1, and emits random ciphertext-pair bits for label 0. Both the
in-memory smoke path and disk-backed remote path must expose identical metadata:

```text
samples_total
positive_rows
negative_rows
dataset_label_mode
key_schedule = per_row_random
```

The task runner receives independent split sizes:

```text
train_samples_total
validation_samples_total
final_test_samples_total
final_test_repeats
```

The existing top-level `metrics` remains the restored best-checkpoint validation
metric. A separate `final_evaluation` block stores every fresh-test metric and
the mean/std used for comparison with the paper.

## Frozen Tiny Smoke

```text
cipher                     = PRESENT-80
target round               = r9
curriculum                 = [5,6,7,8] -> 9
model                      = autond_dbitnet2023
input difference           = 0x000000000d000000
feature encoding           = ciphertext_pair_bits
pairs per row              = 1
dataset_label_mode         = random_labels_total
negative_mode              = random_ciphertext
key_rotation_interval      = 1
train_samples_total        = 32
validation_samples_total   = 16
epochs per round           = 1
optimizer transition       = carry_across_stages
checkpoint metric          = val_loss
final_test_repeats         = 5
final_test_samples_total   = 12 per repeat
device                     = cpu
```

The smoke is a protocol readiness check only. Its accuracy/AUC is not research
evidence.

## Tiny Smoke Gates

All gates are mandatory:

```text
train rows exactly 32 at r5-r9
validation rows exactly 16 at r5-r9
labels are random-label mode with both classes present for the frozen seeds
key schedule reports per_row_random
negative mode reports random_ciphertext
r5 optimizer reused=false and step_before=0
r6-r9 optimizer reused=true with continuous increasing steps
checkpoint metric is val_loss for r5-r9
five final test rows exist with 12 samples each
final accuracy mean/std equals aggregation of the five raw metrics
plan validation returns 1/1 rows and no mismatches
disk cache metadata matches exact totals and is reusable
```

## Paper-Scale Plan Gate

Only after the tiny smoke passes may a remote package request:

```text
train_samples_total        = 10000000 per round
validation_samples_total   = 1000000 per round
pretrain_round_sequence    = [5,6,7,8]
target round               = 9
epochs                     = 40
pretrain_epochs            = 40
final_test_repeats         = 5
final_test_samples_total   = 1000000 per repeat
seed                       = 0
dataset cache              = required, disk-backed, chunked, reusable
```

The remote readiness checker must reject a config that changes any locked
public-code field or omits disk-backed cache/progress.

## Implementation Plan

### Task 1: Exact-total random-label datasets

**Files:**

- Modify `src/blockcipher_nd/data/differential/config.py`
- Modify `src/blockcipher_nd/data/differential/generator.py`
- Modify `src/blockcipher_nd/data/differential/metadata.py`
- Modify `src/blockcipher_nd/data/differential/validation.py`
- Modify `src/blockcipher_nd/data/cache/disk.py`
- Modify `src/blockcipher_nd/engine/datasets.py`
- Test `tests/test_autond_public_protocol.py`

Steps:

1. Write failing tests for exact total rows, deterministic random labels,
   per-row key metadata, random-ciphertext negatives, disk creation, and reuse.
2. Run the new tests and confirm failures are caused by missing
   `samples_total`/`dataset_label_mode` support.
3. Add `samples_total: int | None` and `dataset_label_mode: str` to
   `DifferentialDatasetConfig`; defaults preserve all existing behavior.
4. Add a mixed-label chunk path to the disk cache without changing the current
   balanced positive-then-negative path.
5. Run the new dataset tests and existing cache-worker tests.

### Task 2: Split-size planning and progress metadata

**Files:**

- Modify `src/blockcipher_nd/engine/matrix_runner.py`
- Modify `src/blockcipher_nd/planning/matrix.py`
- Modify `src/blockcipher_nd/engine/task_config.py`
- Modify `src/blockcipher_nd/engine/task_runner.py`
- Modify `src/blockcipher_nd/engine/pretraining.py`
- Modify `src/blockcipher_nd/engine/progress.py`
- Modify `src/blockcipher_nd/engine/results.py`
- Test `tests/test_autond_public_protocol.py`

Steps:

1. Write failing parser and task-runner tests for exact train/validation totals.
2. Parse `train_samples_total`, `validation_samples_total`,
   `final_test_samples_total`, `final_test_repeats`, and `dataset_label_mode`
   from CSV/CLI while leaving historical plans unchanged.
3. Pass the same exact train/validation totals through every curriculum stage.
4. Record requested and observed totals plus label counts in progress and result
   metadata.
5. Run the new task-runner tests and the existing AutoND suite.

### Task 3: Five-fresh-test evaluation

**Files:**

- Create `src/blockcipher_nd/engine/final_evaluation.py`
- Modify `src/blockcipher_nd/engine/task_runner.py`
- Modify `src/blockcipher_nd/engine/results.py`
- Test `tests/test_autond_public_protocol.py`

Steps:

1. Write a failing end-to-end test requesting five fresh deterministic test
   datasets and checking their distinct seeds and exact row counts.
2. Implement `run_final_evaluation` using the restored target model,
   `make_task_dataset`, and `evaluate_binary_classifier`.
3. Return raw repeat metrics, exact test seeds, row counts, and accuracy/AUC
   mean and population standard deviation.
4. Keep validation metrics at top level and store paper-comparison metrics under
   `final_evaluation`.
5. Run the final-evaluation test and existing training metric tests.

### Task 4: Tiny smoke and paper-scale static package

**Files:**

- Create `configs/experiment/innovation1/innovation1_spn_present_autond_public_code_readiness_smoke_seed0.csv`
- Create `configs/experiment/innovation1/innovation1_spn_present_autond_public_code_paperscale_seed0.csv`
- Modify `src/blockcipher_nd/cli/check_remote_readiness.py`
- Modify this experiment record with smoke artifacts and gate outcome
- Test `tests/test_autond_public_protocol.py`

Steps:

1. Write failing plan/readiness tests that lock all frozen smoke and paper-scale
   fields.
2. Add both one-row plans and the AutoND public-code readiness invariant.
3. Run the CPU smoke with disk cache, progress JSONL, result validation, SVG,
   and history CSV.
4. Audit exact split counts, optimizer continuity, cache reuse, and five-test
   aggregation from generated artifacts.
5. If any readiness gate fails, stop before creating a remote launcher. If all
   pass, create and test the remote JSON/launcher/monitor from the pushed commit.

## Stop Rules

- Do not interpret tiny smoke metrics as evidence.
- Do not call a balanced `5000000/class` plan public-code exact.
- Do not launch paper scale if the exact-total disk cache cannot resume/reuse.
- Do not merge strict and public-code negative/key protocols in one result row.
- Do not reopen DDT or the held E1 graph route.
