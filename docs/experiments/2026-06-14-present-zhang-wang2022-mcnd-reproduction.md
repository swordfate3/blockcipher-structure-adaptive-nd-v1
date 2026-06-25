# Zhang/Wang 2022 PRESENT MCND Reproduction Track

This note records the project-local reproduction scaffold for Zhang/Wang 2022,
`Improving Differential-Neural Distinguisher Model For DES, Chaskey, and PRESENT`
(arXiv:2204.06341). The aim is to separate a literature reproduction baseline
from innovation-one structure-adaptive experiments.

## Literature Target

The paper reports improved differential-neural distinguishers for 6-7 round
PRESENT. The PRESENT input difference is represented as four 16-bit words:

```text
(0, 0, 0, 0x9)
```

In this project this is encoded as the 64-bit xor difference:

```text
0x0000000000000009
```

The reproduction scaffold uses the model family:

```text
present_inception_mcnd
```

with configurable Inception branch kernel sizes. For the Zhang/Wang track, the
plan passes:

```json
{"kernel_sizes": [1, 2, 4], "blocks": 3, "dropout": 0.0, "pooling": "attention_mean_max"}
```

## Source And Trust Boundary

Primary paper:

```text
https://arxiv.org/abs/2204.06341
```

The public source-code link found from the paper metadata is a Google Drive
folder, not a GitHub repository:

```text
https://drive.google.com/drive/folders/1i0RciZlGZsEpCyW-wQAy7zzJeOLJNWqL?usp=sharing
```

Current project status:

```text
verification_status: official_code_history_checkpoint_and_smoke_eval_completed
official_saved_history_best_val_acc: 0.722563982
official_best_checkpoint: DATA_Nm_good_trained_nets/present_best_7r_pairs16_distinguisher.h5
official_best_checkpoint_size_bytes: 8067280
official_checkpoint_smoke_eval_raw_pair_count: 16000
official_checkpoint_smoke_eval_grouped_rows: 1000
official_checkpoint_smoke_eval_accuracy: 0.755
official_checkpoint_confirm_eval_raw_pair_count: 160000
official_checkpoint_confirm_eval_grouped_rows: 10000
official_checkpoint_confirm_eval_accuracy: 0.7131
official_checkpoint_large_eval_raw_pair_count: 1000000
official_checkpoint_large_eval_grouped_rows: 62500
official_checkpoint_large_eval_accuracy: 0.721536
project_reproduction_status: official_checkpoint_eval_confirmed_at_reference_level_on_62500_grouped_rows_not_from_scratch_training
strict_reference: Zhang/Wang 2022 Table 4, PRESENT-80, 7-round Case2, m=16, accuracy 0.7205
```

The `0.7205` value is now better than a bare paper citation: the official Drive
code and the saved Case2 r7/m16 training history were located, and that saved
history reports `max(val_acc) = 0.722563982`. On 2026-06-21, the official Case2
r7/m16 best checkpoint was also located and downloaded from the official Drive
snapshot into the isolated audit directory. On the remote Windows workstation,
the checkpoint was loaded with the official `present.py` data generator. A small
smoke evaluation and a larger `10000` grouped-row confirmation evaluation were
completed. This still is not an independently verified paper-scale reproduction
because the confirmation run is below the official validation/evaluation scale.
This distinction is intentionally mirrored in the project evidence-gate rules in
`AGENTS.md` and the Innovation 1 experiment configuration notes under
`configs/experiment/innovation1/`.

Update on 2026-06-24: the official Case2 r7/m16 checkpoint was evaluated again
on `1,000,000` raw official-protocol pairs, corresponding to `62,500` grouped
rows at `m=16`. The checkpoint reached `accuracy = 0.721536`, which is
`+0.001036` above the Table 4 reference `0.7205`. This confirms the official
checkpoint and official data generator can reproduce the reported 7-round
PRESENT signal at a substantially larger independent evaluation scale than the
earlier 10k grouped-row check. It is still a checkpoint reproduction, not a
from-scratch paper-scale retraining run.

## Official Code Audit Snapshot

Audit date: 2026-06-19.

Checkpoint follow-up date: 2026-06-21.

Downloaded audit copy:

```text
/tmp/zhang_wang2022_code_audit
```

Located source files and hashes:

```text
252c271c37bc9ef6a1a3d3f5fd0d595b13733c07c49f9af1e01a49d688063965  deep_net_present.py
2a882cf24711483d977c27c139ca675af57224ee284d3e0408e84e777f8854c6  eval.py
171a07dbb9fb1d0fb379d5a82dcec19bc453a5dca11df00d0e249449d8fb446e  present.py
```

Located official Case2 r7/m16 checkpoint:

```text
Drive file id: 1T_TWw-eJVZYS5Uh7nDxCIwVp0KKY1xB-
Downloaded path: /tmp/zhang_wang2022_code_audit/DATA_Nm_good_trained_nets/present_best_7r_pairs16_distinguisher.h5
Size: 8067280 bytes
```

Project wrapper:

```text
scripts/evaluate-zhang-wang-checkpoint
outputs/zhang_wang_official_checkpoint_audit.json
outputs/official_checkpoint_audit.json
outputs/official_checkpoint_eval_smoke_16000.json
outputs/official_checkpoint_eval_160000.json
```

Remote TensorFlow environment used for the smoke evaluation:

```text
remote root: G:\lxy\zhang_wang2022_code_audit
python: F:\Anaconda\envs\tensorflow_gpu2\python.exe
tensorflow: 2.6.0
h5py: 3.1.0
```

The relevant official configuration is:

```python
train_present_distinguisher(
    20,
    num_rounds=7,
    diff=0x9,
    group_size=16,
    depth=5,
)
```

Official Case2 corresponds to `DATA_Nm_good_trained_nets`. The saved
`present_hist7r_pairs16_nm.p` history reports:

```text
max(val_acc) = 0.7225639820098877
argmax = epoch 20
min(val_loss) = 0.18307027220726013
argmin = epoch 20
```

Important protocol notes from the official code:

- Positive pair: `x1 = x0 ^ 0x9`.
- Negative pair: `x1` is another random plaintext which is then encrypted; this
  is not random ciphertext.
- Each basic pair receives an independent random PRESENT-80 key. The `m=16`
  sample is made by reshaping independent pairs into a group, not by sharing one
  key or one base plaintext across all 16 pairs.
- The training script prints `max(val_acc)` while the checkpoint is selected by
  `val_loss`.
- The default `eval.py` points to `DATA_N_good_trained_nets`; Case2 evaluation
  requires switching it to `DATA_Nm_good_trained_nets` or using a wrapper.
- Local checkpoint/test evaluation was not run because the current local
  environment lacks TensorFlow/Keras/HDF5. The remote `torch310` environment was
  also checked on 2026-06-21 and lacks TensorFlow/HDF5, so checkpoint evaluation
  requires a separate isolated TensorFlow/Keras reproduction environment. The
  remote `tensorflow_gpu2` environment is usable for checkpoint evaluation, but
  TensorFlow falls back to CPU there because `cufft64_10.dll` is missing.
- Project alignment note: use `sample_structure=zhang_wang_case2_official_mcnd`
  for future official-protocol reproduction attempts. Earlier
  `zhang_wang_case2_mcnd` rows used a base-plaintext/public-mask scaffold and do
  not match the official pair-independent-key grouping.

## Official Checkpoint Smoke Evaluation

Command:

```bash
F:\Anaconda\envs\tensorflow_gpu2\python.exe \
  G:/lxy/zhang_wang2022_code_audit/evaluate_zhang_wang_official_checkpoint.py \
  --audit-root G:/lxy/zhang_wang2022_code_audit \
  --raw-pair-count 16000 \
  --batch-size 1000 \
  --run-eval \
  --output G:/lxy/zhang_wang2022_code_audit/outputs/official_checkpoint_eval_smoke_16000.json
```

Result:

```text
status = evaluated
raw_pair_count = 16000
grouped_eval_rows = 1000
positive_rows = 500
negative_rows = 500
accuracy = 0.755
tpr = 0.748
tnr = 0.762
mse = 0.1702931863
```

Interpretation:

```text
This is a checkpoint/data-generator smoke test, not a paper-scale reproduction.
It confirms the official Case2 r7/m16 checkpoint can be loaded and produces a
strong signal on fresh official-protocol samples. The sample size is too small
to treat 0.755 as a stable reproduction of the paper's 0.7205 reference.
```

## Official Checkpoint 10k-Row Confirmation

Command:

```bash
F:\Anaconda\envs\tensorflow_gpu2\python.exe \
  G:/lxy/zhang_wang2022_code_audit/evaluate_zhang_wang_official_checkpoint.py \
  --audit-root G:/lxy/zhang_wang2022_code_audit \
  --raw-pair-count 160000 \
  --batch-size 1000 \
  --run-eval \
  --output G:/lxy/zhang_wang2022_code_audit/outputs/official_checkpoint_eval_160000.json
```

Result:

```text
status = evaluated
raw_pair_count = 160000
grouped_eval_rows = 10000
positive_rows = 5000
negative_rows = 5000
accuracy = 0.7131
accuracy_minus_reference = -0.0074
tpr = 0.6972
tnr = 0.7290
mse = 0.1856265315
```

Interpretation:

```text
This is stronger than the smoke test because it uses 10000 grouped rows, but it
is still below paper scale. The result supports that the official checkpoint and
official data generator produce a real r7 signal, but it does not independently
match or exceed the paper's 0.7205 reference at this smaller confirmation scale.
```

## Official Checkpoint 62.5k-Row Confirmation

Command:

```bash
F:\Anaconda\envs\tensorflow_gpu2\python.exe \
  G:/lxy/zhang_wang2022_code_audit/evaluate_zhang_wang_official_checkpoint.py \
  --audit-root G:/lxy/zhang_wang2022_code_audit \
  --raw-pair-count 1000000 \
  --batch-size 1000 \
  --run-eval \
  --output G:/lxy/zhang_wang2022_code_audit/outputs/official_checkpoint_eval_1000000.json
```

Retrieved local artifact:

```text
outputs/remote_results/zhang_wang_present_r7_m16_checkpoint_eval_1000000/official_checkpoint_eval_1000000.json
```

Result:

```text
status = evaluated
raw_pair_count = 1000000
grouped_eval_rows = 62500
positive_rows = 31250
negative_rows = 31250
accuracy = 0.721536
accuracy_minus_reference = +0.001036
tpr = 0.704288
tnr = 0.738784
mse = 0.1829306853
```

Interpretation:

```text
This is the strongest checkpoint reproduction currently recorded in this
project. It confirms that the official Zhang/Wang 2022 Case2 r7/m16 checkpoint,
when evaluated with the official PRESENT data generator on 62500 fresh grouped
rows, reaches the reported 0.7205-level 7-round signal. This should be cited as
official-checkpoint verification, not as a from-scratch PyTorch or paper-scale
training reproduction.
```

## Official Code Audit Checklist

Before treating the Zhang/Wang number as verified, complete this checklist:

- Download the official Drive package into an isolated audit directory and record
  package names, file hashes, and directory structure.
- Identify the exact PRESENT Case2 `m=16`, 7-round training/evaluation entrypoint.
- Confirm whether the reported `0.7205` is accuracy on an independent test set,
  validation accuracy, best checkpoint validation accuracy, or another metric.
- Audit positive sample construction: input difference, grouping rule, whether
  the 16 pairs share a base plaintext, public masks, key schedule, and row order.
- Audit negative sample construction: random ciphertext, encrypted random
  plaintexts, same-key/cross-key policy, and whether any ciphertext/plaintext
  relation leaks labels.
- Audit split hygiene: train/validation/test separation, key reuse policy,
  random seed handling, shuffle order, and checkpoint selection.
- Audit model and training details: Inception branches, input shape, loss,
  optimizer, cyclic learning-rate schedule, epoch count, batch size, and early
  stopping.
- Run the official script unchanged and record the achieved r6/r7 metrics.
- Run `scripts/evaluate-zhang-wang-checkpoint --run-eval` inside an isolated
  TensorFlow/Keras environment and record the
  checkpoint evaluation accuracy.
- Run a paper-scale or at least larger checkpoint evaluation using
  `raw_pair_count=1000000` if CPU time is acceptable, or an intermediate
  `raw_pair_count=320000`/`1000000` confirmation first.
- Only after the official script reproduces the paper-scale result, port the
  exact data protocol into this project and rerun our local reproduction.

Interpretation:

- If official code reaches approximately `0.7205` and our port does not, our
  local protocol is still misaligned.
- If official code does not reach the reported value under its own settings, the
  Zhang/Wang reference should remain an unverified or unreliable literature
  target.
- If official code shows data leakage or an evaluation mismatch, use it only as
  related work context, not as a strict single-sample SOTA target.

## Project Protocol

New difference profile:

```text
present_zhang_wang2022_mcnd -> 0x0000000000000009
```

New sample structure:

```text
zhang_wang_case2_mcnd
```

This sample structure creates one MCND sample from `m = pairs_per_sample`
ciphertext pairs. The pairs share one random base plaintext and use random public
plaintext masks before applying the fixed input difference. This is intended to
move away from fully independent pair-set generation toward the grouped MCND
Case-2 style used by the literature.

Current caveat: this is still a reproduction scaffold, not a confirmed exact
line-by-line reproduction. The next validation criterion is empirical: 6-round
PRESENT should separate clearly before spending large GPU time on 7-round or
8-round experiments.

## PyTorch Official-Protocol Training Track

Current project-side reproduction uses:

```text
model: present_zhang_wang_keras_mcnd
sample_structure: zhang_wang_case2_official_mcnd
negative_mode: encrypted_random_plaintexts
feature_encoding: ciphertext_pair_bits
pairs_per_sample: 16
loss: mse
```

Important implementation fixes now in `main`:

```text
efcc36d fix: match official present mcnd pair keys
69371d9 fix: add official present keras mcnd model
```

These align the data protocol and model layout with the official Case2
checkpoint more closely than the earlier scaffold:

- `zhang_wang_case2_official_mcnd` uses an independent random PRESENT-80 key for
  every basic ciphertext pair inside the `m=16` sample.
- `present_zhang_wang_keras_mcnd` follows the official Keras-layout MCND model:
  grouped pair tensors, nibble/word permutation, `Conv1D` branches with kernel
  sizes `1/2/4`, five residual convolution blocks, global pooling, and a
  sigmoid-style binary head.

### Retrieved PyTorch Diagnostics

The old pre-fix 8k/class anchor runs were near random and are useful only as
diagnostics:

```text
old model, before per-pair-key fix:
  r6 accuracy=0.5033, AUC=0.5008
  r7 accuracy=0.4983, AUC=0.4978

old model, after per-pair-key fix:
  r6 accuracy=0.5005, AUC=0.4962
  r7 accuracy=0.4979, AUC=0.4963
```

After switching to the Keras-layout model, the same 8k/class budget started to
learn the official-protocol signal:

```text
run_id: zhang_wang_pytorch_official_anchor_diag20_keraslayout_20260624
r6 accuracy=0.816406, AUC=0.897294, calibrated_accuracy=0.817871
r7 accuracy=0.513672, AUC=0.536711, calibrated_accuracy=0.527588
```

The first retrieved r7 64k/class medium diagnostic completed on 2026-06-24:

```text
run_id: zhang_wang_present_r7_64k_keraslayout_20260624
local artifacts: outputs/remote_results/zhang_wang_present_r7_64k_keraslayout_20260624/
source commit: 69371d9141a7d3798b8b52b6606c57b6f0fdb279
samples_per_class: 65536
pairs_per_sample: 16
batch_size: 512
learning_rate: 0.0001
lr_scheduler: none
checkpoint_metric: val_loss
restore_best_checkpoint: true
epochs_ran: 10
best_epoch: 2
accuracy: 0.611740
calibrated_accuracy: 0.614655
AUC: 0.658023
best history AUC: 0.674439 at epoch 4
```

Interpretation:

```text
This is a medium diagnostic, not formal training and not a successful
from-scratch reproduction of the 0.7205 paper reference. It is nevertheless a
strong alignment signal: r7 improved from near-random 8k/class diagnostics to
AUC 0.658 and calibrated accuracy 0.615 at 65536/class after the pair-key and
Keras-layout fixes.
```

The 64k curve also shows training instability: training AUC eventually reaches
about `0.99`, while validation AUC peaks near `0.674` and threshold accuracy
falls back toward chance in later epochs. This motivates training-protocol
alignment before scaling to 1M/class.

### Retrieved 262k/Class Official-Cyclic Diagnostic

The single-row 262144/class diagnostic completed on 2026-06-24 and was
retrieved into the local result archive:

```text
plan: configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_262k.csv
remote config: configs/remote/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_262k_gpu1_20260624.json
run_id: zhang_wang_present_r7_262k_official_cyclic_20260624
local artifacts: outputs/remote_results/zhang_wang_present_r7_262k_official_cyclic_20260624/
samples_per_class: 262144
batch_size: 1024
lr_scheduler: official_cyclic
learning_rate: 0.0001
max_learning_rate: 0.002
checkpoint_metric: val_auc
restore_best_checkpoint: true
dataset storage: disk-backed cache
source commit: 80e849db33f388ef0dd53f1f7050775af4eb853c
gate: pass
```

This run tested three changes together against the retrieved 64k result:

- use an official-style 10-epoch high-to-low cyclic learning-rate schedule;
- select the checkpoint by `val_auc` rather than `val_loss`;
- increase batch size from `512` to `1024`.

Final restored best-checkpoint result:

```text
epochs requested: 20
epochs_ran: 14
stopped_epoch: 14
best_epoch: 6
accuracy: 0.7109107971
calibrated_accuracy: 0.7119331360
AUC: 0.7862925224
loss: 0.5540635363
best_accuracy: 0.7119331360
calibrated_threshold: 0.5347244740
best_checkpoint_metric: 0.7862925224
selected_checkpoint: best
```

Per-epoch validation behavior:

```text
epoch 1:  val_acc=0.679062, val_auc=0.766090, val_loss=0.593941
epoch 2:  val_acc=0.698204, val_auc=0.778329, val_loss=0.571884
epoch 3:  val_acc=0.709324, val_auc=0.782754, val_loss=0.561044
epoch 4:  val_acc=0.705853, val_auc=0.785527, val_loss=0.562508
epoch 5:  val_acc=0.654308, val_auc=0.784695, val_loss=0.642250
epoch 6:  val_acc=0.710911, val_auc=0.786293, val_loss=0.554064
epoch 7:  val_acc=0.685722, val_auc=0.783938
epoch 8:  val_acc=0.709194, val_auc=0.782776
epoch 9:  val_acc=0.697941, val_auc=0.779502
epoch 10: val_acc=0.701427, val_auc=0.773793
epoch 11: val_acc=0.705391, val_auc=0.780267
epoch 12: val_acc=0.682362, val_auc=0.778218
epoch 13: val_acc=0.704628, val_auc=0.776908
epoch 14: val_acc=0.699818, val_auc=0.773312
```

Scale comparison:

```text
64k/class Keras-layout:
  accuracy            = 0.611740
  calibrated_accuracy = 0.614655
  AUC                 = 0.658023

262k/class official-cyclic:
  accuracy            = 0.710911
  calibrated_accuracy = 0.711933
  AUC                 = 0.786293

improvement over 64k:
  accuracy            = +0.099171
  calibrated_accuracy = +0.097278
  AUC                 = +0.128270
```

Reference comparison:

```text
Zhang/Wang Table 4 PRESENT-80 r7 Case2 m=16 accuracy: 0.7205
PyTorch 262k/class official-cyclic accuracy:              0.710911
gap:                                                       -0.009589
PyTorch 262k/class calibrated_accuracy:                   0.711933
calibrated gap:                                           -0.008567
```

Interpretation:

```text
This is a successful medium diagnostic, not a formal reproduction or
breakthrough claim. The official-protocol PyTorch route is now close enough to
the 0.7205 reference that the next useful baseline is a 1000000/class
single-seed run. Multi-seed evidence is still required before making a formal
from-scratch reproduction claim.
```

### Next Paper-Scale Single-Seed Baseline

The next planned run is a single-row 1000000/class official-protocol baseline:

```text
plan: configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_1m.csv
remote config: configs/remote/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_1m_gpu0_20260625.json
run_id: zhang_wang_present_r7_1m_official_cyclic_seed0_20260625
samples_per_class: 1000000
pairs_per_sample: 16
batch_size: 1024
lr_scheduler: official_cyclic
learning_rate: 0.0001
max_learning_rate: 0.002
checkpoint_metric: val_auc
restore_best_checkpoint: true
negative_mode: encrypted_random_plaintexts
sample_structure: zhang_wang_case2_official_mcnd
```

Expected status:

```text
This is paper-scale for sample count but still single-seed. It can validate
whether the from-scratch PyTorch implementation reaches the 0.7205-level
reference under the official protocol, but publication-style claims still need
multiple seeds and a clean result archive.
```

## Interpretation Rules

- If 6-round remains near random, do not claim a Zhang/Wang reproduction.
- If 6-round becomes strong but 7-round remains weak, increase training scale
  and check exact Case-1/Case-2 details.
- Only after a credible 7-round baseline should innovation-one aligned inputs or
  P-aligned/integral variants be attached to this reproduction baseline.
