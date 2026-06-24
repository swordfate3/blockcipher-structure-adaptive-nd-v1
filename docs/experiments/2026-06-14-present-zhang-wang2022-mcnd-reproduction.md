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
project_reproduction_status: official_checkpoint_eval_confirmed_below_reference_at_10k_grouped_rows_not_paper_scale
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
This distinction is intentionally mirrored in
`experiments/innovation1/audit_spn_claim_gate.py` and
`experiments/innovation1/summarize_spn_route_queue.py`.

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
experiments/innovation1/evaluate_zhang_wang_official_checkpoint.py
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
- Run `experiments/innovation1/evaluate_zhang_wang_official_checkpoint.py
  --run-eval` inside an isolated TensorFlow/Keras environment and record the
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

## Plans

Smoke plan:

```text
experiments/innovation1/plans/innovation1_spn_present_zhang_wang2022_mcnd_smoke.csv
```

Medium plan:

```text
experiments/innovation1/plans/innovation1_spn_present_zhang_wang2022_mcnd_medium.csv
```

Medium run protocol:

```text
rounds: 6, 7
seeds: 0, 1, 2
samples_per_class: 8192
pairs_per_sample: 16
feature_encoding: ciphertext_pair_bits
negative_mode: encrypted_random_plaintexts
sample_structure: zhang_wang_case2_mcnd
key_rotation_interval: 1024
```

## Interpretation Rules

- If 6-round remains near random, do not claim a Zhang/Wang reproduction.
- If 6-round becomes strong but 7-round remains weak, increase training scale
  and check exact Case-1/Case-2 details.
- Only after a credible 7-round baseline should innovation-one aligned inputs or
  P-aligned/integral variants be attached to this reproduction baseline.
