# Innovation 1 Feistel / SM4 Position-Preserving Paper-Family Calibration Plan

**Date:** 2026-07-15

**Status:** implementation and readiness passed; 2048/class calibration ready

## Research Question

Does replacing the existing global-average classifier with the
position-preserving residual/flatten/dense path described by Yu/Wu/Zhang recover
SM4-r5 signal at the same `2048/class` budget?

The preceding audit proved that the current cipher and dataset pipeline learns
r3 almost perfectly under fixed and rotating keys, while the existing
`multiscale_dense_resnet` remains near chance at r5 under both. The current
baseline pools away all 128 serialized bit positions before classification.
That is a material mismatch from the paper's described residual network with
dropout and two 64-unit fully connected layers.

## Paper-To-Code Boundary

The available paper description supports the following mechanism-aligned port:

```text
256 raw ciphertext-pair bits
  -> reshape as two ciphertext channels x 128 ordered bit positions
  -> 32-channel convolutional stem
  -> 32-channel residual tower
  -> flatten all channel/position features
  -> dense 64 + dropout 0.5
  -> dense 64 + dropout 0.5
  -> one output logit
```

The exact convolution kernel sequence, residual-block count, batch-normalization
placement, and public source code have not been located. The project freezes
five kernel-3 residual blocks as a conservative Gohr-family interpretation and
names the model `sm4_yu2023_position_resnet`, not an exact reproduction.

## One Variable And Controls

| Role | Model | Position handling |
| --- | --- | --- |
| Candidate baseline | `sm4_yu2023_position_resnet` | flatten preserves all 128 positions |
| Existing control | `multiscale_dense_resnet` | adaptive global average pooling |

Both receive the same raw 256 bits and the same training protocol. The matrix
crosses these two architectures with fixed and rotating keys at r5, producing
four rows. It calibrates the literature anchor only; the SM4 recurrence
candidate remains frozen and absent.

## Frozen Protocol

```text
cipher                 = SM4-r5
difference             = (0, 0, 0, 1)
pairs/sample           = 1
input                   = 256 raw ciphertext-pair bits
negative                = encrypted_random_plaintexts
train                   = 2048/class
validation              = 1024/class
fresh final test        = 3 x 2048/class
seed                    = 0
epochs                  = 10
batch size              = 128
loss                    = MSE
optimizer               = Adam, fixed 1e-4
checkpoint              = best val_auc restored
new model channels      = 32
new model blocks        = 5
new model dense widths  = 64, 64
new model dropout       = 0.5
key protocols           = one fixed key; one random key per row
```

## Readiness Gate

The two-row readiness matrix runs only the new model at r3 and r5 fixed key,
`64/class`, seed0, two epochs, and one fresh test. It validates input shape,
forward/backward, checkpoint restoration, parameter metadata, and gate
plumbing. Its AUC values are not research evidence.

Readiness completed on 2026-07-15:

```text
result rows                 = 2/2
validation errors           = []
position model parameters   = 297921 at r3 and r5
decision                    = feistel_sm4_position_resnet_readiness_passed
research_decision_applied   = false
next action                 = run_sm4_position_resnet_calibration_2048
recent result index         = 001
```

The tiny r3/r5 fresh AUCs were `0.557373047/0.473388672`; they are readiness
diagnostics only. Artifacts are under
`outputs/local_smoke/i1_feistel_sm4_position_resnet_readiness_seed0/`.

Local PyTorch cannot initialize CUDA on the workstation's RTX 5080, and the
flattened classifier makes CPU calibration disproportionately slow. The frozen
`2048/class` matrix may therefore execute on the remote A6000 from a pushed
commit. This changes only the device, not the data/model/training protocol, and
remains a small calibration rather than paper-scale training.

## Frozen Calibration Gate

For each key protocol, calculate:

```text
position-preserving AUC
position-preserving - global-pool AUC
```

Signal requires AUC `>= 0.55`; improvement requires a margin `>= +0.01`.

```text
new model signals under rotating keys and improves baseline
  -> retain the position-preserving anchor and redesign the recurrence
     candidate on the same backbone before a new attribution gate

new model signals only under fixed key and improves baseline
  -> retain a paper-style calibration anchor with explicit fixed-key scope;
     audit cross-key generalization before architecture claims

new model has no r5 signal
  -> architecture repair alone is insufficient at 2048/class;
     authorize one local 8192/class baseline-only scale check as a specific
     data-scarcity diagnostic exception

invalid capacity, plan, key, history, or final-test metadata
  -> no calibration decision
```

## Explicit Stops

- no SM4 recurrence-candidate changes in this calibration;
- no remote run or 65536/class scale;
- no random-ciphertext negatives;
- no claim of exact Yu/Wu/Zhang reproduction;
- no r6-r8 sweep or cross-Feistel conclusion.

## Evidence-Backed Next Action

Implement and unit-test the position-preserving paper-family baseline, run the
two-row readiness gate, then automatically run the frozen four-row calibration
if readiness is valid. Produce JSONL, progress, validation, Chinese SVG,
history, gate, and recent-result index artifacts. The resulting gate, not the
paper's external `0.999`, determines the next local action.
