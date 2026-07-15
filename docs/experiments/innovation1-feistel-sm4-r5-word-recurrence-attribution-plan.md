# Innovation 1 Feistel / SM4-r5 Word-Recurrence Attribution Plan

**Date:** 2026-07-15

**Status:** 2048/class completed; no strict cross-key signal; protocol audit required

## Research Question

Does preserving SM4's ordered four-word recurrence improve a neural
distinguisher over both an exactly equal-capacity wrong-structure control and a
same-protocol convolutional residual baseline?

This is the first generalization test after the retained DES-r5 canonical
mapping cell. It does not reuse DES's two-half representation. SM4 updates one
word through

```text
X[i+4] = X[i] xor T(X[i+1] xor X[i+2] xor X[i+3] xor rk[i]).
```

The hypothesis is that a model should restore chronological state-word order,
keep the four word roles distinct, and expose the three-word round-function
input relation and SM4 linear-diffusion rotations before learned mixing.

## Paper-To-Code Boundary

The literature anchor is Yu, Wu, and Zhang, *Analysis of SM4 Algorithm Based on
Convolutional Residual Network* (2023). The project-held protocol audit records:

```text
plaintext difference = (0, 0, 0, 1)
input                = one 256-bit ciphertext pair
train                = 1,000,000 total rows
test                 = 100,000 total rows
epochs               = 25
optimizer/loss       = Adam 1e-4 / MSE
batch size           = 5000
reported r5 accuracy = 0.999
```

No official implementation or complete layer-by-layer specification has been
located. The existing `multiscale_dense_resnet` is therefore retained only as
a Yu/Wu/Zhang-family convolutional residual baseline, not an exact paper port.
The paper's key-sampling wording is also insufficient to freeze an exact key
reproduction. This experiment deliberately uses row-level key rotation for all
roles so fixed-key memorization cannot decide the comparison.

## One Variable And Controls

| Role | Model key | Input interpretation | Capacity relation |
| --- | --- | --- | --- |
| Candidate | `sm4_word_recurrence_true` | Restore chronological words and real SM4 bit positions | exactly matched to shuffled |
| Attribution control | `sm4_word_recurrence_shuffled` | Fixed wrong word/bit mapping before the same recurrence network | exactly matched to candidate |
| Same-budget baseline | `multiscale_dense_resnet` | Raw 256-bit ciphertext pair | independent literature-family baseline |

The candidate and attribution control receive the identical raw bits, use the
same deterministic feature construction, and have identical trainable
parameters. Only the fixed mapping from serialized ciphertext bits to SM4 word
roles changes. The candidate constructs:

```text
two ciphertexts x four chronological 32-bit words
  -> raw word-role channels: C0, C1, C0 xor C1
  -> feedback channels for X[i]
  -> three-word channels for X[i+1] xor X[i+2] xor X[i+3]
  -> circularly aligned copies at rotations 2, 10, 18, 24
  -> shared residual bit-position encoder and global classifier
```

## Frozen Protocol

```text
cipher                 = SM4
rounds                 = 5
difference             = 0x00000000000000000000000000000001
difference profile     = sm4_yu2023_conv_resnet
pairs/sample           = 1
input                   = 256 raw ciphertext-pair bits
negative                = encrypted_random_plaintexts
sample structure        = independent_pairs
key rotation interval   = 1 row
train                   = 2048/class = 4096 total rows
validation              = 1024/class = 2048 total rows
fresh final test        = 3 x 2048/class = 3 x 4096 total rows
seeds                   = 0, 1
epochs                  = 10
batch size              = 128
loss                    = MSE
optimizer               = Adam, fixed 1e-4
checkpoint              = best val_auc restored
candidate channels      = 32
candidate blocks        = 3
candidate classifier    = 64
candidate dropout       = 0.5
```

This is a small local attribution diagnostic, not paper-scale reproduction.
The paper reference uses `500000/class` train and `50000/class` test if its
balanced one-million/one-hundred-thousand totals are interpreted per class.

## Readiness Gate

```text
plan       = configs/experiment/innovation1/innovation1_feistel_sm4_r5_word_recurrence_attribution_readiness_seed0.csv
scale      = 64/class, seed0, 2 epochs
validation = 64/class
fresh test = 1 x 64/class
rows       = candidate, shuffled, baseline
```

Readiness passes only if all three rows are plan-aligned, finite, restorable,
and report rotating-key metadata. Candidate and shuffled parameter counts must
match exactly, their fixed mapping indices must differ, and
`research_decision_applied=false`. Readiness metrics are not research evidence.

Readiness completed on 2026-07-15:

```text
result rows                 = 3/3
validation errors           = []
candidate parameters        = 28161
shuffled parameters         = 28161
baseline parameters         = 59809
decision                    = feistel_sm4_word_recurrence_readiness_passed
research_decision_applied   = false
next action                 = run_sm4_r5_word_recurrence_attribution_2048
recent result index         = 001
```

The readiness candidate, shuffled control, and baseline fresh-test AUCs were
`0.537109375`, `0.501953125`, and `0.404296875`. These tiny `64/class` values
are recorded only to diagnose the runnable path and must not be used as a model
or architecture result. Artifacts are under
`outputs/local_smoke/i1_feistel_sm4_r5_word_recurrence_attribution_readiness_seed0/`.

## Frozen Research Gate

For each seed independently:

```text
candidate - shuffled >= +0.005 fresh-test AUC
candidate             >= 0.55 fresh-test AUC
candidate - baseline  >= -0.002 fresh-test AUC
```

Complete pass:

```text
decision    = feistel_sm4_r5_word_recurrence_attributed
next action = freeze a 65536/class two-seed remote medium diagnostic with
              disk-backed cache, then synthesize DES/SM4 mapping rules
```

Other outcomes:

```text
signal without mapping margin -> retain strongest baseline; redesign the
                                 structural control locally
mapping but not competitive   -> retain attribution-only evidence; reject the
                                 candidate as an architecture improvement
no candidate signal           -> stop SM4 scale and audit data/model semantics
invalid protocol              -> no research decision
```

## Explicit Stops

- no dense DDT input or differential-path branch;
- no DES two-half mapping reused for SM4;
- no SM4-r6/r7/r8 sweep before the r5 attribution gate;
- no paper-scale or remote launch from readiness metrics;
- no cross-Feistel or thesis-level architecture claim from a single SM4 cell;
- no claim that `multiscale_dense_resnet` exactly reproduces the 2023 paper.

## Evidence-Backed Next Action

Implement the two equal-capacity recurrence models, add a dedicated SM4 gate
and focused tests, then run readiness. If readiness is valid, automatically run
the frozen six-row `2048/class` local matrix with one unique writer and produce
results JSONL, progress JSONL, validation JSON, Chinese SVG, history CSV, gate
JSON, and the numbered recent-result index. Only a complete local gate decides
whether remote medium scale is justified.

## Completed 2048/Class Result

The six-row matrix completed on 2026-07-15 with exact plan alignment and no
validation errors:

| Seed | Recurrence candidate | Shuffled control | Conv-ResNet baseline |
| ---: | ---: | ---: | ---: |
| 0 | 0.496304949 | 0.495662808 | 0.491743843 |
| 1 | 0.488717238 | 0.503069679 | 0.503065666 |

The controlled margins were:

```text
seed0 candidate - shuffled = +0.000642141
seed1 candidate - shuffled = -0.014352441
seed0 candidate - baseline = +0.004561106
seed1 candidate - baseline = -0.014348427
```

The gate returned:

```text
signal_present       = false
topology_attributed  = false
baseline_competitive = false
decision             = feistel_sm4_word_recurrence_not_ready
next_action          = stop_scale_and_audit_sm4_data_model_semantics
```

The candidate training AUC rose to about `0.77` on both seeds while validation
and fresh-test AUC remained near chance. The immediate failure mode is therefore
generalization, not a model that failed to optimize. This result rejects remote
scale for the current strict rotating-key cell, but it does not establish an
SM4 architecture ceiling: the Yu/Wu/Zhang paper used roughly 244 times more
training rows and its key sampling remains unresolved.

Artifacts are under
`outputs/local_diagnostic/i1_feistel_sm4_r5_word_recurrence_attribution_2048_seed0_seed1/`
and the completed result is entry `001` in `outputs/00_RECENT_RESULTS.md`.

The executable next step is the frozen four-row key-schedule/signal audit in
`docs/experiments/innovation1-feistel-sm4-key-schedule-signal-audit-plan.md`.
It changes no architecture and compares `r3/r5 x fixed/rotating key` on the
same Conv-ResNet. This determines whether a paper-aligned fixed-key attribution
cell, a limited data-scarcity exception, or a data/model repair is justified.
