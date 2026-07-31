# Innovation 1 uKNIT r5 Neural Architecture Ablation K1-BS

**Date:** 2026-07-31
**Status:** completed / pass / uKNIT structure expert retained
**Execution:** local CPU, sub-medium architecture diagnostic; device was frozen before the 2026-07-31 local-GPU-first rule

## Research question

The completed uKNIT r5 experiments already contain semantic and structural
controls. K1-V showed that the exact uKNIT structure model reached cross-key
AUC `0.902423` and `0.932539` with sixteen ciphertext pairs, while its
wrong-S-box controls stayed near chance. Those rows establish that correct
S-box semantics matter, but they do not answer a separate model-selection
question:

> Under exactly the same uKNIT r5 data, query, key, seed, epoch and optimizer
> budget, is the current uKNIT structure expert better than established or
> generic neural architectures?

K1-BS is a local architecture diagnostic. It is not formal training, a
paper-protocol reproduction, a capacity-matched structure attribution result,
or a SOTA claim.

## Single variable

Only the trainable architecture changes:

| Condition | Model key | Role |
|---|---|---|
| uKNIT structure expert | `runtime_spn_ct_k1t_position_histogram_true` | current same-protocol anchor |
| AutoND DBitNet | `autond_dbitnet2023` | published cipher-agnostic architecture baseline |
| SPN Cell PairSet DBitNet | `spn_pairset_dbitnet_v2` | generic cell-aware SPN baseline |
| SPN Token Mixer PairSet | `spn_token_mixer_pairset` | generic position-preserving SPN baseline |

The networks are compared as complete model choices and therefore have
different parameter counts. K1-BS records those counts and must not interpret
an end-to-end win as capacity-matched proof that topology semantics alone caused
the difference.

## Frozen protocol

Train eight independent rows: two seeds times four architectures.

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Difference | cell11 role1, `0x0000400000000000` |
| Train | `2048/class`, `4096` total rows |
| Cross-key validation | `1024/class`, `2048` total rows |
| Seeds | `3`, `4` |
| Pairs/sample | `16` independent ciphertext pairs |
| Input width | `2048` bits/sample, `128` bits/pair |
| Negative definition | encrypted random plaintexts |
| Train/validation keys | K1-V seed-matched fixed cross-key pairs |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | restored best validation AUC |
| Device | local CPU |

The same seed must reuse identical disk-backed train and validation datasets
across all four architectures. Difference position, data generation, labels,
negative samples, keys, pair count, sample count and metric are immutable.

## Readiness gate

Before any optimizer step require:

1. exactly eight frozen plan rows;
2. all models accept `[B, 2048]`, representing sixteen 128-bit pairs;
3. finite equal-shape logits and a finite nonzero backward gradient;
4. recorded parameter counts match the instantiated models;
5. no data, key, label, metric or optimizer field differs across architectures.

Any failure prohibits training. Repair only the failed invariant and rerun the
same readiness gate.

## Result gate

For each seed separately require:

```text
structure expert AUC                 >= 0.550
structure expert - best generic AUC  >= +0.010
```

Both seeds must satisfy both clauses before remote confirmation is authorized.
Seed averaging cannot hide a failed seed.

## Decisions

- **Expert passes both clauses on both seeds:** retain it and remotely compare
  only the expert and the strongest generic baseline at `65536/class`.
- **Expert signal reproduces but margin fails:** the specialist architecture is
  not necessary at this evidence scale. Freeze the strongest generic model and
  run one fresh-seed local confirmation before spending a remote slot.
- **Expert itself falls below `0.550`:** audit optimization parity against the
  completed K1-V anchor. Do not redesign data or scale the run.
- **Protocol invalid:** repair only the failed plan, cache, checkpoint, shape or
  artifact binding and rerun unchanged.

Blocked inside K1-BS: new differences, pair counts, keys, ciphers, round counts,
losses, epochs, negative definitions, model tuning, capacity matching, and
remote-running all four models.

## Planned artifacts

```text
run_id = i1_uknit_r5_neural_architecture_ablation_k1bs_16pair_2048_seed3_seed4_20260731
```

The result root must contain readiness, disk caches, progress JSONL, eight best
checkpoints, results JSONL, architecture comparison CSV, gate, validation,
summary, history CSV, Chinese SVG, plot report, and a visual QA marker. After a
completed result, refresh both recent-result indexes and update this document
with the measured metrics and an executable next action.

## Completed result

The frozen eight-row run completed on 2026-07-31. Its evidence root is:

```text
outputs/local_diagnostic/
  i1_uknit_r5_neural_architecture_ablation_k1bs_16pair_2048_seed3_seed4_20260731/
```

| Architecture | Parameters | seed3 AUC | seed4 AUC |
|---|---:|---:|---:|
| uKNIT structure expert | `214316` | `0.902801514` | `0.932538986` |
| AutoND DBitNet | `985985` | `0.511321068` | `0.526423454` |
| Generic SPN Cell-PairSet | `1045763` | `0.511973858` | `0.504181862` |
| Generic SPN Token Mixer | `313634` | `0.508666992` | `0.506958961` |

The strongest generic row differed by seed: Cell-PairSet was seed3's best at
`0.511973858`, while AutoND DBitNet was seed4's best at `0.526423454`. The
preregistered per-seed margins were:

```text
seed3 expert - best generic = +0.390827656
seed4 expert - best generic = +0.406115532
```

Both expert AUCs exceeded `0.550`, and both margins exceeded `+0.010`. All
protocol checks passed: eight result rows completed, input width remained
`2048` bits as sixteen 128-bit pairs, train/validation rows remained
`4096/2048`, all checkpoints restored the best validation AUC, and the shared
disk cache recorded four creations plus twelve parameter-matched reuses.

```text
gate status  = pass
decision     = innovation1_uknit_k1bs_structure_expert_retained
remote_scale = candidate
```

The narrow supported conclusion is that the current uKNIT-specific structure
expert is a much stronger end-to-end model choice than these three generic
architectures under the same local r5 data and training budget. More parameters
did not rescue the generic models. Because parameter counts differ, K1-BS does
not prove a capacity-matched causal contribution from topology alone. It also
does not constitute formal-scale or paper-scale evidence.

The Chinese SVG passed rendered-pixel visual QA: no text overlap, clipping,
missing glyphs, ambiguous title, or unreadable near-chance values were found.

## Executable next action

Run a remote A6000 medium confirmation with only two models:

```text
research question = does the large local expert advantage survive 65536/class?
same-budget anchor = AutoND DBitNet, strongest generic model by two-seed mean AUC
one variable       = neural architecture
cipher / rounds    = uKNIT-BC r5
difference         = cell11 role1, 0x0000400000000000
train / validation = 65536/class / 16384/class
pairs              = 16 independent ciphertext pairs
seeds              = 3,4
epochs / batch     = 10 / 64
models             = structure expert, AutoND DBitNet
execution          = remote A6000 from a pushed commit
```

Before launch require a two-model readiness pass, generated Windows scripts
using `cmd.exe /c`, a clean run-owned remote clone or clean fast-forward gate,
and parameter-matched disk cache/progress/reuse under `G:\lxy`. Advance the
specialist route only if the expert reaches AUC `>=0.550` and beats AutoND by
`>=+0.010` on both seeds. Hold if either seed fails. Do not remotely run all
four models, add epochs or pairs, change the difference, or call
`65536/class` formal evidence.
