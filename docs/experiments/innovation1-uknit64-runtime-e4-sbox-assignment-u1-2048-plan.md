# Innovation 1 uKNIT Runtime-E4 S-box Assignment U1 Plan

Date: 2026-07-24

## Status

```text
stage    = preregistered
run_id   = i1_rtg1_uknit64_runtime_e4_sbox_assignment_u1_2048_seed0_seed1_20260724
result   = not run
decision = pending
```

## Research Question

Does the runtime-parameterized E4 backbone benefit from the true mapping between
uKNIT-BC cells and their cell-specific S-boxes, rather than merely receiving the
same S-box multiset or a global S-box summary?

This is the first trained attribution diagnostic for the real non-round-aligned
uKNIT descriptor. It changes only how the already implemented external S-box
truth tables are attached to cell tokens. It does not add DDT, trail, inverse
round, partial-decryption or handcrafted difference features.

## Fixed Protocol

| Field | Frozen value |
| --- | --- |
| Cipher | uKNIT-BC |
| Reduced-round target | prefix r4 |
| Train / validation keys | `0x00...00` / `0x11...11`, 128 bits each |
| Engineering input difference | `0x0000000000000040` |
| Runtime descriptor | `configs/runtime/spn/uknit64.json` |
| Runtime window | `round_start=2`, `processor_steps=2` |
| Loaded transitions | S2/L2 and S3/L3 |
| Train scale | `2048/class`, `4096` total rows per seed |
| Validation scale | `1024/class`, `2048` total rows per seed |
| Seeds | `0,1` |
| Pairs per sample | `4` |
| Feature encoding | raw ciphertext-pair bits |
| Negative definition | encrypted random plaintexts |
| Sample structure | independent pairs |
| Backbone size | hidden bits `64`, pair embedding `128` |
| Training | MSE, Adam, lr `1e-4`, weight decay `1e-5` |
| Checkpoint | best validation AUC |
| Epochs | `10` |
| Execution | local CPU, sub-medium diagnostic |

The difference is an engineering calibration choice, not a literature-backed
uKNIT differential claim. This budget is a local diagnostic, not formal,
paper-scale or ceiling evidence.

## Three Matched Rows

For each seed train exactly these rows on the same disk-backed train and
validation datasets:

| Role | Model | S-box context | Structural change |
| --- | --- | --- | --- |
| Candidate | correct runtime structure | `late_cell` | none |
| Same-budget anchor | correct runtime structure | `late_pair` | cell ownership is averaged before classification |
| Required control | S-box-assignment shuffled structure | `late_cell` | only S-box-to-cell ownership changes |

The shuffled control preserves every GF(2) linear matrix and each round's S-box
multiset. All three models must have identical parameter names and shapes. The
runner reseeds model construction from the row seed, so equal-geometry rows use
matched initialization. Dataset cache identity excludes model choice, forcing
all three rows for one seed to reuse the exact same feature and label arrays.

Config:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_sbox_assignment_u1_2048_seed0_seed1.csv
```

## Readiness Gate

Before training require:

1. Six plan rows parse as two seeds times three roles.
2. All rows build with identical parameter geometry.
3. Correct and shuffled structures have identical linear matrices and per-round
   S-box multisets but different S-box-to-cell assignments.
4. The descriptor reports `available_rounds=11`, `round_start=2` and
   `loaded_rounds=2`.
5. A copied-weight forward check is insensitive to assignment shuffle in
   `late_pair` mode and sensitive above `1e-6` in `late_cell` mode.
6. Strict encrypted-random-plaintext negatives and disk-backed cache settings
   are present.

## Advance And Stop Gates

For each seed independently define:

```text
candidate_auc          = correct late_cell validation AUC
candidate_minus_anchor = candidate_auc - correct late_pair AUC
candidate_minus_shuffle= candidate_auc - shuffled-assignment late_cell AUC
```

Advance only if every condition holds on both seeds:

```text
candidate_auc           >= 0.520
candidate_minus_anchor  >= 0.005
candidate_minus_shuffle >= 0.005
all protocol checks     = true
```

The thresholds are frozen before training. A pass means only that true uKNIT
S-box ownership is supported at this local diagnostic budget. It authorizes a
separate same-budget window replication plan before any scale increase.

If either seed misses any research gate, classify U1 as `hold/redesign-local`.
Do not increase samples, add epochs, tune the thresholds after reveal, launch a
remote run, add DDT/trail features, or reinterpret the output as a uKNIT attack.

## Execution And Artifacts

```text
output root = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u1_2048_seed0_seed1_20260724
cache root  = outputs/local_cache/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u1_2048_seed0_seed1_20260724
```

The completed run must contain `results.jsonl`, `progress.jsonl`, checkpoints,
`validation.json`, `gate.json`, `summary.json`, `history.csv` and `curves.svg`.
After result validation and visual QA, refresh both recent-result indexes.

## Required Next Action After Completion

- If both seeds pass: keep `late_cell`; next test the unchanged three-row matrix
  on one different valid uKNIT transition window at the same `2048/class`
  budget before considering remote scale.
- If only one seed passes: hold the branch; run a deterministic activation and
  gradient attribution audit comparing correct and shuffled ownership, without
  more training data.
- If neither seed passes or both candidate AUCs are near chance: discard this
  direct late-cell injection and redesign the S-box/topology interaction
  locally; do not mechanically try 8192/class or remote execution.
