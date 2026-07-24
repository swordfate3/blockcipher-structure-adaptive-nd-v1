# Innovation 1 uKNIT Runtime-E4 Recurrent Window R5 Conditional Plan

Date: 2026-07-25

## Status

```text
phase = deterministic readiness only
readiness = pass
training = not started
execution_authorized = no
blocking_decision = RTG3-A seed0 and conditional seed1 adjudication
```

## Research Question

Can one cipher-name-free Runtime-E4 backbone use two distinct earlier uKNIT
S-box/GF(2) transitions, rather than gaining only from applying the shared
processor twice?

The single changed hypothesis is `round_window_mode` plus the actual earlier
transition content. Data, optimization, parameter geometry and the final
transition remain fixed.

## Frozen Panel

The same five roles are repeated at seeds 0 and 1:

| Role | Runtime mode | Window | Purpose |
| --- | --- | --- | --- |
| anchor | `last_transition` | correct full descriptor | strongest same-budget old behavior |
| candidate | `recurrent_window` | correct heterogeneous descriptor | multi-round hypothesis |
| repeat-last | `recurrent_window` | final transition repeated twice | equal-depth control |
| corrupted | `recurrent_window` | deterministic corrupted topology | topology control |
| no-topology | `recurrent_window` | correct descriptor, relation disabled | no-topology control |

Frozen protocol:

```text
cipher                  = uKNIT-BC prefix r5
runtime_round_start     = 3
runtime_rounds          = 2
processor_steps         = 2
runtime transitions     = S3/L3 and S4/L4
input difference        = 0x40
pairs_per_sample        = 4
samples_per_class       = 2048
validation              = 1024/class through the standard runner
seeds                   = 0, 1
epochs                  = 10
loss                    = MSE
optimizer               = Adam, lr 1e-4, weight decay 1e-5
checkpoint              = best validation AUC
negative definition     = encrypted random plaintext pairs
train key               = 0x00000000000000000000000000000000
validation key          = 0x11111111111111111111111111111111
cell input              = two-input state_triplet
execution               = local sub-medium diagnostic
```

Plan CSV:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv
```

## Readiness Gate

Before any data generation, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/check-runtime-spn-recurrent-window-readiness \
  --run-id i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725 \
  --plan configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv \
  --output-root outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725
```

Readiness passes only if the full candidate is heterogeneous, repeat-last is
homogeneous, both share the same final transition but not the same ordered
window hash, all roles share parameter shapes, and every data/training field
matches the frozen contract. This gate has no AUC threshold.

## Conditional Execution And Decision

Do not train this matrix until RTG3-A has a completed local adjudication. If
RTG3-A seed0 stops its conditional seed1, use that explicit route decision
instead of waiting for a nonexistent second result.

Once authorized, the empirical advance rule must be preregistered before
reading AUCs. It must require both seeds to keep useful absolute signal and the
candidate to beat repeat-last, corrupted and no-topology controls. The exact
thresholds will be set from the frozen same-budget last-transition anchor and
RTG3 evidence, not from this future matrix's observed values.

Stop and redesign locally if either seed is near chance, if repeat-last matches
the candidate, or if corrupted/no-topology controls invalidate attribution.
Do not mechanically scale samples or move this panel to the remote GPU.

## Explicitly Blocked Routes

- Do not use PRESENT, GIFT or SKINNY homogeneous windows as evidence of distinct earlier topology use.
- Do not restore the stopped final-transition delta-U query; recurrent semantics are undefined for it.
- Do not add DDT, trail, related-key or partial-decryption features to this one-variable test.
- Do not change the negative definition, keys, data split or optimizer while comparing roles.
- Do not claim an attack, SOTA, universal SPN transfer or a breakthrough from this local diagnostic.

## Completed Readiness Result

The deterministic gate completed with:

```text
run_id = i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725
status = pass
decision = innovation1_runtime_spn_recurrent_window_readiness_passed
manifest_rows = 10
expected_rows = 10
build_errors = []
training_performed = false
```

All protocol checks passed. Every role has `442466` parameters and the same
named-parameter shape fingerprint. The correct full window has two unique
transitions and hash
`b1e4f9bbdc584111c267d4dea576374565c4c854f8f53d4c800410276a9468ba`;
the repeated-final control has one unique transition and hash
`6babb65cb5d8e71b994a1edb305c4ba94c3462648115f4ee20af4b9122e4d006`.
Both end in transition
`3de856d9cc7da00f6fd986d42be309bdac42906011da7d1cda077ab5c8bfe6b5`.
The corrupted window is separately fingerprinted as
`889818870924236f6a0536c01f8ef3f10f930bf271db1f90b92166a06a445029`.

Artifacts:

```text
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/manifest.jsonl
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/gate.json
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/validation.json
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/summary.json
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/progress.jsonl
```

This proves plan/model construction readiness only. It adds no AUC evidence
and does not change `execution_authorized = no`.

## Recommended Next Action

Complete deterministic readiness now and keep training held. When RTG3-A is
adjudicated, record whether its decision authorizes this local mechanism test;
only then freeze the AUC gate and execute the unchanged ten-row CSV.
