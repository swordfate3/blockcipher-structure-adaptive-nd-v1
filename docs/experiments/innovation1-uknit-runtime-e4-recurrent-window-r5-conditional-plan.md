# Innovation 1 uKNIT Runtime-E4 Recurrent Window R5 Conditional Plan

Date: 2026-07-25

## Status

```text
phase = deterministic readiness only
readiness = pass; re-adjudicated after strict no-topology edge-gate repair
training = not started
execution_authorized = no
blocking_decision = RTG3-A seed0 and conditional seed1 adjudication
result_gate = implemented and preregistered; no U3 AUC has been read
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
| no-topology | `recurrent_window` | correct descriptor, identity cell adjacency | no-linear-topology control with S-box self-gating retained |

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

The empirical advance rule is now preregistered before training or reading any
U3 AUC. Both seeds must satisfy all five checks:

```text
candidate AUC >= 0.520
candidate - anchor AUC >= +0.005
candidate - repeat-last AUC >= +0.005
candidate - corrupted AUC >= +0.005
candidate - no-topology AUC >= +0.005
```

The `0.520` absolute floor comes from the completed same-budget uKNIT U2-C
state-triplet evidence, where both candidates retained AUC above `0.535`. The
`+0.005` attribution margin is the already frozen project convention used by
the uKNIT and Runtime-E4 control gates. RTG3-A determines whether the general
GF(2) runtime route is credible enough to authorize U3; its SKINNY AUC does not
tune a threshold for a different cipher after the fact.

After training, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/gate-runtime-spn-recurrent-window \
  --run-id i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_20260725 \
  --run-root outputs/local_diagnostic/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_20260725
```

The gate fails closed unless all ten rows preserve the frozen data, optimizer,
checkpoint, disk-cache, runtime-window and role contracts. It also checks all
ten ordered epoch histories and requires each reported AUC to equal the best
restored validation checkpoint.

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

The re-adjudicated readiness also executes a fixed `512-bit` forward/backward
probe for every model. All outputs have shape `[2, 1]` and are finite. Every
model has finite gradients for all `54/54` trainable parameter tensors and all
`442466/442466` trainable parameter elements. The five seed0 output hashes,
after repairing the edge-gate no-topology semantics, are:

```text
anchor       = 972e46a9524703d21c01fc3f55a56e96138ff55bf3df5d738ef01284ca396c41
candidate    = 78e4f6d56d6cee1741e418a5e1154ccf53a30e10df633428245de30115e67add
repeat_last  = ae765da771c122502c08bf208671d2db6ddf727e9136ebab2c83c86c357ad91f
corrupted    = 90fc5c8172fe8bf546e6005d137372f03021f114413bbd83b933bba1e59fe1cc
no_topology  = eb3ff3369d716ed8c3aba0dfa6064889a4d05f8383dbacf990c8bbf1b8452397
```

The same role hashes reproduce at seed1 because the readiness probe freezes
model and input initialization independently of the future training seed.
Their separation proves that each planned intervention reaches the computation
graph; it does not rank the roles or provide performance evidence.

The previous no-topology probe hash
`6c4f0088adb6689c0c2a7ed32ea5146fcf7095d7ae522fea3ce63c878bde6adb`
is superseded. That path disabled exact inverse-linear views but still read the
runtime linear graph inside `edge_gate`, so it was not a strict no-topology
control. The repaired control uses identity cell adjacency: it keeps the same
S-box self-gating operation and parameter geometry but cannot read cross-cell
linear edges. End-to-end regression tests require both last-transition and
recurrent-window independent logits to remain bit-exact when only the runtime
linear topology is changed, while the corresponding true-topology logits must
change. No U3 training had started, so no AUC result was invalidated or needs
reproduction.

Artifacts:

```text
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/manifest.jsonl
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/gate.json
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/validation.json
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/summary.json
outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725/progress.jsonl
```

This proves plan/model construction plus finite forward/backward readiness
only. It adds no AUC evidence and does not change
`execution_authorized = no`.

## Recommended Next Action

Keep training held while RTG3-A is unresolved. Its completed local decision
must explicitly authorize or stop this route. If authorized, execute the
unchanged ten-row CSV and apply the already frozen result gate without changing
thresholds. A pass advances only to a same-checkpoint window-swap attribution
audit; a hold stops scale-up and returns to local redesign.
