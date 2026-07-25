# Innovation 1 Dialga-128 Runtime-E4 D2 Same-Checkpoint Plan

Date: 2026-07-25

## Status

```text
phase = completed inference-only attribution audit
source = completed plan-aligned D1 two-seed pass
training = prohibited
remote_scale = prohibited
result = pass
decision = innovation1_dialga_runtime_e4_d2_functional_topology_use_supported
```

## Research Question

Do the two D1 correct-topology checkpoints functionally use Dialga's exact
runtime structure at inference time, or did the separately trained D1 roles
only acquire different solutions because of optimization variance?

D2 changes exactly one variable: the runtime structure intervention used by a
frozen checkpoint. Weights, validation features, labels, batch protocol and
metric computation remain fixed within each seed.

## Frozen Source

```text
source_run = i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725
source_root = outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725
source_decision = innovation1_dialga_runtime_e4_d1_two_seed_supported
source_model = runtime_spn_e4_equivariant_true
source_checkpoints = row0001 seed0 and row0004 seed1
```

Before evaluation, D2 must recompute the D1 gate from `results.jsonl`, require
an exact match with the persisted D1 gate, and require every D1 protocol and
research check to pass. File existence or a copied decision string is not
enough.

## Frozen Panel

For each seed, load its correct D1 best checkpoint once and evaluate the exact
same validation cache under three conditions:

| Condition | Runtime structure | Relation mode | Purpose |
| --- | --- | --- | --- |
| correct | exact Dialga rounds 2 and 3 | `true` | reference |
| corrupted | deterministic source-bit corruption, seed `20260725` | `true` | wrong-topology control |
| no-topology | exact Dialga descriptor | `independent` | identity cell-adjacency control |

The no-topology condition intentionally retains Dialga's cell/S-box
description while disabling inverse-linear recovery and cross-cell topology.
Its structure fingerprint therefore matches `correct`, but its intervention
fingerprint must differ because the relation mode is different.

Frozen protocol:

```text
cipher = Dialga-128 prefix r4
runtime descriptor = configs/runtime/spn/dialga128.json
runtime window = round_start 2, rounds 2
cell input = state_triplet
S-box context = edge_gate
round window mode = recurrent_window
pairs per sample = 4
validation = the exact D1 seed-specific 1024/class cache
validation total = 2048 rows per seed
input width = 1024 bits
pair width = 256 bits
negative definition = encrypted random plaintext pairs
checkpoint = restored D1 best validation-AUC checkpoint
seeds = 0, 1
training performed = false
execution = local CPU inference-only audit
```

## Protocol Gate

D2 is valid only if all of the following hold:

1. Exactly six rows exist: two seeds by three conditions.
2. All three rows within a seed share the exact checkpoint, feature, label and
   cache-metadata SHA256 values.
3. The two seed groups use the expected distinct D1 correct checkpoints.
4. Checkpoints report `selected_checkpoint=best`, load with
   `strict=True`, and retain the frozen `442466`-parameter geometry.
5. The source D1 result and gate hashes are present and identical across all
   rows, and the recomputed D1 source gate passes.
6. Descriptor name, SHA256, two-round window, two heterogeneous transitions,
   condition modes and intervention fingerprints match the frozen panel.
7. The validation cache is Dialga-128 r4 with `2048` rows, `1024` input bits,
   four pairs, difference `0x40`, fixed validation key, and encrypted random
   plaintext negatives.
8. The correct-condition AUC exactly reproduces the corresponding D1 reported
   best validation AUC within `1e-12`.
9. Every AUC and prediction-delta value is finite and no training occurs.

## Research Gate

Both seeds must independently satisfy:

```text
correct AUC >= 0.520
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
max probability change under corrupted structure > 1e-6
max probability change under no-topology structure > 1e-6
```

Do not average away a failed seed. Probability sensitivity alone is not a pass;
the correct structure must also retain both AUC margins.

## Decision Routes

- Pass: classify exact Dialga runtime-topology use as supported at local D2
  scale, then preregister one same-budget adjacent-window replication on
  Dialga prefix r5. Do not increase samples first.
- Protocol failure: repair only the audit implementation or source binding;
  keep all checkpoints, caches and thresholds unchanged.
- Sensitivity without margins: retain evidence that the checkpoint reads the
  intervention, but classify discriminative attribution as unsupported and
  redesign locally.
- No prediction sensitivity: classify the D1 training-role advantage as not
  functionally attributable under the frozen checkpoint and stop Dialga scale
  advancement.

## Explicitly Blocked

- No retraining, checkpoint reselection or calibration inside D2.
- No new validation examples, keys, differences, pairs, epochs or seeds.
- No remote GPU launch or medium/formal scale-up.
- No DDT, trail, partial-decryption or guessed-key features.
- No claim of a Dialga attack, paper reproduction, SOTA result, formal
  cross-cipher result or universal SPN breakthrough.

## Execution

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-runtime-spn-dialga-d2 \
  --run-id i1_dialga128_runtime_e4_d2_same_checkpoint_20260725 \
  --d1-root outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725 \
  --output-root outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725 \
  --device cpu
```

After execution, visually inspect the rendered SVG with `visual-qa-redraw`,
refresh both recent-result indexes, record the metrics and evidence-backed next
action here, then run the focused regression suite before commit and push.

## Completed Result

```text
run_id = i1_dialga128_runtime_e4_d2_same_checkpoint_20260725
status = pass
decision = innovation1_dialga_runtime_e4_d2_functional_topology_use_supported
result_rows = 6
training_performed = false
source_gate_recomputed = true
```

| Seed | Correct | Corrupted | No topology | Correct - corrupted | Correct - no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.958417 | 0.925256 | 0.517403 | +0.033161 | +0.441014 |
| 1 | 0.958679 | 0.918380 | 0.526351 | +0.040299 | +0.432328 |

Maximum absolute probability changes relative to the correct structure:

| Seed | Corrupted structure | No topology |
| ---: | ---: | ---: |
| 0 | 0.912204 | 0.918676 |
| 1 | 0.902221 | 0.869230 |

Every protocol check passed. Within each seed, all three conditions share the
same checkpoint, feature, label and metadata hashes. The correct condition
reproduces the corresponding D1 AUC exactly, both checkpoints load strictly,
the source D1 gate recomputes exactly, and no training or new data generation
occurs. The condition and intervention fingerprints prove that `corrupted`
changes the runtime matrices while `no_topology` keeps the exact descriptor but
changes the relation mode to `independent`.

Both seeds exceed the absolute AUC floor and both `+0.005` margins. Prediction
probabilities also change far above `1e-6`. D2 therefore supports the narrow
claim that the frozen D1 checkpoints functionally use Dialga's externally
supplied runtime topology on the prefix-r4 validation protocol; D1's advantage
cannot be explained only by separately trained model variance.

The final SVG was rendered to pixels and passed `visual-qa-redraw` after its
legend was moved away from the seed0 bar labels and the remaining English
`checkpoint` title was replaced with `同一检查点`. The delivered figure has no
text overlap, clipping, ambiguous legend, unreadable labels, misleading scale
or missing glyphs.

Artifacts:

```text
outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725/results.jsonl
outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725/progress.jsonl
outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725/validation.json
outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725/gate.json
outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725/summary.json
outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725/curves.svg
```

## Recommended Next Action

Preregister D3 as one same-budget Dialga prefix-r5 adjacent-window replication:
keep Runtime-E4, input difference, pairs, `2048/class`, validation size, loss,
optimizer, epochs, keys, seeds and all three controls unchanged; change only
the encrypted prefix from r4 to r5 and align the two-transition runtime window
from rounds 2/3 to rounds 3/4. Run locally with disk cache.

D3 should advance only if both seeds again satisfy `correct AUC >= 0.520`,
`correct - corrupted >= +0.005` and `correct - no-topology >= +0.005`, followed
by the same-checkpoint audit if training passes. A miss classifies the Dialga
mechanism as window-specific. Do not increase the sample budget, move to the
remote GPU, add features, or mix a network redesign into this replication.
