# Innovation 1 uKNIT-Family CT-SPN Endpoint-Alignment K1-A

**Date:** 2026-07-28
**Run ID:** `i1_uknit_family_ctspn_endpoint_alignment_k1a_20260728`
**Status:** completed / pass
**Prerequisite:** completed K1 `hold` gate with all protocol checks passing

## 1. Question

K1 showed two different failures under the same frozen protocol:

- uKNIT-BC prefix-r5: CT-SPN stayed near chance and fell below Runtime-E4;
- Dialga-128 prefix-r4: CT-SPN reached about `0.9635` AUC, but repeating the
  final transition or rotating the two-transition schedule changed the AUC by
  effectively zero.

K1-A tests one exact mismatch before any further training:

> Does the current edge-permutation-invariant pooling remove native endpoint
> placement and transition identity even though the supplied runtime structure
> and raw canonical edge values change?

The audit does not test a new model, data source, difference, loss, seed, epoch
count or scale. It replays the selected K1 candidate checkpoints without an
optimizer step.

## 2. Frozen Evidence

```text
source plan       = innovation1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1.csv
source gate       = K1 hold / protocol-clean
ciphers           = uKNIT-BC prefix-r5, Dialga-128 prefix-r4
seeds             = 0, 1
source role       = CT-SPN K1 candidate only
checkpoints       = selected best val_auc checkpoints from K1
probe rows        = 32 deterministic binary samples per cipher and seed
probe seed base   = 20260728
controls          = repeat_last, rotated
training rows     = 0
optimizer steps   = 0
device            = local CPU
```

For every cipher, seed and control, K1-A compares the correct ordered model and
the controlled model at four stages:

1. native endpoint identity `(target cell, target bit role, source cell,
   source bit role)`;
2. the 12-value canonical edge input before the learned encoder;
3. the mean/max/RMS edge-pooled transition summary after the frozen K1 edge
   encoder and mixer;
4. the final frozen K1 logit.

Every control must strict-load the exact same K1 state dictionary. The audit
must record the checkpoint and state-dictionary hashes and must not call a
training loop.

## 3. Gate

Protocol validity requires:

- exact K1 run id, `hold` decision and all K1 protocol checks passing;
- four candidate checkpoints, one per cipher and seed, selected by `val_auc`;
- exactly eight result rows: two ciphers x two seeds x two controls;
- the current K1 edge encoder has exactly 12 input values and therefore no
  native endpoint identity channel;
- identical learned state for correct and controlled evaluations;
- deterministic finite probe metrics and zero optimizer steps.

The alignment-loss hypothesis is supported only if both seeds show:

```text
Dialga repeat_last native endpoint fraction changed >= 0.45
Dialga rotated     native endpoint fraction changed >= 0.95
Dialga raw edge values change under both controls
Dialga pooled transition-summary max delta <= 1e-5
Dialga final-logit max delta <= 1e-4
K1 Dialga candidate-minus-repeat/rotated AUC magnitude <= 1e-5

uKNIT repeat_last native endpoint fraction changed >= 0.45
uKNIT rotated     native endpoint fraction changed >= 0.95
K1 uKNIT candidate AUC remains below the registered 0.520 floor
```

These checks deliberately do not require uKNIT's pooled summary to collapse.
K1 already showed that its candidate is near chance. K1-A asks whether the
representation omits the native placement signal that the next candidate would
need, while Dialga supplies the exact collapse attribution surface.

## 4. Decisions

- **Gate passes:** retain the canonical exact-state views, but replace the
  edge-identity-free token with a fixed-width native endpoint representation.
  Plan K1-B at the identical `2048/class`, two-seed, four-pair, ten-epoch
  protocol. Change only the edge token by adding normalized native cell
  position and four-way input/output bit-role channels. Require uKNIT
  improvement and correct-order dominance, while retaining Dialga AUC.
- **Endpoint identities do not change:** repair the factor-to-native indexing;
  do not train K1-B.
- **Dialga summaries or logits do not collapse:** reject the proposed root
  cause and inspect the temporal aggregation or control construction instead;
  do not train K1-B.
- **Protocol failure:** repair only the replay/audit implementation and rerun.

## 5. Blocked Routes

- Do not increase samples, pairs, epochs or model width from K1-A.
- Do not launch remote training from a zero-training audit.
- Do not start K2 S-box composition until a position-preserving K1 candidate
  passes both ciphers and both seeds.
- Do not add MoE, DDT, trail, partial decryption, guessed keys or cipher-id
  routing.
- Do not use Dialga's high absolute AUC to hide uKNIT's failed seed-level gate.
- Do not include generalized-Feistel MSX in this CT-SPN claim.

## 6. Artifacts

```text
outputs/local_audit/i1_uknit_family_ctspn_endpoint_alignment_k1a_20260728/
  results.jsonl
  progress.jsonl
  validation.json
  gate.json
  summary.json
```

The required next action must be written into the gate, summary, this document
and the user-facing report. The completed audit must be added to
`outputs/00_RECENT_RESULTS.md` and `outputs/00_RECENT_RESULTS.json` before it is
reported.

## 7. Completed Result

K1-A completed on 2026-07-28 with four selected K1 candidate checkpoints,
eight control rows, zero training rows and zero optimizer steps. All nine
protocol checks and all thirty research checks passed:

```text
decision = innovation1_uknit_family_ctspn_endpoint_alignment_loss_confirmed

Dialga repeat_last:
  native endpoint fraction changed = 0.489583
  transition-summary max delta      = 7.15e-7
  final-logit max delta              = 2.62e-6 / 2.38e-6

Dialga rotated:
  native endpoint fraction changed = 0.979167
  transition-summary max delta      = 7.15e-7
  final-logit max delta              = 3.46e-6 / 2.07e-6

uKNIT repeat_last native endpoint fraction changed = 0.492188
uKNIT rotated     native endpoint fraction changed = 0.984375
```

Raw canonical edge values changed by `1.0` under every control, so the collapse
does not come from an unchanged descriptor or failed intervention. It occurs
between the raw edge values and the edge-permutation-invariant pooled summary.

The result is entry `001` in the recent-result index at completion time. It
authorizes only K1-B's fixed-width native endpoint token at the identical local
budget. It does not authorize remote scale, K2, MoE or any benchmark change.
