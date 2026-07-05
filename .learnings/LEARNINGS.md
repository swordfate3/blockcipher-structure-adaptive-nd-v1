## [LRN-20260621-001] correction

**Logged**: 2026-06-21T20:45:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: docs

### Summary
Do not treat SPN/PRESENT small or medium sample runs as formal evidence that a model or route has failed.

### Details
The user corrected an important experimental interpretation error: prior SPN/PRESENT work was discussed too much like formal training, but local completed metrics show PRESENT/SPN results had only reached about `65536` samples per class. Several logs mentioning `131072` or `262144` rows were often total rows across both classes, cache/progress rows, queue plans, or incomplete runs, not completed `>100000/class` formal training.

Correct project distinction:

- Smoke/screen: below `65536/class`.
- Medium diagnostic: `65536/class` through about `524288/class`.
- Formal training: at least `1000000/class`, preferably multi-seed, fixed protocol, completed, retrieved, and plan-aligned.

For SPN/PRESENT, before claiming a route has hit its ceiling or failed, require completed and retrieved scale evidence. A `32k/class` or `65k/class` result may reject only obviously dead variants; it must not be used as a definitive conclusion that the overall route cannot improve.

Current factual baseline from the 2026-06-21 audit:

- ARX/SPECK has completed results above `100000/class`, including `131072/class` and `262144/class`.
- SPN/PRESENT completed metric rows found locally maxed out around `65536/class`.
- Therefore, prior SPN/PRESENT accuracy bottleneck claims were under-supported by large-scale evidence.

### Suggested Action
For future SPN/PRESENT experiments, always state the scale class in reports and labels. Use small runs only as screens. Before making negative claims about accuracy ceilings, run and retrieve at least a medium scale ladder such as `65536/class -> 262144/class`, and reserve "formal result" language for `>=1000000/class` multi-seed completed runs.

### Metadata
- Source: user_feedback
- Related Files: outputs/, experiments/innovation1/plans/, experiments/innovation1/configs/remote/
- Tags: spn, present, experiment-scale, formal-training, accuracy-interpretation
- Pattern-Key: innovation1.spn_present.formal_scale_required
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260705-002] correction

**Logged**: 2026-07-05T23:14:37+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not treat a small set of similar SPN/PRESENT neural models as sufficient evidence for the "multiple neural networks combined" route.

### Details
The user corrected the interpretation of the neural ensemble direction: combining several nearby SPN-aware variants, such as Zhang/Wang-style raw MCND, InvP-only, and DDT graph, is only a near-neighbor score aggregation diagnostic. It is not the full "multiple diverse neural networks" idea.

Correct framing:

- Near-neighbor ensemble rows may show whether closely related SPN views have any error complementarity.
- A real diverse expert pool should intentionally combine different representation and architecture families: raw bit/MCND, InvP cell tokens, DDT/P-layer graph priors, pair-evidence pooling, inverse-round/integral matrix features, and projection/truncated weak features when they are weak-positive and low-correlation.
- The selection rule should depend on both per-expert quality and pairwise diversity/error overlap, not just model count.
- Do not mechanically add more models from the same family when the current pool is weak positive but below gate.

Current evidence motivating this rule:

```text
run_id = i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705
best_single = present_nibble_ddt_graph, AUC 0.789112608414
best_ensemble = probability_mean, AUC 0.790061685257
delta = +0.000949076843, below the +0.001 gate
decision = weak_neural_ensemble_positive_below_gate
```

This shows mild complementarity but does not prove that the broader diverse-neural-network route is exhausted.

### Suggested Action
Create and use a `diverse expert pool` plan for future SPN/PRESENT ensemble work. Require candidate expert-family metadata and diversity gates, for example max error Jaccard and at least one low-overlap non-neighbor expert, before deciding to scale an ensemble route.

### Metadata
- Source: user_feedback
- Related Files: docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md, docs/experiments/innovation1-present-diverse-expert-pool-plan.md
- Tags: innovation1, spn, present, neural-ensemble, diverse-experts, evidence-gates
- See Also: LRN-20260628-004, LRN-20260630-001
- Pattern-Key: innovation1.spn_present.diverse_expert_pool_not_near_neighbor_ensemble
- Recurrence-Count: 1
- First-Seen: 2026-07-05
- Last-Seen: 2026-07-05

---

## [LRN-20260705-003] correction

**Logged**: 2026-07-05T23:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research

### Summary
Do not automatically promote the user's proposed SPN route; independently rank it against current literature and local evidence.

### Details
The user corrected the collaboration style for Innovation 1: the agent should not merely follow the latest suggested direction, such as diverse neural-network aggregation. It should actively inspect local evidence, re-check relevant literature, and decide whether that route deserves the next experiment slot.

Correct framing:

- Treat user-proposed routes as hypotheses, not commands to promote them into the main branch.
- Compare candidate routes against the strongest current evidence and recent literature before spending remote GPU time.
- Keep diverse neural aggregation as a valid route, but only as a secondary validator until compatible weak-positive, low-correlation non-neighbor experts exist.
- Prefer SPN-aware data/feature representation when literature and local evidence point there more strongly than model-family aggregation.

Current route correction after the 2026-07-05 literature refresh:

```text
SPN feature/input search > structure-aware architecture > diverse ensemble
```

The same turn also retrieved a completed r8 integral/inverse screen:

```text
run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705
raw integral anchor AUC = 0.999995831400
InvP matrix AUC = 0.513465017546
InvP+Sinv matrix AUC = 0.505787684582
decision = stop_integral_inverse_feature_screen_for_now
interpretation = integral/multiset data structure signal, not inverse-round architecture gain
```

### Suggested Action
Before launching the next Innovation 1 SPN/PRESENT experiment, write or update the route document with a ranked decision that includes: local result status, same-budget baseline, literature support, expected claim scope, and why the chosen route beats the alternatives. Do not spend the next remote slot on a wider ensemble unless a non-neighbor expert family already has compatible weak-positive score artifacts.

### Metadata
- Source: user_feedback
- Related Files: docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md, sources/research_spn_adaptation_20260705.md, docs/experiments/innovation1-present-diverse-expert-pool-plan.md
- Tags: innovation1, spn, route-selection, literature-refresh, independent-judgment, neural-ensemble
- See Also: LRN-20260705-002, LRN-20260630-001
- Pattern-Key: innovation1.spn_present.independent_literature_ranked_route_selection
- Recurrence-Count: 3
- First-Seen: 2026-07-05
- Last-Seen: 2026-07-06
- Promoted: AGENTS.md

### Resolution
- **Resolved**: 2026-07-06T00:00:00+08:00
- **Commit/PR**: pending
- **Notes**: Promoted a concise rule to `AGENTS.md`: user-proposed routes are hypotheses that must be ranked against literature, local evidence, same-budget baselines, and controls before consuming meaningful experiment slots. Reaffirmed on 2026-07-06 with an independent SPN route re-rank and a baseline-gated neural follow-up smoke plan.

---

## [LRN-20260706-001] best_practice

**Logged**: 2026-07-06T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Interpret the r8 matched-negative raw-pair integral signal as a deterministic SPN/multiset feature candidate, not a neural architecture win.

### Details
After the explicit pair-xor parity leak was removed with `plaintext_integral_nibble_matched_negative`, two local tiny smoke probes still showed raw-pair signal:

```text
seed0 raw-pair AUC = 0.805480957031
seed1 raw-pair AUC = 0.877990722656
```

Simple pair-alignment statistics did not explain the residual, but the follow-up deterministic feature-bank audit did. At `2048/class`, the same scalar statistic dominated both checked audit seeds:

```text
seed0 best statistic = pair_xor_column_sum_variance
seed0 best threshold accuracy = 0.979248046875
seed1 best statistic = pair_xor_column_sum_variance
seed1 best threshold accuracy = 0.982421875
```

Correct interpretation:

- The raw matched-negative neural smoke likely learned a simple pair-xor column-distribution variance statistic.
- This is useful SPN-aware input/feature evidence, but not evidence of an architecture breakthrough.
- The route should not consume a remote neural training slot until additional controls pass.

Pair-order scramble control then tested whether the statistic came only from same-index fixed-difference pairing:

```text
matched anchor pair_xor_column_sum_variance accuracy = 0.979248046875
scrambled-positive pair_xor_column_sum_variance accuracy = 0.8818359375
```

This weakens but does not remove the statistic. Same-index fixed-difference pairing appears to be a major amplifier, while the active-nibble integral multiset construction still leaves a strong residual column-distribution signal.

Clean active/difference variation with `plaintext_integral_nibble_difference_matched_negative` then removed the left/right column-sum mismatch for off-active differences. At `2048/class`, audit seed `17`:

```text
active0 + Zhang/Wang diff 0x9 accuracy = 0.81494140625
active1/7/15 + Zhang/Wang diff 0x9 accuracy ~= 0.52
active0 + AutoND / entropy / Wang-Jain differences accuracy ~= 0.51-0.53
```

This narrows the route further: the deterministic statistic is not a generic integral-multiset signal. It is strongest when the active integral nibble is aligned with the fixed input-difference support.

Aligned active-difference controls then showed that the effect is not unique to Zhang/Wang `0x9`; it also appears for other single-nibble differences when active nibble is aligned:

```text
Zhang/Wang diff 0x9 active0 accuracy = 0.81494140625
AutoND diff 0x0d000000 active6 accuracy = 0.804443359375
Entropy diff 0x00d00000 active5 accuracy = 0.804443359375
Wang/Jain two-nibble diff active2/14 accuracy ~= 0.518
```

Current narrower interpretation: the route is a single-nibble aligned active-difference deterministic feature candidate. It does not currently support a two-nibble difference under the one-active-nibble construction.

A second aligned active-difference audit seed preserved the split:

```text
Zhang/Wang diff 0x9 active0 accuracy = 0.805908203125
AutoND diff 0x0d000000 active6 accuracy = 0.79296875
Entropy diff 0x00d00000 active5 accuracy = 0.8056640625
Wang/Jain two-nibble diff active2/14 accuracy ~= 0.518
```

This strengthens the local deterministic route decision: first make `pair_xor_column_sum_variance` an explicit baseline and design a multi-active-cell control for multi-nibble differences; do not spend the next meaningful slot on a wider neural ensemble.

The fixed deterministic baseline evaluator has now been implemented:

```text
script = scripts/evaluate-integral-deterministic-baseline
api = integral_deterministic_baseline_from_task
default statistic = pair_xor_column_sum_variance
```

Future neural follow-ups should compare against this fixed statistic and should not treat a best-of-feature-bank result as neural architecture evidence.

The fixed baseline now reports AUC. At `2048/class`, audit seed `23`, the
single-nibble aligned controls have strong deterministic AUC:

```text
Zhang/Wang aligned active0 AUC = 0.8878759145736694
AutoND aligned active6 AUC = 0.8747416734695435
Entropy aligned active5 AUC = 0.8852955102920532
```

Future neural follow-ups on this route must beat or explain this fixed baseline
instead of merely showing a raw-pair neural signal.

The multi-active-cell control for Wang/Jain two-nibble differences was then
implemented and tested locally:

```text
sample_structure = plaintext_integral_multi_nibble_difference_matched_negative
pairs_per_sample = 256
seed29 pair_xor_column_sum_variance accuracy = 0.58203125
seed31 pair_xor_column_sum_variance accuracy = 0.59765625
seed29 feature-bank best accuracy = 0.58203125
```

This weak result means the two-nibble Wang/Jain integral route should not take
the next meaningful remote slot; keep the single-nibble aligned active-
difference route primary unless a different multi-cell statistic emerges.

### Suggested Action
Before scaling the r8 matched-negative integral route, use `pair_xor_column_sum_variance` AUC as an explicit deterministic baseline. Keep single-nibble aligned active-difference as the primary deterministic feature route; do not spend the next meaningful slot on the tested Wang/Jain two-nibble integral route unless a new multi-cell statistic or representation changes the evidence.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-integral-parity-control-plan.md, docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md, src/blockcipher_nd/cli/audit_integral_parity_signal.py
- Tags: innovation1, spn, present, integral, deterministic-feature, matched-negative
- See Also: LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.matched_negative_raw_pair_feature_bank_explains_signal
- Recurrence-Count: 8
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-002] best_practice

**Logged**: 2026-07-06T01:30:57+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not re-add PRESENT single-step structural inverse-S as a new GPD feature; it already exists as the Sinv encoding.

### Details
During the independent SPN route re-check, the next plausible literature-backed
direction was Generic Partial Decryption / partial inverse feature engineering.
Code inspection showed that the repository already has the key single-step
zero-key inverse route:

```text
present_pair_xor_paligned_sinv_cell_matrix_bits
```

Its helper computes:

```text
S^{-1}(P^{-1}(C)) xor S^{-1}(P^{-1}(C'))
```

Therefore adding a new feature under a GPD name with the same semantics would
duplicate existing evidence and risk falsely presenting an old route as a new
SPN adaptation. The better local screen is to use existing multi-round DDT /
partial-inverse candidate-path encodings, such as:

```text
present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits
present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits
```

A first seed0 local smoke at `128/class` found only a tiny weak-positive beam
candidate:

```text
InvP control AUC = 0.496337890625
Sinv control AUC = 0.44287109375
DDT beam AUC = 0.5145263671875
DDT beamstats AUC = 0.462158203125
```

This is not remote-launch evidence. It only suggests that expanded DDT beam
paths are a more plausible next local repeat than compressed beamstats.

The seed1 repeat used the same protocol and changed only `seed = 1`:

```text
InvP control AUC = 0.54296875
Sinv control AUC = 0.5361328125
DDT beam AUC = 0.527587890625
DDT beamstats AUC = 0.606689453125
```

This keeps the GPD-style branch alive as a local representation candidate, but
it also shows the `128/class` validation setting is high variance: beamstats
was the best seed1 row but below random on seed0, while the expanded DDT beam
row was weak-positive in both seeds but did not consistently beat controls.
The route should be held for a larger local diagnostic, not remote-launched.

A `512/class` local diagnostic then reduced the variance and changed the route
reading:

```text
InvP control AUC = 0.540496826171875
Sinv control AUC = 0.5286407470703125
DDT beam AUC = 0.562957763671875
DDT beamstats AUC = 0.5418472290039062
```

The expanded DDT beam now beats all controls in this local diagnostic, while
the compressed beamstats seed1 spike did not reproduce. This is still local
diagnostic evidence only, but the next GPD-style step should be a `512/class`
seed1 repeat of the same matrix rather than dropping the branch or jumping
straight to remote scale.

The `512/class` seed1 repeat then corrected that narrow reading:

```text
InvP control AUC = 0.5263595581054688
Sinv control AUC = 0.56329345703125
DDT beam AUC = 0.51806640625
DDT beamstats AUC = 0.5724639892578125
```

Across the two `512/class` diagnostics, expanded DDT beam is not stable: it
wins seed0 but loses to both Sinv and beamstats on seed1. The compressed
beamstats row now has the best two-seed mean AUC (`0.5571556091308594`), but
its 128/class behavior was highly volatile and its 512 seed0 margin over InvP
was only about `+0.00135` AUC. Therefore the branch remains local diagnostic
only. Beamstats may be kept as a lightweight local candidate or future
non-neighbor expert source, but not as remote scale-up evidence yet.

### Suggested Action
When continuing the GPD-style branch, compare against the existing Sinv control
and prefer multi-round DDT/partial-inverse path statistics only when they beat
controls across repeated local diagnostics. After the `512/class` seed1 repeat,
do not launch this GPD-style branch remotely. If continuing it, run a lean local
confirmation or attribution check for the compressed beamstats row; demote the
expanded DDT beam unless a new attribution explains the seed1 failure.

### Metadata
- Source: conversation
- Related Files: src/blockcipher_nd/features/encoders/present_matrix.py, src/blockcipher_nd/features/encoders/present_sbox_ddt.py, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke.csv, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke_seed1.csv, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed0.csv, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv, docs/experiments/innovation1-present-r8-gpd-style-beamstats-plan.md
- Tags: innovation1, spn, present, gpd, partial-inverse, sinv, ddt-beam
- See Also: LRN-20260705-003, LRN-20260706-001
- Pattern-Key: innovation1.spn_present.gpd_do_not_duplicate_existing_sinv
- Recurrence-Count: 4
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06T02:35:00+08:00

---

## [LRN-20260706-003] best_practice

**Logged**: 2026-07-06T01:10:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat the r8 GPD-style beamstats row as an unstable local weak-expert candidate, not as a remote scale-up route.

### Details
The GPD-style beamstats diagnostic produced a weak two-seed local hint, but
semantic attribution did not identify a stable simple driver:

```text
512/class seed0 beamstats AUC = 0.5418472290039062
512/class seed1 beamstats AUC = 0.5724639892578125
2048/class attribution seed0 best scalar = confidence_std, AUC 0.5216715335845947
2048/class attribution seed1 best scalar = cumulative_mean, AUC 0.523328423500061
```

The best semantic scalar changes across seeds and stays near chance. This means
beamstats may still be useful as a future non-neighbor score artifact, but it is
not interpretable or stable enough to justify a `65536/class` remote launch
from the current branch.

Correct framing:

- Keep beamstats as a possible future diverse-expert family only if compatible
  weak-positive score artifacts and low-overlap/error-correlation evidence are
  produced.
- Do not use the current beamstats result to justify mechanically wider
  near-neighbor ensemble work.
- Prefer controlled SPN feature/input attribution before remote scaling.

### Suggested Action
If continuing this branch, run a lean local composite probe or diversity-score
check first. Otherwise return to the stronger controlled SPN feature/input
route and require any neural follow-up to beat or explain its deterministic
baseline.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-present-r8-gpd-style-beamstats-plan.md, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, gpd, beamstats, attribution, diverse-experts
- See Also: LRN-20260705-002, LRN-20260705-003, LRN-20260706-001, LRN-20260706-002
- Pattern-Key: innovation1.spn_present.beamstats_local_candidate_not_scale_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260630-001] correction

**Logged**: 2026-06-30T00:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research

### Summary
Long-term Innovation 1 goals should be framed at the SPN method level, not as an overly detailed experiment SOP.

### Details
The user corrected the goal framing: the real objective is to develop SPN-structure-adaptive neural networks or SPN-adaptive data/feature representations, mainly for PRESENT/SPN, not to encode one specific run protocol, seed sequence, metric gate, remote launcher, or postprocess checklist as the long-term goal.

Correct framing:

- Long-term goal: iterate toward neural distinguishers that genuinely exploit SPN structure.
- Innovation can come from SPN-aware network architecture or SPN-aware data/feature construction.
- Concrete experiment details such as run id, seed, scale, remote config, checkpoint metric, and branch gate belong in `docs/experiments/`.
- Broader research route, method hypothesis, and literature synthesis belong in `docs/research/`.
- Positive results should move the project from exploration into confirmation, attribution, ablation, and formal evidence, rather than ending the research prematurely.

The long-term goal should answer "what kind of method are we trying to create?" The experiment plan should answer "which exact run tests the next hypothesis?"

### Suggested Action
Promote a concise rule to `AGENTS.md`: frame long-term Innovation 1 goals around SPN-structure-adaptive neural distinguishers and keep concrete run protocol details in experiment plans. Use `docs/自动化目标规则.md` as the high-level goal document, not as a replacement for per-run plans.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/自动化目标规则.md, docs/experiments/, docs/research/
- Tags: innovation1, spn, goal-framing, research-workflow, experiment-plans
- Pattern-Key: innovation1.goal_framing.method_level_not_run_sop
- Recurrence-Count: 1
- First-Seen: 2026-06-30
- Last-Seen: 2026-06-30
- Promoted: AGENTS.md

---

## [LRN-20260621-003] best_practice

**Logged**: 2026-06-21T20:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Promote only durable rules from `memory/` into `AGENTS.md`; keep experiment history and run-specific details in `memory/`.

### Details
The user approved the memory cleanup approach: do not merge all `memory/` content into `AGENTS.md`. Instead, extract compact, stable rules that affect future agent behavior. During the 2026-06-21 cleanup, repeated memory/task-plan rules were promoted for remote Windows GPU hygiene, monitor/retrieval workflow, evidence claim gates, and verification/workspace hygiene.

Specific promoted rule groups:

- Remote artifacts and generated project files must stay under `G:\lxy`.
- Windows remote schedule commands must use `cmd.exe /c`, not `cmd.exe /k`.
- After remote launch/handoff, main thread should not SSH-poll; use tmux/watchers/monitors and controlled gates.
- Result reports must distinguish planned, running, completed remotely, fallback-retrieved, verified-branch retrieved, and plan-aligned.
- Strict SPN/PRESENT claims require claim gates, encrypted-random-plaintext negatives, and explicit qualification for multi-query/application-level evidence.
- Use `uv run pytest ...`; keep project-root `tmp_*` clean.

### Suggested Action
When future memory files grow, periodically scan for repeated durable rules and promote only those concise rules to `AGENTS.md`. Leave detailed run ids, timestamps, and transient experiment states in `memory/` or `progress.md`.

### Metadata
- Source: user_feedback
- Related Files: memory/, task_plan.md, progress.md, AGENTS.md
- Tags: memory, agents, remote-workflow, evidence-gates, workspace-hygiene
- See Also: LRN-20260621-002
- Pattern-Key: workflow.memory_to_agents.promote_only_durable_rules
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260621-002] best_practice

**Logged**: 2026-06-21T20:43:13+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
When reading old conversation, handoff, progress, or memory documents, persist important corrections with the self-improvement workflow.

### Details
The user clarified that previous memory-document reading should use the `self-improvement` skill to store durable memory. Important findings from old dialogue, `memory/`, handoff summaries, `progress.md`, or result audits must not remain only in transient model context. If a finding changes future experimental interpretation, remote workflow, reporting language, or agent behavior, log it to `.learnings/LEARNINGS.md` using the skill format and promote concise operational rules to `AGENTS.md` when broadly applicable.

This is especially important for long-running Innovation 1 work because context-window loss and thread restarts have already caused confusion about what had actually completed, what scale counts as formal, and whether remote results were retrieved.

### Suggested Action
Before and after reading historical memory files for a major task, check whether any conclusion should be persisted. Use `.learnings/LEARNINGS.md` for detailed context and `AGENTS.md` for short rules that future agents should obey immediately.

### Metadata
- Source: user_feedback
- Related Files: memory/, progress.md, task_plan.md, .learnings/LEARNINGS.md, AGENTS.md
- Tags: memory, handoff, self-improvement, context-window, project-rules
- See Also: LRN-20260621-001
- Pattern-Key: workflow.memory_reading.persist_with_self_improvement
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260622-001] correction

**Logged**: 2026-06-22T11:40:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: infra

### Summary
After every completed repository modification, make a scoped git commit and push when a remote exists so the workspace does not accumulate dirty state.

### Details
The user corrected a workflow failure: remote experiments are intended to pull code from GitHub, but the workspace had accumulated many uncommitted changes. To avoid committing unrelated dirty files, a remote run was started with `scp` overlays into `G:\lxy`, which made the run less reproducible than a clean GitHub commit-based launch.

Correct workflow:

- Complete any repository edit, including code, config, scripts, tests, README/docs, `.learnings/`, `AGENTS.md`, generated project files, or memory-rule updates.
- Run appropriate verification.
- Commit only the scoped files for the completed task.
- Push the branch to the remote repository when a remote is configured; if no remote exists, report that push is not possible.
- Keep the workspace clean for agent-authored changes before starting new work or launching remote experiments.
- Remote experiments should default to a GitHub-pushed commit. Dirty/scp overlay launches are emergency-only and must be explicitly labeled as such in status reports and handoff notes.

This rule does not authorize reverting or committing unrelated user changes. If unrelated dirty files already exist, isolate the task's files in a scoped commit and report the remaining unrelated dirty state separately.

### Suggested Action
Promote this to `AGENTS.md` under workspace hygiene and remote workflow. After any file modification, run the relevant verification, make a scoped commit for the files just changed, and push if a remote is configured. Before future remote launches, run `git status --short`, ensure required files are committed and pushed, and avoid relying on scp overlays for normal experiment reproducibility.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, experiments/innovation1/configs/remote/, scripts/generated/remote/
- Tags: git, commit, push, workspace-hygiene, remote-reproducibility
- See Also: LRN-20260621-003
- Pattern-Key: workflow.git_commit_push_after_code_changes
- Recurrence-Count: 2
- First-Seen: 2026-06-22
- Last-Seen: 2026-06-24
- Promoted: AGENTS.md

### Recurrence Update
- **Updated**: 2026-06-24T15:08:01+08:00
- **Source**: user_feedback
- **Notes**: User explicitly reminded that every modification should be committed, not only code/config/script edits. This includes README/docs, `.learnings/`, `AGENTS.md`, and other memory or documentation updates. Push remains required when a remote is configured; if no remote exists, commit locally and report that push cannot be performed.

---

## [LRN-20260622-002] best_practice

**Logged**: 2026-06-22T13:05:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat SPN/PRESENT active-nibble accuracy as an auxiliary trail-activity signal, not as real-vs-random accuracy; promote it into an explicit active-pattern distinguisher route.

### Details
The user identified that the high `val_active_nibble_bit_accuracy` from the multitask active-nibble run should not be left as a side metric. It measures per-position correctness for 16 active/inactive nibble labels per sample, where "active" means a 4-bit differential/trail cell is non-zero. It is not ciphertext-value nonzero-ness, not whole-sample trail accuracy, and not real-vs-random classification accuracy.

Correct interpretation:

- `active nibble` means a nibble of a differential or candidate trail state is non-zero.
- The metric is averaged over all sample-position binary labels, e.g. validation rows times 16 positions.
- It can be inflated by inactive-class imbalance; always compare against all-inactive baseline and report active precision, recall, F1, and per-position rates.
- The research opportunity is to convert this auxiliary structure recognition into explicit real-vs-random evidence: active count, position frequency, candidate-trail disagreement, confidence, margin, pair-set consistency, and trail-family match scores.
- Strict evidence still requires `encrypted_random_plaintexts` negatives and separate reporting of single-sample raw accuracy, AUC, and multi-query aggregation.

### Suggested Action
Implement the active-pattern route as a staged Innovation 1 SPN plan:

1. Add deterministic active-pattern/statistics extraction from existing PRESENT beamstats/candidate-trail feature encodings.
2. Add diagnostics for active-label imbalance and real-vs-random distribution separation.
3. Train active-only and active-plus-candidate-statistics baselines before any large neural model.
4. If small/medium evidence is positive, run the route at 262144/class and then >=1000000/class multi-seed before making formal claims.

### Metadata
- Source: user_feedback
- Related Files: outputs/remote_results_incomplete/innovation1-spn-present-multitask-active-nibble-fast-gate-r7-gpu1-20260618/, src/blockcipher_ai_eval/features/pair_features.py, docs/superpowers/plans/
- Tags: innovation1, spn, present, active-nibble, trail-activity, distinguisher, evidence-gates
- See Also: LRN-20260621-001, LRN-20260621-002
- Pattern-Key: innovation1.spn_present.active_pattern_distinguisher
- Recurrence-Count: 2
- First-Seen: 2026-06-22
- Last-Seen: 2026-06-23

---

## [LRN-20260623-001] research

**Logged**: 2026-06-23T10:23:47+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
SPN/PRESENT active-only 24-dim active-pattern screen failed as a real-vs-random distinguisher despite the earlier high auxiliary active-nibble signal.

### Details
The active-pattern route completed and was retrieved from a verified result branch:

- Run ID: `innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622`
- Result branch: `results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622`
- Result commit: `6c3243137b821d5fc39d266d8aa5f39622ad4fdd`
- Local archive: `outputs/remote_results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622/`
- Gate: `result_lines=2`, `expected_rows=2`, `runner_exit_code=0`, `archive_integrity=pass`
- Config: `rounds=7`, `samples_per_class=65536`, `pairs_per_sample=16`, `negative_mode=encrypted_random_plaintexts`, `sample_structure=zhang_wang_case2_mcnd`
- Mean result: `accuracy=0.500000`, `AUC=0.491578`, `feature_dim=24`
- Seed 0: `val_accuracy=0.5`, `val_auc=0.4894024282693863`
- Seed 1: `val_accuracy=0.5`, `val_auc=0.4937528520822525`

Failure interpretation:

- The auxiliary target and the distinguisher target are different. Active-nibble prediction asks whether each 4-bit trail/difference cell is non-zero; real-vs-random classification asks whether a pair set came from the target differential encryption process or encrypted-random-plaintext negatives.
- The 24-dim active summary is too coarse. It keeps only active position frequencies and aggregate density statistics, but discards value-level S-box/DDT evidence, candidate trail scores, top-k margins, confidence, per-pair ordering, and cross-pair consistency.
- Under this representation, real and random samples can share almost the same active-density and position-frequency distribution, i.e. `P(active-summary | real) ~= P(active-summary | random)`. The retrieved AUC below `0.5` is direct evidence that the current active-only summary gives little or no separation.
- Earlier high `active_nibble_bit_accuracy` may be partly inflated by inactive-class imbalance. Per-position bit accuracy can be high when many positions are inactive, so active precision, recall, F1, balanced accuracy, and all-inactive baselines are required before treating it as strong structural evidence.
- Therefore, active-pattern should be used as auxiliary supervision or one component of richer candidate-trail evidence, not as a standalone final feature family.

This was only a screen/medium diagnostic at `65536/class`, not formal `>=1000000/class` evidence. However, because the active-only baseline is already at chance with strict `encrypted_random_plaintexts` negatives, do not scale this exact 24-dim linear route as the main next experiment.

### Suggested Action
Retire the active-only 24-dim linear baseline as a main scaling route. Keep active-nibble information, but combine it with features that measure whether candidate trails actually support the observed sample:

- Top-1/top-2 trail score and top-k margin
- Candidate score entropy and confidence
- Candidate disagreement across pairs
- Active-pattern-to-top-trail match
- Pair-set trail-family consistency
- Transition-spectrum features and multi-query score aggregation

For future reports, state plainly: high active-nibble auxiliary accuracy proves the model can learn trail-activity propagation patterns; it does not prove real-vs-random distinguishability.

### Metadata
- Source: conversation
- Related Files: src/blockcipher_ai_eval/features/spn_active_pattern.py, experiments/innovation1/run_spn_active_pattern_baseline.py, outputs/remote_results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622/
- Tags: innovation1, spn, present, active-pattern, active-nibble, real-vs-random, failure-analysis, evidence-gates
- See Also: LRN-20260622-002
- Pattern-Key: innovation1.spn_present.active_pattern_distinguisher
- Recurrence-Count: 1
- First-Seen: 2026-06-23
- Last-Seen: 2026-06-23

---

## [LRN-20260706-006] best_practice

**Logged**: 2026-07-06T02:40:04+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Before widening SPN/PRESENT ensembles, mine stable SPN-sensitive axes and score-distribution features.

### Details
The user corrected the collaboration style again: the agent should not simply
follow the user's current hypothesis, including the broader diverse-neural-
network aggregation idea. Re-checking local evidence and the literature notes
shows that the next main bottleneck is not model count. It is the absence of a
compatible, weak-positive, low-correlation non-neighbor expert family.

Current route reading:

```text
near-neighbor r7 ensemble = weak-positive below gate
projection v2 = unstable local priors
GPD/beamstats = weak local candidate only
candidate-trail = stopped at medium scale, weak local axes only
aligned integral route = explained by deterministic baseline
```

The better next route is a score-guided / sensitivity-guided SPN projection
audit: use candidate-evidence, beamstats, InvP(delta), and trail-family score
axes to select stable masks only if they survive seed/key stability and false-
family controls. Only then should the route train a small projection probe or
enter the diverse expert pool.

### Suggested Action
Create an SGP local audit before any new remote launch or wider ensemble. Gate
candidate masks by top-k composite AUC, seed/key stability, and shuffled/false-
family controls. Keep diverse expert aggregation as the secondary validator
after a genuine non-neighbor weak-positive score artifact exists.

### Metadata
- Source: user_feedback
- Related Files: docs/research/innovation1-spn-independent-route-recheck-20260706.md, docs/experiments/innovation1-present-diverse-expert-pool-plan.md, docs/experiments/innovation1-present-truncated-projection-feature-plan.md
- Tags: innovation1, spn, present, route-selection, projection, score-distribution, sensitivity, diverse-experts
- See Also: LRN-20260705-002, LRN-20260705-003, LRN-20260706-002
- Pattern-Key: innovation1.spn_present.stable_axis_before_diverse_ensemble
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-007] best_practice

**Logged**: 2026-07-06T03:40:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not advance raw-axis SGP projection after the r8 source sweep; try orbit/grouped InvP stability first.

### Details
The first PRESENT r8 SGP stable-axis source sweep ran at `2048/class`, seeds
`0` and `1`, strict `encrypted_random_plaintexts` negatives, and
`zhang_wang_case2_official_mcnd` sample structure. It tested three source
families:

```text
candidate_cell_structured
candidate_aggregate
invp_delta_bits
```

All three source reports returned `sgp_stable_axis_hold`:

| Source | Min composite AUC | Top-k Jaccard | Control delta |
|---|---:|---:|---:|
| `candidate_cell_structured` | `0.5262441635131836` | `0.14285714285714285` | `-0.0010590553283691406` |
| `candidate_aggregate` | `0.5145435333251953` | `0.16363636363636364` | `0.014543533325195312` |
| `invp_delta_bits` | `0.5609222650527954` | `0.0` | `0.06092226505279541` |

Correct interpretation:

- Candidate-cell evidence is weak and not better than the shuffled-cell control.
- Candidate-aggregate evidence is too weak to justify a projection smoke.
- InvP(delta) has weak-positive composite evidence, but exact flat bit-axis
  identity is unstable across seeds. This should not be forced through by
  weakening the raw-axis Jaccard gate.
- The next more plausible route is orbit/grouped stability over InvP(delta)
  axes, e.g. grouping by pair slot, SPN cell, P-layer orbit, or bit role.
- Multi-source SGP currently regenerates candidate evidence in memory without
  progress output; do not scale this audit path until SGP cache/progress
  support exists.

### Suggested Action
Hold raw-axis `sgp_top32_stable` projection and do not launch remote SGP. Add a
small orbit/grouped stability audit for InvP(delta) before any SGP projection
smoke. If grouped stability also fails, retire SGP as a projection route and
return to stronger representation priors.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-sgp-stable-axis-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_sgp_stable_axis_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, sgp, stable-axis, projection, invp, route-selection
- See Also: LRN-20260706-006, LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.sgp_raw_axis_hold_orbit_group_next
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-008] best_practice

**Logged**: 2026-07-06T04:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not promote grouped SGP after the InvP(delta) grouped/orbit follow-up; convert broad weak signal into intentional statistics instead.

### Details
The PRESENT r8 grouped/orbit SGP follow-up tested `invp_delta_bits` at
`2048/class`, seeds `0` and `1`, strict `encrypted_random_plaintexts`
negatives, and `zhang_wang_case2_official_mcnd` sample structure.

Final non-degenerate top4 artifact:

```text
outputs/local_audits/i1_present_r8_sgp_grouped_axis_audit_2048_top4.json
decision = sgp_grouped_axis_hold
best_group_scheme = word_bit_role
```

Summary table:

| Group scheme | Min composite AUC | Top-k Jaccard | Mask fraction |
|---|---:|---:|---:|
| `pair_word_cell` | `0.5344549417495728` | `0.0` | `0.0078125` |
| `word_cell` | `0.6075923442840576` | `0.14285714285714285` | `0.125` |
| `cell` | `0.6401443481445312` | `0.14285714285714285` | `0.25` |
| `word_bit_role` | `0.685741662979126` | `0.14285714285714285` | `0.5` |
| `p_layer_orbit` | `0.5724446773529053` | `0.0` | `0.09375` |

Correct interpretation:

- InvP(delta) contains broad weak separation, especially under coarse cell or
  bit-role aggregation.
- Exact pair-slot/cell and P-layer orbit groups are not stable across seeds.
- Coarse `word_bit_role` looks strongest but is too broad to be a projection
  expert; a degenerate full-width mask initially looked candidate-like when
  all 8 role groups were selected, so grouped SGP now has a
  `max_selected_axis_fraction` guard.
- This is not evidence for a remote SGP projection run and not a valid diverse
  expert for ensemble aggregation yet.

### Suggested Action
Retire SGP as the next immediate projection route. Use the broad weak
InvP(delta) signal as a hint for explicit pair/global statistics or
bit-role/cell distribution features, then compare that representation against
existing pairset/global-stat anchors before any remote launch. Keep diverse
neural aggregation secondary until a genuinely non-neighbor, weak-positive,
low-overlap expert exists.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-sgp-stable-axis-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_sgp_grouped_axis_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, sgp, grouped-axis, invp, route-selection, degeneracy-gate
- See Also: LRN-20260706-007, LRN-20260706-006, LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.sgp_grouped_hold_statistics_next
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-009] best_practice

**Logged**: 2026-07-06T05:20:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not promote existing InvP global activity statistics after the r8 local audit; they average away the grouped SGP signal.

### Details
After raw/grouped SGP held, the next local diagnostic tested explicit
`present_global_pairset_statistics` over `ciphertext_xor_spn_paligned_bits`.
Protocol:

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
seeds = 0, 1
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
```

Artifact:

```text
outputs/local_audits/i1_present_r8_invp_global_stats_audit_2048.json
decision = invp_global_stats_hold
```

Result:

| Metric | Value |
|---|---:|
| Stat feature dim | `148` |
| Best stat AUC min | `0.5180071592330933` |
| Composite AUC min | `0.5185081958770752` |
| Composite AUC mean | `0.5251665115356445` |
| Top-k Jaccard min | `0.06666666666666667` |

Correct interpretation:

- The existing global activity statistics are too coarse for the signal found
  by grouped SGP.
- They should not trigger a neural smoke, remote launch, or diverse expert
  inclusion.
- If continuing the statistics route, use a targeted group-distribution feature
  bank over cell/word-cell/bit-role/orbit group activities, including variance,
  span, top-k means, and pair-slot consistency.

### Suggested Action
Keep `present_pairset_global_stats` as an existing model/control, not as a
validated next route from this audit. Next local work should test group-level
distribution statistics that preserve the structure where SGP showed broad weak
signal.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-invp-global-stats-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_invp_global_stats_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, invp, global-stats, distribution-statistics, route-selection
- See Also: LRN-20260706-008, LRN-20260706-007, LRN-20260705-003
- Pattern-Key: innovation1.spn_present.invp_global_stats_hold_group_distribution_next
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-010] best_practice

**Logged**: 2026-07-06T05:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not keep adding deterministic InvP(delta) aggregate statistics after group-distribution audit held.

### Details
After generic InvP global statistics held, a more targeted deterministic
group-distribution bank was tested over:

```text
pair_word_cell
word_cell
cell
word_bit_role
p_layer_orbit
```

For each scheme it audited:

```text
activity_mean
activity_std
activity_max
top2_activity_mean
top4_activity_mean
bottom2_activity_mean
bottom4_activity_mean
activity_span
```

Protocol:

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
seeds = 0, 1
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
```

Artifact:

```text
outputs/local_audits/i1_present_r8_invp_group_distribution_audit_2048.json
decision = invp_group_distribution_hold
```

Result:

| Metric | Value |
|---|---:|
| Stat feature dim | `40` |
| Best stat AUC min | `0.514545202255249` |
| Composite AUC min | `0.5135400295257568` |
| Composite AUC mean | `0.5136241912841797` |
| Top-k Jaccard min | `0.18518518518518517` |

Correct interpretation:

- Grouped SGP's higher composite AUC does not translate into stable simple
  unsupervised distribution statistics.
- The broad InvP(delta) weak signal is too weak/unstable for another hand-built
  aggregate-stat pass.
- Do not create a deterministic group-distribution representation smoke, remote
  launch, or ensemble expert from this evidence.

### Suggested Action
Stop deterministic InvP(delta) aggregation for now. If continuing this family,
use a learned pair/group-interaction representation that directly consumes group
activities, or shift the next local slot to data/difference search. Do not add
more handwritten aggregate statistics around the same failed evidence without a
new reason.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-invp-group-distribution-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_invp_group_distribution_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, invp, group-distribution, deterministic-statistics, route-selection
- See Also: LRN-20260706-009, LRN-20260706-008, LRN-20260706-007, LRN-20260705-003
- Pattern-Key: innovation1.spn_present.invp_group_distribution_hold_stop_hand_stats
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260623-002] correction

**Logged**: 2026-06-23T21:56:37+08:00
**Priority**: critical
**Status**: promoted
**Area**: infra

### Summary
Remote training that generates datasets or derived features must use disk-backed cache/progress/reuse before launch; do not run large remote jobs with pure in-memory one-shot generation.

### Details
The user corrected a recurring workflow mistake: prior project work had already established that dataset generation should write reusable artifacts such as `features.npy`, `labels.npy`, metadata, CSV/JSONL summaries, and progress logs. The candidate-evidence route violated this principle by launching `65536/class` remotely through a new prototype runner that built `features: list[np.ndarray]` in memory and only wrote a final result after all feature generation, training, and evaluation finished.

Correct rule:

- Any remote training or medium/large screen that generates datasets, feature matrices, candidate-evidence features, trail statistics, or other derived training inputs must have disk-backed cache before launch.
- The cache must be under `G:\lxy` on the remote and normally inside the run directory or approved run cache root.
- Required artifacts include cache metadata, feature/label arrays or equivalent chunked files, progress JSONL/logging, and reuse/resume behavior when parameters match.
- New runners and new feature routes are not exempt. If they bypass `run_innovation_one_matrix.py`, they must implement an equivalent route-specific cache before remote scale-up.
- Smoke-only local experiments may use in-memory generation, but remote launches at `65536/class` or above must not.
- Do not call a remote experiment ready to launch until this cache/progress gate has been checked explicitly.

The immediate failure mode was the candidate-evidence baseline: positive local fast screens led to a remote `65536/class` run, but its feature generation was pure Python/in-memory and produced no progress or reusable cache, making the remote appear stalled and wasting time.

### Suggested Action
Promote this rule to `AGENTS.md` under Remote Windows GPU Rules / Verification. Add cache/progress support to `experiments/innovation1/run_spn_candidate_evidence_baseline.py` before relaunching scaled candidate-evidence experiments:

- `--feature-cache-root`
- `--feature-cache-chunk-size`
- `--progress-output`
- disk-backed `features.npy` / `labels.npy` / `metadata.json`
- cache identity including rounds, seeds, samples_per_class, pairs_per_sample, negative mode, sample structure, difference profile, key rotation, beam width, depth, source, and feature dimension
- chunk progress events and cache reuse

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, experiments/innovation1/run_spn_candidate_evidence_baseline.py, experiments/run_innovation_one_matrix.py, src/blockcipher_ai_eval/data/cache/disk.py
- Tags: remote-training, dataset-cache, feature-cache, progress-logging, innovation1, spn, present, candidate-evidence
- See Also: LRN-20260621-003
- Pattern-Key: remote_training.must_use_disk_cache_for_generated_data
- Recurrence-Count: 1
- First-Seen: 2026-06-23
- Last-Seen: 2026-06-23
- Promoted: AGENTS.md

---

## [LRN-20260624-001] correction

**Logged**: 2026-06-24T16:03:00+08:00
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
Generated result plots must include human-readable coordinate axes, tick labels, and grid lines; bare lines without coordinate values are not acceptable.

### Details
The user corrected the first SVG output from `scripts/plot-results`: it drew train/validation curves, but the plot did not have enough coordinate values to interpret the metric scale and epoch positions like a normal deep learning training curve. A result figure should support human inspection without reading the raw JSON/CSV.

Correct plotting expectations:

- X axis should show epoch tick values, not only first/last labels.
- Y axis should show metric tick values, including intermediate values such as `0.25`, `0.5`, `0.75` for accuracy/AUC.
- Light grid lines should align with tick labels so the reader can estimate values.
- Axis labels such as `epoch`, `accuracy`, `auc`, and `loss` should be visible.
- Train and validation curves should remain visually distinct and annotated.

### Suggested Action
Keep visualization tests that assert generated SVG contains readable axis labels and intermediate tick values. When adding new plots, inspect the rendered artifact or its SVG text for axes, tick labels, and grid lines before calling the visualization complete.

### Metadata
- Source: user_feedback
- Related Files: src/blockcipher_nd/evaluation/plots.py, scripts/plot-results, tests/test_project_structure.py
- Tags: visualization, svg, training-curves, docs, result-reporting
- Pattern-Key: visualization.training_curves.require_readable_axes
- Recurrence-Count: 1
- First-Seen: 2026-06-24
- Last-Seen: 2026-06-24

---

## [LRN-20260624-002] correction

**Logged**: 2026-06-24T20:30:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
Remote experiment launches should not be personally supervised from the main thread; use tmux monitors to watch and retrieve results automatically.

### Details
The user clarified the intended remote workflow: after a remote GPU experiment is launched or handed off, the main agent should not keep personally supervising it through repeated SSH checks or manual polling. The correct pattern is to start a local tmux monitor/watcher that waits for completion artifacts, retrieves the result archive or raw fallback outputs, writes local logs/markers, and then lets the main thread continue or report only from local artifacts.

This strengthens the existing "do not SSH-poll from the main thread" rule. Manual remote contact should be reserved for controlled exceptions such as a local monitor health failure, a dry-run gate that explicitly allows SSH, or a user-requested repair. Normal long-running training should be monitored by tmux and retrieval scripts, not by interactive supervision.

### Suggested Action
For future remote runs, always launch or verify a supported local tmux monitor immediately after the remote command is handed off. Report the monitor session/log/marker paths to the user. Inspect results only after the monitor has pulled them back locally, unless the controlled gate says remote inspection is allowed.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, scripts/monitor_remote_results.py, outputs/remote_results/
- Tags: remote-training, tmux, monitoring, retrieval, ssh, workflow
- See Also: LRN-20260621-003, LRN-20260622-001
- Pattern-Key: remote_training.tmux_monitor_retrieves_results
- Recurrence-Count: 1
- First-Seen: 2026-06-24
- Last-Seen: 2026-06-24
- Promoted: AGENTS.md

---

## [LRN-20260625-001] best_practice

**Logged**: 2026-06-25T15:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Use `blockcipher-auto-research` as the project research workflow and Karpathy-style guidelines as the default implementation discipline for this repository.

### Details
The user explicitly requested that future project work remember and follow the behavior methods from two skills:

- `skills/blockcipher-auto-research/SKILL.md` should drive the project-local research loop: define the question, identify the same-budget baseline, change one hypothesis at a time, run fixed-budget experiments, generate JSONL/CSV/SVG/gate artifacts, apply evidence-scale language, document keep/discard/crash/diagnostic status, and commit/push scoped repository changes after verification.
- `karpathy-guidelines` should guide coding behavior: read relevant code before editing, state uncertainty when evidence is incomplete, prefer simple boring implementations, avoid unnecessary abstractions/dependencies/frameworks, edit precisely, protect user work, validate against observable success criteria, and debug actual failures rather than guessing.

Correct default going forward:

- Treat `blockcipher-auto-research` as the workflow skeleton for Innovation 1 experiments and related reproduction work.
- Treat Karpathy-style guidelines as the execution style for code/config/docs changes inside that workflow.
- Use the combination to avoid untraceable experiment drift: do not change benchmark protocol, validation data, labels, negative-sample definition, metric computation, or plan-alignment logic while also changing model/feature hypotheses unless the user explicitly requests benchmark redesign.
- Keep changes attributable to one research hypothesis whenever possible, then verify, commit, and push.

### Suggested Action
Promote a concise version to `AGENTS.md` under a project research execution section so future agents default to this combined method. Continue using `paper-code-reproducer` for honest paper-reproduction boundaries and `remote-windows-gpu-conda-ssh` for remote A6000 launch/monitor/retrieval rules when those tasks apply.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, skills/blockcipher-auto-research/SKILL.md, /home/fate/.agents/skills/karpathy-guidelines/SKILL.md
- Tags: workflow, research, coding-discipline, innovation1, experiments, karpathy, auto-research
- See Also: LRN-20260622-001, LRN-20260624-002
- Pattern-Key: workflow.blockcipher_auto_research_plus_karpathy_guidelines
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260625-002] best_practice

**Logged**: 2026-06-25T16:10:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Use clear documentation destinations: formal experiment plans in `docs/experiments/`, research blueprints in `docs/research/`, and historical agent plans only in archive-style locations.

### Details
The user clarified the intended documentation organization after discussing whether `$using-superpowers` creates Markdown plans under `docs/superpowers/`. The correct project convention is:

- New formal experiment plans, reproduction records, result analyses, and next-step execution plans belong under `docs/experiments/`.
- New research blueprints, literature syntheses, theory notes, and broad method proposals belong under `docs/research/`.
- Historical agent execution plans may remain under `docs/superpowers/plans/`, or be migrated to `docs/archive/agent-plans/` if the archive is reorganized.

This separates current project-facing research artifacts from historical superpowers/planning workflow outputs. In particular, do not create new current experiment plans under `docs/superpowers/`; that directory should be treated as historical plan archive unless explicitly reorganized.

### Suggested Action
Promote this convention to `AGENTS.md` under documentation organization. When writing new project docs, choose the destination before creating the file:

- `docs/experiments/` for executable experiment plans and result records.
- `docs/research/` for broad research strategy and literature-backed blueprints.
- `docs/superpowers/plans/` or `docs/archive/agent-plans/` only for historical agent execution plans.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, docs/research/, docs/superpowers/plans/
- Tags: docs, documentation-organization, experiments, research, archive, superpowers
- See Also: LRN-20260625-001
- Pattern-Key: docs.destination.experiments_research_agent_archive
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260625-003] correction

**Logged**: 2026-06-25T20:30:07+08:00
**Priority**: critical
**Status**: promoted
**Area**: docs

### Summary
Do not answer project implementation or tooling facts from memory; inspect source, config, logs, or artifacts before reporting them.

### Details
The user corrected a hallucinated implementation claim: the assistant previously stated that project plotting already used Matplotlib, but the actual implementation at that time was hand-written SVG in `src/blockcipher_nd/evaluation/plots.py`. The later visualization change did convert plotting to Matplotlib, but the earlier answer was still wrong because it was made without first checking the code.

Correct behavior:

- Before saying "the project currently uses X" or "this is implemented by Y", inspect relevant files with `rg`, `sed`, config/lockfile reads, tests, logs, or result artifacts.
- This applies especially to dependencies, plotting/rendering libraries, training protocols, remote launch scripts, artifact paths, experiment status, metrics, checkpoint selection, and result gates.
- If the evidence has not been checked in the current turn or handoff, state that the answer is an assumption or check before reporting.
- When a prior statement is found to be wrong, correct it explicitly and separate the old false claim from the newly verified state.
- Do not rely on a recent memory of a change unless the repository state or command output confirms it.

### Suggested Action
Promote a concise factual-reporting rule to `AGENTS.md`. For future implementation/status answers, first cite or consult the relevant source/config/artifact. If verification is not possible, qualify the uncertainty instead of making a definitive claim.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, src/blockcipher_nd/evaluation/plots.py, pyproject.toml, uv.lock, scripts/plot-results
- Tags: hallucination, factual-reporting, source-verification, implementation-facts, visualization
- See Also: LRN-20260624-001, LRN-20260625-001
- Pattern-Key: workflow.factual_reporting.verify_before_claim
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260626-001] best_practice

**Logged**: 2026-06-26T11:07:22+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Update `docs/experiments/` for meaningful experiment lifecycle events, and update `docs/research/` only for research-direction changes.

### Details
The user clarified that future agents should not mechanically write both `docs/experiments/` and `docs/research/` after every action. The correct documentation workflow is:

- New formal or medium-scale experiment plans should be recorded under `docs/experiments/`.
- Completed meaningful experiments should update the relevant `docs/experiments/` record with run id, configuration, evidence gate, metrics, result status, and next action.
- Smoke tests, temporary debug runs, and local implementation checks do not require experiment documentation unless they change an evidence judgment, expose an important failure mode, or become part of the research record.
- `docs/research/` is for research blueprints, theory notes, literature syntheses, method proposals, and major route changes. It should not become a run-by-run log.
- When unsure, choose the smallest durable documentation update that preserves the evidence chain without creating noise.

### Suggested Action
Promote this as an execution rule in `AGENTS.md` under Documentation Organization. Future experiment launches and completions should first decide whether the event is documentation-worthy, then update only the appropriate destination.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, docs/research/
- Tags: docs, experiments, research, documentation-workflow, evidence-chain
- See Also: LRN-20260625-002
- Pattern-Key: docs.update_lifecycle.experiments_vs_research
- Recurrence-Count: 1
- First-Seen: 2026-06-26
- Last-Seen: 2026-06-26
- Promoted: AGENTS.md

---

## [LRN-20260627-001] correction

**Logged**: 2026-06-27T11:35:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
When a meaningful remote experiment result is completed and retrieved, automatically update `docs/experiments/` before reporting; do not ask whether to document it.

### Details
The user corrected the reporting workflow after the `I1-SPN-001` 1M/class single-seed result completed. The assistant initially suggested documenting the result as a next step instead of doing it immediately. For this project, meaningful experiment results are part of the evidence chain and should be written to the relevant experiment document as soon as the result is retrieved and parsed.

Correct behavior:

- If a meaningful experiment reaches a valid result gate and artifacts are retrieved, update the relevant `docs/experiments/` record in the same task turn.
- Include run id, protocol scale, gate status, local/remote artifact paths, metrics, deltas versus baseline, claim scope, and next action.
- Do not ask the user whether to write the result document; ask only if there is genuine ambiguity about which experiment record should own the result.
- Smoke tests, temporary debug checks, and local implementation checks still do not need docs unless they change the evidence judgment or expose an important failure mode.

### Suggested Action
Promote this rule to `AGENTS.md` under Documentation Organization. Future remote-result handling should follow this sequence: retrieve artifacts, parse metrics, generate missing local plots/history if needed, update `docs/experiments/`, verify, commit, push, then report.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, outputs/remote_results/
- Tags: docs, experiments, remote-results, evidence-chain, workflow
- See Also: LRN-20260626-001, LRN-20260624-002
- Pattern-Key: docs.results.auto_update_experiment_record
- Recurrence-Count: 1
- First-Seen: 2026-06-27
- Last-Seen: 2026-06-27
- Promoted: AGENTS.md

---

## [LRN-20260627-002] research

**Logged**: 2026-06-27T19:41:43+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
After the 262k N1-v2 structure ablation, do not scale the current gated-MCND route as the main Innovation 1 path; prioritize SPN-only and transition-aware SPN backbones.

### Details
The completed and locally retrieved run `i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627` tested N0 Zhang/Wang MCND, N1-v1 late fusion, SPN-only, true-P gated MCND, and shuffled-P gated MCND at `262144/class` with strict `encrypted_random_plaintexts` negatives and Zhang/Wang Case2 official MCND sample structure.

Observed AUC:

- `present_zhang_wang_keras_mcnd`: `0.784541`
- `present_nibble_paligned_mcnd`: `0.784299`
- `present_nibble_paligned_spn_only`: `0.791488`
- `present_nibble_paligned_gated_mcnd`: `0.784897`
- `present_nibble_shuffled_paligned_gated_mcnd`: `0.784281`

The true-P gated model was weakly positive against N0, N1-v1, and shuffled control, but it failed all planned continuation gates:

- N1-v2 AUC vs N0: `+0.000356`, below required `+0.002`.
- N1-v2 AUC vs N1-v1: `+0.000598`, below required `+0.001`.
- N1-v2 AUC vs shuffled: `+0.000615`, below required `+0.001`.
- N1-v2 calibrated accuracy vs N0: `-0.000530`.

The strongest diagnostic signal came from `SPN-only`, not MCND fusion. This suggests the PRESENT nibble/P-layer structure view contains useful real-vs-random signal, but the current gate/fusion design dilutes or fails to inject it into the MCND backbone.

This is medium diagnostic evidence, not formal multi-seed `>=1000000/class` evidence. However, it is enough to reject scaling the exact current `present_nibble_paligned_gated_mcnd` design as the main next route.

### Suggested Action
For the next Innovation 1 experiment, design a minimal SPN transition-aware backbone around the SPN-only signal. Keep the benchmark fixed and compare at the same `262144/class` evidence scale:

1. Keep N0 baseline and current SPN-only as anchors.
2. Add N2 transition-aware models that treat 16 PRESENT nibbles and P-layer transitions as the primary representation.
3. Include true-P versus shuffled-P controls for attribution.
4. Do not change validation data, labels, negative mode, metric computation, or Zhang/Wang Case2 sample construction.
5. Only consider 1M/multi-seed scaling if N2 beats SPN-only and the true-P route beats shuffled-P by the predeclared gate.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-n1v2-structure-ablation-plan.md, outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/
- Tags: innovation1, spn, present, n1v2, spn-only, transition-aware-backbone, evidence-gates
- See Also: LRN-20260621-001, LRN-20260623-001, LRN-20260627-001
- Pattern-Key: innovation1.spn_present.prioritize_transition_backbone_after_n1v2
- Recurrence-Count: 1
- First-Seen: 2026-06-27
- Last-Seen: 2026-06-27

---

## [LRN-20260628-001] correction

**Logged**: 2026-06-28T00:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
After a new training experiment smoke test passes, automatically continue to the corresponding non-smoke remote training launch instead of stopping at the smoke report.

### Details
The user corrected the experiment workflow after the SPN-only attribution smoke test. For this project, when the user asks to "推进" a new training experiment and a smoke matrix is part of the setup, the smoke is only a gate. If smoke passes and code/config/docs have been verified, committed, and pushed, the agent should proceed to launch the planned medium/formal remote run using the pushed GitHub commit and the established tmux monitor/retrieval workflow.

Correct behavior:

- Treat local smoke as a readiness gate, not the final deliverable, unless the user explicitly asks for smoke/local verification only.
- After smoke passes, commit and push scoped changes.
- Create or verify the matching remote config/launcher for the non-smoke run.
- Audit remote rules: `cmd.exe /c`, all artifacts under `G:\lxy`, disk-backed dataset cache/progress for `>=65536/class`, GitHub-pushed commit source.
- Launch remote training and start/verify a local tmux monitor for automatic retrieval.
- Report the run as `running`/`planned` with run id and monitor details, not as complete.

### Suggested Action
Promote this to `AGENTS.md` under research execution or remote workflow. Future "推进训练实验" tasks should run the full pipeline: implement -> smoke -> verify -> commit/push -> remote launch -> tmux monitor handoff.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, configs/experiment/innovation1/, configs/remote/, scripts/generated/remote/, docs/experiments/
- Tags: experiments, smoke, remote-training, tmux-monitor, workflow, innovation1
- See Also: LRN-20260622-001, LRN-20260627-001
- Pattern-Key: workflow.training.smoke_then_remote_launch
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260628-002] research

**Logged**: 2026-06-28T17:40:34+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
After SPN-only attribution, prioritize an InvP-centered PRESENT/SPN distinguisher route over generic DeltaC+InvP concatenation.

### Details
The completed remote run `i1_spn_only_attr_r7_262k_seed0_gpu1_20260628` showed that the strongest diagnostic row was `present_nibble_invp_only_spn_only`, not the previous `DeltaC + InvP(DeltaC)` anchor.

At `262144/class`, strict `encrypted_random_plaintexts` negatives, Zhang/Wang Case2 official MCND protocol:

- `present_zhang_wang_keras_mcnd`: AUC `0.783228`
- `present_nibble_delta_only_spn_only`: AUC `0.782918`
- `present_nibble_shuffled_paligned_spn_only`: AUC `0.784487`
- `present_nibble_paligned_spn_only`: AUC `0.790665`
- `present_nibble_invp_only_spn_only`: AUC `0.792536`

Key attribution deltas:

- InvP-only vs baseline: AUC `+0.009308`
- InvP-only vs DeltaC-only: AUC `+0.009617`
- InvP-only vs shuffled-P: AUC `+0.008048`
- InvP-only vs DeltaC+InvP anchor: AUC `+0.001871`

This supports the interpretation that inverse-P aligned `DeltaC` is the dominant useful signal in the current SPN-only family. Raw `DeltaC` concatenation may dilute or fail to improve the signal under the simple token-mixer architecture.

This is medium diagnostic single-seed evidence, not formal `>=1000000/class` multi-seed evidence.

### Suggested Action
Advance Innovation 1 as an InvP-centered route:

1. Add a compact InvP pair-set consistency model that reuses the existing InvP-only encoder and changes only pair aggregation.
2. Compare against baseline, current SPN-only anchor, InvP-only, DeltaC-only, and shuffled-P under the same `262144/class` protocol.
3. Treat local smoke as a launch gate; if smoke passes, automatically commit/push and launch the remote medium diagnostic with tmux monitor retrieval.
4. Only consider 1M/class multi-seed scaling after InvP-only or InvP-centered consistency remains stable across at least one additional 262k seed or clearly beats the current InvP-only anchor.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-spn-only-attribution-plan.md, outputs/remote_results/i1_spn_only_attr_r7_262k_seed0_gpu1_20260628/
- Tags: innovation1, spn, present, invp, p-layer, attribution, pair-consistency
- See Also: LRN-20260627-002, LRN-20260628-001
- Pattern-Key: innovation1.spn_present.invp_centered_route
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28

---

## [LRN-20260628-003] correction

**Logged**: 2026-06-28T18:20:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Before starting or advancing meaningful training tasks, proactively write or update the experiment plan instead of waiting for the user to request it.

### Details
The user corrected the experiment workflow: future training launches should not depend on the user saying "写计划" first. For this project, a meaningful training task starts with a project-facing plan under `docs/experiments/`, then proceeds through implementation, smoke/readiness validation, scoped commit/push, remote launch from the pushed commit, local tmux monitoring, retrieval, result parsing, experiment-doc update, and final scoped commit/push.

Correct behavior:

- If the task is a meaningful new run, route, scale-up, or ablation, create or update the relevant `docs/experiments/` plan before launching.
- The plan should include the research question, fixed protocol, same-budget baseline, rows/models, scale, evidence gate, artifact paths, cache/progress expectation, remote device/run id expectations, and next action.
- Smoke/local checks remain readiness gates; if they pass and the user did not request smoke-only, continue to the planned non-smoke run.
- Do not mechanically update `docs/research/` for every run; update it only when the broader research route, theory, or method blueprint changes.
- Do not ask whether to write the plan for a meaningful training task; ask only when ownership of the document is genuinely ambiguous.

### Suggested Action
Promote this to `AGENTS.md` under Research Execution Style. Future experiment advancement should follow: plan first -> implement -> smoke -> commit/push -> remote launch -> tmux monitor -> retrieve -> update experiment docs -> commit/push.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, .learnings/LEARNINGS.md
- Tags: experiments, planning, docs, remote-training, workflow
- See Also: LRN-20260628-001, LRN-20260627-001
- Pattern-Key: workflow.training.proactive_experiment_plan_first
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260628-004] correction

**Logged**: 2026-06-28T19:25:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research

### Summary
Do not default every remote training experiment to large multi-model comparison matrices; keep incremental runs lean.

### Details
The user pointed out that repeatedly comparing many models in every remote experiment is cumbersome. The correct workflow is to separate attribution/audit experiments from incremental model-selection experiments:

- Full comparison matrices are useful for stage gates, attribution studies, protocol audits, and checking whether a control invalidates a route.
- Incremental model changes should usually compare only the new candidate against the strongest current same-protocol anchor, plus the minimum necessary baseline/control rows.
- A normal incremental remote matrix should target 2-3 models and rarely exceed 4.
- Do not keep re-running historical weaker controls in every new experiment once the attribution route is stable, unless the research question specifically requires them.
- Existing already-launched remote jobs should normally be allowed to finish; apply the lean-matrix rule to the next planned runs.

For Innovation 1, after InvP attribution has already shown `InvP-only` as the current strongest route, follow-up architecture experiments should usually compare:

```text
new candidate
current strongest InvP anchor
optional Zhang/Wang baseline or one critical control
```

instead of automatically including DeltaC-only, shuffled-P, old DeltaC+InvP, and every previous baseline in each run.

### Suggested Action
Promote this to `AGENTS.md` under Research Execution Style. Future experiment plans should explicitly justify any matrix larger than 3-4 rows and prefer lean comparisons for iteration speed and clarity.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, configs/experiment/innovation1/
- Tags: experiments, model-matrix, remote-training, efficiency, innovation1
- See Also: LRN-20260628-002, LRN-20260628-003
- Pattern-Key: workflow.training.lean_experiment_matrix
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260629-001] correction

**Logged**: 2026-06-29T14:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
Remote tmux monitoring should be delegated to a sub-agent or watcher; the main thread should not repeatedly loop over tmux status.

### Details
The user corrected a workflow drift: after remote GPU launch, I repeatedly checked tmux sessions and monitor logs from the main thread. This is not the intended project workflow. The correct pattern is:

- Main thread launches or verifies exactly enough to hand off the remote run.
- A tmux monitor, watcher, or sub-agent owns the monitoring loop and retrieval.
- Main thread should not repeatedly inspect `tmux ls`, `monitor.log`, or remote progress just to see whether training finished.
- Main thread resumes result processing only when local artifacts have arrived, when the user explicitly asks for a status check, or when a monitor health failure is detected by the delegated watcher/sub-agent.
- If additional post-processing is needed after retrieval, it should also be delegated to a watcher/sub-agent where possible, not manually polled from the main thread.

This refines the existing "do not SSH-poll from the main thread" rule: do not replace SSH polling with main-thread tmux polling. Monitoring loops belong outside the main research/implementation thread.

### Suggested Action
Promote to `AGENTS.md` under Remote Monitoring And Retrieval. Future remote launches should report the monitor/sub-agent handoff and then continue with non-monitoring work or wait for retrieved artifacts. If the user asks "continue" while a remote run is active, avoid repetitive tmux checks; do a single local artifact check if needed, then proceed with planning/implementation that does not require the running result.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, outputs/remote_results/
- Tags: remote-training, tmux, subagent, monitoring, workflow
- See Also: LRN-20260624-001
- Pattern-Key: remote_training.delegate_tmux_monitoring_to_subagent
- Recurrence-Count: 1
- First-Seen: 2026-06-29
- Last-Seen: 2026-06-29
- Promoted: AGENTS.md

---

## [LRN-20260705-001] best_practice

**Logged**: 2026-07-05T19:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Update project guidance to match the current `self-improvement` workflow and its split from `self-healing`.

### Details
The user asked to update the current `self-improvement` usage. The installed skill now separates active runtime recovery from passive learning capture:

- Use `self-healing` for active failures that need diagnosis, patching, verification, and verified `HEAL-` entries during the current task.
- Use `self-improvement` for user corrections, outdated knowledge, recurring best practices, historical or external/tool failures, missing capabilities, and promotion of recurring self-healing handoffs.
- Route learnings by artifact type: `.learnings/LEARNINGS.md` for corrections, knowledge gaps, and best practices; `.learnings/ERRORS.md` for historical or external/tool failures; `.learnings/FEATURE_REQUESTS.md` for missing capabilities.
- Search existing `.learnings/` before adding a new entry, link related records with `See Also`, and use `Pattern-Key` plus recurrence fields for recurring patterns.
- Promote concise prevention rules only when they are broadly useful; for recurring patterns, prefer the current threshold of `Recurrence-Count >= 3`, at least two distinct tasks, and a 30-day window.
- Consider skill extraction when a learning becomes a reusable, verified, non-obvious workflow.

### Suggested Action
Promote this concise workflow into `AGENTS.md` so future agents apply the current self-improvement/self-healing split and promotion threshold without relying on chat context.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, /home/fate/.agents/skills/self-improvement/SKILL.md
- Tags: self-improvement, self-healing, learnings, workflow, project-rules
- See Also: LRN-20260621-002
- Pattern-Key: workflow.self_improvement.current_usage
- Recurrence-Count: 1
- First-Seen: 2026-07-05
- Last-Seen: 2026-07-05
- Promoted: AGENTS.md

---

## [LRN-20260706-004] best_practice

**Logged**: 2026-07-06T02:20:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not advance the aligned active-difference integral route into neural residual or ensemble work until a non-baseline residual signal appears.

### Details
The composite residual probe tested whether deterministic feature-bank
combinations improve over the fixed `pair_xor_column_sum_variance` baseline on
three aligned single-nibble PRESENT r8 routes. At `2048/class`, seed `23`:

```text
Zhang/Wang active0 baseline AUC = 0.8878759145736694
Zhang/Wang best baseline+one delta = 0.0
AutoND active6 baseline AUC = 0.8747416734695435
AutoND best baseline+one delta = 0.0
Entropy active5 baseline AUC = 0.8852955102920532
Entropy best baseline+one delta = 0.0
```

The equal-weight composite diluted the signal to about `0.603-0.619` AUC, and
the best `baseline + one additional statistic` scan gave no improvement over
the baseline. This makes the route a deterministic SPN/multiset feature
baseline, not a current neural residual or diverse-expert candidate.

Correct framing:

- Keep `pair_xor_column_sum_variance` as the explicit comparator for this
  aligned active-difference route.
- Do not spend the next remote or neural slot on residual learning for this
  route unless a later local probe finds non-baseline signal.
- Continue searching for genuinely different SPN feature/input routes before
  diverse expert aggregation.

### Suggested Action
Use the composite residual audit as a gate before neural residual follow-up.
If both equal composite and best baseline+one fail to beat the fixed baseline,
record the route as deterministic-only and move to a different representation
family.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-integral-parity-control-plan.md, src/blockcipher_nd/cli/audit_integral_parity_signal.py
- Tags: innovation1, spn, present, integral, residual-probe, deterministic-baseline
- See Also: LRN-20260706-001, LRN-20260706-003, LRN-20260705-003
- Pattern-Key: innovation1.spn_present.aligned_active_difference_no_composite_residual
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-005] best_practice

**Logged**: 2026-07-06T02:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not reopen candidate-trail as a main route from weak local low-dimensional feature probes alone.

### Details
After the aligned active-difference route was reduced to a deterministic
baseline, candidate-evidence was rechecked as a possible non-neighbor feature
family. The local low-dimensional probe on PRESENT r7 official Case2 found
repeated weak top-axis composite signal:

```text
seed17 top-axis composite AUC = 0.5382152795791626
seed17 best axis = 285, AUC advantage = 0.03317534923553467
seed18 top-axis composite AUC = 0.5390714406967163
seed18 best axis = 64, AUC advantage = 0.025446653366088867
```

This is useful route-selection information, but it does not overturn the
retrieved 262144/class candidate-trail gate:

```text
best candidate-trail AUC = 0.703854276799
InvP anchor AUC = 0.793651987187
shuffled-cell control AUC = 0.702488259296
decision = stop_candidate_trail_route
```

Correct framing:

- The local probe is weak-positive but semantically/positionally unstable.
- It is not enough to reopen candidate-trail seed1 or remote scale.
- Candidate-trail remains stopped as a main route unless a genuinely different
  representation or control beats the InvP anchor and shuffled-cell control.

### Suggested Action
Use candidate-evidence low-dimensional probes only as cheap route-selection
screens. Before spending remote time, require stability across seeds and a
control that separates true candidate evidence from shuffled-cell evidence.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-candidate-trail-consistency-plan.md, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, candidate-trail, feature-probe, diverse-experts
- See Also: LRN-20260706-004, LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.candidate_trail_lowdim_probe_not_reopen_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---
