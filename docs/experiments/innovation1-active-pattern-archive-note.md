# Innovation 1 Active-Pattern Archive Note

**Date:** 2026-06-30

**Status:** archived historical screen / not launchable

## Purpose

This note prevents the old active-pattern screen from being mistaken for a
current Innovation 1 launch candidate.

The active-pattern idea remains relevant as a possible SPN-adaptive data or
auxiliary-signal route, but the existing 2026-06-22 screen assets are not aligned
with the current remote workflow or current PRESENT/SPN evidence protocol.

## Archived Assets

```text
config = configs/experiment/innovation1/innovation1_spn_present_active_pattern_r7_screen.csv
remote_config = configs/remote/innovation1_spn_present_active_pattern_r7_screen_gpu1_20260622.json
script = scripts/spn-active-pattern
```

The remote config is now explicitly disabled:

```text
launch_enabled = false
archive_status = archived_historical_screen_only
```

`scripts/check-remote-readiness` must fail this config before launch.

## Why It Is Archived

The old screen used:

```text
sample_structure = zhang_wang_case2_mcnd
scale = 65536/class
route = standalone 24-dimensional active-pattern linear baseline
```

The current `scripts/spn-active-pattern` and `scripts/audit-spn-features`
defaults now use `zhang_wang_case2_official_mcnd` for ad hoc local diagnostics.
This does not make the archived 2026-06-22 remote config launchable.

Current Innovation 1 PRESENT/SPN comparisons use:

```text
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
current repository = swordfate3/blockcipher-structure-adaptive-nd-v1
remote artifacts = G:\lxy
meaningful launch = pushed commit + disk-backed cache + tmux watcher
```

Also, active-pattern metrics are auxiliary structure signals. A strong active
label metric would not by itself prove real-vs-random distinguisher strength.

## Future Re-entry Rule

If active-pattern returns as a current branch, create a new
`docs/experiments/` plan first. The new plan should not reuse the archived
remote config. It should test one of these narrower hypotheses:

```text
1. active-pattern features as auxiliary head for a real-vs-random model
2. active count / position-frequency statistics as attribution diagnostics
3. trail-consistency features combined with InvP or DDT graph evidence
```

Required protocol:

```text
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
same-budget baseline = current strongest InvP/graph route at the same scale
primary metric = val_auc
claim scope = smoke or diagnostic until 1000000/class multi-seed evidence
```

Do not launch the old 2026-06-22 active-pattern remote config.
