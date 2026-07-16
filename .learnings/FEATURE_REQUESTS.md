## [FEAT-20260714-001] recency_sorted_experiment_result_index

**Logged**: 2026-07-14T19:42:06+08:00
**Priority**: high
**Status**: resolved
**Area**: docs

### Requested Capability
Make the large `outputs/` tree understandable by providing a stable, numbered
recent-result entry point where `001` always means the latest completed
experiment.

### User Context
Raw run directories use technical IDs and are spread across local smoke,
retrieved remote, and incomplete remote roots. Sorting directory names does not
show which result is newest, and renaming evidence directories would break
paths already referenced by configs, documentation, and gates.

On 2026-07-16, the fixed 30-entry cap proved too small during a dense experiment
week. The index must retain every result from at least the latest seven days so
same-week evidence cannot disappear merely because more than 30 runs completed.

### Complexity Estimate
medium

### Suggested Implementation
Scan only result-bearing roots, rank run directories by gate, validation, then
result completion timestamps, and generate both a human-readable Markdown
index and a structured JSON index. Include Chinese experiment descriptions,
status/decision summaries, and direct artifact links. Refresh the index after
every completed result-producing run or re-adjudication, whether local or
retrieved from a remote worker. Treat the configured count as a minimum and
retain all entries within a configurable time window, defaulting to seven days.

### Metadata
- Frequency: recurring
- Related Features: training curve SVG, result gates, result retrieval
- Pattern-Key: outputs.navigation.recency_sorted_result_index

### Resolution
- **Resolved**: 2026-07-14T19:42:06+08:00
- **Commit/PR**: included with the recent-result index implementation
- **Notes**: Added `scripts/index-results`, Markdown/JSON outputs, deterministic
  completion-evidence sorting, E4 Chinese labels, tests, and project workflow
  rules. The scope was subsequently strengthened to require same-turn refresh
  for every completed local or retrieved remote result, including smoke,
  readiness, fallback-incomplete retrieval, and re-adjudication outputs. The
  default index now keeps at least 30 entries plus every result completed within
  seven days of the newest result.

---
