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

## [FEAT-20260716-002] visual_qa_redraw_workflow_availability

**Logged**: 2026-07-16T21:20:00+08:00
**Priority**: high
**Status**: resolved
**Area**: tests

### Requested Capability
Make the `visual-qa-redraw` workflow available and mandatory for checking every
generated or regenerated user-facing visualization before it is reported as
complete.

### User Context
Experiment figures can be syntactically valid and nonempty while still having
overlapping fonts, cramped curves, ambiguous technical titles, clipped labels,
or scales that hide decision-relevant differences. The user explicitly requires
future SVG/PNG/chart generation to pass `visual-qa-redraw`, not only plotting
tests or a file-size check.

The first installed-skill search did not find the workflow. The user then
provided the installed skill explicitly at
`/home/fate/.agents/skills/visual-qa-redraw/SKILL.md`; the exact render-inspect-
redraw workflow is now available and was applied to the latest Innovation 2
joint chart.

### Complexity Estimate
medium

### Suggested Implementation
Install or expose a vetted `visual-qa-redraw` skill in the Codex skill catalog,
document its invocation and output contract, and have experiment result handling
run it after plot generation and before index/document completion. The gate
should inspect rendered pixels for overlap, clipping, title clarity, curve and
marker separation, axis readability, legend completeness, and responsive framing
where relevant, then redraw and repeat until it passes.

### Metadata
- Frequency: recurring
- Related Features: training curve SVG, result plots, result retrieval, visual regression
- See Also: LRN-20260624-001
- Pattern-Key: visualization.generated_artifacts.require_visual_qa_redraw

### Resolution
- **Resolved**: 2026-07-16T21:45:00+08:00
- **Commit/PR**: ed5bba5
- **Notes**: Promoted the mandatory workflow to `AGENTS.md`; verified the
  installed skill path; rendered the latest Innovation 2 joint SVG at
  `1800x958`, found the bottom verdict overlapping the legends, separated the
  legend/verdict regions, re-rendered, and passed full-frame plus bottom-detail
  pixel inspection.

---
