# Research Sources

This directory stores successful web/API captures and concise research lookup
records that support documents under `docs/research/`.

Keep a capture here only when its contents match its filename and intended
source. API rate-limit responses, HTTP error pages, access-denied pages, and
empty downloads are not research evidence. Record reusable retrieval failures
in `.learnings/ERRORS.md`; if a raw failure payload must be retained for
diagnosis, give it an explicit `.error.*` name so literature tooling cannot
mistake it for a valid source.

Downloaded papers and their parsed text belong under `papers/`. Literature
syntheses, method comparisons, and route decisions belong under
`docs/research/` rather than this raw-source directory.
