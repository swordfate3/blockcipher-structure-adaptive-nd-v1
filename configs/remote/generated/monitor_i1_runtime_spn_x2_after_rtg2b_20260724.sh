#!/usr/bin/env bash
set -uo pipefail

PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
JOINT_GATE="$PROJECT/outputs/remote_results_incomplete/i1_rtg2b_skinny64_general_gf2_scale_262144_joint_seed0_seed1_20260724/gate.json"
SEED1_VISUAL_GATE="$PROJECT/outputs/remote_results/i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724/visual_qa_passed.marker"
RUN_ID=i1_rtg1_gift_to_skinny_frozen_backbone_target_head_x2_seed0_seed1_20260724
OUTPUT_ROOT="$PROJECT/outputs/local_diagnostic/$RUN_ID"
MONITOR_ROOT="$PROJECT/outputs/local_diagnostic/${RUN_ID}_monitor"

mkdir -p "$MONITOR_ROOT"
if [[ -f "$MONITOR_ROOT/x2_complete.marker" && -f "$OUTPUT_ROOT/gate.json" ]]; then
    printf '%s x2_already_complete\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"
    exit 0
fi

printf '%s waiting_for_rtg2b_joint_gate\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"
while true; do
    if [[ -f "$JOINT_GATE" ]]; then
        GATE_STATE=$(python -c 'import json,sys; g=json.load(open(sys.argv[1], encoding="utf-8")); print("|".join((str(g.get("phase")), str(g.get("status")), str(g.get("decision")), str(all(g.get("protocol_checks", {}).values())), str(all(g.get("research_checks", {}).values())))))' "$JOINT_GATE" 2>> "$MONITOR_ROOT/gate_stderr.log")
        if [[ $? -eq 0 ]]; then
            break
        fi
    fi
    sleep 60
done

printf '%s joint_gate=%s\n' "$(date --iso-8601=seconds)" "$GATE_STATE" >> "$MONITOR_ROOT/monitor.log"
if [[ "$GATE_STATE" != "rtg2b|pass|innovation1_rtg2b_skinny_scale_two_seed_supported|True|True" ]]; then
    printf '%s x2_skipped_by_joint_gate\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"
    touch "$MONITOR_ROOT/x2_skipped.marker"
    exit 0
fi

printf '%s waiting_for_seed1_visual_qa\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"
while [[ ! -f "$SEED1_VISUAL_GATE" ]]; do
    sleep 60
done
printf '%s seed1_visual_qa_passed\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"

printf '%s x2_started\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"
cd "$PROJECT" || exit 2
if UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run-runtime-spn-cross-cipher-head-adaptation \
    --dependency-gate "$JOINT_GATE" \
    --output-root "$OUTPUT_ROOT" \
    --device cpu \
    > "$MONITOR_ROOT/x2_stdout.log" \
    2> "$MONITOR_ROOT/x2_stderr.log"; then
    printf '%s x2_training_and_gate_complete\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"
else
    STATUS=$?
    printf '%s x2_failed exit=%s\n' "$(date --iso-8601=seconds)" "$STATUS" >> "$MONITOR_ROOT/monitor.log"
    touch "$MONITOR_ROOT/x2_failed.marker"
    exit "$STATUS"
fi

if UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    > "$MONITOR_ROOT/index_stdout.log" \
    2> "$MONITOR_ROOT/index_stderr.log"; then
    touch "$OUTPUT_ROOT/visual_qa_pending.marker"
    touch "$MONITOR_ROOT/x2_complete.marker"
    printf '%s x2_indexed_visual_qa_pending\n' "$(date --iso-8601=seconds)" >> "$MONITOR_ROOT/monitor.log"
else
    STATUS=$?
    printf '%s x2_index_failed exit=%s\n' "$(date --iso-8601=seconds)" "$STATUS" >> "$MONITOR_ROOT/monitor.log"
    touch "$MONITOR_ROOT/x2_index_failed.marker"
    exit "$STATUS"
fi
