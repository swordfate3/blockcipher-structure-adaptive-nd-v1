#!/usr/bin/env bash
set -u

run_id="i2_present_r9_atm_split333_resumable_generation_20260720"
remote_root="G:/lxy/blockcipher-structure-adaptive-nd-runs/${run_id}"
project_root="/home/fate/gitproject/blockcipher-structure-adaptive-nd-v1"
local_root="${project_root}/outputs/remote_results_incomplete/${run_id}"
history_root="${project_root}/outputs/remote_results_incomplete_history/${run_id}"
monitor_log="/tmp/e104_remote_watcher.log"
retrieved_marker="/tmp/e104_remote_watcher_retrieved.marker"
unhealthy_marker="/tmp/e104_remote_watcher_unhealthy.marker"
timeout_marker="/tmp/e104_remote_watcher_timeout.marker"
verified_root="${project_root}/outputs/remote_results/${run_id}"
checkpoint_root="${project_root}/outputs/local_readiness/i2_present_r9_atm_e99_coordinate_checkpoint_replay_seed0_seed1_20260720"
public_results_root="${checkpoint_root}/public_results"
e105_output_root="${project_root}/outputs/local_diagnostic/i2_present_r9_atm_split333_source_heldout_ranking_seed0_seed1_20260720"
max_checks=480
failures=0

record() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" >> "$monitor_log"
}

retrieve() {
    local terminal="$1"
    local stamp
    local incoming
    local archive
    stamp=$(date +%Y%m%d-%H%M%S)
    incoming="${local_root}/.incoming-${stamp}"
    mkdir -p "$local_root"
    mkdir -p "$incoming"
    if scp -r "lxy-a6000:${remote_root}/logs" "$incoming/" >> "$monitor_log" 2>&1 && \
       scp -r "lxy-a6000:${remote_root}/results" "$incoming/" >> "$monitor_log" 2>&1; then
        archive="${history_root}/${stamp}"
        shopt -s dotglob nullglob
        existing=("${local_root}"/*)
        shopt -u dotglob nullglob
        if [[ ${#existing[@]} -gt 1 ]] || \
           [[ ${#existing[@]} -eq 1 && "${existing[0]}" != "$incoming" ]]; then
            mkdir -p "$archive"
            for path in "${existing[@]}"; do
                if [[ "$path" != "$incoming" ]]; then
                    mv "$path" "$archive/"
                fi
            done
            record "archived_prior_raw_snapshot root=${archive}"
        fi
        mv "$incoming/logs" "$local_root/logs"
        mv "$incoming/results" "$local_root/results"
        rmdir "$incoming"
        printf '%s\n' \
            "Raw fallback retrieval from lxy-a6000:${remote_root}." \
            "Terminal marker: ${terminal}." \
            "This is not yet a verified result branch or plan-aligned final archive." \
            > "${local_root}/RAW_RETRIEVAL_NOTICE.txt"
        printf 'terminal=%s\nlocal_root=%s\nretrieved_at=%s\n' \
            "$terminal" "$local_root" "$(date --iso-8601=seconds)" > "$retrieved_marker"
        record "retrieved terminal=${terminal} local_root=${local_root}"
        return 0
    fi
    record "retrieval_failed terminal=${terminal}"
    return 1
}

record "watcher_started run_id=${run_id}"
for ((check=1; check<=max_checks; check++)); do
    state=$(ssh -o BatchMode=yes -o ConnectTimeout=10 lxy-a6000 \
        'powershell.exe -NoProfile -Command "$runId=\"i2_present_r9_atm_split333_resumable_generation_20260720\"; $root=\"G:\lxy\blockcipher-structure-adaptive-nd-runs\$runId\"; Get-ChildItem \"$root\logs\" -Filter \"*.marker\" -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { \"MARKER=$($_.Name)\" }; Get-ChildItem \"$root\results\" -Filter \"*.marker\" -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { \"MARKER=$($_.Name)\" }; $candidates=@(Get-ChildItem \"$root\results\search_state\candidate_results\" -Filter \"*.json\" -File -ErrorAction SilentlyContinue); $candidateBytes=[long]0; foreach ($candidate in $candidates) { $candidateBytes += $candidate.Length }; \"CANDIDATE_COUNT=$($candidates.Count)\"; \"CANDIDATE_BYTES=$candidateBytes\"; $progress=Get-Item \"$root\results\search_state\progress.jsonl\" -ErrorAction SilentlyContinue; if ($progress) { \"SEARCH_PROGRESS_UPDATED=$($progress.LastWriteTime.ToString(\"s\"))\" }; $task=Get-ScheduledTask -TaskName \"i2_e104_present_r9_split333_20260720\" -ErrorAction SilentlyContinue; if ($task) { \"TASK_STATE=$($task.State)\"; $info=Get-ScheduledTaskInfo -TaskName $task.TaskName; \"TASK_RESULT=$($info.LastTaskResult)\" }; $processes=@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like \"*$runId*\" }); $cpuSeconds=[double]0; $workingBytes=[long]0; foreach ($process in $processes) { $live=Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue; if ($live) { $cpuSeconds += $live.CPU; $workingBytes += $live.WorkingSet64 } }; $cpuRounded=[math]::Round($cpuSeconds,1); $workingMiB=[math]::Round($workingBytes/1MB,1); \"PROCESS_COUNT=$($processes.Count)\"; \"PROCESS_CPU_SECONDS=$cpuRounded\"; \"PROCESS_WORKING_MIB=$workingMiB\""' \
        2>> "$monitor_log")
    rc=$?
    state=${state//$'\r'/}
    if [[ $rc -ne 0 ]]; then
        failures=$((failures + 1))
        record "ssh_failed check=${check} consecutive=${failures}"
        if [[ $failures -ge 6 ]]; then
            printf 'consecutive_ssh_failures=%s\nfailed_at=%s\n' \
                "$failures" "$(date --iso-8601=seconds)" > "$unhealthy_marker"
            exit 2
        fi
        sleep 300
        continue
    fi
    failures=0
    compact=$(printf '%s' "$state" | tr '\n' ',')
    record "check=${check} markers=${compact}"

    terminal=""
    for candidate in pipeline_passed.marker probe_failed.marker resource_cap_hit.marker pipeline_failed.marker setup_failed.marker; do
        if [[ "$state" == *"$candidate"* ]]; then
            terminal="$candidate"
            break
        fi
    done
    if [[ -n "$terminal" ]]; then
        if retrieve "$terminal"; then
            cd "$project_root" || exit 4
            UV_CACHE_DIR=/tmp/uv-cache \
                uv run python scripts/index-results >> "$monitor_log" 2>&1
            index_rc=$?
            record "raw_result_index_done rc=${index_rc} terminal=${terminal}"
            if [[ $index_rc -ne 0 ]]; then
                exit 5
            fi
            if [[ "$terminal" == "pipeline_passed.marker" ]]; then
                MPLCONFIGDIR=/tmp/matplotlib-e105 \
                UV_CACHE_DIR=/tmp/uv-cache \
                uv run python scripts/postprocess-innovation2-present-r9-atm-split333 \
                    --raw-root "$local_root" \
                    --verified-root "$verified_root" \
                    --checkpoint-root "$checkpoint_root" \
                    --public-results-root "$public_results_root" \
                    --e105-output-root "$e105_output_root" \
                    --device cpu >> "$monitor_log" 2>&1
                postprocess_rc=$?
                record "postprocess_done rc=${postprocess_rc} output=${e105_output_root}"
                if [[ $postprocess_rc -ne 0 ]]; then
                    exit 4
                fi
            fi
            exit 0
        fi
    fi
    sleep 300
done

printf 'checks=%s\ntimed_out_at=%s\n' "$max_checks" "$(date --iso-8601=seconds)" > "$timeout_marker"
record "watcher_timeout checks=${max_checks}"
retrieve "watcher_timeout" || true
exit 3
