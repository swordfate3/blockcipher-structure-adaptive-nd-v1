#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

SOURCE_COMMIT="${1:-}"
SEED0_ID="i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725"
SEED1_ID="i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725"
SEED0_ROOT="outputs/remote_results/${SEED0_ID}"
SEED0_MONITOR="outputs/remote_results_incomplete/${SEED0_ID}_monitor"
GATE_ROOT="outputs/local_readiness/i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_launch_gate_20260725"
MONITOR_ROOT="outputs/remote_results_incomplete/i1_rtg3a_seed1_after_seed0_20260725_monitor"
SEED1_MONITOR_SCRIPT="configs/remote/generated/monitor_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725.sh"
SEED1_TMUX="i1_rtg3a_skinny64_formal_seed1_monitor"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <pushed-source-commit>" >&2
  exit 6
fi

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

stop_successor() {
  local reason="$1"
  printf '%s\n' "${reason}" > "${MONITOR_ROOT}/seed1_not_launched.marker"
  echo "$(timestamp) seed1_not_launched reason=${reason}" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
}

required_seed0_artifacts_ready() {
  local required
  for required in \
    checkpoint-verification.local.json \
    curves.svg \
    gate.local.json \
    history.local.csv \
    results.jsonl \
    retrieved_from_verified_result_branch.marker \
    validation.local.json \
    visual_qa_passed.marker; do
    [[ -f "${SEED0_ROOT}/${required}" ]] || return 1
  done
}

while true; do
  if [[ -f "${SEED0_MONITOR}/remote_failed.marker" ]]; then
    stop_successor "seed0_remote_failed"
  fi

  if [[ -f "${SEED0_ROOT}/gate.local.json" ]]; then
    seed0_decision="$(python -c "import json,pathlib; print(json.loads(pathlib.Path(r'${SEED0_ROOT}/gate.local.json').read_text(encoding='utf-8')).get('decision',''))" 2>/dev/null || true)"
    if [[ "${seed0_decision}" != "innovation1_rtg3a_skinny_formal_seed0_supported" ]]; then
      stop_successor "seed0_gate_${seed0_decision:-invalid}"
    fi

    if required_seed0_artifacts_ready; then
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-runtime-spn-skinny-rtg3a-seed1-launch \
        --seed0-root "${SEED0_ROOT}" \
        --source-commit "${SOURCE_COMMIT}" \
        --upstream-ref origin/main \
        --repository . \
        --output-root "${GATE_ROOT}" \
        >> "${MONITOR_ROOT}/gate.log" 2>> "${MONITOR_ROOT}/gate_stderr.log" || {
          touch "${MONITOR_ROOT}/seed1_gate_failed.marker"
          echo "$(timestamp) seed1_launch_gate_failed" >> "${MONITOR_ROOT}/monitor.log"
          exit 4
        }
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
        >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 5
      python -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${GATE_ROOT}/gate.json').read_text(encoding='utf-8')); sys.exit(0 if g.get('launch_authorized') is True else 1)" || exit 7
      if tmux has-session -t "${SEED1_TMUX}" 2>/dev/null; then
        touch "${MONITOR_ROOT}/seed1_monitor_already_running.marker"
      else
        tmux new-session -d -s "${SEED1_TMUX}" -c "${PROJECT_ROOT}" \
          bash "${SEED1_MONITOR_SCRIPT}" "${SOURCE_COMMIT}" "${GATE_ROOT}/gate.json" || exit 8
      fi
      touch "${MONITOR_ROOT}/seed1_monitor_started.marker"
      echo "$(timestamp) seed1_gate_passed_local_monitor_started" >> "${MONITOR_ROOT}/monitor.log"
      exit 0
    fi
  fi

  echo "$(timestamp) waiting_for_complete_local_seed0_evidence" >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done
