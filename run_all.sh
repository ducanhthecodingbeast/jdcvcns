#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PHASE_RUN_LOG:-${SCRIPT_DIR}/.phase_state/default/run.log}"
PID_FILE="${PHASE_RUN_PID:-${SCRIPT_DIR}/.phase_state/default/run.pid}"

RUN_BACKGROUND=false
if [[ "${1:-}" == "--background" || "${1:-}" == "background" || "${1:-}" == "runbackground" ]]; then
  RUN_BACKGROUND=true
  shift
fi

if [[ "${RUN_BACKGROUND}" == "true" ]]; then
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "Phased suite is already running."
      echo "PID: ${old_pid}"
      echo "Log: ${LOG_FILE}"
      echo "Monitor: tail -f ${LOG_FILE}"
      exit 0
    fi
  fi

  mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"
  touch "${LOG_FILE}"
  (
    cd "${SCRIPT_DIR}"
    nohup "${SCRIPT_DIR}/scripts/phased_suite.py" "$@" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "$!" >"${PID_FILE}"
  )

  pid="$(cat "${PID_FILE}")"
  echo "Started phased suite in background."
  echo "PID: ${pid}"
  echo "Log: ${LOG_FILE}"
  echo "Monitor: tail -f ${LOG_FILE}"
  exit 0
fi

exec "${SCRIPT_DIR}/scripts/phased_suite.py" "$@"
