#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_all.sh"
LOG_FILE="${RUN_ALL_LOG:-${SCRIPT_DIR}/run_all.log}"
PID_FILE="${RUN_ALL_PID:-${SCRIPT_DIR}/run_all.pid}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Missing executable runner: ${RUN_SCRIPT}" >&2
  echo "Fix with: chmod +x ${RUN_SCRIPT}" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Dataset run is already running."
    echo "PID: ${old_pid}"
    echo "Log: ${LOG_FILE}"
    echo "Monitor: tail -f ${LOG_FILE}"
    exit 0
  fi
fi

mkdir -p "$(dirname "${LOG_FILE}")"
touch "${LOG_FILE}"

cd "${SCRIPT_DIR}"
nohup "${RUN_SCRIPT}" >"${LOG_FILE}" 2>&1 </dev/null &
pid="$!"
echo "${pid}" >"${PID_FILE}"

echo "Started Dataset run in background."
echo "PID: ${pid}"
echo "Log: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
