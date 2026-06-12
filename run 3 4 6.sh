#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${RUN_346_MODE:-local}"
CHECK_ONLY=false
CONTINUE_ON_ERROR=false
RUN_BACKGROUND=false
ACTION="run"
VERSIONS=("3.0" "4.0" "6.0")
SELECTED_VERSIONS=()
LOG_FILE="${RUN_346_LOG:-${SCRIPT_DIR}/run 3 4 6.log}"
PID_FILE="${RUN_346_PID:-${SCRIPT_DIR}/run 3 4 6.pid}"

usage() {
  cat <<'EOF'
Usage: ./run\ 3\ 4\ 6.sh [options] [all|3.0|4.0|6.0 ...]
       ./run\ 3\ 4\ 6.sh status
       ./run\ 3\ 4\ 6.sh stop

One-command organizer for the 3.0, 4.0, and 6.0 benchmark suites.
With no version arguments, it runs 3.0, then 4.0, then the safe 6.0 default.

Modes:
  local   Uses each version's local run.sh. Default and recommended.
  docker  Uses each version's compose.yaml test service.

Options:
  --background          Run the organizer in the background.
  --check-only          Validate files/data/tools, then exit.
  --continue-on-error   Continue to the next suite if one suite fails.
  --mode local|docker   Select runner mode.
  -h, --help            Show this help.

Environment:
  RUN_346_MODE=local|docker
  QDRANT_UPSERT_BATCH_SIZE=1
  QDRANT_TIMEOUT=300
  BM25_REGEX_TOKENIZER=1
  PYTHON_BIN=python3.11        Used by 6.0 local mode.
  RUN_346_LOG=path             Background log path.
  RUN_346_PID=path             Background PID path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|background|runbackground)
      RUN_BACKGROUND=true
      shift
      ;;
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=true
      shift
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --mode. Use local or docker." >&2
        usage >&2
        exit 2
      fi
      MODE="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    status)
      ACTION="status"
      shift
      ;;
    stop)
      ACTION="stop"
      shift
      ;;
    all)
      SELECTED_VERSIONS=("3.0" "4.0" "6.0")
      shift
      ;;
    3|3.0)
      SELECTED_VERSIONS+=("3.0")
      shift
      ;;
    4|4.0)
      SELECTED_VERSIONS+=("4.0")
      shift
      ;;
    6|6.0)
      SELECTED_VERSIONS+=("6.0")
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#SELECTED_VERSIONS[@]} -gt 0 ]]; then
  VERSIONS=("${SELECTED_VERSIONS[@]}")
fi

export QDRANT_UPSERT_BATCH_SIZE="${QDRANT_UPSERT_BATCH_SIZE:-1}"
export QDRANT_TIMEOUT="${QDRANT_TIMEOUT:-300}"
export QDRANT_UPSERT_RETRIES="${QDRANT_UPSERT_RETRIES:-3}"
export BM25_REGEX_TOKENIZER="${BM25_REGEX_TOKENIZER:-1}"

background_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  printf '%s\n' "${pid}"
}

status_run() {
  local pid
  if pid="$(background_pid)"; then
    echo "Organizer is running."
    echo "PID: ${pid}"
    echo "Log: ${LOG_FILE}"
    return 0
  fi

  echo "Organizer is not running."
  echo "PID file: ${PID_FILE}"
  echo "Log: ${LOG_FILE}"
}

stop_run() {
  local pid
  if ! pid="$(background_pid)"; then
    echo "Organizer is not running."
    rm -f "${PID_FILE}"
    return 0
  fi

  echo "Stopping organizer PID ${pid}"
  kill "${pid}"
  sleep 2
  if kill -0 "${pid}" 2>/dev/null; then
    echo "PID ${pid} is still running; sending SIGKILL."
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

check_file() {
  [[ -f "$1" ]] || fail "Missing required file: $1"
}

check_data() {
  local jd_ok=false
  local cv_ok=false

  [[ -s "${SCRIPT_DIR}/Data/jd.csv" || -s "${SCRIPT_DIR}/Data/JOB_DATA_FINAL.csv" ]] && jd_ok=true
  [[ -s "${SCRIPT_DIR}/Data/mockcv.csv" || -s "${SCRIPT_DIR}/Data/cv.csv" || -s "${SCRIPT_DIR}/Data/USER_DATA_FINAL.csv" ]] && cv_ok=true

  if [[ "${jd_ok}" != "true" || "${cv_ok}" != "true" ]]; then
    echo "Data preflight failed." >&2
    echo "Required JD input: Data/jd.csv or Data/JOB_DATA_FINAL.csv" >&2
    echo "Required CV input: Data/mockcv.csv, Data/cv.csv, or Data/USER_DATA_FINAL.csv" >&2
    echo "Current Data files:" >&2
    find "${SCRIPT_DIR}/Data" -maxdepth 1 -type f -printf '  %f (%s bytes)\n' 2>/dev/null | sort >&2 || true
    fail "Prepare shared CV/JD CSVs before running 3.0, 4.0, and 6.0."
  fi
}

preflight() {
  case "${MODE}" in
    docker|local) ;;
    *) fail "Invalid mode '${MODE}'. Use docker or local." ;;
  esac

  check_file "${SCRIPT_DIR}/scripts/compose"
  for version in "${VERSIONS[@]}"; do
    [[ -d "${SCRIPT_DIR}/${version}" ]] || fail "Missing version directory: ${version}"
    check_file "${SCRIPT_DIR}/${version}/run.sh"
    check_file "${SCRIPT_DIR}/${version}/requirements.txt"
    check_file "${SCRIPT_DIR}/${version}/compose.yaml"
  done

  check_file "${SCRIPT_DIR}/3.0/bgmewdranttesting3.0.py"
  check_file "${SCRIPT_DIR}/4.0/virankertesting4.0.py"
  check_file "${SCRIPT_DIR}/6.0/jobberttesting6.0.py"
  check_file "${SCRIPT_DIR}/6.0/jobberttesting6.1.py"
  check_file "${SCRIPT_DIR}/6.0/bm25testing6.2.py"

  check_data

  if [[ "${MODE}" == "docker" ]]; then
    command -v docker >/dev/null 2>&1 || fail "Docker is required for --mode docker."
    docker compose version >/dev/null 2>&1 || fail "Docker Compose V2 is required for --mode docker."
  fi

  echo "Preflight OK."
  echo "Mode: ${MODE}"
  echo "Versions: ${VERSIONS[*]}"
  echo "Qdrant writes: batch=${QDRANT_UPSERT_BATCH_SIZE}, timeout=${QDRANT_TIMEOUT}s"
  echo "6.0 default: BM25 regex tokenizer"
}

start_background() {
  local pid
  if pid="$(background_pid)"; then
    echo "Organizer is already running."
    echo "PID: ${pid}"
    echo "Log: ${LOG_FILE}"
    echo "Monitor: tail -f '${LOG_FILE}'"
    exit 0
  fi

  mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"
  touch "${LOG_FILE}"
  (
    cd "${SCRIPT_DIR}"
    nohup "${SCRIPT_DIR}/run 3 4 6.sh" "${VERSIONS[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "$!" >"${PID_FILE}"
  )
  pid="$(cat "${PID_FILE}")"
  echo "Started organizer in background."
  echo "PID: ${pid}"
  echo "Log: ${LOG_FILE}"
  echo "Monitor: tail -f '${LOG_FILE}'"
}

run_version() {
  local version="$1"
  echo
  echo "=== Running ${version} ==="

  if [[ "${MODE}" == "docker" ]]; then
    (cd "${SCRIPT_DIR}/${version}" && "${SCRIPT_DIR}/scripts/compose" run --rm test)
    return
  fi

  case "${version}" in
    3.0|4.0)
      (cd "${SCRIPT_DIR}/${version}" && ./run.sh)
      ;;
    6.0)
      (cd "${SCRIPT_DIR}/6.0" && ./run.sh 6.2 -- --regex-tokenizer)
      ;;
    *)
      fail "Unknown version: ${version}"
      ;;
  esac
}

case "${ACTION}" in
  status)
    status_run
    exit 0
    ;;
  stop)
    stop_run
    exit 0
    ;;
esac

preflight

if [[ "${CHECK_ONLY}" == "true" ]]; then
  exit 0
fi

if [[ "${RUN_BACKGROUND}" == "true" ]]; then
  start_background
  exit 0
fi

failed_versions=()
for version in "${VERSIONS[@]}"; do
  if ! run_version "${version}"; then
    failed_versions+=("${version}")
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      echo "Stopping after ${version} failure." >&2
      exit 1
    fi
  fi
done

if [[ ${#failed_versions[@]} -gt 0 ]]; then
  echo "Failed versions: ${failed_versions[*]}" >&2
  exit 1
fi

echo
echo "All requested benchmark suites completed."
