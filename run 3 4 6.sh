#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${RUN_346_MODE:-docker}"
CHECK_ONLY=false
CONTINUE_ON_ERROR=false
VERSIONS=("3.0" "4.0" "6.0")

usage() {
  cat <<'EOF'
Usage: ./run\ 3\ 4\ 6.sh [--check-only] [--mode docker|local] [--continue-on-error]

Runs the 3.0, 4.0, and 6.0 benchmark suites in order.

Modes:
  docker  Uses each version's compose.yaml test service. Default and recommended.
  local   Uses each version's local run.sh.

Environment:
  RUN_346_MODE=docker|local
  PYTHON_BIN=python3.11        Used by 6.0 local mode.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=true
      shift
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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
      (cd "${SCRIPT_DIR}/6.0" && ./run.sh all)
      ;;
    *)
      fail "Unknown version: ${version}"
      ;;
  esac
}

preflight

if [[ "${CHECK_ONLY}" == "true" ]]; then
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
