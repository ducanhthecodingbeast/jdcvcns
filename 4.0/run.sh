#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOCAL_ENV="${SCRIPT_DIR}/.env.local"
CACHE_DIR="${SCRIPT_DIR}/.cache"
LOG_FILE="${RUN_LOG:-${SCRIPT_DIR}/run.log}"
PID_FILE="${RUN_PID:-${SCRIPT_DIR}/run.pid}"

RUN_BACKGROUND=false
if [[ "${1:-}" == "--background" || "${1:-}" == "background" || "${1:-}" == "runbackground" ]]; then
  RUN_BACKGROUND=true
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "Unknown argument(s): $*" >&2
  echo "Usage: ./run.sh [--background]" >&2
  exit 2
fi

if [[ "${RUN_BACKGROUND}" == "true" ]]; then
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "Run is already active."
      echo "PID: ${old_pid}"
      echo "Log: ${LOG_FILE}"
      echo "Monitor: tail -f ${LOG_FILE}"
      exit 0
    fi
  fi

  mkdir -p "$(dirname "${LOG_FILE}")"
  touch "${LOG_FILE}"
  cd "${SCRIPT_DIR}"
  nohup "${SCRIPT_DIR}/run.sh" >"${LOG_FILE}" 2>&1 </dev/null &
  pid="$!"
  echo "${pid}" >"${PID_FILE}"

  echo "Started run in background."
  echo "PID: ${pid}"
  echo "Log: ${LOG_FILE}"
  echo "Monitor: tail -f ${LOG_FILE}"
  exit 0
fi

create_venv() {
  echo "Creating virtual environment: ${VENV_DIR}"
  if ! python3 -m venv "${VENV_DIR}"; then
    echo "python3 -m venv failed. Retrying without pip bootstrap..." >&2
    rm -rf "${VENV_DIR}"
    python3 -m venv --without-pip "${VENV_DIR}" || {
      echo "Could not create virtual environment." >&2
      echo "On Ubuntu/Debian, install venv support first: sudo apt install python3-venv" >&2
      exit 1
    }
  fi
}

bootstrap_pip() {
  local get_pip
  get_pip="$(mktemp)"

  echo "Bootstrapping pip inside ${VENV_DIR}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${get_pip}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${get_pip}" https://bootstrap.pypa.io/get-pip.py
  else
    python3 - "${get_pip}" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", sys.argv[1])
PY
  fi

  "${VENV_DIR}/bin/python" "${get_pip}"
  rm -f "${get_pip}"
}

if [[ -f "${LOCAL_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LOCAL_ENV}"
  set +a
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  create_venv
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  bootstrap_pip
fi

mkdir -p "${CACHE_DIR}/huggingface" "${CACHE_DIR}/sentence-transformers"
export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${CACHE_DIR}/sentence-transformers}"

cd "${SCRIPT_DIR}"

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

if [[ -f "${REPO_ROOT}/scripts/compose" ]]; then
  "${REPO_ROOT}/scripts/compose" up -d
fi

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/virankertesting4.0.py"
