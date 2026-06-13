#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOCAL_ENV="${SCRIPT_DIR}/.env.local"
DATA_DIR="${REPO_ROOT}/Data"
CACHE_DIR="${SCRIPT_DIR}/.cache"
LOG_FILE="${RUN_LOG:-${SCRIPT_DIR}/run.log}"
PID_FILE="${RUN_PID:-${SCRIPT_DIR}/run.pid}"

RUN_BACKGROUND=false
CHECK_ONLY=false
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --background|background|runbackground)
      RUN_BACKGROUND=true
      shift
      ;;
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Unknown argument(s): $*" >&2
  echo "Usage: ./run.sh [--background] [--check-only]" >&2
  exit 2
fi

if [[ "${RUN_BACKGROUND}" == "true" ]]; then
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
  nohup "${SCRIPT_DIR}/run.sh" >"${LOG_FILE}" 2>&1 </dev/null &
  pid="$!"
  echo "${pid}" >"${PID_FILE}"

  echo "Started Dataset run in background."
  echo "PID: ${pid}"
  echo "Log: ${LOG_FILE}"
  echo "Monitor: tail -f ${LOG_FILE}"
  exit 0
fi

create_venv() {
  echo "Creating Dataset virtual environment: ${VENV_DIR}"
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

if [[ "${CHECK_ONLY}" != "true" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "Missing KAGGLE_USERNAME or KAGGLE_KEY." >&2
  echo "Export them before running, or put them in ${LOCAL_ENV}:" >&2
  echo "KAGGLE_USERNAME=your_username" >&2
  echo "KAGGLE_KEY=your_api_key" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}"
mkdir -p "${CACHE_DIR}/huggingface" "${CACHE_DIR}/sentence-transformers"
export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${CACHE_DIR}/sentence-transformers}"

cd "${REPO_ROOT}"

if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
  echo "Skipping pip install because SKIP_PIP_INSTALL=1"
else
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

if [[ "${CHECK_ONLY}" == "true" ]]; then
  "${VENV_DIR}/bin/python" -m Dataset.mockcv \
    --jd-path "${DATA_DIR}/jd.csv" \
    --cv-path "${DATA_DIR}/cv.csv" \
    --target-dir "${DATA_DIR}" \
    --check-only
  exit 0
fi

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/data_preprocessing.py"

if [[ ! -s "${DATA_DIR}/jd.csv" || ! -s "${DATA_DIR}/cv.csv" ]]; then
  echo "Preprocessing did not create non-empty ${DATA_DIR}/jd.csv and ${DATA_DIR}/cv.csv." >&2
  echo "Check Kaggle credentials and the downloaded dataset filenames." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m Dataset.mockcv "${MOCKCV_ARGS:---force}"
