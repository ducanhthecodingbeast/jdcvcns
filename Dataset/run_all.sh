#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOCAL_ENV="${SCRIPT_DIR}/.env.local"
DATA_DIR="${REPO_ROOT}/Data"

if [[ -f "${LOCAL_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LOCAL_ENV}"
  set +a
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing Dataset virtual environment: ${VENV_DIR}" >&2
  echo "Create it first: cd ${SCRIPT_DIR} && python3 -m venv .venv" >&2
  exit 1
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "Missing KAGGLE_USERNAME or KAGGLE_KEY." >&2
  echo "Export them before running, or put them in ${LOCAL_ENV}:" >&2
  echo "KAGGLE_USERNAME=thecapybaracoder" >&2
  echo "KAGGLE_KEY=KGAT_040b5e85a3503e30560d5a05ab5039b3" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}"

cd "${REPO_ROOT}"

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/data_preprocessing.py"

if [[ ! -s "${DATA_DIR}/jd.csv" || ! -s "${DATA_DIR}/cv.csv" ]]; then
  echo "Preprocessing did not create non-empty ${DATA_DIR}/jd.csv and ${DATA_DIR}/cv.csv." >&2
  echo "Check Kaggle credentials and the downloaded dataset filenames." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m Dataset.mockcv "${MOCKCV_ARGS:---force}"
