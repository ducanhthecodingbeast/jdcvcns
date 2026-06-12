#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOCAL_ENV="${SCRIPT_DIR}/.env.local"
CACHE_DIR="${SCRIPT_DIR}/.cache"
LOG_FILE="${RUN_LOG:-${SCRIPT_DIR}/run.log}"
PID_FILE="${RUN_PID:-${SCRIPT_DIR}/run.pid}"
ENV_QDRANT_TIMEOUT="${QDRANT_TIMEOUT:-}"
ENV_QDRANT_UPSERT_BATCH_SIZE="${QDRANT_UPSERT_BATCH_SIZE:-}"
ENV_HF_HOME="${HF_HOME:-}"
ENV_SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-}"

RUN_BACKGROUND=false
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --background|background|runbackground)
      RUN_BACKGROUND=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done
RUN_ARGS=("$@")

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
  nohup "${SCRIPT_DIR}/run.sh" "${RUN_ARGS[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
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

choose_cache_dir() {
  local version_name
  local fallback_dir
  local tmp_dir

  version_name="$(basename "${SCRIPT_DIR}")"
  fallback_dir="${XDG_CACHE_HOME:-${HOME}/.cache}/jdcvcns/${version_name}"
  tmp_dir="${TMPDIR:-/tmp}/jdcvcns-cache-${USER:-user}/${version_name}"

  if mkdir -p "${CACHE_DIR}" 2>/dev/null && [[ -w "${CACHE_DIR}" ]]; then
    printf '%s\n' "${CACHE_DIR}"
    return
  fi

  if mkdir -p "${fallback_dir}" 2>/dev/null && [[ -w "${fallback_dir}" ]]; then
    echo "Project cache is not writable; using ${fallback_dir}" >&2
    printf '%s\n' "${fallback_dir}"
    return
  fi

  mkdir -p "${tmp_dir}" || {
    echo "Could not create a writable cache directory." >&2
    exit 1
  }
  echo "Project and home caches are not writable; using ${tmp_dir}" >&2
  printf '%s\n' "${tmp_dir}"
}

if [[ -f "${LOCAL_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LOCAL_ENV}"
  set +a
fi

export QDRANT_UPSERT_BATCH_SIZE="${ENV_QDRANT_UPSERT_BATCH_SIZE:-1}"
export QDRANT_TIMEOUT="${ENV_QDRANT_TIMEOUT:-300}"
export QDRANT_UPSERT_RETRIES="${QDRANT_UPSERT_RETRIES:-3}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
HARDWARE_TARGET_PERCENT="${HARDWARE_TARGET_PERCENT:-95}"
CPU_THREADS="${CPU_THREADS:-$(( ($(nproc 2>/dev/null || echo 1) * HARDWARE_TARGET_PERCENT + 99) / 100 ))}"
if [[ "${CPU_THREADS}" -lt 1 ]]; then
  CPU_THREADS=1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${CPU_THREADS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-${CPU_THREADS}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.95}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export BGE_BATCH_SIZE="${BGE_BATCH_SIZE:-64}"
RUNTIME_CACHE_DIR="$(choose_cache_dir)"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  create_venv
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  bootstrap_pip
fi

mkdir -p "${RUNTIME_CACHE_DIR}/huggingface" "${RUNTIME_CACHE_DIR}/sentence-transformers"
export HF_HOME="${ENV_HF_HOME:-${RUNTIME_CACHE_DIR}/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${ENV_SENTENCE_TRANSFORMERS_HOME:-${RUNTIME_CACHE_DIR}/sentence-transformers}"

cd "${SCRIPT_DIR}"

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

if [[ -f "${REPO_ROOT}/scripts/compose" ]]; then
  "${REPO_ROOT}/scripts/compose" up -d
fi

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/bgmewdranttesting3.0.py" "${RUN_ARGS[@]}"
