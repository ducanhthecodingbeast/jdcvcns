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
HARDWARE_TARGET_PERCENT="${RUN_346_HARDWARE_TARGET_PERCENT:-${HARDWARE_TARGET_PERCENT:-95}}"
RUN_PARALLEL="${RUN_346_PARALLEL:-auto}"
SIX_VARIANT="${RUN_346_6_VARIANT:-${RUN_VARIANT:-all}}"
CPU_TOTAL="unknown"
CPUSET=""
GPU_MEMORY_MIB="unknown"
RUN_PARALLEL_RESOLVED=false

usage() {
  cat <<'EOF'
Usage: ./run\ 3\ 4\ 6.sh [options] [all|3.0|4.0|6.0 ...]
       ./run\ 3\ 4\ 6.sh status
       ./run\ 3\ 4\ 6.sh stop

One-command organizer for the 3.0, 4.0, and 6.0 benchmark suites.
With no version arguments, it runs 3.0, then 4.0, then all 6.0 variants.

Modes:
  local   Uses each version's local run.sh. Default and recommended.
  docker  Uses each version's compose.yaml test service.

Options:
  --background          Run the organizer in the background.
  --check-only          Validate files/data/tools, then exit.
  --continue-on-error   Continue to the next suite if one suite fails.
  --parallel            Run selected suites at the same time.
  --sequential          Run selected suites one after another.
  --6-variant value     6.0 variant: all, 6.0, 6.1, 6.2, cosine, dot, dot-product, bm25.
  --mode local|docker   Select runner mode.
  -h, --help            Show this help.

Environment:
  RUN_346_MODE=local|docker
  RUN_346_PARALLEL=auto|true|false
  RUN_346_6_VARIANT=all|6.0|6.1|6.2
  RUN_346_HARDWARE_TARGET_PERCENT=95
  RUN_346_CPU_THREADS=auto
  CUDA_VISIBLE_DEVICES=0
  QDRANT_UPSERT_BATCH_SIZE=auto
  QDRANT_TIMEOUT=300
  BM25_REGEX_TOKENIZER=1
  BGE_BATCH_SIZE=auto
  VIRANKER_BATCH_SIZE=auto
  JOBBERT_BATCH_SIZE=auto
  SCORE_BATCH_SIZE=auto
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
    --parallel)
      RUN_PARALLEL=true
      shift
      ;;
    --sequential)
      RUN_PARALLEL=false
      shift
      ;;
    --6-variant|--six-variant)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1. Use all, 6.0, 6.1, 6.2, cosine, dot, or bm25." >&2
        usage >&2
        exit 2
      fi
      SIX_VARIANT="${2:-}"
      shift 2
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

detect_cpu_total() {
  local total
  total="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  if [[ ! "${total}" =~ ^[0-9]+$ ]] || [[ "${total}" -lt 1 ]]; then
    total=1
  fi
  printf '%s\n' "${total}"
}

detect_gpu_memory_mib() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  local gpu_index
  gpu_index="${CUDA_VISIBLE_DEVICES:-0}"
  gpu_index="${gpu_index%%,*}"
  if [[ -z "${gpu_index}" || "${gpu_index}" == "all" ]]; then
    gpu_index=0
  fi

  nvidia-smi -i "${gpu_index}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NR == 1 { gsub(/ /, "", $1); print int($1) }'
}

default_if_auto() {
  local value="$1"
  local default_value="$2"
  case "${value}" in
    ""|auto|AUTO)
      printf '%s\n' "${default_value}"
      ;;
    *)
      printf '%s\n' "${value}"
      ;;
  esac
}

set_batch_defaults_for_gpu() {
  local memory_mib="$1"
  if [[ ! "${memory_mib}" =~ ^[0-9]+$ ]]; then
    memory_mib=0
  fi

  if [[ "${memory_mib}" -ge 44000 ]]; then
    DEFAULT_BGE_BATCH_SIZE=128
    DEFAULT_VIRANKER_BATCH_SIZE=128
    DEFAULT_JOBBERT_BATCH_SIZE=256
    DEFAULT_SCORE_BATCH_SIZE=4096
    DEFAULT_QDRANT_UPSERT_BATCH_SIZE=128
    DEFAULT_PREFETCH_MULTIPLIER=8
  elif [[ "${memory_mib}" -ge 24000 ]]; then
    DEFAULT_BGE_BATCH_SIZE=96
    DEFAULT_VIRANKER_BATCH_SIZE=96
    DEFAULT_JOBBERT_BATCH_SIZE=192
    DEFAULT_SCORE_BATCH_SIZE=3072
    DEFAULT_QDRANT_UPSERT_BATCH_SIZE=96
    DEFAULT_PREFETCH_MULTIPLIER=8
  elif [[ "${memory_mib}" -ge 16000 ]]; then
    DEFAULT_BGE_BATCH_SIZE=64
    DEFAULT_VIRANKER_BATCH_SIZE=64
    DEFAULT_JOBBERT_BATCH_SIZE=128
    DEFAULT_SCORE_BATCH_SIZE=2048
    DEFAULT_QDRANT_UPSERT_BATCH_SIZE=64
    DEFAULT_PREFETCH_MULTIPLIER=6
  elif [[ "${memory_mib}" -ge 10000 ]]; then
    DEFAULT_BGE_BATCH_SIZE=48
    DEFAULT_VIRANKER_BATCH_SIZE=48
    DEFAULT_JOBBERT_BATCH_SIZE=96
    DEFAULT_SCORE_BATCH_SIZE=1024
    DEFAULT_QDRANT_UPSERT_BATCH_SIZE=48
    DEFAULT_PREFETCH_MULTIPLIER=6
  elif [[ "${memory_mib}" -ge 7000 ]]; then
    DEFAULT_BGE_BATCH_SIZE=24
    DEFAULT_VIRANKER_BATCH_SIZE=24
    DEFAULT_JOBBERT_BATCH_SIZE=48
    DEFAULT_SCORE_BATCH_SIZE=512
    DEFAULT_QDRANT_UPSERT_BATCH_SIZE=32
    DEFAULT_PREFETCH_MULTIPLIER=4
  else
    DEFAULT_BGE_BATCH_SIZE=16
    DEFAULT_VIRANKER_BATCH_SIZE=16
    DEFAULT_JOBBERT_BATCH_SIZE=32
    DEFAULT_SCORE_BATCH_SIZE=256
    DEFAULT_QDRANT_UPSERT_BATCH_SIZE=16
    DEFAULT_PREFETCH_MULTIPLIER=4
  fi
}

resolve_parallel_mode() {
  case "${RUN_PARALLEL}" in
    true|1|yes|on)
      RUN_PARALLEL_RESOLVED=true
      ;;
    false|0|no|off)
      RUN_PARALLEL_RESOLVED=false
      ;;
    auto)
      if [[ "${MODE}" == "local" && "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ && "${GPU_MEMORY_MIB}" -ge 44000 && "${#VERSIONS[@]}" -gt 1 ]]; then
        RUN_PARALLEL_RESOLVED=true
      else
        RUN_PARALLEL_RESOLVED=false
      fi
      ;;
    *)
      fail "Invalid RUN_346_PARALLEL='${RUN_PARALLEL}'. Use auto, true, or false."
      ;;
  esac
}

configure_hardware() {
  [[ "${HARDWARE_TARGET_PERCENT}" =~ ^[0-9]+$ ]] || fail "RUN_346_HARDWARE_TARGET_PERCENT must be an integer."
  if [[ "${HARDWARE_TARGET_PERCENT}" -lt 1 || "${HARDWARE_TARGET_PERCENT}" -gt 100 ]]; then
    fail "RUN_346_HARDWARE_TARGET_PERCENT must be between 1 and 100."
  fi

  case "${SIX_VARIANT}" in
    all|6.0|6.1|6.2|cosine|dot|dotproduct|dot-product|bm25) ;;
    *) fail "Invalid 6.0 variant '${SIX_VARIANT}'. Use all, 6.0, 6.1, 6.2, cosine, dot, dot-product, or bm25." ;;
  esac

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"

  CPU_TOTAL="$(detect_cpu_total)"
  CPU_THREADS="${RUN_346_CPU_THREADS:-${CPU_THREADS:-}}"
  CPU_THREADS="$(default_if_auto "${CPU_THREADS}" "")"
  if [[ -z "${CPU_THREADS}" ]]; then
    CPU_THREADS="$((CPU_TOTAL * HARDWARE_TARGET_PERCENT / 100))"
  fi
  [[ "${CPU_THREADS}" =~ ^[0-9]+$ ]] || fail "RUN_346_CPU_THREADS must be an integer."
  if [[ "${CPU_THREADS}" -lt 1 ]]; then
    CPU_THREADS=1
  fi
  if [[ "${CPU_THREADS}" -gt "${CPU_TOTAL}" ]]; then
    CPU_THREADS="${CPU_TOTAL}"
  fi

  CPUSET="${RUN_346_CPUSET:-${CPUSET:-}}"
  if [[ -z "${CPUSET}" && "${CPU_THREADS}" -lt "${CPU_TOTAL}" ]]; then
    if [[ "${CPU_THREADS}" -eq 1 ]]; then
      CPUSET="0"
    else
      CPUSET="0-$((CPU_THREADS - 1))"
    fi
  fi

  GPU_MEMORY_MIB="$(detect_gpu_memory_mib || true)"
  if [[ -z "${GPU_MEMORY_MIB}" ]]; then
    GPU_MEMORY_MIB="unknown"
  fi
  set_batch_defaults_for_gpu "${GPU_MEMORY_MIB}"

  export RUN_346_MODE="${MODE}"
  export RUN_346_6_VARIANT="${SIX_VARIANT}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${CPU_THREADS}}"
  export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-${CPU_THREADS}}"
  export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${CPU_THREADS}}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
  export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
  export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-${CPU_THREADS}}"
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-${HARDWARE_TARGET_PERCENT}}"
  export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
  export GPU_MEMORY_FRACTION="${RUN_346_GPU_MEMORY_FRACTION:-${GPU_MEMORY_FRACTION:-0.95}}"

  export BGE_BATCH_SIZE
  BGE_BATCH_SIZE="$(default_if_auto "${RUN_346_BGE_BATCH_SIZE:-${BGE_BATCH_SIZE:-}}" "${DEFAULT_BGE_BATCH_SIZE}")"
  export VIRANKER_BATCH_SIZE
  VIRANKER_BATCH_SIZE="$(default_if_auto "${RUN_346_VIRANKER_BATCH_SIZE:-${VIRANKER_BATCH_SIZE:-}}" "${DEFAULT_VIRANKER_BATCH_SIZE}")"
  export JOBBERT_BATCH_SIZE
  JOBBERT_BATCH_SIZE="$(default_if_auto "${RUN_346_JOBBERT_BATCH_SIZE:-${JOBBERT_BATCH_SIZE:-}}" "${DEFAULT_JOBBERT_BATCH_SIZE}")"
  export SCORE_BATCH_SIZE
  SCORE_BATCH_SIZE="$(default_if_auto "${RUN_346_SCORE_BATCH_SIZE:-${SCORE_BATCH_SIZE:-}}" "${DEFAULT_SCORE_BATCH_SIZE}")"
  export QDRANT_UPSERT_BATCH_SIZE
  QDRANT_UPSERT_BATCH_SIZE="$(default_if_auto "${QDRANT_UPSERT_BATCH_SIZE:-}" "${DEFAULT_QDRANT_UPSERT_BATCH_SIZE}")"
  export QDRANT_TIMEOUT
  QDRANT_TIMEOUT="$(default_if_auto "${QDRANT_TIMEOUT:-}" "300")"
  export QDRANT_UPSERT_RETRIES
  QDRANT_UPSERT_RETRIES="$(default_if_auto "${QDRANT_UPSERT_RETRIES:-}" "3")"
  export BM25_REGEX_TOKENIZER="${BM25_REGEX_TOKENIZER:-1}"
  export PREFETCH_MULTIPLIER
  PREFETCH_MULTIPLIER="$(default_if_auto "${PREFETCH_MULTIPLIER:-}" "${DEFAULT_PREFETCH_MULTIPLIER}")"
  export VIRANKER_MAX_LENGTH
  VIRANKER_MAX_LENGTH="$(default_if_auto "${VIRANKER_MAX_LENGTH:-}" "1024")"
  export VIRANKER_QUERY_MAX_LENGTH
  VIRANKER_QUERY_MAX_LENGTH="$(default_if_auto "${VIRANKER_QUERY_MAX_LENGTH:-}" "384")"
  export JOBBERT_MAX_LENGTH
  JOBBERT_MAX_LENGTH="$(default_if_auto "${JOBBERT_MAX_LENGTH:-}" "512")"

  resolve_parallel_mode
}

run_limited() {
  if [[ -n "${CPUSET}" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "${CPUSET}" "$@"
    return
  fi

  "$@"
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
  echo "Hardware target: ${HARDWARE_TARGET_PERCENT}%"
  echo "CPU: threads=${CPU_THREADS}/${CPU_TOTAL}, cpuset=${CPUSET:-all}"
  echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, memory=${GPU_MEMORY_MIB} MiB, MPS active thread=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}%"
  echo "Batch sizes: BGE=${BGE_BATCH_SIZE}, ViRanker=${VIRANKER_BATCH_SIZE}, JobBERT=${JOBBERT_BATCH_SIZE}, score=${SCORE_BATCH_SIZE}"
  echo "Parallel suites: ${RUN_PARALLEL_RESOLVED}"
  echo "6.0 variant: ${SIX_VARIANT}, BM25 regex tokenizer=${BM25_REGEX_TOKENIZER}"
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
  local args=("--mode" "${MODE}" "--6-variant" "${SIX_VARIANT}")
  if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
    args+=("--continue-on-error")
  fi
  case "${RUN_PARALLEL}" in
    true|1|yes|on) args+=("--parallel") ;;
    false|0|no|off) args+=("--sequential") ;;
  esac
  args+=("${VERSIONS[@]}")
  (
    cd "${SCRIPT_DIR}"
    nohup "${SCRIPT_DIR}/run 3 4 6.sh" "${args[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
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
  local bm25_args=()
  echo
  echo "=== Running ${version} ==="

  if [[ "${MODE}" == "docker" ]]; then
    (cd "${SCRIPT_DIR}/${version}" && run_limited "${SCRIPT_DIR}/scripts/compose" run --rm test)
    return
  fi

  case "${BM25_REGEX_TOKENIZER}" in
    1|true|TRUE|yes|YES|on|ON)
      bm25_args+=("--regex-tokenizer")
      ;;
  esac

  case "${version}" in
    3.0)
      (
        cd "${SCRIPT_DIR}/3.0"
        run_limited ./run.sh -- \
          --batch-size "${BGE_BATCH_SIZE}" \
          --upsert-batch-size "${QDRANT_UPSERT_BATCH_SIZE}" \
          --prefetch-multiplier "${PREFETCH_MULTIPLIER}" \
          --qdrant-timeout "${QDRANT_TIMEOUT}" \
          --qdrant-upsert-retries "${QDRANT_UPSERT_RETRIES}"
      )
      ;;
    4.0)
      (
        cd "${SCRIPT_DIR}/4.0"
        run_limited ./run.sh -- \
          --batch-size "${BGE_BATCH_SIZE}" \
          --upsert-batch-size "${QDRANT_UPSERT_BATCH_SIZE}" \
          --reranker-batch-size "${VIRANKER_BATCH_SIZE}" \
          --reranker-max-length "${VIRANKER_MAX_LENGTH}" \
          --reranker-query-max-length "${VIRANKER_QUERY_MAX_LENGTH}" \
          --prefetch-multiplier "${PREFETCH_MULTIPLIER}" \
          --qdrant-timeout "${QDRANT_TIMEOUT}" \
          --qdrant-upsert-retries "${QDRANT_UPSERT_RETRIES}"
      )
      ;;
    6.0)
      (
        cd "${SCRIPT_DIR}/6.0"
        run_limited ./run.sh "${SIX_VARIANT}" -- \
          --batch-size "${JOBBERT_BATCH_SIZE}" \
          --score-batch-size "${SCORE_BATCH_SIZE}" \
          --max-length "${JOBBERT_MAX_LENGTH}" \
          "${bm25_args[@]}"
      )
      ;;
    *)
      fail "Unknown version: ${version}"
      ;;
  esac
}

run_versions_parallel() {
  local version
  local index
  local pids=()
  local labels=()

  for version in "${VERSIONS[@]}"; do
    run_version "${version}" &
    pids+=("$!")
    labels+=("${version}")
  done

  failed_versions=()
  for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
      failed_versions+=("${labels[${index}]}")
    fi
  done
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

configure_hardware
preflight

if [[ "${CHECK_ONLY}" == "true" ]]; then
  exit 0
fi

if [[ "${RUN_BACKGROUND}" == "true" ]]; then
  start_background
  exit 0
fi

failed_versions=()
if [[ "${RUN_PARALLEL_RESOLVED}" == "true" ]]; then
  run_versions_parallel
else
  for version in "${VERSIONS[@]}"; do
    if ! run_version "${version}"; then
      failed_versions+=("${version}")
      if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
        echo "Stopping after ${version} failure." >&2
        exit 1
      fi
    fi
  done
fi

if [[ ${#failed_versions[@]} -gt 0 ]]; then
  echo "Failed versions: ${failed_versions[*]}" >&2
  exit 1
fi

echo
echo "All requested benchmark suites completed."
