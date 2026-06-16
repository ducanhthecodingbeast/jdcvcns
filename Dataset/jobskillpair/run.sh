#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Export kaggle username and token here (copy them from your env file):
# export KAGGLE_USERNAME="..."
# export KAGGLE_KEY="..."

# Alternatively, automatically load them from an .env file if it exists
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
elif [[ -f "Dataset/.env.local" ]]; then
    set -a
    source "Dataset/.env.local"
    set +a
fi

# Run the python scripts using venv python if it exists, else system python
if [[ -f "Dataset/.venv/bin/python3" ]]; then
    PYTHON_EXEC="Dataset/.venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

run_pipeline() {
    "${PYTHON_EXEC}" -m Dataset.jobskillpair.jobskillpair "$@"
    "${PYTHON_EXEC}" -m Dataset.jobskillpair.datapreprocessing "$@"
}

# Check for --background flag
if [[ "${1:-}" == "--background" ]]; then
    shift
    LOG_FILE="Dataset/Data/jobskillpair.log"
    echo "Starting jobskillpair pipeline in background using ${PYTHON_EXEC}..."
    nohup bash -c '"$0" -m Dataset.jobskillpair.jobskillpair "${@:2}" && "$0" -m Dataset.jobskillpair.datapreprocessing "${@:2}"' "${PYTHON_EXEC}" _ "$@" > "$LOG_FILE" 2>&1 &
    echo "Background process started! PID: $!"
    echo "You can monitor the progress by running: tail -f $LOG_FILE"
    exit 0
fi

run_pipeline "$@"
