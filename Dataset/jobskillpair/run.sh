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

# Run the python script directly using system python
# Assuming user has packages installed globally or is in their own venv.
python3 -m Dataset.jobskillpair.jobskillpair "$@"
