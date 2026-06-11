#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Starting all environments in the background"
echo "=========================================="

echo "Starting Dataset preparation..."
"${SCRIPT_DIR}/Dataset/run.sh" --background

echo "Starting 1.0 test suite..."
"${SCRIPT_DIR}/1.0/run.sh" --background

echo "Starting 2.0 test suite..."
"${SCRIPT_DIR}/2.0/run.sh" --background

echo "Starting 3.0 test suite..."
"${SCRIPT_DIR}/3.0/run.sh" --background

echo "Starting 4.0 test suite..."
"${SCRIPT_DIR}/4.0/run.sh" --background

echo "=========================================="
echo "All scripts have been launched!"
echo "Check the run.log file in each folder to monitor progress."
echo "Example: tail -f 4.0/run.log"
