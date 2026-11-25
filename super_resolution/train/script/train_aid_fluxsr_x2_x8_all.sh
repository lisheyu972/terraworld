#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Prevent duplicate training sequences.
MASTER_LOCK_FILE="${MASTER_LOCK_FILE:-logs/.aid_fluxsr_x2_x8_master.lock}"
mkdir -p "$(dirname "${MASTER_LOCK_FILE}")"
exec 8>"${MASTER_LOCK_FILE}"
if ! flock -n 8; then
  echo "The x2/x8 master training sequence is already running." >&2
  echo "Lock file: ${MASTER_LOCK_FILE}" >&2
  exit 3
fi
printf '%s\n' "$$" 1>&8

# Run each experiment on the selected GPUs in sequence.
EXPERIMENTS="${EXPERIMENTS:-x2,x2_glor,x8,x8_glor}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOG_ROOT="${LOG_ROOT:-logs/aid_fluxsr_x2_x8_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_ROOT}"

IFS=',' read -r -a EXPERIMENT_ARRAY <<< "${EXPERIMENTS}"
for experiment in "${EXPERIMENT_ARRAY[@]}"; do
  echo "============================================================"
  echo "Starting ${experiment}; log=${LOG_ROOT}/${experiment}.log"
  echo "============================================================"
  EXPERIMENT="${experiment}" \
  GPU_IDS="${GPU_IDS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  bash train/script/train_aid_fluxsr_scale_ablation.sh \
    2>&1 | tee "${LOG_ROOT}/${experiment}.log"
done

echo "All requested experiments completed: ${EXPERIMENTS}"
echo "Console logs: ${LOG_ROOT}"
