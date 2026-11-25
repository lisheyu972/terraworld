#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"
cd "${REPO_ROOT}"

# Prevent two DDP experiments from claiming the same GPUs at once.
TRAIN_LOCK_FILE="${TRAIN_LOCK_FILE:-logs/.aid_fluxsr_gpu012.lock}"
mkdir -p "$(dirname "${TRAIN_LOCK_FILE}")"
exec 9>"${TRAIN_LOCK_FILE}"
if ! flock -n 9; then
  echo "Another AID FluxSR scale experiment is already running." >&2
  echo "Lock file: ${TRAIN_LOCK_FILE}" >&2
  exit 3
fi
printf '%s\n' "$$" 1>&9

EXPERIMENT="${EXPERIMENT:-${1:-}}"
case "${EXPERIMENT}" in
  x2)
    DEFAULT_CONFIG="train/config/aid_fluxsr_ftd_rsbackbone_x2.yaml"
    ;;
  x2_glor)
    DEFAULT_CONFIG="train/config/aid_fluxsr_ftd_rsbackbone_x2_glor.yaml"
    ;;
  x8)
    DEFAULT_CONFIG="train/config/aid_fluxsr_ftd_rsbackbone_x8.yaml"
    ;;
  x8_glor)
    DEFAULT_CONFIG="train/config/aid_fluxsr_ftd_rsbackbone_x8_glor.yaml"
    ;;
  *)
    echo "Usage: $0 {x2|x2_glor|x8|x8_glor}" >&2
    exit 2
    ;;
esac

SR_TRANSFORMER_DIR="$(bundle_abspath "${TRANSFORMER_PATH:-${SR_TRANSFORMER_DIR}}")"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"

if [[ ! -d "${SR_TRANSFORMER_DIR}" ]]; then
  echo "Remote-sensing FLUX transformer not found: ${SR_TRANSFORMER_DIR}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Training config not found: ${CONFIG}" >&2
  exit 2
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must contain comma-separated GPU indices." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMINI_NUM_DEVICES="${#GPU_ARRAY[@]}"
export OMINI_BATCH_SIZE="${BATCH_SIZE}"
OMINI_CONFIG="$(bundle_prepare_config sr "${CONFIG}")"
export OMINI_CONFIG
export OMINI_TRANSFORMER_PATH="${SR_TRANSFORMER_DIR}"
unset OMINI_INIT_LORA_PATH || true
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Experiment: ${EXPERIMENT}"
echo "Config: ${OMINI_CONFIG}"
echo "Transformer: ${OMINI_TRANSFORMER_PATH}"
echo "Physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "DDP processes: ${OMINI_NUM_DEVICES}"
echo "Initial LoRA: none; a new rank-16 LoRA is trained"
echo "GPU lock acquired: ${TRAIN_LOCK_FILE}"

python -m omini.train_flux.train_aid_fluxsr_ftd
