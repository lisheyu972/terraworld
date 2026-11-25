#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"
cd "${REPO_ROOT}"

SR_TRANSFORMER_DIR="$(bundle_abspath "${TRANSFORMER_PATH:-${SR_TRANSFORMER_DIR}}")"
GPU_IDS="${GPU_IDS:-0}"
INIT_LORA_PATH="${INIT_LORA_PATH:-}"

if [[ ! -d "${SR_TRANSFORMER_DIR}" ]]; then
  echo "Remote-sensing FLUX transformer not found: ${SR_TRANSFORMER_DIR}" >&2
  exit 2
fi
if [[ -n "${INIT_LORA_PATH}" ]]; then
  if [[ -d "${INIT_LORA_PATH}" ]]; then
    INIT_WEIGHT="${INIT_LORA_PATH}/default.safetensors"
  else
    INIT_WEIGHT="${INIT_LORA_PATH}"
  fi
  if [[ ! -f "${INIT_WEIGHT}" ]]; then
    echo "Initial LoRA weight not found: ${INIT_WEIGHT}" >&2
    exit 2
  fi
fi

if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must contain comma-separated GPU indices." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMINI_NUM_DEVICES="${#GPU_ARRAY[@]}"
OMINI_CONFIG="$(bundle_prepare_config sr "${CONFIG:-${OMINI_CONFIG:-train/config/aid_fluxsr_ftd_rsbackbone_best.yaml}}")"
export OMINI_CONFIG
export OMINI_TRANSFORMER_PATH="${SR_TRANSFORMER_DIR}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${INIT_LORA_PATH}" ]]; then
  export OMINI_INIT_LORA_PATH="${INIT_LORA_PATH}"
else
  unset OMINI_INIT_LORA_PATH || true
fi

echo "Method: FluxSR best Y-fidelity recipe with remote-sensing FLUX backbone"
echo "Base pipeline components: FLUX.1-dev"
echo "Transformer override: ${SR_TRANSFORMER_DIR}"
echo "Config: ${OMINI_CONFIG}"
echo "Physical GPUs: ${GPU_IDS}"
echo "DDP processes: ${OMINI_NUM_DEVICES}"
echo "Initial LoRA: ${INIT_LORA_PATH:-none (new LoRA)}"

python -m omini.train_flux.train_aid_fluxsr_ftd
