#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"
cd "${REPO_ROOT}"

SR_TRANSFORMER_DIR="$(bundle_abspath "${TRANSFORMER_PATH:-${SR_TRANSFORMER_DIR}}")"
GPU_IDS="${GPU_IDS:-0}"

if [[ ! -d "${SR_TRANSFORMER_DIR}" ]]; then
  echo "Remote-sensing FLUX transformer not found: ${SR_TRANSFORMER_DIR}" >&2
  exit 2
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must contain comma-separated GPU indices." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMINI_NUM_DEVICES="${#GPU_ARRAY[@]}"
OMINI_CONFIG="$(bundle_prepare_config sr "${CONFIG:-${OMINI_CONFIG:-train/config/aid_fluxsr_ftd_rsbackbone_glor.yaml}}")"
export OMINI_CONFIG
export OMINI_TRANSFORMER_PATH="${SR_TRANSFORMER_DIR}"
# Start the GLoR comparison with fresh LoRA weights.
unset OMINI_INIT_LORA_PATH || true
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Method: One-step FluxSR-FTD + sparse GLoR observation retrieval"
echo "Transformer: ${SR_TRANSFORMER_DIR}"
echo "Initial LoRA: none (new rank-16 LoRA)"
echo "Config: ${OMINI_CONFIG}"
echo "Physical GPUs: ${GPU_IDS}"
echo "DDP processes: ${OMINI_NUM_DEVICES}"

python -m omini.train_flux.train_aid_fluxsr_ftd
