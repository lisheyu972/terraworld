#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"

GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-1000}"
PRINT_EVERY_N_STEPS="${PRINT_EVERY_N_STEPS:-10}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-41365}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMINI_CONFIG="$(bundle_prepare_config nir "${OMINI_CONFIG:-${REPO_ROOT}/train/config/nir_to_optical_glor_control.yaml}")"
export OMINI_LORA_TARGET_PRESET=expanded
export OMINI_KEEP_OMINI_CONDITION="${OMINI_KEEP_OMINI_CONDITION:-true}"
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_VERBOSITY=error
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
export OMINI_BATCH_SIZE="${BATCH_SIZE}"
export OMINI_ACCUMULATE_GRAD_BATCHES="${GRAD_ACCUM_STEPS}"
export OMINI_MAX_STEPS="${MAX_STEPS}"
export OMINI_SAVE_INTERVAL="${SAVE_EVERY_N_STEPS}"
export OMINI_PRINT_EVERY_N_STEPS="${PRINT_EVERY_N_STEPS}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
MULTI_GPU_ARGS=()
if (( NUM_PROCESSES > 1 )); then MULTI_GPU_ARGS+=(--multi_gpu); fi

echo "Using config: ${OMINI_CONFIG}"
echo "Training mode: paired NIR -> optical at native 512x512 + GLoR-Control + expanded LoRA"
echo "GLoR-Control: global memory + local retrieval injected into middle/late DiT blocks"
echo "LoRA branches: text + generated image + NIR condition"
echo "Keep original Omini condition: ${OMINI_KEEP_OMINI_CONDITION}"
echo "GPU IDs: ${GPU_IDS} (${NUM_PROCESSES} processes)"
echo "Effective global batch: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_PROCESSES))"
accelerate launch \
  "${MULTI_GPU_ARGS[@]}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m omini.train_flux.train_nir_to_optical
