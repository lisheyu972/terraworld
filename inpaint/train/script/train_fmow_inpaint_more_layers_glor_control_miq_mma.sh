#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"

GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-1000}"
PRINT_EVERY_N_STEPS="${PRINT_EVERY_N_STEPS:-10}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-41373}"
PRETRAINED_TRANSFORMER="$(bundle_abspath "${PRETRAINED_TRANSFORMER:-${INPAINT_TRANSFORMER_DIR}}")"

if [[ ! -f "${PRETRAINED_TRANSFORMER}/config.json" ]]; then
  echo "Missing Diffusers transformer config: ${PRETRAINED_TRANSFORMER}/config.json" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_TRANSFORMER}/diffusion_pytorch_model.safetensors.index.json" ]]; then
  echo "Missing sharded transformer index below ${PRETRAINED_TRANSFORMER}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMINI_CONFIG="$(bundle_prepare_config inpaint "${OMINI_CONFIG:-${REPO_ROOT}/train/config/fmow_inpaint_glor_control_miq_mma_zero_gate.yaml}")"
export OMINI_TRANSFORMER_PATH="${PRETRAINED_TRANSFORMER}"
export OMINI_LORA_TARGET_PRESET=expanded
export OMINI_KEEP_OMINI_CONDITION=true
export OMINI_BATCH_SIZE="${BATCH_SIZE}"
export OMINI_ACCUMULATE_GRAD_BATCHES="${GRAD_ACCUM_STEPS}"
export OMINI_MAX_STEPS="${MAX_STEPS}"
export OMINI_SAVE_INTERVAL="${SAVE_EVERY_N_STEPS}"
export OMINI_PRINT_EVERY_N_STEPS="${PRINT_EVERY_N_STEPS}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
MULTI_GPU_ARGS=()
if (( NUM_PROCESSES > 1 )); then MULTI_GPU_ARGS+=(--multi_gpu); fi

echo "Training mode: mask-aware GLoR-Control + MIQ-MMA fMOW inpainting"
echo "Pretrained transformer: ${PRETRAINED_TRANSFORMER}"
echo "Metadata injection: target -> main Q; masked source -> condition K"
echo "Checkpoint extras: glor_control.pt + metadata_qkv.pt"
echo "GPU IDs: ${GPU_IDS} (${NUM_PROCESSES} processes)"

accelerate launch \
  "${MULTI_GPU_ARGS[@]}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m omini.train_flux.train_fmow_inpaint
