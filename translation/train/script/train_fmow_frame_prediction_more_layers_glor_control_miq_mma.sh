#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"

GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-1000}"
PRINT_EVERY_N_STEPS="${PRINT_EVERY_N_STEPS:-10}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-41370}"
PRETRAINED_TRANSFORMER="$(bundle_abspath "${PRETRAINED_TRANSFORMER:-${FRAME_TRANSFORMER_DIR}}")"

if [[ ! -f "${PRETRAINED_TRANSFORMER}/config.json" ]]; then
  echo "Missing Diffusers transformer config: ${PRETRAINED_TRANSFORMER}/config.json" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_TRANSFORMER}/diffusion_pytorch_model.safetensors.index.json" ]]; then
  echo "Missing sharded transformer weight index: ${PRETRAINED_TRANSFORMER}/diffusion_pytorch_model.safetensors.index.json" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMINI_CONFIG="$(bundle_prepare_config frame "${OMINI_CONFIG:-${REPO_ROOT}/train/config/fmow_frame_prediction_glor_control_miq_mma_zero_gate.yaml}")"
export OMINI_TRANSFORMER_PATH="${PRETRAINED_TRANSFORMER}"
export OMINI_LORA_TARGET_PRESET=expanded
export OMINI_KEEP_OMINI_CONDITION=true
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
echo "Training mode: fMOW frame prediction + temporal GLoR-Control + MIQ-MMA"
echo "Pretrained transformer: ${OMINI_TRANSFORMER_PATH}"
echo "GLoR: dual-frame global memory + temporally tagged local retrieval"
echo "MIQ-MMA: target -> main Q; each context -> its condition K"
echo "Keep original Omini conditions: true (required by MIQ-MMA)"
echo "Checkpoint extras: glor_control.pt + metadata_qkv.pt"
echo "GPU IDs: ${GPU_IDS} (${NUM_PROCESSES} processes)"
echo "Effective global batch: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_PROCESSES))"

accelerate launch \
  "${MULTI_GPU_ARGS[@]}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m omini.train_flux.train_fmow_frame_prediction_glor_miq_mma
