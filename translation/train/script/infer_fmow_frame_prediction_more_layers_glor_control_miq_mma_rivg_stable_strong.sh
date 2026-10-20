#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"

GPU_IDS="${GPU_IDS:-${GPU_ID:-4,5,6,7}}"
OUTPUT_DIR="$(bundle_abspath "${OUTPUT_DIR:-${REPO_ROOT}/outputs/fmow_frame_prediction_glor_control_miq_mma_12_rivg_stable_strong}")"
SPLIT="${SPLIT:-test}"
TARGET_SLOT="${TARGET_SLOT:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
STEPS="${STEPS:-28}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.0}"
CONFIG="$(bundle_prepare_config frame "${CONFIG:-${REPO_ROOT}/train/config/fmow_frame_prediction_glor_control_miq_mma_zero_gate.yaml}")"
RUNS_DIR="$(bundle_abspath "${RUNS_DIR:-${REPO_ROOT}/runs/fmow_frame_prediction_glor_control_miq_mma_zero_gate32}")"
CKPT="$(bundle_abspath "${CKPT:-/data/dxc/186_backups/OminiControl-main/runs/fmow_frame_prediction_glor_control_miq_mma_zero_gate32/20260801-230350/ckpt/12000}")"
GLOR_CKPT="$(bundle_abspath "${GLOR_CKPT:-${CKPT}/glor_control.pt}")"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"

# Strong Stable settings selected from the perceptual-metric experiment.
WEAK_KERNEL_SIZE="${WEAK_KERNEL_SIZE:-9}"
WEAK_SIGMA="${WEAK_SIGMA:-2.0}"
INFO_LOWER_QUANTILE="${INFO_LOWER_QUANTILE:-0.35}"
INFO_UPPER_QUANTILE="${INFO_UPPER_QUANTILE:-0.95}"
RIVG_INFO_FUSION="${RIVG_INFO_FUSION:-max}"
RIVG_LAMBDA_MAX="${RIVG_LAMBDA_MAX:-0.40}"
RIVG_TIME_POWER="${RIVG_TIME_POWER:-1.5}"
RIVG_START_FRACTION="${RIVG_START_FRACTION:-0.60}"
RESIDUAL_NORM_RATIO="${RESIDUAL_NORM_RATIO:-0.65}"
SAVE_RIVG_DEBUG="${SAVE_RIVG_DEBUG:-false}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing inference config: ${CONFIG}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_VERBOSITY=error
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"

CHECKPOINT_ARGS=(--lora-preset expanded --runs-dir "${RUNS_DIR}")
if [[ -n "${CKPT}" ]]; then
  for required in text.safetensors main.safetensors condition.safetensors glor_control.pt metadata_qkv.pt; do
    if [[ ! -f "${CKPT}/${required}" ]]; then
      echo "Missing combined checkpoint file: ${CKPT}/${required}" >&2
      exit 1
    fi
  done
  CHECKPOINT_ARGS=(--checkpoint "${CKPT}")
fi

GLOR_ARGS=()
if [[ -n "${GLOR_CKPT}" ]]; then
  GLOR_ARGS=(--glor-checkpoint "${GLOR_CKPT}")
fi
DEBUG_ARGS=()
if [[ "${SAVE_RIVG_DEBUG}" == "true" ]]; then
  DEBUG_ARGS=(--save-rivg-debug)
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
MULTI_GPU_ARGS=()
if (( NUM_PROCESSES > 1 )); then MULTI_GPU_ARGS+=(--multi_gpu); fi

echo "Inference mode: temporal GLoR-Control + MIQ-MMA + Strong RIVG-Stable"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT:-latest combined checkpoint below ${RUNS_DIR}}"
echo "GLoR checkpoint: ${GLOR_CKPT:-auto beside checkpoint}"
echo "RIVG: lambda=${RIVG_LAMBDA_MAX}, start=${RIVG_START_FRACTION}, time_power=${RIVG_TIME_POWER}, residual ratio=${RESIDUAL_NORM_RATIO}"
echo "Dataset partition: ${SPLIT}; target slot: ${TARGET_SLOT}"
echo "GPU IDs: ${GPU_IDS} (${NUM_PROCESSES} inference processes)"
echo "Distributed rendezvous: ${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
echo "Output directory: ${OUTPUT_DIR}"

accelerate launch \
  "${MULTI_GPU_ARGS[@]}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend "${DYNAMO_BACKEND}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${REPO_ROOT}/tools/infer_fmow_frame_prediction_glor_control.py" \
  "${CHECKPOINT_ARGS[@]}" \
  "${GLOR_ARGS[@]}" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --target-slot "${TARGET_SLOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --image-guidance-scale "${IMAGE_GUIDANCE_SCALE}" \
  --keep-omini-condition \
  --rivg-method stable_rivg \
  --weak-kernel-size "${WEAK_KERNEL_SIZE}" \
  --weak-sigma "${WEAK_SIGMA}" \
  --info-lower-quantile "${INFO_LOWER_QUANTILE}" \
  --info-upper-quantile "${INFO_UPPER_QUANTILE}" \
  --rivg-info-fusion "${RIVG_INFO_FUSION}" \
  --rivg-lambda-max "${RIVG_LAMBDA_MAX}" \
  --rivg-time-power "${RIVG_TIME_POWER}" \
  --rivg-start-fraction "${RIVG_START_FRACTION}" \
  --residual-norm-ratio "${RESIDUAL_NORM_RATIO}" \
  "${DEBUG_ARGS[@]}" \
  "$@"
