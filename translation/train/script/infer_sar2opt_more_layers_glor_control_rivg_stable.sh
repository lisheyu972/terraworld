#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"

DATA_ROOT="$(bundle_abspath "${DATA_ROOT:-${SAR_DATA_ROOT}}")"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
OUTPUT_DIR="$(bundle_abspath "${OUTPUT_DIR:-${REPO_ROOT}/outputs/sar2opt_glor_control_16000_rivg_stable}")"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
STEPS="${STEPS:-28}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.0}"
RUNS_DIR="$(bundle_abspath "${RUNS_DIR:-${REPO_ROOT}/runs/sar_to_optical_glor_control}")"
CONFIG="${CONFIG:-${REPO_ROOT}/train/config/sar_to_optical_glor_control.yaml}"
CKPT="$(bundle_abspath "${CKPT:-${BUNDLE_ROOT}/assets/checkpoints/sar/16000}")"
GLOR_CKPT="$(bundle_abspath "${GLOR_CKPT:-${CKPT}/glor_control.pt}")"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"

# RIVG-Stable defaults.
WEAK_KERNEL_SIZE="${WEAK_KERNEL_SIZE:-9}"
WEAK_SIGMA="${WEAK_SIGMA:-2.0}"
INFO_LOWER_QUANTILE="${INFO_LOWER_QUANTILE:-0.40}"
INFO_UPPER_QUANTILE="${INFO_UPPER_QUANTILE:-0.95}"
RIVG_LAMBDA_MAX="${RIVG_LAMBDA_MAX:-0.25}"
RIVG_TIME_POWER="${RIVG_TIME_POWER:-2.0}"
RIVG_START_FRACTION="${RIVG_START_FRACTION:-0.70}"
RESIDUAL_NORM_RATIO="${RESIDUAL_NORM_RATIO:-0.50}"
SAVE_RIVG_DEBUG="${SAVE_RIVG_DEBUG:-false}"

CONFIG="$(bundle_abspath "${CONFIG}")"
for required_dir in json images/sar images/optical; do
  if [[ ! -d "${DATA_ROOT}/${required_dir}" ]]; then
    echo "Missing SAR2Opt dataset directory: ${DATA_ROOT}/${required_dir}" >&2
    exit 1
  fi
done
for required_file in "${CONFIG}" "${GLOR_CKPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done
for adapter in text main condition; do
  if [[ ! -f "${CKPT}/${adapter}.safetensors" ]]; then
    echo "Missing checkpoint weight: ${CKPT}/${adapter}.safetensors" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_VERBOSITY=error
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"

CHECKPOINT_ARGS=(--lora-preset expanded --runs-dir "${RUNS_DIR}")
if [[ -n "${CKPT}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${CKPT}" --runs-dir "${RUNS_DIR}")
fi
DEBUG_ARGS=()
if [[ "${SAVE_RIVG_DEBUG}" == "true" ]]; then
  DEBUG_ARGS=(--save-rivg-debug)
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
MULTI_GPU_ARGS=()
if (( NUM_PROCESSES > 1 )); then MULTI_GPU_ARGS+=(--multi_gpu); fi

echo "Inference mode: SAR2Opt + GLoR-Control + RIVG-Stable (training-free)"
echo "Dataset root: ${DATA_ROOT}; split: ${SPLIT}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT:-latest expanded checkpoint below ${RUNS_DIR}}"
echo "GLoR checkpoint: ${GLOR_CKPT}"
echo "RIVG: lambda=${RIVG_LAMBDA_MAX}, start=${RIVG_START_FRACTION}, residual ratio=${RESIDUAL_NORM_RATIO}"
echo "Weak SAR: Gaussian ${WEAK_KERNEL_SIZE}x${WEAK_KERNEL_SIZE}, sigma=${WEAK_SIGMA}"
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
  "${REPO_ROOT}/tools/infer_sar2opt_rivg.py" \
  "${CHECKPOINT_ARGS[@]}" \
  --glor-control \
  --keep-omini-condition \
  --glor-checkpoint "${GLOR_CKPT}" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --image-guidance-scale "${IMAGE_GUIDANCE_SCALE}" \
  --rivg-method stable_rivg \
  --weak-kernel-size "${WEAK_KERNEL_SIZE}" \
  --weak-sigma "${WEAK_SIGMA}" \
  --info-lower-quantile "${INFO_LOWER_QUANTILE}" \
  --info-upper-quantile "${INFO_UPPER_QUANTILE}" \
  --rivg-lambda-max "${RIVG_LAMBDA_MAX}" \
  --rivg-time-power "${RIVG_TIME_POWER}" \
  --rivg-start-fraction "${RIVG_START_FRACTION}" \
  --residual-norm-ratio "${RESIDUAL_NORM_RATIO}" \
  "${DEBUG_ARGS[@]}" \
  "$@"
