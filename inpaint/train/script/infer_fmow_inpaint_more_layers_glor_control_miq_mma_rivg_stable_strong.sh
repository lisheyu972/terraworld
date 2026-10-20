#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"

GPU_IDS="${GPU_IDS:-${GPU_ID:-4,5,6,7}}"
PRETRAINED_TRANSFORMER="$(bundle_abspath "${PRETRAINED_TRANSFORMER:-${INPAINT_TRANSFORMER_DIR}}")"
OUTPUT_DIR="$(bundle_abspath "${OUTPUT_DIR:-${REPO_ROOT}/outputs/fmow_inpaint_glor_control_miq_mma_test_rivg_stable_strong}")"
TEST_ROOT="$(bundle_abspath "${TEST_ROOT:-${FMOW_TEST_ROOT}}")"
TEST_INDEX="$(bundle_abspath "${TEST_INDEX:-${FMOW_TEST_INDEX}}")"
FRAME_MODE="${FRAME_MODE:-first}"
MASK_TYPE="${MASK_TYPE:-random}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
STEPS="${STEPS:-28}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.0}"
CONFIG="$(bundle_prepare_config inpaint "${CONFIG:-${REPO_ROOT}/train/config/fmow_inpaint_glor_control_miq_mma_zero_gate.yaml}")"
RUNS_DIR="$(bundle_abspath "${RUNS_DIR:-${REPO_ROOT}/runs/fmow_inpaint_glor_control_miq_mma_zero_gate}")"
CKPT="$(bundle_abspath "${CKPT:-/data/dxc/grsm/OminiControl-main-183/runs/fmow_inpaint_glor_control_miq_mma_zero_gate/20260802-024258/ckpt/10000}")"
GLOR_CKPT="$(bundle_abspath "${GLOR_CKPT:-${CKPT}/glor_control.pt}")"

# Strong Stable settings selected from the reference perceptual-metric experiment.
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

for required in config.json diffusion_pytorch_model.safetensors.index.json; do
  if [[ ! -f "${PRETRAINED_TRANSFORMER}/${required}" ]]; then
    echo "Missing pretrained transformer file: ${PRETRAINED_TRANSFORMER}/${required}" >&2
    exit 1
  fi
done
if [[ -n "${CKPT}" ]]; then
  for required in text.safetensors main.safetensors condition.safetensors glor_control.pt metadata_qkv.pt; do
    if [[ ! -f "${CKPT}/${required}" ]]; then
      echo "Missing combined checkpoint file: ${CKPT}/${required}" >&2
      exit 1
    fi
  done
fi

GPU_IDS="${GPU_IDS//[[:space:]]/}"
if [[ -z "${GPU_IDS}" ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
for GPU in "${GPU_ARRAY[@]}"; do
  if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS must be a comma-separated list of GPU ids: ${GPU_IDS}" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

CHECKPOINT_ARGS=(--lora-preset expanded --runs-dir "${RUNS_DIR}")
if [[ -n "${CKPT}" ]]; then CHECKPOINT_ARGS=(--checkpoint "${CKPT}"); fi
GLOR_ARGS=()
if [[ -n "${GLOR_CKPT}" ]]; then GLOR_ARGS=(--glor-checkpoint "${GLOR_CKPT}"); fi
DEBUG_ARGS=()
if [[ "${SAVE_RIVG_DEBUG}" == "true" ]]; then
  DEBUG_ARGS=(--save-rivg-debug)
fi

echo "Inference mode: mask-aware GLoR-Control + MIQ-MMA + Strong RIVG-Stable"
echo "Pretrained transformer: ${PRETRAINED_TRANSFORMER}"
echo "Checkpoint: ${CKPT:-latest compatible checkpoint under ${RUNS_DIR}}"
echo "GLoR checkpoint: ${GLOR_CKPT:-glor_control.pt beside LoRA checkpoint}"
echo "Output: ${OUTPUT_DIR}"
echo "RIVG: lambda=${RIVG_LAMBDA_MAX}, start=${RIVG_START_FRACTION}, time_power=${RIVG_TIME_POWER}, residual ratio=${RESIDUAL_NORM_RATIO}"
echo "GPUs: ${GPU_IDS} (${NUM_GPUS} processes)"
echo "Distributed rendezvous: standalone on 127.0.0.1 (automatic free port)"

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --local-addr=127.0.0.1 \
  --nproc_per_node="${NUM_GPUS}" \
  "${REPO_ROOT}/tools/infer_fmow_inpaint.py" \
  "${CHECKPOINT_ARGS[@]}" \
  "${GLOR_ARGS[@]}" \
  --config "${CONFIG}" \
  --transformer-path "${PRETRAINED_TRANSFORMER}" \
  --test-root "${TEST_ROOT}" \
  --test-index "${TEST_INDEX}" \
  --frame-mode "${FRAME_MODE}" \
  --mask-type "${MASK_TYPE}" \
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
  --rivg-info-fusion "${RIVG_INFO_FUSION}" \
  --rivg-lambda-max "${RIVG_LAMBDA_MAX}" \
  --rivg-time-power "${RIVG_TIME_POWER}" \
  --rivg-start-fraction "${RIVG_START_FRACTION}" \
  --residual-norm-ratio "${RESIDUAL_NORM_RATIO}" \
  "${DEBUG_ARGS[@]}" \
  "$@"
