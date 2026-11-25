#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
CKPT="${CKPT:?Set CKPT to a FluxSR checkpoint directory.}"
SCALE="${SCALE:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/aid_fluxsr_ftd_test}"

case "${SCALE}" in
  2|8) DEFAULT_CONFIG="train/config/aid_fluxsr_ftd_rsbackbone_x${SCALE}_glor.yaml" ;;
  4) DEFAULT_CONFIG="train/config/aid_fluxsr_ftd_rsbackbone_glor.yaml" ;;
  *) echo "SCALE must be 2, 4, or 8." >&2; exit 2 ;;
esac
RUNTIME_CONFIG="$(bundle_prepare_config sr "${CONFIG:-${DEFAULT_CONFIG}}")"
DATASET_ROOT="$(bundle_abspath "${DATASET_ROOT:-${AID_DATA_ROOT}}")"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

python tools/infer_aid_fluxsr_ftd.py \
  --checkpoint "${CKPT}" \
  --config "${RUNTIME_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
