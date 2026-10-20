#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../common.sh"
cd "${REPO_ROOT}"

# Reproduce the settings recorded by the two existing GLoR-Control outputs,
# then add training-free RIVG-Stable. With PRESET=both, the two jobs run
# sequentially and each job uses all six selected GPUs.
PRESET="${PRESET:-both}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-1,2,3,4,5,6,7}}"
DATA_ROOT="$(bundle_abspath "${DATA_ROOT:-${SAR_DATA_ROOT}}")"

if [[ "${PRESET}" == "both" ]]; then
  for preset in 30000 16000; do
    echo "===== Starting SAR-to-optical GLoR + RIVG-Stable preset ${preset} ====="
    env \
      -u CONFIG \
      -u CKPT \
      -u GLOR_CKPT \
      -u OUTPUT_DIR \
      PRESET="${preset}" \
      GPU_IDS="${GPU_IDS}" \
      DATA_ROOT="${DATA_ROOT}" \
      bash "${SCRIPT_DIR}/infer_sar_to_optical_more_layers_glor_control_rivg_stable.sh" "$@"
  done
  exit 0
fi

case "${PRESET}" in
  30000)
    DEFAULT_CONFIG="${REPO_ROOT}/train/config/sar_to_optical_glor_control_30000.yaml"
    DEFAULT_CKPT="/data/dxc/186_backups/OminiControl-main/runs/sar_to_optical_glor_control/20260726-030518/ckpt/30000"
    DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs/sar_to_optical_glor_control8000_30000_rivg_stable"
    ;;
  16000)
    DEFAULT_CONFIG="${REPO_ROOT}/train/config/sar_to_optical_glor_control.yaml"
    DEFAULT_CKPT="/data/dxc/186_backups/OminiControl-main/runs/sar_to_optical_glor_control/20260724-020936/ckpt/16000"
    DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs/sar_to_optical_glor_control_16000_rivg_stable"
    ;;
  *)
    echo "Unsupported PRESET=${PRESET}; expected 30000, 16000, or both." >&2
    exit 2
    ;;
esac

export DATA_ROOT
export GPU_IDS
export RUNS_DIR="$(bundle_abspath "${RUNS_DIR:-${REPO_ROOT}/runs/sar_to_optical_glor_control}")"
export CONFIG="$(bundle_prepare_config sar "${CONFIG:-${DEFAULT_CONFIG}}")"
export CKPT="$(bundle_abspath "${CKPT:-${DEFAULT_CKPT}}")"
export GLOR_CKPT="$(bundle_abspath "${GLOR_CKPT:-${CKPT}/glor_control.pt}")"
export OUTPUT_DIR="$(bundle_abspath "${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}")"

echo "Selected preset: ${PRESET}"

exec bash \
  "${SCRIPT_DIR}/infer_sar2opt_more_layers_glor_control_rivg_stable.sh" \
  "$@"
