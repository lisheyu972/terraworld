#!/usr/bin/env bash
# Shared launcher setup. PROJECT_ROOT is set by each entry script.
set -euo pipefail

BUNDLE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"
if [[ -f "${BUNDLE_ROOT}/paths.env" ]]; then
  set -a
  source "${BUNDLE_ROOT}/paths.env"
  set +a
fi

FLUX_MODEL_DIR="${FLUX_MODEL_DIR:-/data/dxc/model_zoo/models--black-forest-labs--FLUX.1-dev/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21}"
FRAME_TRANSFORMER_DIR="${FRAME_TRANSFORMER_DIR:-/data/dxc/grsm/flux1/runs/flux_git10m_continue_from_8000/checkpoint-16000-diffusers}"
SAR_TRANSFORMER_DIR="${SAR_TRANSFORMER_DIR:-/data/dxc/grsm/flux1/runs/flux_git10m_full/checkpoint-8000-diffusers}"
INPAINT_TRANSFORMER_DIR="${INPAINT_TRANSFORMER_DIR:-/data/dxc/grsm/flux1/outputs/checkpoint-3000}"
FMOW_TRAIN_ROOT="${FMOW_TRAIN_ROOT:-/data/zjz/fmow_processed}"
FMOW_TRAIN_INDEX="${FMOW_TRAIN_INDEX:-${FMOW_TRAIN_ROOT}/index.jsonl}"
FMOW_TEST_ROOT="${FMOW_TEST_ROOT:-/data/zjz/fmow_test_processed}"
FMOW_TEST_INDEX="${FMOW_TEST_INDEX:-${FMOW_TEST_ROOT}/index.jsonl}"
NIR_DATA_ROOT="${NIR_DATA_ROOT:-/data/dxc/grsm/datasets/lin/grsm_opt_nir_whu}"
SAR_DATA_ROOT="${SAR_DATA_ROOT:-/data/dxc/grsm/datasets/lin/grsm_opt_sar}"
export FLUX_MODEL_DIR FRAME_TRANSFORMER_DIR SAR_TRANSFORMER_DIR INPAINT_TRANSFORMER_DIR
export FMOW_TRAIN_ROOT FMOW_TRAIN_INDEX FMOW_TEST_ROOT FMOW_TEST_INDEX NIR_DATA_ROOT SAR_DATA_ROOT

bundle_abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(os.path.join(sys.argv[1], os.path.expanduser(sys.argv[2]))))' "${PROJECT_ROOT}" "$1"
}

bundle_prepare_config() {
  local task="$1" template="$2" output
  mkdir -p "${PROJECT_ROOT}/runtime/config"
  output="${PROJECT_ROOT}/runtime/config/${task}-${BASHPID}.yaml"
  python3 "${BUNDLE_ROOT}/prepare_config.py" --task "${task}" --project-root "${PROJECT_ROOT}" --template "$(bundle_abspath "${template}")" --output "${output}"
  printf '%s\n' "${output}"
}
