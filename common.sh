#!/usr/bin/env bash
# Shared launcher setup. PROJECT_ROOT is set by each entry script.
set -euo pipefail

BUNDLE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"
if [[ -f "${BUNDLE_ROOT}/paths.env" ]]; then
  set -a
  source "${BUNDLE_ROOT}/paths.env"
  set +a
fi

FLUX_MODEL_DIR="${FLUX_MODEL_DIR:-${BUNDLE_ROOT}/assets/models/flux}"
FRAME_TRANSFORMER_DIR="${FRAME_TRANSFORMER_DIR:-${BUNDLE_ROOT}/assets/models/frame-transformer}"
SAR_TRANSFORMER_DIR="${SAR_TRANSFORMER_DIR:-${BUNDLE_ROOT}/assets/models/sar-transformer}"
INPAINT_TRANSFORMER_DIR="${INPAINT_TRANSFORMER_DIR:-${BUNDLE_ROOT}/assets/models/inpaint-transformer}"
SR_TRANSFORMER_DIR="${SR_TRANSFORMER_DIR:-${BUNDLE_ROOT}/assets/models/sr-transformer}"
FMOW_TRAIN_ROOT="${FMOW_TRAIN_ROOT:-${BUNDLE_ROOT}/assets/datasets/fmow/train}"
FMOW_TRAIN_INDEX="${FMOW_TRAIN_INDEX:-${FMOW_TRAIN_ROOT}/index.jsonl}"
FMOW_TEST_ROOT="${FMOW_TEST_ROOT:-${BUNDLE_ROOT}/assets/datasets/fmow/test}"
FMOW_TEST_INDEX="${FMOW_TEST_INDEX:-${FMOW_TEST_ROOT}/index.jsonl}"
NIR_DATA_ROOT="${NIR_DATA_ROOT:-${BUNDLE_ROOT}/assets/datasets/nir}"
SAR_DATA_ROOT="${SAR_DATA_ROOT:-${BUNDLE_ROOT}/assets/datasets/sar}"
AID_DATA_ROOT="${AID_DATA_ROOT:-${BUNDLE_ROOT}/assets/datasets/aid}"
export FLUX_MODEL_DIR FRAME_TRANSFORMER_DIR SAR_TRANSFORMER_DIR INPAINT_TRANSFORMER_DIR SR_TRANSFORMER_DIR
export FMOW_TRAIN_ROOT FMOW_TRAIN_INDEX FMOW_TEST_ROOT FMOW_TEST_INDEX NIR_DATA_ROOT SAR_DATA_ROOT AID_DATA_ROOT

bundle_abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(os.path.join(sys.argv[1], os.path.expanduser(sys.argv[2]))))' "${PROJECT_ROOT}" "$1"
}

bundle_prepare_config() {
  local task="$1" template="$2" output
  mkdir -p "${PROJECT_ROOT}/runtime/config" || return
  output="${PROJECT_ROOT}/runtime/config/${task}-${BASHPID}.yaml"
  python3 "${BUNDLE_ROOT}/prepare_config.py" --task "${task}" --project-root "${PROJECT_ROOT}" --template "$(bundle_abspath "${template}")" --output "${output}" || return
  printf '%s\n' "${output}"
}
