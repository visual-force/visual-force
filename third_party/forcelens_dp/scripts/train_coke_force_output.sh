#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VISUALFORCE_ROOT="${VISUALFORCE_ROOT:-${DP_ROOT}/../..}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"

SOFT_DATASET="${SOFT_DATASET:-data/pick/coke_soft_0821}"
HARD_DATASET="${HARD_DATASET:-data/pick/coke_hard_0821}"
DATASET_PATH="${DATASET_PATH:-data/pick/coke_mixed_0821_force_output}"
PSEUDO_LABEL_NAME="${PSEUDO_LABEL_NAME:-visualforce_pseudo_force_fz.npz}"
VISUALFORCE_CKPT="${VISUALFORCE_CKPT:-}"
SAM2_ROOT="${SAM2_ROOT:-${VISUALFORCE_ROOT}/third_party/sam2}"
SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_small.pt}"
PREVIEW_DIR="${PREVIEW_DIR:-${VISUALFORCE_ROOT}/reports/coke_force_output_preview}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
WANDB_MODE="${WANDB_MODE:-offline}"

usage() {
  cat <<'EOF'
Train the 9D Coke action-force policy (8 robot actions + |Fz|).

Usage:
  scripts/train_coke_force_output.sh build|label|validate|train|all|dry-run

Required for label/all:
  VISUALFORCE_CKPT=/path/to/force-estimator.pt

The generated force-output view is separate from the raw collections and is
never committed by VisualForce.
EOF
}

dataset_abs_path() {
  if [[ "${DATASET_PATH}" == /* ]]; then
    printf '%s' "${DATASET_PATH}"
  else
    printf '%s/%s' "${DP_ROOT}" "${DATASET_PATH}"
  fi
}

require_estimator() {
  if [[ -z "${VISUALFORCE_CKPT}" || ! -f "${VISUALFORCE_CKPT}" ]]; then
    echo "VISUALFORCE_CKPT must point to the external force-estimator checkpoint." >&2
    return 1
  fi
}

build_dataset() {
  cd "${DP_ROOT}"
  "${PYTHON_BIN}" tools/coke/build_coke_mixed_dataset.py \
    --soft-root "${SOFT_DATASET}" \
    --hard-root "${HARD_DATASET}" \
    --output "${DATASET_PATH}" \
    --overwrite
}

label_dataset() {
  require_estimator
  local dataset_abs
  dataset_abs="$(dataset_abs_path)"
  if [[ ! -d "${dataset_abs}/episodes" ]]; then
    echo "Dataset view not found: ${dataset_abs}; run build first." >&2
    return 1
  fi
  cd "${VISUALFORCE_ROOT}"
  "${PYTHON_BIN}" scripts/precompute_flip_pseudo_force.py \
    --dataset-path "${dataset_abs}" \
    --visualforce-ckpt "${VISUALFORCE_CKPT}" \
    --output-name "${PSEUDO_LABEL_NAME}" \
    --force-keys Fz \
    --mask-mode sam2 \
    --sam2-model small \
    --sam2-repo "${SAM2_ROOT}" \
    --sam2-ckpt "${SAM2_CKPT}" \
    --device "${DEVICE}" \
    --batch-size 64 \
    --preview-dir "${PREVIEW_DIR}" \
    --preview-frames 6
}

validate_labels() {
  local dataset_abs episode_count label_count
  dataset_abs="$(dataset_abs_path)"
  episode_count="$(find "${dataset_abs}/episodes" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  label_count="$(find "${dataset_abs}/episodes" -mindepth 2 -maxdepth 2 -type f -name "${PSEUDO_LABEL_NAME}" | wc -l)"
  printf 'episodes: %s\nlabels: %s\n' "${episode_count}" "${label_count}"
  [[ "${episode_count}" -gt 0 && "${label_count}" -eq "${episode_count}" ]]
}

train_policy() {
  local train_python="${PYTHON_BIN}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    train_python=/bin/echo
  else
    validate_labels
  fi
  cd "${DP_ROOT}"
  cd "${DP_ROOT}"
  WANDB_MODE="${WANDB_MODE}" WANDB_DISABLE_SERVICE=true \
    "${train_python}" train.py \
      --config-name=train_diffusion_unet_pick_coke_hybrid_workspace \
      name=train_diffusion_unet_pick_coke_hybrid_force_output \
      task.name=coke_mixed_force_output_image \
      task.dataset.dataset_path="${DATASET_PATH}" \
      'task.shape_meta.action.shape=[9]' \
      +task.dataset.append_force_to_action=true \
      +task.dataset.force_label_name="${PSEUDO_LABEL_NAME}" \
      +task.dataset.force_key=Fz \
      +task.dataset.force_mode=magnitude \
      task.dataset.action_normalizer_mode=quantile \
      +task.dataset.gripper_action_normalizer_mode=limits \
      task.dataset.agent_pos_normalizer_mode=quantile \
      training.device="${DEVICE}" \
      training.num_epochs="${NUM_EPOCHS}" \
      training.checkpoint_every="${CHECKPOINT_EVERY}" \
      training.resume=false \
      logging.mode="${WANDB_MODE}" \
      'logging.tags=[coke,force_output,pseudo_fz,quantile]'
}

case "${1:-help}" in
  build) build_dataset ;;
  label) label_dataset ;;
  validate) validate_labels ;;
  train) train_policy ;;
  all) build_dataset; label_dataset; validate_labels; train_policy ;;
  dry-run) DRY_RUN=1 train_policy ;;
  help|-h|--help) usage ;;
  *) echo "Unknown stage: $1" >&2; usage >&2; exit 2 ;;
esac
