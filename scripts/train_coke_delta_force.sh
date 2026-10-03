#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/critic_pipeline.sh"
DP_ROOT="${REPO_ROOT}/third_party/forcelens_dp"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"

SOFT_DATASET="${SOFT_DATASET:-data/pick/coke_soft_0821}"
HARD_DATASET="${HARD_DATASET:-data/pick/coke_hard_0821}"
DYNAMICS_DATASET="${DYNAMICS_DATASET:-data/pick/coke_dyn_0821}"
LABEL_CACHE="${LABEL_CACHE:-data/pick/coke_mixed_0821_force_output}"
DATASET_PATH="${DATASET_PATH:-data/pick/coke_delta_force_all_0821}"
PSEUDO_LABEL_NAME="${PSEUDO_LABEL_NAME:-visualforce_pseudo_force_fz.npz}"
VISUALFORCE_CKPT="${VISUALFORCE_CKPT:-}"
SAM2_ROOT="${SAM2_ROOT:-${REPO_ROOT}/third_party/sam2}"
SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_small.pt}"
PREVIEW_DIR="${PREVIEW_DIR:-${REPO_ROOT}/reports/coke_delta_force_all_0821_preview}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/coke_delta_force_all_0821_final_delta_v1}"
TARGET_MODE="${TARGET_MODE:-final_delta}"
EPOCHS="${EPOCHS:-100}"
REBUILD="${REBUILD:-0}"

usage() {
  cat <<'EOF'
Build labels and train the Coke action-conditioned delta-force critic.

Usage:
  bash scripts/train_coke_delta_force.sh build
  bash scripts/train_coke_delta_force.sh label
  bash scripts/train_coke_delta_force.sh validate
  bash scripts/train_coke_delta_force.sh train
  bash scripts/train_coke_delta_force.sh all
  bash scripts/train_coke_delta_force.sh dry-run

The generated view combines all three Coke collections:
  coke_soft_0821 (20), coke_hard_0821 (20), coke_dyn_0821 (12)

The build reuses the 40 soft/hard pseudo-force files already present in the
label cache. The label stage therefore runs frozen VisualForce only for missing
episodes (normally the 12 dynamics episodes).

The default critic predicts the final force change over a 16-action candidate
chunk. This matches the earlier delta-force TTS scoring path. It does not add a
force head to the diffusion policy; the DP and this critic remain separate.

Environment overrides:
  DEVICE, PYTHON_BIN, EPOCHS, TARGET_MODE, OUTPUT_DIR, REBUILD,
  SOFT_DATASET, HARD_DATASET, DYNAMICS_DATASET, LABEL_CACHE, DATASET_PATH,
  VISUALFORCE_CKPT, SAM2_ROOT, SAM2_CKPT, PREVIEW_DIR, PSEUDO_LABEL_NAME.
EOF
}

dataset_abs_path() {
  if [[ "${DATASET_PATH}" == /* ]]; then
    printf '%s' "${DATASET_PATH}"
  else
    printf '%s/%s' "${DP_ROOT}" "${DATASET_PATH}"
  fi
}

build_dataset() {
  cd "${DP_ROOT}"
  "${PYTHON_BIN}" tools/coke/build_coke_delta_force_dataset.py build \
    --soft-root "${SOFT_DATASET}" \
    --hard-root "${HARD_DATASET}" \
    --dynamics-root "${DYNAMICS_DATASET}" \
    --label-cache "${LABEL_CACHE}" \
    --label-name "${PSEUDO_LABEL_NAME}" \
    --output "${DATASET_PATH}" \
    --overwrite
}

ensure_dataset() {
  local dataset_abs
  dataset_abs="$(dataset_abs_path)"
  if [[ "${REBUILD}" == "1" || ! -d "${dataset_abs}/episodes" ]]; then
    build_dataset
  else
    printf 'Reusing combined dataset: %s\n' "${dataset_abs}"
  fi
}

label_dataset() {
  local dataset_abs
  dataset_abs="$(dataset_abs_path)"
  if [[ ! -d "${dataset_abs}/episodes" ]]; then
    echo "Dataset view not found: ${dataset_abs}; run the build stage first." >&2
    return 1
  fi
  local -a PRECOMPUTE_PROFILE_ARGS=()
  DATASET_PATH="${dataset_abs}" critic_precompute_labels
}

validate_dataset() {
  cd "${DP_ROOT}"
  "${PYTHON_BIN}" tools/coke/build_coke_delta_force_dataset.py validate \
    --dataset "$(dataset_abs_path)" \
    --label-name "${PSEUDO_LABEL_NAME}" \
    --force-key Fz
}

train_critic() {
  local runner="${PYTHON_BIN}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    runner=/bin/echo
  else
    validate_dataset
  fi
  local -a TRAIN_PROFILE_ARGS=(
    --pred-horizon 16
    --image-normalization minus_one_one
    --critic-image-mode rgb
    --include-agent-pos
    --batch-size 32
  )
  CUM_LOSS_WEIGHT=0.0 \
    NO_WANDB=1 \
    CRITIC_RUNNER="${runner}" \
    DATASET_PATH="$(dataset_abs_path)" \
    critic_train
}

case "${1:-help}" in
  build) build_dataset ;;
  label) label_dataset ;;
  validate) validate_dataset ;;
  train) train_critic ;;
  all)
    ensure_dataset
    label_dataset
    validate_dataset
    train_critic
    ;;
  dry-run) DRY_RUN=1 train_critic ;;
  help|-h|--help) usage ;;
  *) echo "Unknown stage: $1" >&2; usage >&2; exit 2 ;;
esac
