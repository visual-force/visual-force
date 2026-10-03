#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/critic_pipeline.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:1}"
REBUILD="${REBUILD:-0}"
PRECOMPUTE_OVERWRITE="${PRECOMPUTE_OVERWRITE:-0}"

FORCELENS_DP_ROOT="${REPO_ROOT}/third_party/forcelens_dp"
PSEUDO_LABEL_NAME="${PSEUDO_LABEL_NAME:-visualforce_pseudo_force_fz.npz}"
VISUALFORCE_CKPT="${VISUALFORCE_CKPT:-}"
SAM2_ROOT="${SAM2_ROOT:-${REPO_ROOT}/third_party/sam2}"
SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_small.pt}"
EPOCHS="${EPOCHS:-100}"

usage() {
  cat <<'EOF'
Build data, generate labels, and train a Berry force-prediction critic.

Usage:
  scripts/train_berry_force_critic.sh PROFILE [STAGE] [TARGET_MODE]

Profiles:
  standard     RGB critic, 16-step prediction horizon
  action_only  Action-only critic, 4-step prediction horizon
  edge         Edge-image critic, 16-step prediction horizon
  h4           RGB critic, 4-step prediction horizon

Stages:
  build        Build the deterministic verifier dataset
  label        Generate missing VisualForce pseudo-labels
  train        Train from an already built and labelled dataset
  all          Build if needed, label, then train (default)
  dry-run      Print the resolved training command

TARGET_MODE defaults to final_delta for standard, future_peak for action_only,
and max_delta for edge and h4. For compatibility, TARGET_MODE may be supplied
directly as the second argument; this implies the all stage.

Environment overrides include PYTHON_BIN, DEVICE, EPOCHS, OUTPUT_DIR, REBUILD,
PRECOMPUTE_OVERWRITE, PSEUDO_LABEL_NAME, VISUALFORCE_CKPT, SAM2_ROOT, and
SAM2_CKPT.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PROFILE="$1"
shift

BUILD_ARGS=()
PRECOMPUTE_PROFILE_ARGS=()
TRAIN_PROFILE_ARGS=()

case "${PROFILE}" in
  standard)
    DEFAULT_TARGET_MODE=final_delta
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all"
    OUTPUT_SUFFIX=
    PREVIEW_DIR="${PREVIEW_DIR:-${REPO_ROOT}/reports/berry_force_prediction_pseudo_force_preview}"
    BUILD_ARGS+=(--overwrite)
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 16
      --image-normalization minus_one_one
      --include-agent-pos
      --batch-size 32
    )
    ;;
  action_only)
    DEFAULT_TARGET_MODE=future_peak
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all_action_only"
    OUTPUT_SUFFIX=_action_only
    PREVIEW_DIR="${PREVIEW_DIR:-${REPO_ROOT}/reports/berry_force_prediction_action_only_pseudo_force_preview}"
    BUILD_ARGS+=(
      --policy-output data/pick/berry_stage2_policy_action_only_tmp
      --verifier-output data/pick/berry_force_prediction_all_action_only
      --overwrite
    )
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 4
      --critic-image-mode none
      --batch-size 64
    )
    ;;
  edge)
    DEFAULT_TARGET_MODE=max_delta
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all_edge"
    OUTPUT_SUFFIX=_edge
    PREVIEW_DIR="${PREVIEW_DIR:-${REPO_ROOT}/reports/berry_force_prediction_edge_pseudo_force_preview}"
    EDGE_OBS_NAME="${EDGE_OBS_NAME:-visualforce_edge_obs.npz}"
    BUILD_ARGS+=(
      --policy-output data/pick/berry_stage2_policy_edge_tmp
      --verifier-output data/pick/berry_force_prediction_all_edge
      --overwrite
    )
    PRECOMPUTE_PROFILE_ARGS+=(--edge-output-name "${EDGE_OBS_NAME}")
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 16
      --image-normalization zero_one
      --critic-image-mode edge
      --edge-obs-name "${EDGE_OBS_NAME}"
      --include-agent-pos
      --batch-size 32
    )
    ;;
  h4)
    DEFAULT_TARGET_MODE=max_delta
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all_h4"
    OUTPUT_SUFFIX=_h4
    PREVIEW_DIR="${PREVIEW_DIR:-${REPO_ROOT}/reports/berry_force_prediction_h4_pseudo_force_preview}"
    BUILD_ARGS+=(
      --policy-output data/pick/berry_stage2_policy_h4_tmp
      --verifier-output data/pick/berry_force_prediction_all_h4
      --overwrite
    )
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 4
      --image-normalization minus_one_one
      --critic-image-mode rgb
      --include-agent-pos
      --batch-size 32
    )
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown Berry critic profile: ${PROFILE}" >&2
    usage >&2
    exit 2
    ;;
esac

STAGE=all
if [[ $# -gt 0 ]]; then
  case "$1" in
    build|label|train|all|dry-run)
      STAGE="$1"
      shift
      ;;
  esac
fi

TARGET_MODE="${1:-${DEFAULT_TARGET_MODE}}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

case "${TARGET_MODE}" in
  delta_trajectory|final_delta|max_delta|mean_delta|future_peak) ;;
  *) echo "Unknown target mode: ${TARGET_MODE}" >&2; usage >&2; exit 2 ;;
esac

if [[ "${PROFILE}" == "action_only" && "${TARGET_MODE}" == "future_peak" ]]; then
  TRAIN_PROFILE_ARGS+=(--include-current-force)
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/berry_force_prediction_${TARGET_MODE}${OUTPUT_SUFFIX}_v1}"

build_dataset() {
  cd "${FORCELENS_DP_ROOT}"
  "${PYTHON_BIN}" tools/berry/build_berry_staged_datasets.py "${BUILD_ARGS[@]}"
}

ensure_dataset() {
  if [[ "${REBUILD}" == "1" || ! -d "${DATASET_PATH}/episodes" ]]; then
    build_dataset
  else
    printf 'Reusing existing verifier dataset: %s\n' "${DATASET_PATH}"
  fi
}

train_critic() {
  if [[ "${STAGE}" == "dry-run" ]]; then
    CRITIC_RUNNER=/bin/echo
  fi
  CUM_LOSS_WEIGHT=0.0
  NO_WANDB=1
  critic_train
}

case "${STAGE}" in
  build) build_dataset ;;
  label) critic_precompute_labels ;;
  train|dry-run) train_critic ;;
  all)
    ensure_dataset
    critic_precompute_labels
    train_critic
    ;;
esac
