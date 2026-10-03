#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${REPO_ROOT}/scripts/critic_pipeline.sh"

usage() {
  cat <<'EOF'
Build pseudo-force labels and train a Flip action-conditioned force critic.

Usage:
  scripts/train_flip_force_critic.sh PROFILE [STAGE] [TARGET_MODE]

Profiles:
  stage2           Stage-2 chunk critic (alias: stage2_chunk)
  stage2_delta     Original per-step delta trajectory critic
  dynamics         Object-4 dynamics chunk critic

Stages:
  label            Generate missing VisualForce pseudo-labels
  train            Train from an already labelled dataset
  all              Label, then train (default)
  dry-run          Print the resolved training command

TARGET_MODE defaults to final_delta for chunk profiles and delta_trajectory
for stage2_delta. For compatibility, TARGET_MODE may be supplied directly as
the second argument; this implies the all stage.

Environment overrides:
  PYTHON_BIN, DEVICE, DATASET_PATH, PSEUDO_LABEL_NAME, VISUALFORCE_CKPT,
  SAM2_ROOT, SAM2_CKPT, OUTPUT_DIR, PREVIEW_DIR, EPOCHS, BATCH_SIZE,
  PRECOMPUTE_OVERWRITE, WANDB_PROJECT, and NO_WANDB.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PROFILE="$1"
shift

case "${PROFILE}" in
  stage2|stage2_chunk)
    PROFILE=stage2
    DEFAULT_DEVICE=cuda:1
    DEFAULT_DATASET_PATH=third_party/forcelens_dp/data/flip/flip_obj_4_stage2_all
    DEFAULT_TARGET_MODE=final_delta
    DEFAULT_OUTPUT_DIR=checkpoints/flip_chunk_force_obj4_stage2_final_delta_v1
    OUTPUT_PREFIX=checkpoints/flip_chunk_force_obj4_stage2
    PRED_HORIZON=16
    CUM_LOSS_WEIGHT=0.0
    DEFAULT_NO_WANDB=0
    WANDB_RUN_PREFIX=flip_obj4_stage2_chunk_force
    DEFAULT_WANDB_RUN=flip_obj4_stage2_chunk_force_final_delta_v1
    DEFAULT_PREVIEW_DIR=
    ;;
  stage2_delta)
    DEFAULT_DEVICE=cuda:1
    DEFAULT_DATASET_PATH=third_party/forcelens_dp/data/flip/flip_obj_4_stage2_all
    DEFAULT_TARGET_MODE=delta_trajectory
    DEFAULT_OUTPUT_DIR=checkpoints/flip_delta_force_obj4_stage2_v2_dp_obs
    OUTPUT_PREFIX=checkpoints/flip_delta_force_obj4_stage2
    PRED_HORIZON=16
    CUM_LOSS_WEIGHT=1.0
    DEFAULT_NO_WANDB=0
    WANDB_RUN_PREFIX=flip_obj4_stage2_delta_force
    DEFAULT_WANDB_RUN=flip_obj4_stage2_delta_force_v2_dp_obs
    DEFAULT_PREVIEW_DIR=
    ;;
  dynamics)
    DEFAULT_DEVICE=cuda:0
    DEFAULT_DATASET_PATH=third_party/forcelens_dp/data/flip/obj4_dynamics
    DEFAULT_TARGET_MODE=final_delta
    DEFAULT_OUTPUT_DIR=checkpoints/flip_chunk_force_obj4_dynamics_final_delta_v1
    OUTPUT_PREFIX=checkpoints/flip_chunk_force_obj4_dynamics
    PRED_HORIZON=16
    CUM_LOSS_WEIGHT=0.0
    DEFAULT_NO_WANDB=1
    WANDB_RUN_PREFIX=flip_obj4_dynamics_chunk_force
    DEFAULT_WANDB_RUN=flip_obj4_dynamics_chunk_force_final_delta_v1
    DEFAULT_PREVIEW_DIR=reports/flip_obj4_dynamics_pseudo_force_preview
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown Flip critic profile: ${PROFILE}" >&2
    usage >&2
    exit 2
    ;;
esac

STAGE=all
if [[ $# -gt 0 ]]; then
  case "$1" in
    label|train|all|dry-run)
      STAGE="$1"
      shift
      ;;
  esac
fi

TARGET_MODE="${1:-${DEFAULT_TARGET_MODE}}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${TARGET_MODE}" in
  delta_trajectory|final_delta|max_delta|mean_delta|future_peak) ;;
  *)
    echo "Unknown target mode: ${TARGET_MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
if [[ $# -gt 0 ]]; then
  echo "Unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

DEVICE="${DEVICE:-${DEFAULT_DEVICE}}"
DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
PSEUDO_LABEL_NAME="${PSEUDO_LABEL_NAME:-visualforce_pseudo_force_fz.npz}"
VISUALFORCE_CKPT="${VISUALFORCE_CKPT:-}"
SAM2_ROOT="${SAM2_ROOT:-third_party/sam2}"
SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_small.pt}"
PREVIEW_DIR="${PREVIEW_DIR-${DEFAULT_PREVIEW_DIR}}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NO_WANDB="${NO_WANDB:-${DEFAULT_NO_WANDB}}"
WANDB_PROJECT="${WANDB_PROJECT:-force_estimation}"
PRECOMPUTE_PROFILE_ARGS=()
TRAIN_PROFILE_ARGS=(
  --pred-horizon "${PRED_HORIZON}"
  --image-normalization minus_one_one
  --include-agent-pos
  --batch-size "${BATCH_SIZE}"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  RESOLVED_OUTPUT_DIR="${OUTPUT_DIR}"
elif [[ "${TARGET_MODE}" == "${DEFAULT_TARGET_MODE}" ]]; then
  RESOLVED_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
else
  RESOLVED_OUTPUT_DIR="${OUTPUT_PREFIX}_${TARGET_MODE}_v1"
fi

train_critic() {
  WANDB_RUN="${WANDB_RUN_PREFIX}_${TARGET_MODE}_v1"
  if [[ "${TARGET_MODE}" == "${DEFAULT_TARGET_MODE}" ]]; then
    WANDB_RUN="${DEFAULT_WANDB_RUN}"
  fi
  if [[ "${STAGE}" == "dry-run" ]]; then
    CRITIC_RUNNER=/bin/echo
  fi
  OUTPUT_DIR="${RESOLVED_OUTPUT_DIR}"
  critic_train
}

case "${STAGE}" in
  label) critic_precompute_labels ;;
  train|dry-run) train_critic ;;
  all)
    critic_precompute_labels
    train_critic
    ;;
esac
