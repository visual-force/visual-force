#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VISUALFORCE_ROOT="${VISUALFORCE_ROOT:-${DP_ROOT}/../..}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RAW_DATASET="${RAW_DATASET:-data/flip/flip_obj_4}"
DATASET_PATH="${DATASET_PATH:-data/flip/flip_obj_4_stage2_force_output}"
PSEUDO_LABEL_NAME="${PSEUDO_LABEL_NAME:-visualforce_pseudo_force_fz.npz}"
VISUALFORCE_CKPT="${VISUALFORCE_CKPT:-}"
SAM2_ROOT="${SAM2_ROOT:-${VISUALFORCE_ROOT}/third_party/sam2}"
SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_small.pt}"
PREVIEW_DIR="${PREVIEW_DIR:-${VISUALFORCE_ROOT}/reports/flip_obj4_stage2_force_output_preview}"
MASK_MODE="${MASK_MODE:-sam2}"
LABEL_DEVICE="${LABEL_DEVICE:-${DEVICE:-cuda:0}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE:-cuda:1}}"
NUM_EPOCHS="${NUM_EPOCHS:-300}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WANDB_MODE="${WANDB_MODE:-offline}"
DATA_MATERIALIZATION="${DATA_MATERIALIZATION:-symlink}"
GRIPPER_ACTION_MAPPING="${GRIPPER_ACTION_MAPPING:-identity}"
TRAIN_NAME="${TRAIN_NAME:-train_diffusion_unet_flip_obj4_stage2_force_output_quantile}"
TASK_NAME="${TASK_NAME:-flip_obj4_stage2_force_output_image}"
LOGGING_TAGS="${LOGGING_TAGS:-[flip_obj4,stage2,force_output,pseudo_fz,obs_anchor,quantile]}"

usage() {
  cat <<'EOF'
Train the stage-2 flipping diffusion policy with a ninth pseudo-force output.

Usage:
  scripts/train_flip_force_output.sh build
  scripts/train_flip_force_output.sh label
  scripts/train_flip_force_output.sh validate
  scripts/train_flip_force_output.sh prepare
  scripts/train_flip_force_output.sh train [Hydra overrides...]
  scripts/train_flip_force_output.sh all [Hydra overrides...]
  scripts/train_flip_force_output.sh dry-run [Hydra overrides...]

Stages:
  build     Create a validated manifest-backed view of object-4 stage-2 data.
            Existing valid pseudo-Fz files are copied from the prior run.
  label     Reuse valid cached labels, or generate only missing labels.
  validate  Check manifest membership, frame alignment, finite labels, and
            VisualForce checkpoint/mask provenance.
  prepare   Run build, label, and validate without starting policy training.
  train     Train a 9D policy: eight robot actions plus pseudo-|Fz|.
  all       Prepare the dataset and then train.
  dry-run   Print the fully resolved train.py command without training.

The robot-action representation matches the previous stage-2 flip policy:
relative obs-anchor positions with quantile action/state normalization. The
ninth output is trained jointly as part of the diffusion-policy action vector;
this is separate from the existing action-conditioned delta-force critic.

Environment overrides:
  RAW_DATASET, DATASET_PATH, PSEUDO_LABEL_NAME, VISUALFORCE_CKPT,
  SAM2_ROOT, SAM2_CKPT, PREVIEW_DIR, MASK_MODE, LABEL_DEVICE,
  TRAIN_DEVICE, DEVICE, NUM_EPOCHS, CHECKPOINT_EVERY, BATCH_SIZE,
  WANDB_MODE, PYTHON_BIN, TRAIN_NAME, TASK_NAME, LOGGING_TAGS, and RELABEL.

Set DATA_MATERIALIZATION=copy and
GRIPPER_ACTION_MAPPING=legacy_ease_0p080_to_linear_0p088 to create an
independent July-data copy whose action labels match the current controller.

Set RELABEL=1 only to intentionally replace copied labels inside the generated
training view. Raw collection labels are never overwritten.
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
  "${PYTHON_BIN}" tools/flip/build_flip_force_output_dataset.py \
    --source "${RAW_DATASET}" \
    --output "${DATASET_PATH}" \
    --label-name "${PSEUDO_LABEL_NAME}" \
    --force-key Fz \
    --materialization "${DATA_MATERIALIZATION}" \
    --gripper-action-mapping "${GRIPPER_ACTION_MAPPING}" \
    --overwrite
}

validate_dataset() {
  cd "${DP_ROOT}"
  "${PYTHON_BIN}" tools/flip/validate_flip_force_output_dataset.py \
    --dataset "${DATASET_PATH}" \
    --label-name "${PSEUDO_LABEL_NAME}" \
    --force-key Fz \
    --visualforce-ckpt "${VISUALFORCE_CKPT}" \
    --mask-mode "${MASK_MODE}" \
    --gripper-action-mapping "${GRIPPER_ACTION_MAPPING}"
}

labels_are_valid() {
  validate_dataset >/dev/null 2>&1
}

label_dataset() {
  local dataset_abs
  local -a overwrite_args=()
  dataset_abs="$(dataset_abs_path)"
  if [[ ! -d "${dataset_abs}/episodes" ]]; then
    echo "Dataset view not found: ${dataset_abs}" >&2
    echo "Run '$0 build' first." >&2
    return 1
  fi
  if [[ "${RELABEL:-0}" != "1" ]] && labels_are_valid; then
    echo "Reusing validated pseudo-force labels in ${DATASET_PATH}"
    return
  fi
  if [[ ! -f "${VISUALFORCE_CKPT}" ]]; then
    echo "VisualForce checkpoint not found: ${VISUALFORCE_CKPT}" >&2
    return 1
  fi
  if [[ ! -f "${SAM2_CKPT}" ]]; then
    echo "SAM2 checkpoint not found: ${SAM2_CKPT}" >&2
    return 1
  fi
  if [[ "${RELABEL:-0}" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi
  cd "${VISUALFORCE_ROOT}"
  "${PYTHON_BIN}" scripts/precompute_flip_pseudo_force.py \
    --dataset-path "${dataset_abs}" \
    --visualforce-ckpt "${VISUALFORCE_CKPT}" \
    --output-name "${PSEUDO_LABEL_NAME}" \
    --force-keys Fz \
    --mask-mode "${MASK_MODE}" \
    --sam2-model small \
    --sam2-repo "${SAM2_ROOT}" \
    --sam2-ckpt "${SAM2_CKPT}" \
    --device "${LABEL_DEVICE}" \
    --batch-size 64 \
    --preview-dir "${PREVIEW_DIR}" \
    --preview-frames 6 \
    "${overwrite_args[@]}"
  if ! validate_dataset; then
    echo "Pseudo-force labels are incomplete or have mixed provenance." >&2
    echo "Inspect the error above; use RELABEL=1 only if replacement is intended." >&2
    return 1
  fi
}

train_policy() {
  local train_python="${PYTHON_BIN}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    train_python=/bin/echo
  else
    validate_dataset
  fi
  cd "${DP_ROOT}"
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}" \
  "${train_python}" train.py \
    --config-name=train_diffusion_unet_pick_berries_hybrid_workspace \
    task=pick_berries_all_image \
    name="${TRAIN_NAME}" \
    task.name="${TASK_NAME}" \
    task.dataset_path="${DATASET_PATH}" \
    'task.shape_meta.action.shape=[9]' \
    task.dataset.relative_position_action=true \
    task.dataset.relative_position_action_mode=obs_anchor \
    task.dataset.relative_gripper_action=false \
    task.dataset.append_force_to_action=true \
    task.dataset.force_label_name="${PSEUDO_LABEL_NAME}" \
    task.dataset.force_key=Fz \
    task.dataset.force_mode=magnitude \
    task.dataset.action_normalizer_mode=quantile \
    task.dataset.gripper_action_normalizer_mode=null \
    task.dataset.agent_pos_normalizer_mode=quantile \
    dataloader.batch_size="${BATCH_SIZE}" \
    val_dataloader.batch_size="${BATCH_SIZE}" \
    training.device="${TRAIN_DEVICE}" \
    training.num_epochs="${NUM_EPOCHS}" \
    training.checkpoint_every="${CHECKPOINT_EVERY}" \
    training.resume=false \
    logging.mode="${WANDB_MODE}" \
    "logging.tags=${LOGGING_TAGS}" \
    "$@"
}

case "${1:-help}" in
  build) build_dataset ;;
  label) label_dataset ;;
  validate) validate_dataset ;;
  prepare)
    build_dataset
    label_dataset
    validate_dataset
    ;;
  train)
    shift
    train_policy "$@"
    ;;
  all)
    shift
    build_dataset
    label_dataset
    validate_dataset
    train_policy "$@"
    ;;
  dry-run)
    shift
    DRY_RUN=1 train_policy "$@"
    ;;
  help|-h|--help) usage ;;
  *) echo "Unknown stage: $1" >&2; usage >&2; exit 2 ;;
esac
