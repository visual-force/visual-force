# Shared command construction for action-conditioned force-critic launchers.
# Callers provide profile-specific values through the documented variables and
# argument arrays before invoking either function.

critic_precompute_labels() {
  local -a preview_args=()
  local -a overwrite_args=()
  if [[ -n "${PREVIEW_DIR:-}" ]]; then
    preview_args+=(--preview-dir "${PREVIEW_DIR}" --preview-frames "${PREVIEW_FRAMES:-6}")
  fi
  if [[ "${PRECOMPUTE_OVERWRITE:-0}" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" scripts/precompute_flip_pseudo_force.py \
    --dataset-path "${DATASET_PATH}" \
    --visualforce-ckpt "${VISUALFORCE_CKPT}" \
    --output-name "${PSEUDO_LABEL_NAME}" \
    "${PRECOMPUTE_PROFILE_ARGS[@]}" \
    --force-keys "${FORCE_KEY:-Fz}" \
    --mask-mode sam2 \
    --sam2-model "${SAM2_MODEL:-small}" \
    --sam2-repo "${SAM2_ROOT}" \
    --sam2-ckpt "${SAM2_CKPT}" \
    --device "${DEVICE}" \
    --batch-size "${PRECOMPUTE_BATCH_SIZE:-64}" \
    "${preview_args[@]}" \
    "${overwrite_args[@]}"
}

critic_train() {
  local runner="${CRITIC_RUNNER:-${PYTHON_BIN}}"
  local -a logging_args=(--no-wandb)
  if [[ "${NO_WANDB:-1}" != "1" ]]; then
    logging_args=(--wandb-project "${WANDB_PROJECT:-force_estimation}" --wandb-run "${WANDB_RUN}")
  fi

  cd "${REPO_ROOT}"
  "${runner}" scripts/train_flip_delta_force.py \
    --dataset-path "${DATASET_PATH}" \
    --pseudo-label-name "${PSEUDO_LABEL_NAME}" \
    --force-key "${FORCE_KEY:-Fz}" \
    --force-mode "${FORCE_MODE:-magnitude}" \
    --action-mode "${ACTION_MODE:-obs_delta_pos_gripper}" \
    --obs-steps "${OBS_STEPS:-2}" \
    --image-size "${IMAGE_HEIGHT:-240}" "${IMAGE_WIDTH:-320}" \
    "${TRAIN_PROFILE_ARGS[@]}" \
    --target-mode "${TARGET_MODE}" \
    --epochs "${EPOCHS:-100}" \
    --lr "${LEARNING_RATE:-1e-4}" \
    --weight-decay "${WEIGHT_DECAY:-1e-4}" \
    --cum-loss-weight "${CUM_LOSS_WEIGHT:-0.0}" \
    --device "${DEVICE}" \
    --output "${OUTPUT_DIR}" \
    "${logging_args[@]}"
}
