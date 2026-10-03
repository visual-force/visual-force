#!/usr/bin/env bash

# Shared TTS profiles. A profile defines the force target semantics and which
# action dimensions TTS is allowed to modify. Task launchers still own physical
# limits because those values must come from that task's demonstrations.

tts_require_nonnegative() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "${name} must be a non-negative number, got: ${value}" >&2
    return 2
  fi
}

tts_require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    return 2
  fi
}

tts_prepare_inference() {
  local task_name="$1"
  local min_free_mib="${2:-12000}"
  viewforce_select_gpu "${min_free_mib}" "${task_name}"
}

tts_build_queue_args() {
  local policy_steps="${POLICY_ACTION_STEPS:-8}"
  local action_steps="${N_ACTION_STEPS:-8}"
  # Replan after most of the current chunk has executed. Refilling at the full
  # six-step capacity resamples every control tick and stitches together the
  # tail of unrelated diffusion samples, which presents as backtracking.
  local refill_steps="${ACTION_REFILL_STEPS:-2}"
  tts_require_positive_integer POLICY_ACTION_STEPS "${policy_steps}"
  tts_require_positive_integer N_ACTION_STEPS "${action_steps}"
  tts_require_positive_integer ACTION_REFILL_STEPS "${refill_steps}"
  if (( action_steps > policy_steps )); then
    echo "N_ACTION_STEPS cannot exceed POLICY_ACTION_STEPS." >&2
    return 2
  fi
  # policy_server reserves two 100-ms actions for latency compensation.
  local buffered_steps=$((action_steps - 2))
  if (( buffered_steps <= 0 )); then
    echo "N_ACTION_STEPS must exceed the two latency-compensation steps." >&2
    return 2
  fi
  if (( refill_steps > buffered_steps )); then
    echo "ACTION_REFILL_STEPS cannot exceed the ${buffered_steps} executable buffered steps." >&2
    return 2
  fi
  TTS_QUEUE_ARGS=(
    --policy-action-steps "${policy_steps}"
    --n-action-steps "${action_steps}"
    --action-refill-steps "${refill_steps}"
  )
}

tts_build_grasp_profile_args() {
  local desired_delta="$1"
  local baseline_max_position="$2"
  local min_position="$3"
  local max_position="$4"
  local close_step="$5"
  local maintain_step="$6"
  local max_lead="$7"
  local release_step="$8"
  local max_force_rate="$9"
  local deadband="${10}"
  local stop_margin="${11}"
  local policy_release_enabled="${12:-1}"
  local baseline_samples="${13:-3}"
  local policy_release_contact_delta="${14:-${desired_delta}}"
  local policy_gripper_approach="${15:-0}"
  local candidates="${SAMPLING_CANDIDATES:-32}"
  local score_steps="${SCORE_STEPS:-4}"

  tts_require_nonnegative desired_force_delta "${desired_delta}"
  tts_require_positive_integer SAMPLING_CANDIDATES "${candidates}"
  tts_require_positive_integer SCORE_STEPS "${score_steps}"
  tts_require_positive_integer baseline_samples "${baseline_samples}"
  tts_require_nonnegative policy_release_contact_delta "${policy_release_contact_delta}"
  local name value
  for name in \
    baseline_max_position min_position max_position close_step maintain_step \
    max_lead release_step max_force_rate deadband stop_margin; do
    value="${!name}"
    tts_require_nonnegative "${name}" "${value}"
  done
  if [[ "${policy_release_enabled}" != "0" && "${policy_release_enabled}" != "1" ]]; then
    echo "policy_release_enabled must be 0 or 1." >&2
    return 2
  fi
  if [[ "${policy_gripper_approach}" != "0" && "${policy_gripper_approach}" != "1" ]]; then
    echo "policy_gripper_approach must be 0 or 1." >&2
    return 2
  fi

  TTS_PROFILE_ARGS=(
    --tts-desired-force "${desired_delta}"
    --tts-force-target-mode baseline_delta
    --tts-close-positive
    --tts-steering-mode sample
    --tts-selection-scope gripper
    --tts-sampling-candidates "${candidates}"
    --tts-sampling-score-steps "${score_steps}"
    --tts-auto-policy-force-output
    --tts-gentle-gripper-control
    --tts-policy-release-contact-delta "${policy_release_contact_delta}"
    --tts-add-gripper-fallback-candidates
    --tts-gripper-release-step "${release_step}"
    --tts-gripper-safety-min-position "${min_position}"
    --tts-contact-force-delta "${desired_delta}"
    --tts-force-filter-window "${TTS_FORCE_FILTER_WINDOW:-3}"
    --tts-force-baseline-samples "${baseline_samples}"
    --tts-force-baseline-max-position "${baseline_max_position}"
    --tts-gripper-close-step "${close_step}"
    --tts-gripper-maintain-step "${maintain_step}"
    --tts-gripper-max-position "${max_position}"
    --tts-gripper-max-lead "${max_lead}"
    --tts-max-force-rate "${max_force_rate}"
    --tts-deadband "${deadband}"
    --tts-stop-margin "${stop_margin}"
    --tts-unsafe-fallback min_close
    --tts-log-candidates
  )
  if [[ "${policy_release_enabled}" == "0" ]]; then
    TTS_PROFILE_ARGS+=(--tts-disable-policy-release)
  fi
  if [[ "${policy_gripper_approach}" == "1" ]]; then
    TTS_PROFILE_ARGS+=(--tts-policy-gripper-approach)
  fi
  TTS_PROFILE_NAME="grasp"
  TTS_PROFILE_TARGET="${desired_delta} N rise"
  TTS_PROFILE_CANDIDATES="${candidates}"
}

tts_build_manipulation_profile_args() {
  local desired_delta="$1"
  local activation_force="${2:-0.0}"
  local candidates="${SAMPLING_CANDIDATES:-32}"
  local score_steps="${SCORE_STEPS:-4}"
  local selection_scope="${TTS_SELECTION_SCOPE:-full}"

  tts_require_nonnegative desired_force_delta "${desired_delta}"
  tts_require_nonnegative activation_force "${activation_force}"
  tts_require_positive_integer SAMPLING_CANDIDATES "${candidates}"
  tts_require_positive_integer SCORE_STEPS "${score_steps}"
  if [[ "${selection_scope}" != "full" && "${selection_scope}" != "gripper" ]]; then
    echo "TTS_SELECTION_SCOPE must be full or gripper." >&2
    return 2
  fi

  TTS_PROFILE_ARGS=(
    --tts-desired-force "${desired_delta}"
    --tts-force-target-mode baseline_delta
    --tts-close-positive
    --tts-steering-mode sample
    --tts-selection-scope "${selection_scope}"
    --tts-activation-force "${activation_force}"
    --tts-sampling-candidates "${candidates}"
    --tts-sampling-score-steps "${score_steps}"
    --tts-log-candidates
  )
  TTS_PROFILE_NAME="manipulation"
  TTS_PROFILE_TARGET="${desired_delta} N rise"
  TTS_PROFILE_CANDIDATES="${candidates}"
}

tts_print_profile() {
  local task_name="$1"
  printf 'TTS profile: task=%s mode=%s target=%s candidates=%s device=%s\n' \
    "${task_name}" \
    "${TTS_PROFILE_NAME}" \
    "${TTS_PROFILE_TARGET}" \
    "${TTS_PROFILE_CANDIDATES}" \
    "${DEVICE}" >&2
}
