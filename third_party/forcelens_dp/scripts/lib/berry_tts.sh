#!/usr/bin/env bash

# Berry force-aware inference and baseline commands.

DEFAULT_DESIRED_FORCE="5.0"
DEFAULT_ACTIVATION_FORCE="0.5"
DEFAULT_FORCE_AGGREGATION="max"
DEFAULT_SAMPLING_CANDIDATES="32"
DEFAULT_SCORE_STEPS="4"
DEFAULT_SELECTION_SCOPE="gripper"
DEFAULT_DIRECT_GRIPPER_SAFETY="0"
DEFAULT_GENTLE_GRIPPER_CONTROL="1"
DEFAULT_GRIPPER_RELEASE_STEP="0.05"
DEFAULT_GRIPPER_SAFETY_MIN_POSITION="0.35"
DEFAULT_CONTACT_FORCE_DELTA="5.0"
DEFAULT_FORCE_FILTER_WINDOW="3"
DEFAULT_FORCE_BASELINE_MAX_POSITION="0.35"
# Persistent increments accumulate until the physical gripper clears its
# actuator deadband; max lead prevents an unresponsive actuator from running
# the command directly to the hard position cap.
DEFAULT_GRIPPER_CLOSE_STEP="0.05"
DEFAULT_GRIPPER_MAINTAIN_STEP="0.005"
DEFAULT_GRIPPER_MAX_POSITION="0.82"
DEFAULT_GRIPPER_MAX_LEAD="0.25"
DEFAULT_MAX_FORCE_RATE="20.0"
DEFAULT_FORCE_SAFETY_LIMIT="10.5"
DEFAULT_GENTLE_DEADBAND="0.5"
DEFAULT_GENTLE_STOP_MARGIN="2.0"

berry_prompt_value() {
  local variable_name="$1"
  local label="$2"
  local current_value="${!variable_name}"
  local entered_value=""
  read -r -p "${label} [${current_value}]: " entered_value
  if [[ -n "${entered_value}" ]]; then
    printf -v "${variable_name}" '%s' "${entered_value}"
  fi
}

berry_validate_tts_parameters() {
  local desired_force="$1"
  local activation_force="$2"
  local aggregation="$3"
  local sampling_candidates="$4"
  local score_steps="$5"
  local selection_scope="$6"

  if [[ ! "${desired_force}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Desired force must be a non-negative number, got: ${desired_force}" >&2
    return 2
  fi
  if [[ ! "${activation_force}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Activation force must be a non-negative number, got: ${activation_force}" >&2
    return 2
  fi
  if [[ "${aggregation}" != "last" && "${aggregation}" != "mean" && "${aggregation}" != "max" ]]; then
    echo "Force aggregation must be last, mean, or max; got: ${aggregation}" >&2
    return 2
  fi
  if [[ ! "${sampling_candidates}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Sampling candidates must be a positive integer, got: ${sampling_candidates}" >&2
    return 2
  fi
  if [[ ! "${score_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Score steps must be a positive integer, got: ${score_steps}" >&2
    return 2
  fi
  if [[ "${selection_scope}" != "gripper" && "${selection_scope}" != "full" ]]; then
    echo "TTS selection scope must be gripper or full; got: ${selection_scope}" >&2
    return 2
  fi
}

berry_common_monitor_args() {
  build_viewforce_common_args
  tts_build_queue_args
  COMMON_ARGS+=(
    "${TTS_QUEUE_ARGS[@]}"
    --gripper-command-min 0.0
    --gripper-command-max 1.0
  )
}

berry_run_tts() {
  local interactive=0
  local desired_force="${DESIRED_FORCE:-${DEFAULT_DESIRED_FORCE}}"
  local activation_force="${ACTIVATION_FORCE:-${DEFAULT_ACTIVATION_FORCE}}"
  local aggregation="${FORCE_AGGREGATION:-${DEFAULT_FORCE_AGGREGATION}}"
  local sampling_candidates="${SAMPLING_CANDIDATES:-${DEFAULT_SAMPLING_CANDIDATES}}"
  local score_steps="${SCORE_STEPS:-${DEFAULT_SCORE_STEPS}}"
  local selection_scope="${TTS_SELECTION_SCOPE:-${DEFAULT_SELECTION_SCOPE}}"
  local direct_gripper_safety="${TTS_DIRECT_GRIPPER_SAFETY:-${DEFAULT_DIRECT_GRIPPER_SAFETY}}"
  local gentle_gripper_control="${TTS_GENTLE_GRIPPER_CONTROL:-${DEFAULT_GENTLE_GRIPPER_CONTROL}}"
  local gripper_release_step="${TTS_GRIPPER_RELEASE_STEP:-${DEFAULT_GRIPPER_RELEASE_STEP}}"
  local gripper_safety_min_position="${TTS_GRIPPER_SAFETY_MIN_POSITION:-${DEFAULT_GRIPPER_SAFETY_MIN_POSITION}}"
  local contact_force_delta="${TTS_CONTACT_FORCE_DELTA:-${DEFAULT_CONTACT_FORCE_DELTA}}"
  local force_filter_window="${TTS_FORCE_FILTER_WINDOW:-${DEFAULT_FORCE_FILTER_WINDOW}}"
  local force_baseline_max_position="${TTS_FORCE_BASELINE_MAX_POSITION:-${DEFAULT_FORCE_BASELINE_MAX_POSITION}}"
  local gripper_close_step="${TTS_GRIPPER_CLOSE_STEP:-${DEFAULT_GRIPPER_CLOSE_STEP}}"
  local gripper_maintain_step="${TTS_GRIPPER_MAINTAIN_STEP:-${DEFAULT_GRIPPER_MAINTAIN_STEP}}"
  local gripper_max_position="${TTS_GRIPPER_MAX_POSITION:-${DEFAULT_GRIPPER_MAX_POSITION}}"
  local gripper_max_lead="${TTS_GRIPPER_MAX_LEAD:-${DEFAULT_GRIPPER_MAX_LEAD}}"
  local max_force_rate="${TTS_MAX_FORCE_RATE:-${DEFAULT_MAX_FORCE_RATE}}"
  local force_safety_limit="${TTS_FORCE_SAFETY_LIMIT:-${DEFAULT_FORCE_SAFETY_LIMIT}}"
  local gentle_deadband="${TTS_GENTLE_DEADBAND:-${DEFAULT_GENTLE_DEADBAND}}"
  local gentle_stop_margin="${TTS_GENTLE_STOP_MARGIN:-${DEFAULT_GENTLE_STOP_MARGIN}}"
  local rollout_dir="${ROLLOUT_DIR:-${VIEWFORCE_ROOT}/rollouts/berry_force_output_tts}"
  local -a extra_args=()

  if (( $# == 0 )) || [[ "${1:-}" == "--interactive" || "${1:-}" == "-i" ]]; then
    interactive=1
    if (( $# > 0 )); then
      shift
    fi
    extra_args=("$@")
    if [[ ! -t 0 ]]; then
      echo "Interactive TTS requires a terminal." >&2
      return 2
    fi
    berry_select_checkpoint_interactive
    berry_prompt_value desired_force "Desired force (N)"
    berry_prompt_value activation_force "ViewForce activation threshold (N)"
    berry_prompt_value aggregation "Force aggregation (last/mean/max)"
    berry_prompt_value sampling_candidates "Number of DP candidates"
    berry_prompt_value score_steps "Candidate scoring steps"
    berry_prompt_value selection_scope "Selection scope (gripper/full)"
  elif [[ "${1:-}" == "--no-interactive" ]]; then
    shift
    extra_args=("$@")
  else
    desired_force="${1:-${desired_force}}"
    activation_force="${2:-${activation_force}}"
    aggregation="${3:-${aggregation}}"
    if (( $# > 3 )); then
      extra_args=("${@:4}")
    fi
  fi

  berry_validate_tts_parameters \
    "${desired_force}" \
    "${activation_force}" \
    "${aggregation}" \
    "${sampling_candidates}" \
    "${score_steps}" \
    "${selection_scope}"

  if [[ -z "${TTS_CONTACT_FORCE_DELTA+x}" ]]; then
    contact_force_delta="${desired_force}"
  fi

  if [[ "${direct_gripper_safety}" != "0" && "${direct_gripper_safety}" != "1" ]]; then
    echo "TTS_DIRECT_GRIPPER_SAFETY must be 0 or 1; got: ${direct_gripper_safety}" >&2
    return 2
  fi
  if [[ "${gentle_gripper_control}" != "0" && "${gentle_gripper_control}" != "1" ]]; then
    echo "TTS_GENTLE_GRIPPER_CONTROL must be 0 or 1; got: ${gentle_gripper_control}" >&2
    return 2
  fi
  if [[ "${direct_gripper_safety}" == "1" && "${gentle_gripper_control}" == "1" ]]; then
    echo "Enable only one of TTS_DIRECT_GRIPPER_SAFETY and TTS_GENTLE_GRIPPER_CONTROL." >&2
    return 2
  fi
  if [[ ! "${gripper_release_step}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "TTS_GRIPPER_RELEASE_STEP must be non-negative; got: ${gripper_release_step}" >&2
    return 2
  fi
  if [[ ! "${gripper_safety_min_position}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "TTS_GRIPPER_SAFETY_MIN_POSITION must be non-negative; got: ${gripper_safety_min_position}" >&2
    return 2
  fi
  local value_name value
  for value_name in \
    contact_force_delta \
    force_baseline_max_position \
    gripper_close_step \
    gripper_maintain_step \
    gripper_max_position \
    gripper_max_lead \
    max_force_rate \
    force_safety_limit \
    gentle_deadband \
    gentle_stop_margin; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "${value_name} must be non-negative; got: ${value}" >&2
      return 2
    fi
  done
  if [[ ! "${force_filter_window}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TTS_FORCE_FILTER_WINDOW must be a positive integer; got: ${force_filter_window}" >&2
    return 2
  fi

  tts_prepare_inference berry "${TTS_MIN_GPU_FREE_MIB:-12000}"
  if (( interactive )); then
    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "Checkpoint not found: ${CKPT_PATH}" >&2
      return 1
    fi
  else
    berry_require_checkpoint
  fi
  if (( interactive )); then
    local checkpoint_display confirmation
    checkpoint_display="$(berry_display_checkpoint_path "${CKPT_PATH}")"
    printf '\nResolved TTS configuration\n'
    printf '  checkpoint:       %s\n' "${checkpoint_display}"
    printf '  device:           %s\n' "${DEVICE}"
    printf '  desired rise:     %s N above episode baseline\n' "${desired_force}"
    printf '  activation force: %s N\n' "${activation_force}"
    printf '  aggregation:      %s\n' "${aggregation}"
    printf '  candidates:       %s\n' "${sampling_candidates}"
    printf '  score steps:      %s\n' "${score_steps}"
    printf '  selection scope:  %s\n\n' "${selection_scope}"
    printf '  direct safety:    %s\n' "${direct_gripper_safety}"
    printf '  force controller: %s\n' "${gentle_gripper_control}"
    printf '  contact-force rise: %s N\n' "${contact_force_delta}"
    printf '  target deadband:  +/- %s N\n' "${gentle_deadband}"
    printf '  stop margin:      %s N\n' "${gentle_stop_margin}"
    printf '  force filter:     %s frames\n' "${force_filter_window}"
    printf '  max force rate:   %s N/s\n' "${max_force_rate}"
    printf '  release step:     %s\n\n' "${gripper_release_step}"
    printf '  safety min pos:   %s\n\n' "${gripper_safety_min_position}"
    printf '  close step:       %s\n' "${gripper_close_step}"
    printf '  max position:     %s\n\n' "${gripper_max_position}"
    printf '  max command lead: %s\n\n' "${gripper_max_lead}"
    read -r -p 'Start inference? [Y/n]: ' confirmation
    case "${confirmation:-y}" in
      y|Y|yes|YES) ;;
      n|N|no|NO) echo 'Cancelled.'; return 0 ;;
      *) echo "Please answer y or n." >&2; return 2 ;;
    esac
  fi
  berry_common_monitor_args
  local -a safety_args=()
  if [[ "${direct_gripper_safety}" == "1" ]]; then
    safety_args=(
      --tts-absolute-gripper-safety
      --tts-gripper-release-step "${gripper_release_step}"
      --tts-gripper-safety-min-position "${gripper_safety_min_position}"
    )
  elif [[ "${gentle_gripper_control}" == "1" ]]; then
    safety_args=(
      --tts-gentle-gripper-control
      --tts-add-gripper-fallback-candidates
      --tts-gripper-release-step "${gripper_release_step}"
      --tts-gripper-safety-min-position "${gripper_safety_min_position}"
      --tts-contact-force-delta "${contact_force_delta}"
      --tts-force-filter-window "${force_filter_window}"
      --tts-force-baseline-max-position "${force_baseline_max_position}"
      --tts-gripper-close-step "${gripper_close_step}"
      --tts-gripper-maintain-step "${gripper_maintain_step}"
      --tts-gripper-max-position "${gripper_max_position}"
      --tts-gripper-max-lead "${gripper_max_lead}"
      --tts-max-force-rate "${max_force_rate}"
      --tts-force-safety-limit "${force_safety_limit}"
      --tts-unsafe-fallback min_close
      --tts-deadband "${gentle_deadband}"
      --tts-stop-margin "${gentle_stop_margin}"
    )
  fi
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" policy_server.py \
    "${COMMON_ARGS[@]}" \
    --tts-experiment-profile berry \
    --tts-desired-force "${desired_force}" \
    --tts-force-target-mode baseline_delta \
    --tts-close-positive \
    --tts-steering-mode sample \
    --tts-selection-scope "${selection_scope}" \
    --tts-activation-force "${activation_force}" \
    --tts-sampling-candidates "${sampling_candidates}" \
    --tts-sampling-score-steps "${score_steps}" \
    --tts-policy-force-output \
    --tts-policy-force-aggregation "${aggregation}" \
    --tts-log-candidates \
    --tts-rollout-dir "${rollout_dir}" \
    "${safety_args[@]}" \
    "${extra_args[@]}"
}

berry_run_baseline() {
  local rollout_dir="${ROLLOUT_DIR:-${VIEWFORCE_ROOT}/rollouts/berry_force_output_baseline}"
  tts_prepare_inference berry-baseline "${TTS_MIN_GPU_FREE_MIB:-12000}"
  berry_require_checkpoint
  berry_common_monitor_args
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" policy_server.py \
    "${COMMON_ARGS[@]}" \
    --tts-desired-force "${DESIRED_FORCE:-8.0}" \
    --tts-steering-mode monitor \
    --tts-rollout-dir "${rollout_dir}" \
    "$@"
}
