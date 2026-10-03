#!/usr/bin/env bash

# Berry checkpoint discovery and interactive selection.

berry_discover_checkpoints() {
  local outputs_dir="${REPO_ROOT}/outputs"
  local -a complete_checkpoints=()
  local -a partial_checkpoints=()
  local checkpoint_path metadata
  if [[ ! -d "${outputs_dir}" ]]; then
    return 0
  fi
  while IFS= read -r checkpoint_path; do
    metadata="$(berry_checkpoint_metadata "${checkpoint_path}")"
    if [[ "${metadata}" == *', complete,'* ]]; then
      complete_checkpoints+=("${checkpoint_path}")
    else
      partial_checkpoints+=("${checkpoint_path}")
    fi
  done < <(
    find "${outputs_dir}" \
      -type f \
      -path '*train_diffusion_unet_berry_stage2_absolute_force_output_quantile*/checkpoints/latest.ckpt' \
      -printf '%T@ %p\n' \
      | sort -nr \
      | sed 's/^[^ ]* //'
  )
  if (( ${#complete_checkpoints[@]} + ${#partial_checkpoints[@]} > 0 )); then
    printf '%s\n' "${complete_checkpoints[@]}" "${partial_checkpoints[@]}"
  fi
}

berry_display_checkpoint_path() {
  repo_relative_path "$1"
}

berry_checkpoint_run_label() {
  local checkpoint_path="$1"
  local run_dir date_dir run_name date_name time_name
  run_dir="$(dirname "$(dirname "${checkpoint_path}")")"
  date_dir="$(dirname "${run_dir}")"
  run_name="$(basename "${run_dir}")"
  date_name="$(basename "${date_dir}")"
  time_name="${run_name:0:8}"
  printf '%s %s' "${date_name//./-}" "${time_name//./:}"
}

berry_checkpoint_metadata() {
  local checkpoint_path="$1"
  local run_dir logs_path config_path last_epoch num_epochs status modified
  run_dir="$(dirname "$(dirname "${checkpoint_path}")")"
  logs_path="${run_dir}/logs.json.txt"
  config_path="${run_dir}/.hydra/config.yaml"
  last_epoch="unknown"
  num_epochs=""
  status="unknown"

  if [[ -f "${logs_path}" ]]; then
    last_epoch="$(
      sed -n 's/.*"epoch": \([0-9][0-9]*\).*/\1/p' "${logs_path}" \
        | sort -n \
        | tail -1
    )"
    last_epoch="${last_epoch:-unknown}"
  fi
  if [[ -f "${config_path}" ]]; then
    num_epochs="$(sed -n 's/^  num_epochs: \([0-9][0-9]*\)$/\1/p' "${config_path}" | head -1)"
  fi
  if [[ "${last_epoch}" != "unknown" && -n "${num_epochs}" ]]; then
    if (( last_epoch >= num_epochs - 1 )); then
      status="complete"
    else
      status="partial"
    fi
  fi
  modified="$(date -r "${checkpoint_path}" '+%Y-%m-%d %H:%M:%S')"
  printf 'epoch %s, %s, %s' "${last_epoch}" "${status}" "${modified}"
}

berry_resolve_checkpoint() {
  if [[ -n "${CKPT_PATH}" ]]; then
    normalize_checkpoint_path
    return 0
  fi

  case "${CKPT_PRESET}" in
    latest) ;;
    recommended)
      echo "CKPT_PRESET=recommended is deprecated; using newest latest.ckpt." >&2
      ;;
    *)
      echo "Unknown CKPT_PRESET=${CKPT_PRESET}; use latest." >&2
      return 2
      ;;
  esac

  local -a checkpoints=()
  mapfile -t checkpoints < <(berry_discover_checkpoints)
  if (( ${#checkpoints[@]} == 0 )); then
    echo "No berry force-output latest.ckpt found under ${REPO_ROOT}/outputs." >&2
    return 1
  fi
  CKPT_PATH="${checkpoints[0]}"
}

berry_list_checkpoints() {
  local -a checkpoints=()
  mapfile -t checkpoints < <(berry_discover_checkpoints)
  if (( ${#checkpoints[@]} == 0 )); then
    echo "No berry force-output latest.ckpt found under ${REPO_ROOT}/outputs." >&2
    return 1
  fi

  printf 'Available berry force-output checkpoints (newest first):\n'
  local index path relative metadata label
  for index in "${!checkpoints[@]}"; do
    path="${checkpoints[index]}"
    relative="$(berry_display_checkpoint_path "${path}")"
    metadata="$(berry_checkpoint_metadata "${path}")"
    label="$(berry_checkpoint_run_label "${path}")"
    printf '  %d) %s  [%s]\n' "$((index + 1))" "${label}" "${metadata}"
    printf '     %s\n' "${relative}"
  done
}

berry_select_checkpoint_interactive() {
  local -a checkpoints=()
  mapfile -t checkpoints < <(berry_discover_checkpoints)
  if (( ${#checkpoints[@]} == 0 )); then
    echo "No berry force-output latest.ckpt found under ${REPO_ROOT}/outputs." >&2
    return 1
  fi

  printf 'Available berry force-output checkpoints (newest first):\n'
  local index path relative metadata label selection custom_path
  for index in "${!checkpoints[@]}"; do
    path="${checkpoints[index]}"
    relative="$(berry_display_checkpoint_path "${path}")"
    metadata="$(berry_checkpoint_metadata "${path}")"
    label="$(berry_checkpoint_run_label "${path}")"
    printf '  %d) %s  [%s]\n' "$((index + 1))" "${label}" "${metadata}"
    printf '     %s\n' "${relative}"
  done
  printf '  c) Custom checkpoint path\n'
  read -r -p 'Checkpoint [1]: ' selection
  selection="${selection:-1}"

  if [[ "${selection}" == "c" || "${selection}" == "custom" ]]; then
    read -r -p 'Custom checkpoint path: ' custom_path
    if [[ -z "${custom_path}" ]]; then
      echo "Custom checkpoint path cannot be empty." >&2
      return 2
    fi
    CKPT_PATH="${custom_path}"
    normalize_checkpoint_path
  elif [[ "${selection}" =~ ^[1-9][0-9]*$ ]] \
      && (( selection <= ${#checkpoints[@]} )); then
    CKPT_PATH="${checkpoints[selection - 1]}"
  else
    echo "Invalid checkpoint selection: ${selection}" >&2
    return 2
  fi
}

berry_require_checkpoint() {
  berry_resolve_checkpoint
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
  fi
}
