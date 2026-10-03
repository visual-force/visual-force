#!/usr/bin/env bash

# Shared helpers for task launchers. This file is sourced by scripts/* and is
# not intended to be executed directly.

viewforce_init_paths() {
  # The estimator is an external artifact; operators must provide its path.
  VIEWFORCE_CKPT="${VIEWFORCE_CKPT:-}"
  SAM2_ROOT="${SAM2_ROOT:-${VIEWFORCE_ROOT}/third_party/sam2}"
  SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_small.pt}"
}

viewforce_select_gpu() {
  local min_free_mib="${1:-12000}"
  local workload_name="${2:-inference}"
  local query_output
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU preflight failed: nvidia-smi is not available." >&2
    return 1
  fi
  if ! query_output="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
  )"; then
    echo "GPU preflight failed: unable to query NVIDIA GPUs." >&2
    return 1
  fi

  local requested_index=""
  if [[ -n "${DEVICE:-}" ]]; then
    if [[ "${DEVICE}" =~ ^cuda:([0-9]+)$ ]]; then
      requested_index="${BASH_REMATCH[1]}"
    else
      echo "GPU preflight failed: DEVICE must look like cuda:0, got ${DEVICE}." >&2
      return 1
    fi
  fi

  local line index free_mib
  local selected_index=""
  local selected_free=-1
  while IFS= read -r line; do
    line="${line//[[:space:]]/}"
    IFS=',' read -r index free_mib <<<"${line}"
    [[ "${index}" =~ ^[0-9]+$ && "${free_mib}" =~ ^[0-9]+$ ]] || continue
    if [[ -n "${requested_index}" ]]; then
      if [[ "${index}" == "${requested_index}" ]]; then
        selected_index="${index}"
        selected_free="${free_mib}"
        break
      fi
    elif (( free_mib > selected_free )); then
      selected_index="${index}"
      selected_free="${free_mib}"
    fi
  done <<<"${query_output}"

  if [[ -z "${selected_index}" ]]; then
    echo "GPU preflight failed: requested GPU ${DEVICE:-auto} is not available." >&2
    return 1
  fi
  if (( selected_free < min_free_mib )); then
    echo "GPU preflight failed: cuda:${selected_index} has ${selected_free} MiB free; ${workload_name} inference requires at least ${min_free_mib} MiB." >&2
    if [[ -n "${requested_index}" ]]; then
      echo "Unset DEVICE to select the GPU with the most free memory." >&2
    fi
    return 1
  fi

  DEVICE="cuda:${selected_index}"
  if [[ -n "${requested_index}" ]]; then
    echo "GPU preflight: using requested ${DEVICE} (${selected_free} MiB free)." >&2
  else
    echo "GPU preflight: selected ${DEVICE} (${selected_free} MiB free)." >&2
  fi
}

normalize_checkpoint_path() {
  if [[ -n "${CKPT_PATH:-}" && "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${REPO_ROOT}/${CKPT_PATH}"
  fi
}

resolve_latest_checkpoint() {
  local checkpoint_pattern="$1"
  if [[ -z "${CKPT_PATH:-}" && -d "${REPO_ROOT}/outputs" ]]; then
    CKPT_PATH="$(
      find "${REPO_ROOT}/outputs" -type f \
        -path "${checkpoint_pattern}" \
        -printf '%T@ %p\n' \
        | sort -nr \
        | sed -n '1s/^[^ ]* //p'
    )"
  fi
  normalize_checkpoint_path
  [[ -n "${CKPT_PATH:-}" && -f "${CKPT_PATH}" ]]
}

repo_relative_path() {
  local path="$1"
  if [[ "${path}" == "${REPO_ROOT}/"* ]]; then
    printf '%s' "${path#${REPO_ROOT}/}"
  else
    printf '%s' "${path}"
  fi
}

build_viewforce_common_args() {
  if [[ -z "${VIEWFORCE_CKPT}" ]]; then
    echo "VIEWFORCE_CKPT must point to the external force-estimator checkpoint." >&2
    return 2
  fi
  COMMON_ARGS=(
    --device "${DEVICE}"
    --ckpt-path "${CKPT_PATH}"
    --tts-viewforce-ckpt "${VIEWFORCE_CKPT}"
    --tts-viewforce-root "${VIEWFORCE_ROOT}"
  )
  if declare -p VIEWFORCE_ARGS_AFTER_ROOT >/dev/null 2>&1; then
    COMMON_ARGS+=("${VIEWFORCE_ARGS_AFTER_ROOT[@]}")
  fi
  COMMON_ARGS+=(
    --tts-force-mode magnitude
    --tts-frame-key wrist_image
    --tts-side-frame-key base_image
  )
  if [[ -n "${VIEWFORCE_FRAME_COLOR_SPACE:-}" ]]; then
    COMMON_ARGS+=(--tts-frame-color-space "${VIEWFORCE_FRAME_COLOR_SPACE}")
  fi
  COMMON_ARGS+=(
    --tts-auto-mask
    --tts-mask-mode sam2
    --tts-sam2-model small
    --tts-sam2-repo "${SAM2_ROOT}"
    --tts-sam2-ckpt "${SAM2_CKPT}"
    --tts-gripper-index 7
  )
}
