#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PREVIEW_DIR="${PREVIEW_DIR:-reports/flip_obj4_stage2_sam2_mask_preview}" \
  exec "${SCRIPT_DIR}/train_flip_force_critic.sh" stage2 label
