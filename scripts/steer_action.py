"""
Offline demo for VisualForce test-time action steering.

This script is intentionally policy-agnostic: pass in a base-policy action vector
and it returns the force-steered action vector.  In the real robot loop, call
src.steering.VisualForceSteeringPipeline directly instead of shelling out.

Example:
    python scripts/steer_action.py \
        --checkpoint checkpoints/run1/best.pt \
        --frame frame.png \
        --mask mask.png \
        --action 0.01 0.0 0.0 0.20 \
        --desired-force 2.0 \
        --gripper-index 3 \
        --close-positive
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.steering import ForceSteeringConfig, VisualForceSteeringPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Steer a base-policy action with VisualForce")
    parser.add_argument("--checkpoint", required=True, help="VisualForce checkpoint path")
    parser.add_argument("--frame", required=True, help="RGB frame image path")
    parser.add_argument("--mask", required=True, help="SAM/SAM2 gripper mask image path")
    parser.add_argument("--action", nargs="+", type=float, required=True, help="Base action vector")
    parser.add_argument("--desired-force", type=float, required=True, help="Desired force in N")
    parser.add_argument("--force-key", default="Fz", help="Force channel used for steering")
    parser.add_argument(
        "--force-mode",
        choices=["magnitude", "signed"],
        default="magnitude",
        help="Use absolute force magnitude or signed channel value",
    )
    parser.add_argument("--gripper-index", type=int, default=None, help="Action index for gripper command")
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--close-positive", action="store_true", help="Positive gripper command closes")
    direction.add_argument("--close-negative", action="store_true", help="Negative gripper command closes")
    parser.add_argument(
        "--motion-indices",
        nargs="*",
        type=int,
        default=[],
        help="Optional action indices to scale down when force is above target",
    )
    parser.add_argument("--deadband", type=float, default=0.10)
    parser.add_argument("--slowdown-band", type=float, default=1.0)
    parser.add_argument("--stop-margin", type=float, default=0.75)
    parser.add_argument("--open-command", type=float, default=0.0)
    parser.add_argument("--close-gain", type=float, default=0.0)
    parser.add_argument("--max-close-command", type=float, default=None)
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda:0 or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    close_positive = not args.close_negative

    config = ForceSteeringConfig(
        desired_force=args.desired_force,
        force_key=args.force_key,
        force_mode=args.force_mode,
        deadband=args.deadband,
        slowdown_band=args.slowdown_band,
        stop_margin=args.stop_margin,
        gripper_index=args.gripper_index,
        close_positive=close_positive,
        open_command=args.open_command,
        close_gain=args.close_gain,
        max_close_command=args.max_close_command,
        motion_indices=tuple(args.motion_indices),
    )
    pipeline = VisualForceSteeringPipeline(args.checkpoint, config, device=args.device)

    from PIL import Image

    frame = Image.open(args.frame)
    mask = Image.open(args.mask)
    result = pipeline.steer_action(frame, mask, args.action)

    print(json.dumps({
        "base_action": result.base_action.tolist(),
        "steered_action": result.action.tolist(),
        "predicted_force": result.predicted_force.values,
        "selected_force": result.predicted_force.selected_force,
        "control_force": result.predicted_force.control_force,
        "force_key": result.predicted_force.selected_key,
        "force_error": result.force_error,
        "close_scale": result.close_scale,
        "motion_scale": result.motion_scale,
        "metadata": result.metadata,
    }, indent=2))


if __name__ == "__main__":
    main()
