"""Attended capture and bounded replay of an ARX paper-cup demo pose.

This server uses the same robot-side policy protocol and iPhone handoff as the
cup force experiment, but it does not load VisualForce or a robot policy. Capture
mode records a stationary Cartesian pose. Goto mode approaches that pose in
small Cartesian increments while holding the initial gripper command.

This is pose interpolation, not collision-free motion planning. Goto therefore
refuses starts that are too far from the saved pose and requires an operator at
the normal stop path.
"""

import argparse
import json
import signal
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from diffusion_policy.real_world.cup_common import (
    cartesian_pose_error,
    finite_scalar,
    finite_vector,
    is_stationary,
    measured_hold_action,
    normalized_quaternion,
)
from policy_server import PolicyServer


POSE_FORMAT_VERSION = 1
POSE_FRAME = 'arx_world'


@dataclass(frozen=True)
class CupPoseCaptureConfig:
    samples: int = 10
    stationary_velocity_limit: float = 0.05

    def __post_init__(self) -> None:
        if (
            isinstance(self.samples, bool)
            or not isinstance(self.samples, (int, np.integer))
            or self.samples <= 0
        ):
            raise ValueError('samples must be a positive integer')
        if (
            not np.isfinite(self.stationary_velocity_limit)
            or self.stationary_velocity_limit <= 0.0
        ):
            raise ValueError('stationary_velocity_limit must be finite and positive')


@dataclass(frozen=True)
class CupPoseGotoConfig:
    max_translation_step_m: float = 0.005
    max_rotation_step_degrees: float = 2.0
    max_start_distance_m: float = 0.10
    max_start_rotation_degrees: float = 20.0
    tracking_translation_limit_m: float = 0.02
    tracking_rotation_limit_degrees: float = 8.0
    goal_translation_tolerance_m: float = 0.003
    goal_rotation_tolerance_degrees: float = 1.0
    settle_samples: int = 5
    stationary_velocity_limit: float = 0.05

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name == 'settle_samples':
                continue
            value = float(getattr(self, field.name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{field.name} must be finite and positive')
        if (
            isinstance(self.settle_samples, bool)
            or not isinstance(self.settle_samples, (int, np.integer))
            or self.settle_samples <= 0
        ):
            raise ValueError('settle_samples must be a positive integer')


def quaternion_step(current: Any, target: Any, max_angle: float) -> np.ndarray:
    """Return a shortest-path quaternion step no larger than ``max_angle``."""

    current_q = normalized_quaternion(current, 'current quaternion')
    target_q = normalized_quaternion(target, 'target quaternion')
    dot = float(np.dot(current_q, target_q))
    if dot < 0.0:
        target_q = -target_q
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    physical_angle = 2.0 * float(np.arccos(dot))
    if physical_angle <= max_angle:
        result = target_q
    else:
        fraction = float(max_angle) / physical_angle
        omega = float(np.arccos(dot))
        if abs(omega) < 1e-8:
            result = (1.0 - fraction) * current_q + fraction * target_q
        else:
            result = (
                np.sin((1.0 - fraction) * omega) / np.sin(omega) * current_q
                + np.sin(fraction * omega) / np.sin(omega) * target_q
            )
    result = result / np.linalg.norm(result)
    if result[3] < 0.0:
        result = -result
    return result


def save_pose(path: Path, position: np.ndarray, quaternion: np.ndarray, grip: float) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    position = finite_vector(position, 3, 'arm_pos')
    quaternion = normalized_quaternion(quaternion)
    grip = finite_scalar(grip, 'gripper_pos_at_capture')
    payload = {
        'format_version': POSE_FORMAT_VERSION,
        'frame': POSE_FRAME,
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'arm_pos': position.tolist(),
        'arm_quat_xyzw': quaternion.tolist(),
        'gripper_pos_at_capture': grip,
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def load_pose(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    path = path.expanduser().resolve()
    with path.open() as file:
        payload = json.load(file)
    if payload.get('format_version') != POSE_FORMAT_VERSION:
        raise ValueError(
            f'unsupported pose format {payload.get("format_version")!r} in {path}'
        )
    if payload.get('frame') != POSE_FRAME:
        raise ValueError(f'pose frame must be {POSE_FRAME} in {path}')
    position = finite_vector(payload.get('arm_pos'), 3, 'saved arm_pos')
    quaternion = normalized_quaternion(
        payload.get('arm_quat_xyzw'), 'saved arm_quat_xyzw'
    )
    return position, quaternion, payload


class CupPoseCapturePolicy:
    """Capture the median of stationary pose samples and otherwise hold."""

    def __init__(
        self,
        pose_file: str,
        *,
        config: Optional[CupPoseCaptureConfig] = None,
        overwrite: bool = False,
    ):
        config = config or CupPoseCaptureConfig()
        self.pose_file = Path(pose_file).expanduser().resolve()
        self.config = config
        if self.pose_file.exists() and not overwrite:
            raise FileExistsError(
                f'pose file already exists: {self.pose_file}; use pose replace '
                'only when intentionally replacing it'
            )
        self._has_written = False
        self.reset()

    def reset(self) -> None:
        if self._has_written:
            print(f'CUP POSE already captured at {self.pose_file}')
            return
        self.positions = []
        self.quaternions = []
        self.grips = []
        self.last_reported_count = -1
        print('CUP POSE CAPTURE — hold the arm still at the desired demo pose.')

    def step(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        action = measured_hold_action(obs, normalize_orientation=True)
        if self._has_written:
            return action
        stationary = is_stationary(
            obs.get('arm_joint_velocity', []),
            self.config.stationary_velocity_limit,
        )
        if not stationary:
            if self.positions:
                for samples in (self.positions, self.quaternions, self.grips):
                    samples.clear()
            if self.last_reported_count != 0:
                print('CUP POSE CAPTURE waiting for the arm to become stationary')
                self.last_reported_count = 0
            return action

        quaternion = action['arm_quat']
        if self.quaternions and np.dot(self.quaternions[0], quaternion) < 0.0:
            quaternion = -quaternion
        self.positions.append(action['arm_pos'])
        self.quaternions.append(quaternion)
        self.grips.append(float(action['gripper_pos'][0]))
        count = len(self.positions)
        if count != self.last_reported_count:
            print(f'CUP POSE CAPTURE sample {count}/{self.config.samples}')
            self.last_reported_count = count
        if count >= self.config.samples:
            position = np.median(np.stack(self.positions), axis=0)
            quaternion = np.mean(np.stack(self.quaternions), axis=0)
            quaternion = normalized_quaternion(quaternion, 'captured quaternion')
            grip = float(np.median(np.asarray(self.grips)))
            save_pose(self.pose_file, position, quaternion, grip)
            self._has_written = True
            print(f'CUP POSE CAPTURE COMPLETE — saved {self.pose_file}')
        return action


class CupPoseGotoPolicy:
    """Approach a saved pose through bounded steps and tracking checks."""

    def __init__(
        self,
        pose_file: str,
        *,
        config: Optional[CupPoseGotoConfig] = None,
    ):
        config = config or CupPoseGotoConfig()
        self.pose_file = Path(pose_file).expanduser().resolve()
        self.target_position, self.target_quaternion, _ = load_pose(self.pose_file)
        self.config = config
        self.max_rotation_step = np.deg2rad(config.max_rotation_step_degrees)
        self.max_start_rotation = np.deg2rad(config.max_start_rotation_degrees)
        self.tracking_rotation_limit = np.deg2rad(
            config.tracking_rotation_limit_degrees
        )
        self.goal_rotation_tolerance = np.deg2rad(
            config.goal_rotation_tolerance_degrees
        )
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.fault_reason = None
        self.last_command_position = None
        self.last_command_quaternion = None
        self.gripper_command = None
        self.settle_count = 0
        self.reached = False
        self.last_mode = None
        print(
            'CUP POSE GOTO — clear the workspace and keep an operator at the '
            'stop path before inference handoff.'
        )

    def _fault(
        self,
        measured: Dict[str, np.ndarray],
        reason: str,
    ) -> Dict[str, np.ndarray]:
        if self.fault_reason is None:
            self.fault_reason = reason
            print(f'CUP POSE FAULT — {reason}; returning measured-pose hold.')
        return measured

    def _report(self, mode: str, distance: float, angle: float) -> None:
        if mode == self.last_mode and mode != 'moving':
            return
        self.last_mode = mode
        print(
            f'CUP POSE mode={mode} distance={distance:.4f} m '
            f'rotation={np.rad2deg(angle):.2f} deg'
        )

    def _target_error(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> Tuple[float, float]:
        return cartesian_pose_error(
            position,
            quaternion,
            self.target_position,
            self.target_quaternion,
        )

    def step(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        measured = measured_hold_action(obs, normalize_orientation=True)
        position = measured['arm_pos']
        quaternion = measured['arm_quat']
        if self.fault_reason is not None:
            return measured

        if not self.initialized:
            distance, angle = self._target_error(position, quaternion)
            if distance > self.config.max_start_distance_m:
                return self._fault(
                    measured,
                    f'start is {distance:.3f} m from saved pose '
                    f'(limit {self.config.max_start_distance_m:.3f} m)',
                )
            if angle > self.max_start_rotation:
                return self._fault(
                    measured,
                    f'start is {np.rad2deg(angle):.1f} deg from saved pose '
                    f'(limit {np.rad2deg(self.max_start_rotation):.1f} deg)',
                )
            self.initialized = True
            self.last_command_position = position.copy()
            self.last_command_quaternion = quaternion.copy()
            self.gripper_command = measured['gripper_pos'].copy()
            self._report('initialized_hold', distance, angle)
            return measured

        tracking_distance, tracking_angle = cartesian_pose_error(
            position,
            quaternion,
            self.last_command_position,
            self.last_command_quaternion,
        )
        if tracking_distance > self.config.tracking_translation_limit_m:
            return self._fault(
                measured,
                f'position tracking error {tracking_distance:.3f} m exceeds '
                f'{self.config.tracking_translation_limit_m:.3f} m',
            )
        if tracking_angle > self.tracking_rotation_limit:
            return self._fault(
                measured,
                f'rotation tracking error {np.rad2deg(tracking_angle):.1f} deg '
                f'exceeds {np.rad2deg(self.tracking_rotation_limit):.1f} deg',
        )

        delta = self.target_position - position
        distance, angle = self._target_error(position, quaternion)
        at_goal = (
            distance <= self.config.goal_translation_tolerance_m
            and angle <= self.goal_rotation_tolerance
        )
        stationary = is_stationary(
            obs.get('arm_joint_velocity', []),
            self.config.stationary_velocity_limit,
        )
        if at_goal:
            self.settle_count = self.settle_count + 1 if stationary else 0
            self.last_command_position = self.target_position.copy()
            self.last_command_quaternion = self.target_quaternion.copy()
            if self.settle_count >= self.config.settle_samples and not self.reached:
                self.reached = True
                print(
                    'CUP SAVED POSE REACHED — return to phone control before '
                    'placing or grasping the cup.'
                )
            self._report('reached_hold' if self.reached else 'settling', distance, angle)
        else:
            if distance <= self.config.max_translation_step_m:
                next_position = self.target_position.copy()
            else:
                next_position = (
                    position
                    + delta / distance * self.config.max_translation_step_m
                )
            next_quaternion = quaternion_step(
                quaternion,
                self.target_quaternion,
                self.max_rotation_step,
            )
            self.last_command_position = next_position
            self.last_command_quaternion = next_quaternion
            self._report('moving', distance, angle)

        return {
            'arm_pos': self.last_command_position.copy(),
            'arm_quat': self.last_command_quaternion.copy(),
            'gripper_pos': self.gripper_command.copy(),
        }


def _config_from_args(args: argparse.Namespace, config_type):
    values = {
        field.name: getattr(args, field.name)
        for field in fields(config_type)
    }
    return config_type(**values)


def build_parser() -> argparse.ArgumentParser:
    capture_defaults = CupPoseCaptureConfig()
    goto_defaults = CupPoseGotoConfig()
    parser = argparse.ArgumentParser(
        description='Capture or approach an attended ARX paper-cup demo pose.'
    )
    parser.add_argument('--mode', choices=('capture', 'goto', 'show'), required=True)
    parser.add_argument('--pose-file', required=True)
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--capture-samples',
        dest='samples',
        type=int,
        default=capture_defaults.samples,
    )
    parser.add_argument(
        '--stationary-velocity-limit',
        type=float,
        default=capture_defaults.stationary_velocity_limit,
    )
    for field in fields(CupPoseGotoConfig):
        if field.name == 'stationary_velocity_limit':
            continue
        default = getattr(goto_defaults, field.name)
        parser.add_argument(
            f'--{field.name.replace("_", "-")}',
            type=type(default),
            default=default,
        )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pose_file = Path(args.pose_file).expanduser().resolve()
    if args.mode == 'show':
        position, quaternion, payload = load_pose(pose_file)
        print(f'Pose file: {pose_file}')
        print(f'Captured: {payload.get("captured_at_utc", "unknown")}')
        print(f'arm_pos: {position.tolist()}')
        print(f'arm_quat_xyzw: {quaternion.tolist()}')
        print(f'gripper_pos_at_capture: {payload.get("gripper_pos_at_capture", "")}')
        return
    if args.mode == 'capture':
        policy = CupPoseCapturePolicy(
            str(pose_file),
            config=_config_from_args(args, CupPoseCaptureConfig),
            overwrite=args.overwrite,
        )
    else:
        policy = CupPoseGotoPolicy(
            str(pose_file),
            config=_config_from_args(args, CupPoseGotoConfig),
        )

    def exit_on_signal(signum, _frame):
        print(f'Received signal {signum}; closing cup pose server')
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    print(f'Cup pose server mode: {args.mode.upper()}; no learned model is loaded.')
    PolicyServer(policy, port=args.port).run()


if __name__ == '__main__':
    main()
