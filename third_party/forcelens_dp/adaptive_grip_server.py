"""No-policy VisualForce server for the ARX paper-cup pouring demo.

The ARX NUC remains responsible for iPhone teleoperation and the attended
handoff.  After handoff this server latches the measured Cartesian pose and the
operator's empty-cup grip, estimates added payload from ARX joint torques, and
schedules only a bounded gripper position command. VisualForce independently
limits excessive grip force. It never loads or samples a robot policy.
"""

import argparse
import csv
import queue
import signal
import sys
import threading
import time
import traceback
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2 as cv
import numpy as np

from diffusion_policy.common.video_encoding import (
    VideoEncodingError,
    encode_h264,
    h264_encoder_available,
)
from diffusion_policy.real_world.cup_adaptive_grip import (
    CupAdaptiveGripConfig,
    CupAdaptiveGripController,
)
from diffusion_policy.real_world.cup_common import (
    GRAVITY_M_S2,
    cartesian_pose_error,
    finite_float,
    finite_scalar,
    finite_vector,
    is_stationary,
    measured_hold_action,
)
from policy_server import (
    PolicyServer,
    Sam2FrameMasker,
    make_hsv_gripper_mask,
)


_CONTROLLER_CLI_FIELDS = tuple(
    field
    for field in fields(CupAdaptiveGripConfig)
    if field.name != 'min_torque_basis_norm'
)


class CupRolloutRecorder:
    """Record an annotated demo video and controller telemetry."""

    METADATA_FIELDS = (
        'mode',
        'reason',
        'calibrated',
        'stationary',
        'measurement_valid',
        'load_valid',
        'mask_fraction',
        'mask_reject_count',
        'raw_force_n',
        'filtered_force_n',
        'baseline_force_n',
        'raw_added_load_n',
        'filtered_added_load_n',
        'effective_added_load_n',
        'scheduled_closure',
        'scheduled_closure_increment',
        'force_limit_n',
        'emergency_force_n',
        'measured_closure',
        'command_closure',
        'gripper_torque',
        'closure_limited',
        'emergency_latched',
        'pose_error_m',
        'rotation_error_deg',
    )
    OBS_VECTOR_FIELDS = {
        'arm_pos': 'arm_pos',
        'arm_quat': 'arm_quat',
        'joint_torque': 'arm_joint_torque',
        'payload_torque_basis': 'arm_payload_torque_basis',
    }
    METADATA_VECTOR_FIELDS = {
        'target_arm_pos': 'target_arm_pos',
        'target_arm_quat': 'target_arm_quat',
    }
    CSV_FIELDS = (
        'frame_idx',
        'time_s',
        *METADATA_FIELDS,
        'water_equivalent_g',
        *OBS_VECTOR_FIELDS,
        *METADATA_VECTOR_FIELDS,
    )

    def __init__(self, output_root: Optional[str], fps: float = 10.0):
        self.output_root = None if not output_root else Path(output_root)
        self.fps = float(fps)
        self.output_dir = None
        self.started_at = None
        self.frame_idx = 0
        self.writers = {}
        self.csv_file = None
        self.csv_writer = None
        self._closed = False

    @staticmethod
    def _put_lines(image_bgr: np.ndarray, lines) -> np.ndarray:
        result = image_bgr.copy()
        y = 28
        for line in lines:
            cv.putText(
                result,
                str(line),
                (12, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                result,
                str(line),
                (12, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.62,
                (20, 20, 20),
                1,
                cv.LINE_AA,
            )
            y += 27
        return result

    def _start(self, wrist_rgb: np.ndarray, demo_rgb: np.ndarray) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output_dir = self.output_root / f'cup_adaptive_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        wrist_size = (wrist_rgb.shape[1], wrist_rgb.shape[0])
        demo_size = (demo_rgb.shape[1], demo_rgb.shape[0])
        self.writers = {
            stem: cv.VideoWriter(
                str(self.output_dir / f'.{stem}_recording.mp4'),
                fourcc,
                self.fps,
                size,
            )
            for stem, size in (
                ('wrist', wrist_size),
                ('mask', wrist_size),
                ('demo', demo_size),
            )
        }
        self.wrist_size = wrist_size
        self.demo_size = demo_size
        self.csv_file = (self.output_dir / 'adaptive_grip.csv').open('w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.CSV_FIELDS)
        self.csv_writer.writeheader()
        self.started_at = time.monotonic()
        print(f'Cup rollout recording: {self.output_dir}')

    @staticmethod
    def _vector_text(value: Any) -> str:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        return ';'.join(f'{item:.8g}' for item in vector)

    def _telemetry_row(
        self,
        metadata: Dict[str, Any],
        obs: Dict[str, Any],
        water_equivalent_g: float,
    ) -> Dict[str, Any]:
        row = {name: metadata.get(name, '') for name in self.METADATA_FIELDS}
        row.update(
            {
                'frame_idx': self.frame_idx,
                'time_s': time.monotonic() - self.started_at,
                'water_equivalent_g': water_equivalent_g,
                **{
                    column: self._vector_text(obs.get(source, []))
                    for column, source in self.OBS_VECTOR_FIELDS.items()
                },
                **{
                    column: self._vector_text(metadata.get(source, []))
                    for column, source in self.METADATA_VECTOR_FIELDS.items()
                },
            }
        )
        row['measured_closure'] = finite_float(
            metadata.get('measured_closure')
        )
        row['command_closure'] = finite_float(metadata.get('command_closure'))
        return row

    def record(
        self,
        *,
        wrist_rgb: np.ndarray,
        mask: np.ndarray,
        side_rgb: Optional[np.ndarray],
        metadata: Dict[str, Any],
        obs: Dict[str, Any],
    ) -> None:
        if self.output_root is None or self._closed:
            return
        wrist_rgb = np.asarray(wrist_rgb, dtype=np.uint8)
        if wrist_rgb.ndim != 3 or wrist_rgb.shape[2] != 3:
            return
        demo_rgb = wrist_rgb if side_rgb is None else np.asarray(side_rgb, dtype=np.uint8)
        if demo_rgb.ndim != 3 or demo_rgb.shape[2] != 3:
            demo_rgb = wrist_rgb
        if self.output_dir is None:
            self._start(wrist_rgb, demo_rgb)

        wrist_bgr = cv.cvtColor(wrist_rgb, cv.COLOR_RGB2BGR)
        if (wrist_bgr.shape[1], wrist_bgr.shape[0]) != self.wrist_size:
            wrist_bgr = cv.resize(wrist_bgr, self.wrist_size)
        mask_bool = np.asarray(mask).astype(bool)
        mask_u8 = mask_bool.astype(np.uint8) * 255
        if mask_u8.shape != wrist_bgr.shape[:2]:
            mask_u8 = cv.resize(
                mask_u8,
                self.wrist_size,
                interpolation=cv.INTER_NEAREST,
            )
            mask_bool = mask_u8.astype(bool)
        mask_overlay = wrist_bgr.copy()
        mask_overlay[mask_bool] = (
            0.45 * mask_overlay[mask_bool]
            + 0.55 * np.array([0, 220, 0])
        ).astype(np.uint8)

        demo_bgr = cv.cvtColor(demo_rgb, cv.COLOR_RGB2BGR)
        if (demo_bgr.shape[1], demo_bgr.shape[0]) != self.demo_size:
            demo_bgr = cv.resize(demo_bgr, self.demo_size)

        load_n = finite_float(metadata.get('filtered_added_load_n'))
        force_n = finite_float(metadata.get('filtered_force_n'))
        force_limit_n = finite_float(metadata.get('force_limit_n'))
        measured_grip = finite_float(metadata.get('measured_closure'))
        command_grip = finite_float(metadata.get('command_closure'))
        scheduled_grip = finite_float(metadata.get('scheduled_closure'))
        water_g = max(0.0, load_n) * 1000.0 / GRAVITY_M_S2
        lines = (
            f"added load: {load_n:+.2f} N  (~{water_g:.0f} g water)",
            f"VisualForce: {force_n:.2f} / limit {force_limit_n:.2f} N",
            (
                f"grip: {measured_grip:.3f} -> {command_grip:.3f} "
                f"(scheduled {scheduled_grip:.3f})"
            ),
            f"mode: {metadata.get('mode', '')}",
        )
        demo_bgr = self._put_lines(demo_bgr, lines)

        self.writers['wrist'].write(wrist_bgr)
        self.writers['mask'].write(mask_overlay)
        self.writers['demo'].write(demo_bgr)

        self.csv_writer.writerow(self._telemetry_row(metadata, obs, water_g))
        self.csv_file.flush()
        self.frame_idx += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.output_dir is None:
            return
        for writer in self.writers.values():
            writer.release()
        if self.csv_file is not None:
            self.csv_file.close()
        print(f'Cup adaptive CSV saved: {self.output_dir / "adaptive_grip.csv"}')
        if not h264_encoder_available():
            print('ffmpeg not found; leaving mp4v cup rollout videos in place')
            return
        for stem in self.writers:
            source = self.output_dir / f'.{stem}_recording.mp4'
            target = self.output_dir / f'{stem}_h264.mp4'
            if not source.exists():
                continue
            try:
                encode_h264(source, target)
                print(f'Cup H.264 video saved: {target}')
            except VideoEncodingError as exc:
                print(f'ffmpeg failed while writing {target.name}:')
                print(exc)


class CupAdaptiveGripPolicy:
    """Latch an ARX pose and run the pure cup controller from fresh telemetry."""

    def __init__(
        self,
        *,
        visualforce_checkpoint: str,
        visualforce_root: str,
        device: str,
        controller_config: CupAdaptiveGripConfig,
        frame_key: str = 'wrist_image',
        side_frame_key: Optional[str] = 'base_image',
        mask_mode: str = 'sam2',
        sam2_model: str = 'small',
        sam2_repo: Optional[str] = None,
        sam2_checkpoint: Optional[str] = None,
        stationary_velocity_limit: float = 0.05,
        min_mask_fraction: float = 0.01,
        max_mask_fraction: float = 0.40,
        max_pose_error_m: float = 0.03,
        max_rotation_error_degrees: float = 8.0,
        rollout_dir: Optional[str] = None,
        rollout_fps: float = 10.0,
    ):
        root = Path(visualforce_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'VisualForce root does not exist: {root}')
        sys.path.insert(0, str(root))
        from src.steering import VisualForceEstimator

        self.estimator = VisualForceEstimator(visualforce_checkpoint, device=device)
        self.controller = CupAdaptiveGripController(controller_config)
        self.frame_key = frame_key
        self.side_frame_key = side_frame_key
        if mask_mode == 'sam2':
            self.masker = Sam2FrameMasker(
                root,
                sam2_model,
                sam2_repo=sam2_repo,
                sam2_ckpt=sam2_checkpoint,
            )
        elif mask_mode == 'hsv':
            self.masker = None
        else:
            raise ValueError("mask_mode must be 'sam2' or 'hsv'")
        self.stationary_velocity_limit = float(stationary_velocity_limit)
        self.min_mask_fraction = float(min_mask_fraction)
        self.max_mask_fraction = float(max_mask_fraction)
        self.max_pose_error_m = float(max_pose_error_m)
        self.max_rotation_error = np.deg2rad(max_rotation_error_degrees)
        self.rollout_root = rollout_dir
        self.rollout_fps = rollout_fps
        self.recorder = CupRolloutRecorder(rollout_dir, rollout_fps)
        self._clear_episode_state()

    def _clear_episode_state(self) -> None:
        self.target_position = None
        self.target_quaternion = None
        self.pose_fault_latched = False
        self.pose_fault_announced = False
        self.last_mode = None
        self.last_status_time = 0.0

    def _action(self, obs: Dict[str, Any], closure: float):
        if self.target_position is not None:
            position = self.target_position
            quaternion = self.target_quaternion
        else:
            position = finite_vector(obs['arm_pos'], 3, 'arm_pos')
            quaternion = finite_vector(obs['arm_quat'], 4, 'arm_quat')
        return {
            'arm_pos': np.asarray(position, dtype=np.float64).copy(),
            'arm_quat': np.asarray(quaternion, dtype=np.float64).copy(),
            'gripper_pos': np.array([float(closure)], dtype=np.float64),
        }

    def reset(self) -> None:
        self.recorder.close()
        self.recorder = CupRolloutRecorder(self.rollout_root, self.rollout_fps)
        self.controller.reset()
        if self.masker is not None:
            self.masker.reset()
        self._clear_episode_state()
        print(
            'Cup controller reset: manually hold the EMPTY cup, stop phone '
            'motion, then wait for CALIBRATION COMPLETE before adding water.'
        )

    def close(self) -> None:
        self.recorder.close()

    def _mask(self, frame: np.ndarray):
        if self.masker is None:
            mask = make_hsv_gripper_mask(frame)
            reject_count = 0
        else:
            mask = self.masker.predict(frame)
            reject_count = int(self.masker.reject_count)
        mask_fraction = float(np.asarray(mask).astype(bool).mean())
        valid = (
            self.min_mask_fraction <= mask_fraction <= self.max_mask_fraction
            and reject_count == 0
        )
        return mask, mask_fraction, reject_count, valid

    def _status(self, metadata: Dict[str, Any]) -> None:
        now = time.monotonic()
        mode = metadata.get('mode')
        if mode == self.last_mode and now - self.last_status_time < 1.0:
            return
        self.last_mode = mode
        self.last_status_time = now
        if mode == 'calibration_complete':
            print('CUP CALIBRATION COMPLETE — empty-cup baselines latched; ready.')
        print(
            'CUP '
            f'mode={mode} '
            f'load={metadata.get("filtered_added_load_n", "")} N '
            f'force={metadata.get("filtered_force_n", "")} N '
            f'limit={metadata.get("force_limit_n", "")} N '
            f'scheduled={metadata.get("scheduled_closure", "")} '
            f'grip={metadata.get("measured_closure", "")}->'
            f'{metadata.get("command_closure", "")}'
        )
        reason = metadata.get('reason')
        if reason:
            print(f'CUP HOLD reason={reason}')

    def step(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        try:
            return self._step(obs)
        except Exception:
            print('Cup adaptive inference failed; holding the latest safe command:')
            traceback.print_exc()
            if self.controller.command_closure is not None:
                return self._action(obs, self.controller.command_closure)
            return measured_hold_action(obs)

    def _step(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        required = ('arm_pos', 'arm_quat', 'gripper_pos', self.frame_key)
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f'cup observation is missing keys: {missing}')

        position = finite_vector(obs['arm_pos'], 3, 'arm_pos')
        quaternion = finite_vector(obs['arm_quat'], 4, 'arm_quat')
        if self.target_position is None:
            self.target_position = position.copy()
            self.target_quaternion = quaternion.copy()

        position_error, rotation_error = cartesian_pose_error(
            position,
            quaternion,
            self.target_position,
            self.target_quaternion,
        )
        if (
            position_error > self.max_pose_error_m
            or rotation_error > self.max_rotation_error
        ):
            self.pose_fault_latched = True
        if self.pose_fault_latched:
            if not self.pose_fault_announced:
                print(
                    'CUP POSE FAULT — switching to measured pose hold; use Return '
                    'to phone control before continuing.'
                )
                self.pose_fault_announced = True
            return measured_hold_action(obs)

        frame = np.asarray(obs[self.frame_key], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f'{self.frame_key} must be an RGB HxWx3 image')
        mask, mask_fraction, reject_count, measurement_valid = self._mask(frame)
        if measurement_valid:
            prediction = self.estimator.predict(
                frame,
                mask,
                force_key='Fz',
                force_mode='magnitude',
            )
            raw_force = float(prediction.control_force)
        else:
            raw_force = 0.0

        stationary = is_stationary(
            obs.get('arm_joint_velocity', []),
            self.stationary_velocity_limit,
        )
        torque = np.asarray(
            obs.get('arm_joint_torque', []), dtype=np.float64
        ).reshape(-1)
        basis = np.asarray(
            obs.get('arm_payload_torque_basis', []), dtype=np.float64
        ).reshape(-1)
        current_closure = finite_scalar(obs['gripper_pos'], 'gripper_pos')
        command, metadata = self.controller.update(
            current_closure=current_closure,
            raw_force=raw_force,
            joint_torque=torque,
            payload_torque_basis=basis,
            stationary=stationary,
            measurement_valid=measurement_valid,
        )
        metadata.update(
            {
                'mask_fraction': mask_fraction,
                'mask_reject_count': reject_count,
                'measured_closure': current_closure,
                'gripper_torque': (
                    finite_scalar(obs['gripper_torque'], 'gripper_torque')
                    if 'gripper_torque' in obs
                    else ''
                ),
                'pose_error_m': position_error,
                'rotation_error_deg': float(np.rad2deg(rotation_error)),
                'target_arm_pos': self.target_position.copy(),
                'target_arm_quat': self.target_quaternion.copy(),
            }
        )
        side_frame = None
        if self.side_frame_key and self.side_frame_key in obs:
            side_frame = np.asarray(obs[self.side_frame_key], dtype=np.uint8)
        self.recorder.record(
            wrist_rgb=frame,
            mask=mask,
            side_rgb=side_frame,
            metadata=metadata,
            obs=obs,
        )
        self._status(metadata)
        return self._action(obs, command)


class AsyncLatestPolicy:
    """Keep the NUC handoff responsive while SAM2/VisualForce runs on a worker."""

    def __init__(self, policy: CupAdaptiveGripPolicy):
        self.policy = policy
        self.pending = queue.Queue(maxsize=1)
        self.state_lock = threading.Lock()
        self.compute_lock = threading.Lock()
        self.generation = 0
        self.last_action = None
        self.last_action_generation = -1
        self.stop_event = threading.Event()
        self.closed = False
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    @staticmethod
    def _fallback(obs):
        return measured_hold_action(obs)

    def reset(self) -> None:
        with self.state_lock:
            self.generation += 1
            self.last_action = None
            self.last_action_generation = -1
        while True:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break

    def step(self, obs):
        with self.state_lock:
            generation = self.generation
            action = (
                self.last_action
                if self.last_action_generation == generation
                else None
            )
        item = (generation, obs)
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(item)
        if action is None:
            return self._fallback(obs)
        return {key: np.asarray(value).copy() for key, value in action.items()}

    def _loop(self):
        active_generation = -1
        while not self.stop_event.is_set():
            try:
                generation, obs = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self.compute_lock:
                    if generation != active_generation:
                        self.policy.reset()
                        active_generation = generation
                    action = self.policy.step(obs)
                with self.state_lock:
                    if generation == self.generation:
                        self.last_action = action
                        self.last_action_generation = generation
            except Exception:
                print('Cup adaptive worker failed; retaining the previous hold action:')
                traceback.print_exc()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.worker.join(timeout=30.0)
        if self.worker.is_alive():
            print('Cup adaptive worker did not stop in 30 seconds; skipping recorder close')
            return
        with self.compute_lock:
            self.policy.close()

    def flush_tts_rollout(self):
        # PolicyServer calls this hook from its finally block.
        self.close()


def _add_controller_arguments(parser: argparse.ArgumentParser) -> None:
    for field in _CONTROLLER_CLI_FIELDS:
        flag = f'--{field.name.replace("_", "-")}'
        default = field.default
        if isinstance(default, bool):
            action = 'store_false' if default else 'store_true'
            parser.add_argument(flag, action=action, default=default)
            continue
        kwargs = {'type': type(default), 'default': default}
        if field.name == 'payload_torque_sign':
            kwargs['choices'] = (-1.0, 1.0)
        parser.add_argument(flag, **kwargs)


def controller_config_from_args(args: argparse.Namespace) -> CupAdaptiveGripConfig:
    values = {
        field.name: getattr(args, field.name)
        for field in _CONTROLLER_CLI_FIELDS
    }
    return CupAdaptiveGripConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Hold an ARX pose and adapt paper-cup grip without a robot policy.'
    )
    parser.add_argument('--visualforce-ckpt', required=True)
    parser.add_argument('--visualforce-root', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--frame-key', default='wrist_image')
    parser.add_argument('--side-frame-key', default='base_image')
    parser.add_argument('--mask-mode', choices=('sam2', 'hsv'), default='sam2')
    parser.add_argument(
        '--sam2-model', choices=('tiny', 'small', 'base', 'large'), default='small'
    )
    parser.add_argument('--sam2-repo', default=None)
    parser.add_argument('--sam2-ckpt', default=None)
    _add_controller_arguments(parser)
    parser.add_argument('--stationary-velocity-limit', type=float, default=0.05)
    parser.add_argument('--min-mask-fraction', type=float, default=0.01)
    parser.add_argument('--max-mask-fraction', type=float, default=0.40)
    parser.add_argument('--max-pose-error-m', type=float, default=0.03)
    parser.add_argument('--max-rotation-error-degrees', type=float, default=8.0)
    parser.add_argument(
        '--rollout-dir',
        default='tts_rollouts/cup_adaptive',
    )
    parser.add_argument('--rollout-fps', type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    controller_config = controller_config_from_args(args)
    policy = CupAdaptiveGripPolicy(
        visualforce_checkpoint=args.visualforce_ckpt,
        visualforce_root=args.visualforce_root,
        device=args.device,
        controller_config=controller_config,
        frame_key=args.frame_key,
        side_frame_key=args.side_frame_key or None,
        mask_mode=args.mask_mode,
        sam2_model=args.sam2_model,
        sam2_repo=args.sam2_repo,
        sam2_checkpoint=args.sam2_ckpt,
        stationary_velocity_limit=args.stationary_velocity_limit,
        min_mask_fraction=args.min_mask_fraction,
        max_mask_fraction=args.max_mask_fraction,
        max_pose_error_m=args.max_pose_error_m,
        max_rotation_error_degrees=args.max_rotation_error_degrees,
        rollout_dir=args.rollout_dir or None,
        rollout_fps=args.rollout_fps,
    )
    wrapped = AsyncLatestPolicy(policy)

    def exit_on_signal(signum, _frame):
        print(f'Received signal {signum}; closing the cup rollout')
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    mode = 'MONITOR ONLY' if args.monitor_only else 'ADAPTIVE GRIP'
    print(f'Cup server mode: {mode}; no diffusion-policy checkpoint is loaded.')
    PolicyServer(wrapped, port=args.port).run()


if __name__ == '__main__':
    main()
