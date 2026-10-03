import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from adaptive_grip_server import (
    CupAdaptiveGripPolicy,
    CupRolloutRecorder,
    build_parser,
    controller_config_from_args,
)
from diffusion_policy.real_world.cup_adaptive_grip import (
    CupAdaptiveGripConfig,
    CupAdaptiveGripController,
)


class _Estimator:
    def __init__(self, force=2.0):
        self.force = force

    def predict(self, _frame, _mask, *, force_key, force_mode):
        if force_key != 'Fz' or force_mode != 'magnitude':
            raise AssertionError('cup server must request Fz magnitude')
        return SimpleNamespace(control_force=self.force)


class _Recorder:
    def __init__(self):
        self.rows = []

    def record(self, **kwargs):
        self.rows.append(kwargs)


class CupAdaptiveGripPolicyTest(unittest.TestCase):
    @staticmethod
    def make_policy():
        policy = CupAdaptiveGripPolicy.__new__(CupAdaptiveGripPolicy)
        policy.estimator = _Estimator()
        policy.controller = CupAdaptiveGripController(
            CupAdaptiveGripConfig(
                calibration_samples=1,
                force_filter_window=1,
                load_filter_window=1,
                load_deadband_n=0.0,
                closure_per_load_n=0.01,
            )
        )
        policy.frame_key = 'wrist_image'
        policy.side_frame_key = 'base_image'
        policy.stationary_velocity_limit = 0.05
        policy.min_mask_fraction = 0.01
        policy.max_mask_fraction = 0.40
        policy.max_pose_error_m = 0.03
        policy.max_rotation_error = np.deg2rad(8.0)
        policy.target_position = None
        policy.target_quaternion = None
        policy.pose_fault_latched = False
        policy.pose_fault_announced = False
        policy.last_mode = None
        policy.last_status_time = 0.0
        policy.recorder = _Recorder()
        policy._mask = lambda frame: (
            np.ones(frame.shape[:2], dtype=np.uint8) * 255,
            0.10,
            0,
            True,
        )
        policy._status = lambda _metadata: None
        return policy

    @staticmethod
    def observation(position=None, closure=0.4, torque=None):
        return {
            'arm_pos': np.zeros(3) if position is None else np.asarray(position),
            'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
            'arm_joint_velocity': np.zeros(2),
            'arm_joint_torque': (
                np.zeros(2) if torque is None else np.asarray(torque)
            ),
            'arm_payload_torque_basis': np.array([1.0, 0.0]),
            'gripper_pos': np.array([closure]),
            'gripper_torque': np.array([0.5]),
            'wrist_image': np.zeros((20, 30, 3), dtype=np.uint8),
            'base_image': np.zeros((20, 30, 3), dtype=np.uint8),
        }

    def test_policy_latches_pose_and_only_closes_for_added_load(self):
        policy = self.make_policy()
        first = policy._step(self.observation())
        second = policy._step(
            self.observation(position=[0.001, 0.0, 0.0], torque=[2.0, 0.0])
        )

        np.testing.assert_allclose(first['arm_pos'], np.zeros(3))
        np.testing.assert_allclose(second['arm_pos'], np.zeros(3))
        self.assertAlmostEqual(float(first['gripper_pos'][0]), 0.4)
        self.assertAlmostEqual(float(second['gripper_pos'][0]), 0.4025)
        self.assertEqual(policy.recorder.rows[-1]['metadata']['mode'], 'close_for_load')

    def test_pose_fault_returns_measured_hold_without_advancing_grip(self):
        policy = self.make_policy()
        policy._step(self.observation())
        moved = self.observation(position=[0.04, 0.0, 0.0], closure=0.41)

        with mock.patch('builtins.print'):
            action = policy._step(moved)

        self.assertTrue(policy.pose_fault_latched)
        np.testing.assert_allclose(action['arm_pos'], moved['arm_pos'])
        self.assertAlmostEqual(float(action['gripper_pos'][0]), 0.41)
        self.assertAlmostEqual(policy.controller.command_closure, 0.4)

    def test_controller_cli_defaults_come_from_controller_config(self):
        args = build_parser().parse_args(['--visualforce-ckpt', 'unused.pt'])

        config = controller_config_from_args(args)

        self.assertEqual(config, CupAdaptiveGripConfig())

    def test_recorder_serializes_shared_metadata_and_vectors(self):
        recorder = CupRolloutRecorder(None)
        recorder.started_at = time.monotonic()
        metadata = {
            'mode': 'close_for_load',
            'measured_closure': np.array([0.4]),
            'command_closure': 0.405,
            'target_arm_pos': np.array([0.1, 0.2, 0.3]),
            'target_arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        }
        obs = self.observation()

        row = recorder._telemetry_row(metadata, obs, 101.0)

        self.assertEqual(set(row), set(recorder.CSV_FIELDS))
        self.assertEqual(row['mode'], 'close_for_load')
        self.assertEqual(row['arm_pos'], '0;0;0')
        self.assertEqual(row['target_arm_pos'], '0.1;0.2;0.3')
        self.assertAlmostEqual(row['measured_closure'], 0.4)
        self.assertAlmostEqual(row['command_closure'], 0.405)
        self.assertAlmostEqual(row['water_equivalent_g'], 101.0)


if __name__ == '__main__':
    unittest.main()
