import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cup_pose_server import (
    CupPoseCaptureConfig,
    CupPoseCapturePolicy,
    CupPoseGotoConfig,
    CupPoseGotoPolicy,
    load_pose,
    quaternion_step,
    save_pose,
)


class CupPoseServerTest(unittest.TestCase):
    @staticmethod
    def observation(position=None, quaternion=None, closure=0.4, velocity=None):
        return {
            'arm_pos': np.zeros(3) if position is None else np.asarray(position),
            'arm_quat': (
                np.array([0.0, 0.0, 0.0, 1.0])
                if quaternion is None
                else np.asarray(quaternion)
            ),
            'gripper_pos': np.array([closure]),
            'arm_joint_velocity': (
                np.zeros(2) if velocity is None else np.asarray(velocity)
            ),
        }

    @staticmethod
    def write_pose(path: Path, position) -> None:
        save_pose(
            path,
            np.asarray(position, dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0]),
            0.4,
        )

    def test_capture_uses_stationary_median_and_normalizes_quaternion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pose_path = Path(temp_dir) / 'cup_pose.json'
            observations = (
                self.observation(position=[0.0, 0.0, 0.0]),
                self.observation(
                    position=[0.9, 0.0, 0.0],
                    velocity=[0.1, 0.0],
                ),
                self.observation(position=[0.01, 0.0, 0.0]),
                self.observation(position=[0.02, 0.0, 0.0]),
                self.observation(position=[0.03, 0.0, 0.0]),
            )

            with mock.patch('builtins.print'):
                policy = CupPoseCapturePolicy(
                    str(pose_path),
                    config=CupPoseCaptureConfig(samples=3),
                )
                for obs in observations:
                    action = policy.step(obs)

            position, quaternion, _ = load_pose(pose_path)
            np.testing.assert_allclose(position, [0.02, 0.0, 0.0])
            np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])
            np.testing.assert_allclose(action['arm_pos'], observations[-1]['arm_pos'])

    def test_goto_bounds_translation_and_keeps_initial_grip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pose_path = Path(temp_dir) / 'cup_pose.json'
            self.write_pose(pose_path, [0.02, 0.0, 0.0])

            with mock.patch('builtins.print'):
                policy = CupPoseGotoPolicy(
                    str(pose_path),
                    config=CupPoseGotoConfig(max_translation_step_m=0.005),
                )
                initial = policy.step(self.observation(closure=0.4))
                moving = policy.step(self.observation(closure=0.5))

            np.testing.assert_allclose(initial['arm_pos'], np.zeros(3))
            np.testing.assert_allclose(moving['arm_pos'], [0.005, 0.0, 0.0])
            self.assertAlmostEqual(float(moving['gripper_pos'][0]), 0.4)

    def test_goto_latches_fault_when_start_is_too_far_away(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pose_path = Path(temp_dir) / 'cup_pose.json'
            self.write_pose(pose_path, [0.2, 0.0, 0.0])
            obs = self.observation(closure=0.42)

            with mock.patch('builtins.print'):
                policy = CupPoseGotoPolicy(str(pose_path))
                action = policy.step(obs)

            self.assertIn('start is', policy.fault_reason)
            np.testing.assert_allclose(action['arm_pos'], obs['arm_pos'])
            self.assertAlmostEqual(float(action['gripper_pos'][0]), 0.42)

    def test_goto_requires_stationary_settle_samples_at_goal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pose_path = Path(temp_dir) / 'cup_pose.json'
            self.write_pose(pose_path, [0.0, 0.0, 0.0])

            with mock.patch('builtins.print'):
                policy = CupPoseGotoPolicy(
                    str(pose_path),
                    config=CupPoseGotoConfig(settle_samples=2),
                )
                policy.step(self.observation())
                policy.step(self.observation(velocity=[0.1, 0.0]))
                self.assertFalse(policy.reached)
                policy.step(self.observation())
                policy.step(self.observation())

            self.assertTrue(policy.reached)

    def test_quaternion_step_obeys_physical_angle_limit(self):
        target = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

        stepped = quaternion_step(
            np.array([0.0, 0.0, 0.0, 1.0]),
            target,
            max_angle=np.deg2rad(10.0),
        )

        expected = np.array(
            [0.0, 0.0, np.sin(np.deg2rad(5.0)), np.cos(np.deg2rad(5.0))]
        )
        np.testing.assert_allclose(stepped, expected)


if __name__ == '__main__':
    unittest.main()
