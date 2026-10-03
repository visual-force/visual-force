import unittest

import numpy as np

from diffusion_policy.real_world.cup_adaptive_grip import (
    CupAdaptiveGripConfig,
    CupAdaptiveGripController,
)


class CupAdaptiveGripControllerTest(unittest.TestCase):
    @staticmethod
    def make_controller(**overrides):
        values = {
            'calibration_samples': 3,
            'force_filter_window': 1,
            'load_filter_window': 1,
            'load_deadband_n': 0.0,
            'closure_per_load_n': 0.01,
            'force_limit_rise_n': 3.0,
            'emergency_margin_n': 1.0,
            'close_step': 0.005,
            'release_step': 0.005,
            'min_initial_closure': 0.2,
            'max_closure': 0.55,
            'max_closure_delta': 0.1,
            'max_command_lead': 0.03,
        }
        values.update(overrides)
        return CupAdaptiveGripController(CupAdaptiveGripConfig(**values))

    @staticmethod
    def calibrate(controller, closure=0.4, force=2.0):
        result = None
        for _ in range(controller.config.calibration_samples):
            result = controller.update(
                current_closure=closure,
                raw_force=force,
                joint_torque=np.zeros(2),
                payload_torque_basis=np.array([1.0, 0.0]),
            )
        return result

    def test_payload_projection_recovers_load_for_non_unit_basis(self):
        baseline = np.array([1.0, -2.0, 0.5])
        basis = np.array([0.25, -0.50, 0.75])
        torque = baseline + 3.2 * basis

        load = CupAdaptiveGripController.project_added_load(
            torque,
            baseline,
            basis,
        )

        self.assertAlmostEqual(load, 3.2)

    def test_config_rejects_non_finite_and_non_integer_windows(self):
        with self.assertRaises(ValueError):
            CupAdaptiveGripConfig(load_deadband_n=np.nan)
        with self.assertRaises(ValueError):
            CupAdaptiveGripConfig(force_filter_window=2.5)

    def test_payload_projection_supports_measured_torque_sign_flip(self):
        load = CupAdaptiveGripController.project_added_load(
            np.array([-2.0, 0.0]),
            np.zeros(2),
            np.array([1.0, 0.0]),
            torque_sign=-1.0,
        )
        self.assertAlmostEqual(load, 2.0)

    def test_empty_cup_calibration_holds_manual_grip(self):
        controller = self.make_controller()

        command, metadata = self.calibrate(controller)

        self.assertTrue(controller.calibrated)
        self.assertEqual(metadata['mode'], 'calibration_complete')
        self.assertAlmostEqual(command, 0.4)
        self.assertAlmostEqual(metadata['baseline_force_n'], 2.0)

    def test_calibration_waits_for_stationary_arm(self):
        controller = self.make_controller()

        command, metadata = controller.update(
            current_closure=0.4,
            raw_force=2.0,
            joint_torque=np.zeros(2),
            payload_torque_basis=np.array([1.0, 0.0]),
            stationary=False,
        )

        self.assertEqual(metadata['mode'], 'calibration_motion_hold')
        self.assertEqual(metadata['calibration_count'], 0)
        self.assertAlmostEqual(command, 0.4)

    def test_added_payload_schedules_and_closes_gripper(self):
        controller = self.make_controller()
        self.calibrate(controller)

        command, metadata = controller.update(
            current_closure=0.4,
            raw_force=2.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'close_for_load')
        self.assertAlmostEqual(metadata['filtered_added_load_n'], 2.0)
        self.assertAlmostEqual(metadata['scheduled_closure'], 0.42)
        self.assertAlmostEqual(metadata['scheduled_closure_increment'], 0.02)
        self.assertAlmostEqual(metadata['force_limit_n'], 5.0)
        self.assertAlmostEqual(command, 0.405)

    def test_load_closes_even_when_force_rises_below_safety_limit(self):
        controller = self.make_controller()
        self.calibrate(controller)

        command, metadata = controller.update(
            current_closure=0.4,
            raw_force=4.5,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'close_for_load')
        self.assertAlmostEqual(command, 0.405)

    def test_empty_cup_force_noise_does_not_cause_closure(self):
        controller = self.make_controller()
        self.calibrate(controller)

        command, metadata = controller.update(
            current_closure=0.4,
            raw_force=1.0,
            joint_torque=np.zeros(2),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'hold_empty_cup')
        self.assertAlmostEqual(command, 0.4)

    def test_close_command_obeys_lead_and_absolute_limits(self):
        controller = self.make_controller(
            close_step=0.02,
            max_command_lead=0.01,
            max_closure=0.52,
            max_closure_delta=0.015,
            closure_per_load_n=1.0,
        )
        self.calibrate(controller, closure=0.5)

        command, metadata = controller.update(
            current_closure=0.5,
            raw_force=2.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'closure_limit_hold')
        self.assertTrue(metadata['closure_limited'])
        self.assertAlmostEqual(command, 0.51)

    def test_emergency_release_is_one_bounded_latched_step(self):
        controller = self.make_controller()
        self.calibrate(controller)
        first_command, _ = controller.update(
            current_closure=0.4,
            raw_force=2.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )
        self.assertAlmostEqual(first_command, 0.405)

        release_command, metadata = controller.update(
            current_closure=0.405,
            raw_force=6.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )
        held_command, held_metadata = controller.update(
            current_closure=0.4,
            raw_force=2.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'emergency_release')
        self.assertAlmostEqual(release_command, 0.4)
        self.assertEqual(held_metadata['mode'], 'emergency_release_hold')
        self.assertAlmostEqual(held_command, release_command)

    def test_force_limit_stops_closing_without_opening(self):
        controller = self.make_controller()
        self.calibrate(controller)

        command, metadata = controller.update(
            current_closure=0.4,
            raw_force=5.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'force_limit_hold')
        self.assertAlmostEqual(command, 0.4)

    def test_falling_load_never_reopens_liquid_filled_cup(self):
        controller = self.make_controller()
        self.calibrate(controller)
        command, _ = controller.update(
            current_closure=0.4,
            raw_force=2.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        held, metadata = controller.update(
            current_closure=command,
            raw_force=2.0,
            joint_torque=np.zeros(2),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'hold_empty_cup')
        self.assertAlmostEqual(held, command)

    def test_monitor_mode_never_changes_grip_even_at_emergency_force(self):
        controller = self.make_controller(monitor_only=True)
        self.calibrate(controller)

        command, metadata = controller.update(
            current_closure=0.4,
            raw_force=10.0,
            joint_torque=np.array([3.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'monitor_emergency_force')
        self.assertFalse(metadata['emergency_latched'])
        self.assertAlmostEqual(command, 0.4)

    def test_monitor_startup_fault_does_not_change_manual_grip(self):
        controller = self.make_controller(monitor_only=True, max_closure=0.55)

        command, metadata = controller.update(
            current_closure=0.6,
            raw_force=2.0,
            joint_torque=np.zeros(2),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'startup_fault_hold')
        self.assertAlmostEqual(command, 0.6)

    def test_invalid_measurement_holds_last_command(self):
        controller = self.make_controller()
        self.calibrate(controller)
        last_command, _ = controller.update(
            current_closure=0.4,
            raw_force=2.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        command, metadata = controller.update(
            current_closure=0.405,
            raw_force=2.0,
            joint_torque=np.array([2.0, 0.0]),
            payload_torque_basis=np.array([1.0, 0.0]),
            measurement_valid=False,
        )

        self.assertEqual(metadata['mode'], 'sensor_hold')
        self.assertAlmostEqual(command, last_command)

    def test_controller_refuses_to_start_without_manual_cup_grip(self):
        controller = self.make_controller()

        command, metadata = controller.update(
            current_closure=0.05,
            raw_force=2.0,
            joint_torque=np.zeros(2),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'startup_fault_hold')
        self.assertIn('manually grip the empty cup', metadata['reason'])
        self.assertAlmostEqual(command, 0.05)

    def test_startup_fault_preserves_manual_grip_above_adaptive_limit(self):
        controller = self.make_controller(max_closure=0.55)

        command, metadata = controller.update(
            current_closure=0.6,
            raw_force=2.0,
            joint_torque=np.zeros(2),
            payload_torque_basis=np.array([1.0, 0.0]),
        )

        self.assertEqual(metadata['mode'], 'startup_fault_hold')
        self.assertAlmostEqual(command, 0.6)


if __name__ == '__main__':
    unittest.main()
