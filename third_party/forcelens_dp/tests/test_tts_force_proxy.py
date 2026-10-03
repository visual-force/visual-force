import unittest

import cv2 as cv
import numpy as np

from policy_server import (
    GentleGripperController,
    _append_gripper_fallback_candidates,
    _apply_absolute_gripper_safety,
    _compose_selected_action,
    _force_target_score,
    _gripper_proxy_features,
    _policy_force_proxies,
    _resolve_policy_force_dynamic_target,
    _resolve_tts_force_target,
    _select_action_refill,
    _select_force_candidate,
    make_hsv_gripper_mask,
)


class TestForceProxy(unittest.TestCase):
    def test_hsv_gripper_mask_removes_small_components(self):
        hsv = np.zeros((60, 60, 3), dtype=np.uint8)
        hsv[10:40, 10:40] = (75, 255, 255)
        hsv[50:54, 50:54] = (75, 255, 255)
        frame = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

        mask = make_hsv_gripper_mask(frame)

        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[20, 20], 255)
        self.assertEqual(mask[52, 52], 0)

    def make_gentle_controller(self, **overrides):
        kwargs = {
            'desired_force_delta': 5.0,
            'deadband': 0.5,
            'stop_margin': 2.0,
            'close_step': 0.05,
            'maintain_step': 0.005,
            'release_step': 0.05,
            'filter_window': 1,
            'baseline_max_position': 0.35,
            'min_closure_position': 0.35,
            'max_closure_position': 0.82,
            'max_command_lead': 0.25,
            'max_force_rate': None,
            'baseline_min_samples': 1,
            'gripper_index': 7,
            'close_positive': True,
            'command_min': 0.0,
            'command_max': 1.0,
        }
        kwargs.update(overrides)
        return GentleGripperController(**kwargs)

    def test_gripper_fallbacks_append_hold_and_open_chunks(self):
        candidates = np.zeros((2, 4, 8), dtype=np.float32)
        candidates[:, :, 7] = 0.9
        result = _append_gripper_fallback_candidates(
            candidates,
            current_gripper=0.6,
            release_step=0.05,
            command_min=0.0,
            command_max=1.0,
        )

        self.assertEqual(result.shape, (4, 4, 8))
        np.testing.assert_allclose(result[-2, :, 7], 0.6)
        np.testing.assert_allclose(result[-1, :, 7], 0.55)
        np.testing.assert_array_equal(result[-2, :, :7], candidates[0, :, :7])

    def test_gentle_controller_accumulates_setpoint_when_measurement_stalls(self):
        controller = self.make_gentle_controller()
        actions = np.zeros((4, 8), dtype=np.float32)
        actions[:, 7] = 0.6

        commands = []
        for _ in range(6):
            controlled, metadata = controller.apply(
                actions,
                current_gripper=0.10,
                raw_force=2.5,
            )
            commands.append(float(controlled[0, 7]))

        np.testing.assert_allclose(commands, [0.15, 0.20, 0.25, 0.30, 0.35, 0.35])
        self.assertEqual(
            metadata['gripper_safety_mode'],
            'close_before_contact_region',
        )
        self.assertFalse(metadata['gripper_safety_closure_ready'])
        self.assertTrue(metadata['gentle_close_limited'])
        self.assertFalse(metadata['preempt_action_queue'])
        self.assertAlmostEqual(metadata['gentle_filtered_force_n'], 2.5)
        self.assertAlmostEqual(metadata['gentle_desired_delta_n'], 5.0)
        self.assertAlmostEqual(metadata['gentle_max_command_lead'], 0.25)

    def test_policy_approach_uses_nominal_gripper_without_fixed_close_step(self):
        controller = self.make_gentle_controller(
            policy_gripper_approach=True,
            max_command_lead=0.10,
        )
        selected = np.zeros((4, 8), dtype=np.float32)
        selected[:, 7] = 0.8
        nominal = selected.copy()
        nominal[:, 7] = 0.18

        controlled, metadata = controller.apply(
            selected,
            current_gripper=0.10,
            raw_force=2.0,
            policy_actions=nominal,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.18)
        self.assertEqual(metadata['gripper_safety_mode'], 'policy_approach')
        self.assertTrue(metadata['gentle_policy_gripper_approach'])
        self.assertFalse(metadata['gentle_force_control_armed'])

    def test_policy_approach_still_bounds_nominal_command_lead_and_cap(self):
        controller = self.make_gentle_controller(
            policy_gripper_approach=True,
            max_closure_position=0.45,
            max_command_lead=0.10,
        )
        selected = np.zeros((4, 8), dtype=np.float32)
        selected[:, 7] = 0.8
        nominal = selected.copy()
        nominal[:, 7] = 0.9

        controlled, metadata = controller.apply(
            selected,
            current_gripper=0.10,
            raw_force=2.0,
            policy_actions=nominal,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.20)
        self.assertTrue(metadata['gentle_close_limited'])

    def test_policy_approach_hands_off_after_contact_evidence(self):
        controller = self.make_gentle_controller(
            desired_force_delta=4.0,
            policy_release_contact_delta=1.5,
            policy_gripper_approach=True,
            min_closure_position=0.25,
            baseline_max_position=0.15,
            max_closure_position=0.45,
            max_command_lead=0.10,
        )
        actions = np.zeros((4, 8), dtype=np.float32)
        actions[:, 7] = 0.30
        controller.apply(actions, 0.10, 2.0, policy_actions=actions)

        controlled, metadata = controller.apply(
            actions,
            current_gripper=0.30,
            raw_force=3.6,
            policy_actions=actions,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.35)
        self.assertEqual(metadata['gripper_safety_mode'], 'close_below_band')
        self.assertTrue(metadata['gentle_force_control_armed'])

    def test_gentle_controller_opens_from_current_position_above_target(self):
        controller = self.make_gentle_controller()
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.65, raw_force=2.0)

        controlled, metadata = controller.apply(
            actions,
            current_gripper=0.69,
            raw_force=10.0,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.65)
        self.assertTrue(metadata['gripper_safety_active'])
        self.assertEqual(metadata['gripper_safety_mode'], 'emergency_release')
        self.assertTrue(metadata['preempt_action_queue'])

    def test_gentle_controller_closes_when_force_rate_is_high_but_force_is_below_band(self):
        controller = self.make_gentle_controller(max_force_rate=20.0)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.40, raw_force=2.0)

        controlled, metadata = controller.apply(
            actions,
            current_gripper=0.42,
            raw_force=4.4,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.50)
        self.assertEqual(metadata['gripper_safety_mode'], 'close_below_band')
        self.assertGreaterEqual(metadata['gentle_force_rate_n_s'], 20.0)
        self.assertFalse(metadata['preempt_action_queue'])

    def test_gentle_controller_holds_at_target(self):
        controller = self.make_gentle_controller()
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.95
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.40, raw_force=2.0)

        controlled, metadata = controller.apply(
            actions,
            current_gripper=0.42,
            raw_force=7.0,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.42)
        self.assertEqual(metadata['gripper_safety_mode'], 'hold_target_band')
        self.assertTrue(metadata['gentle_just_latched'])
        self.assertTrue(metadata['preempt_action_queue'])

    def test_gentle_controller_filters_force_before_comparison(self):
        controller = self.make_gentle_controller(filter_window=3)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.30, raw_force=2.0)

        controlled, metadata = controller.apply(
            actions,
            current_gripper=0.40,
            raw_force=10.0,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.45)
        self.assertEqual(metadata['gripper_safety_mode'], 'close_below_band')
        self.assertAlmostEqual(metadata['gentle_filtered_force_n'], 2.0)

    def test_gentle_controller_respects_physical_close_limit(self):
        controller = self.make_gentle_controller()
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.99
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.command_closure = 0.80

        capped, metadata = controller.apply(
            actions,
            current_gripper=0.81,
            raw_force=2.0,
        )

        np.testing.assert_allclose(capped[:, 7], 0.82)
        self.assertTrue(metadata['gentle_position_capped'])

    def test_gentle_controller_holds_when_force_measurement_is_invalid(self):
        controller = self.make_gentle_controller()
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9

        held, metadata = controller.apply(
            actions,
            current_gripper=0.4,
            raw_force=3.0,
            measurement_valid=False,
        )

        np.testing.assert_allclose(held[:, 7], 0.4)
        self.assertEqual(metadata['gripper_safety_mode'], 'sensor_hold')
        self.assertFalse(metadata['gentle_measurement_valid'])
        self.assertTrue(metadata['preempt_action_queue'])

        held, metadata = controller.apply(
            actions,
            current_gripper=0.4,
            raw_force=3.0,
            measurement_valid=False,
        )
        np.testing.assert_allclose(held[:, 7], 0.4)
        self.assertFalse(metadata['preempt_action_queue'])

    def test_gentle_controller_calibrates_before_closing(self):
        controller = self.make_gentle_controller(baseline_min_samples=3)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.7

        first, first_meta = controller.apply(actions, 0.1, 2.0)
        second, second_meta = controller.apply(actions, 0.1, 3.0)
        third, third_meta = controller.apply(actions, 0.1, 2.5)

        np.testing.assert_allclose(first[:, 7], 0.10)
        np.testing.assert_allclose(second[:, 7], 0.10)
        np.testing.assert_allclose(third[:, 7], 0.15)
        self.assertEqual(first_meta['gripper_safety_mode'], 'calibration_hold')
        self.assertEqual(second_meta['gripper_safety_mode'], 'calibration_hold')
        self.assertTrue(third_meta['gentle_baseline_calibrated'])
        self.assertAlmostEqual(third_meta['gentle_baseline_force_n'], 2.5)
        self.assertAlmostEqual(third_meta['gentle_desired_force_n'], 7.5)

    def test_gentle_controller_holds_without_open_pose_calibration(self):
        controller = self.make_gentle_controller(baseline_min_samples=3)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9

        held, metadata = controller.apply(actions, 0.5, 8.0)

        np.testing.assert_allclose(held[:, 7], 0.50)
        self.assertEqual(metadata['gripper_safety_mode'], 'calibration_hold')
        self.assertFalse(metadata['gentle_baseline_calibrated'])

    def test_gentle_controller_ignores_policy_open_without_calibration(self):
        controller = self.make_gentle_controller(baseline_min_samples=3)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.1

        opened, metadata = controller.apply(actions, 0.5, 8.0)

        np.testing.assert_allclose(opened[:, 7], 0.50)
        self.assertEqual(metadata['gripper_safety_mode'], 'calibration_hold')
        self.assertFalse(metadata['gentle_baseline_calibrated'])

    def test_gentle_controller_subtracts_open_gripper_force_bias(self):
        controller = self.make_gentle_controller()
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9

        controller.apply(actions, current_gripper=0.10, raw_force=3.0)
        controlled, metadata = controller.apply(
            actions,
            current_gripper=0.10,
            raw_force=3.4,
        )

        np.testing.assert_allclose(controlled[:, 7], 0.20)
        self.assertEqual(
            metadata['gripper_safety_mode'],
            'close_before_contact_region',
        )
        self.assertAlmostEqual(metadata['gentle_force_delta_n'], 0.4)

    def test_gentle_controller_latches_fixed_episode_baseline(self):
        controller = self.make_gentle_controller(
            baseline_min_samples=3,
            filter_window=1,
        )
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9

        controller.apply(actions, 0.10, 2.0)
        controller.apply(actions, 0.10, 3.0)
        controller.apply(actions, 0.10, 2.5)
        _, metadata = controller.apply(actions, 0.10, 3.5)

        self.assertAlmostEqual(metadata['gentle_baseline_force_n'], 2.5)
        self.assertAlmostEqual(metadata['gentle_force_delta_n'], 1.0)

    def test_baseline_calibration_survives_small_open_gripper_drift(self):
        controller = self.make_gentle_controller(
            baseline_min_samples=4,
            baseline_max_position=0.10,
            min_closure_position=0.25,
            filter_window=1,
        )
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9

        positions = [0.005, 0.075, 0.094, 0.106]
        for position in positions:
            _, metadata = controller.apply(actions, position, 2.0)

        self.assertTrue(metadata['gentle_baseline_calibrated'])
        self.assertAlmostEqual(metadata['gentle_baseline_force_n'], 2.0)
        self.assertEqual(metadata['gripper_safety_mode'], 'close_before_contact_region')

    def test_baseline_does_not_start_outside_verified_open_region(self):
        controller = self.make_gentle_controller(
            baseline_min_samples=3,
            baseline_max_position=0.10,
            min_closure_position=0.25,
            filter_window=1,
        )
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9

        for _ in range(4):
            _, metadata = controller.apply(actions, 0.15, 2.0)

        self.assertFalse(metadata['gentle_baseline_calibrated'])
        self.assertEqual(metadata['gripper_safety_mode'], 'calibration_hold')

    def test_gentle_controller_ignores_normal_force_before_contact_region(self):
        controller = self.make_gentle_controller(
            filter_window=1,
            max_force_rate=None,
        )
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, 0.10, 2.0)

        closing, metadata = controller.apply(actions, 0.20, 7.2)

        np.testing.assert_allclose(closing[:, 7], 0.25)
        self.assertEqual(
            metadata['gripper_safety_mode'],
            'close_before_contact_region',
        )
        self.assertFalse(controller.contact_seen)

        held, metadata = controller.apply(actions, 0.35, 7.2)
        np.testing.assert_allclose(held[:, 7], 0.25)
        self.assertEqual(metadata['gripper_safety_mode'], 'hold_target_band')
        self.assertTrue(metadata['gripper_safety_closure_ready'])

    def test_gentle_controller_reaches_target_before_initial_hysteresis(self):
        controller = self.make_gentle_controller(
            filter_window=1,
            max_force_rate=None,
        )
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, 0.10, 2.0)

        closing, metadata = controller.apply(actions, 0.40, 6.7)
        np.testing.assert_allclose(closing[:, 7], 0.45)
        self.assertEqual(metadata['gripper_safety_mode'], 'close_below_band')

        held, metadata = controller.apply(actions, 0.45, 7.0)
        np.testing.assert_allclose(held[:, 7], 0.45)
        self.assertEqual(metadata['gripper_safety_mode'], 'hold_target_band')

    def test_gentle_controller_allows_opening_with_invalid_measurement(self):
        controller = self.make_gentle_controller()
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.1

        opened, metadata = controller.apply(
            actions,
            current_gripper=0.5,
            raw_force=8.0,
            measurement_valid=False,
        )

        np.testing.assert_allclose(opened[:, 7], 0.45)
        self.assertEqual(metadata['gripper_safety_mode'], 'sensor_open')
        self.assertTrue(metadata['preempt_action_queue'])

    def test_gentle_controller_ignores_policy_open_below_target(self):
        controller = self.make_gentle_controller()
        close_actions = np.zeros((2, 8), dtype=np.float32)
        close_actions[:, 7] = 0.8
        controller.apply(close_actions, 0.2, 2.0)
        controller.command_closure = 0.6
        open_actions = close_actions.copy()
        open_actions[:, 7] = 0.3

        opened, metadata = controller.apply(open_actions, 0.55, 2.0)

        np.testing.assert_allclose(opened[:, 7], 0.65)
        self.assertEqual(metadata['gripper_safety_mode'], 'close_below_band')
        self.assertFalse(metadata['preempt_action_queue'])

    def test_gentle_controller_can_disable_policy_release(self):
        controller = self.make_gentle_controller(
            policy_release_enabled=False,
            max_force_rate=None,
        )
        close_actions = np.zeros((2, 8), dtype=np.float32)
        close_actions[:, 7] = 0.8
        controller.apply(close_actions, 0.2, 2.0)
        controller.contact_seen = True
        controller.command_closure = 0.3
        open_actions = close_actions.copy()
        open_actions[:, 7] = 0.0

        controlled, metadata = controller.apply(open_actions, 0.3, 2.0)

        np.testing.assert_allclose(controlled[:, 7], 0.305)
        self.assertEqual(metadata['gripper_safety_mode'], 'maintain_below_band')
        self.assertFalse(metadata['gentle_policy_release_enabled'])
        self.assertFalse(metadata['gentle_policy_release_latched'])

    def test_gentle_controller_holds_inside_deadband_without_chatter(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.40, raw_force=2.0)

        held, first_metadata = controller.apply(
            actions,
            current_gripper=0.45,
            raw_force=7.3,
        )
        held_again, second_metadata = controller.apply(
            actions,
            current_gripper=0.45,
            raw_force=6.8,
        )

        np.testing.assert_allclose(held[:, 7], 0.45)
        np.testing.assert_allclose(held_again[:, 7], 0.45)
        self.assertEqual(
            first_metadata['gripper_safety_mode'],
            'hold_target_band',
        )
        self.assertTrue(first_metadata['preempt_action_queue'])
        self.assertFalse(second_metadata['preempt_action_queue'])

    def test_gentle_controller_only_preempts_once_above_target_band(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.40, raw_force=2.0)

        first, first_metadata = controller.apply(actions, 0.45, 8.0)
        second, second_metadata = controller.apply(actions, 0.43, 8.0)

        np.testing.assert_allclose(first[:, 7], 0.40)
        np.testing.assert_allclose(second[:, 7], 0.40)
        self.assertEqual(first_metadata['gripper_safety_mode'], 'open_above_band')
        self.assertTrue(first_metadata['preempt_action_queue'])
        self.assertFalse(second_metadata['preempt_action_queue'])

    def test_gentle_controller_brakes_on_projected_force_overshoot(self):
        controller = self.make_gentle_controller(max_force_rate=20.0)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.40, raw_force=2.0)

        held, metadata = controller.apply(actions, 0.42, 5.2)

        np.testing.assert_allclose(held[:, 7], 0.42)
        self.assertEqual(metadata['gripper_safety_mode'], 'rate_hold')
        self.assertTrue(metadata['preempt_action_queue'])

    def test_gentle_controller_recloses_slowly_after_contact(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.40, raw_force=2.0)
        controller.apply(actions, current_gripper=0.45, raw_force=8.0)

        maintained, metadata = controller.apply(actions, 0.40, 2.0)

        np.testing.assert_allclose(maintained[:, 7], 0.405)
        self.assertEqual(metadata['gripper_safety_mode'], 'maintain_below_band')

    def test_gentle_controller_keeps_recovery_setpoint_in_target_band(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9
        controller.apply(actions, current_gripper=0.20, raw_force=2.0)
        controller.apply(actions, current_gripper=0.40, raw_force=2.0)
        controller.apply(actions, current_gripper=0.45, raw_force=10.0)

        held, metadata = controller.apply(actions, 0.35, 7.0)

        np.testing.assert_allclose(held[:, 7], 0.40)
        self.assertEqual(metadata['gripper_safety_mode'], 'hold_target_band')

    def test_gentle_controller_executes_learned_release_after_contact(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        grasp = np.zeros((8, 8), dtype=np.float32)
        grasp[:, 7] = 0.9
        controller.apply(grasp, current_gripper=0.20, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.40, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.45, raw_force=8.0)

        release = grasp.copy()
        release[:, 7] = [0.70, 0.20, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0]
        opened, metadata = controller.apply(release, 0.45, 7.0)

        np.testing.assert_allclose(opened[:, 7], 0.0)
        self.assertEqual(metadata['gripper_safety_mode'], 'policy_release')
        self.assertTrue(metadata['gentle_policy_release_latched'])
        self.assertEqual(metadata['gentle_policy_release_index'], 1)
        self.assertTrue(metadata['preempt_action_queue'])

        held_open, metadata = controller.apply(grasp, 0.10, 2.0)
        np.testing.assert_allclose(held_open[:, 7], 0.0)
        self.assertEqual(metadata['gripper_safety_mode'], 'policy_release')
        self.assertFalse(metadata['preempt_action_queue'])

    def test_policy_release_arms_below_high_holding_target(self):
        controller = self.make_gentle_controller(
            desired_force_delta=3.0,
            policy_release_contact_delta=1.5,
            filter_window=1,
            max_force_rate=None,
        )
        grasp = np.zeros((8, 8), dtype=np.float32)
        grasp[:, 7] = 0.9
        controller.apply(grasp, current_gripper=0.10, raw_force=2.0)

        closing, metadata = controller.apply(
            grasp,
            current_gripper=0.40,
            raw_force=3.6,
        )
        np.testing.assert_allclose(closing[:, 7], 0.45)
        self.assertEqual(metadata['gripper_safety_mode'], 'close_below_band')
        self.assertTrue(metadata['gentle_policy_release_armed'])
        self.assertFalse(metadata['gentle_policy_release_latched'])

        release = grasp.copy()
        release[:, 7] = [0.70, 0.20, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0]
        opened, metadata = controller.apply(release, 0.45, 3.6)

        np.testing.assert_allclose(opened[:, 7], 0.0)
        self.assertEqual(metadata['gripper_safety_mode'], 'policy_release')
        self.assertTrue(metadata['gentle_policy_release_latched'])

    def test_gentle_controller_uses_future_release_intent(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        grasp = np.zeros((8, 8), dtype=np.float32)
        grasp[:, 7] = 0.9
        controller.apply(grasp, current_gripper=0.20, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.40, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.45, raw_force=8.0)

        future_release = grasp.copy()
        future_release[:, 7] = [0.70, 0.60, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0]
        opened, metadata = controller.apply(future_release, 0.45, 7.0)

        np.testing.assert_allclose(opened[:, 7], 0.0)
        self.assertEqual(metadata['gripper_safety_mode'], 'policy_release')
        self.assertTrue(metadata['gentle_policy_release_latched'])
        self.assertEqual(metadata['gentle_policy_release_index'], 2)

    def test_gentle_controller_uses_nominal_release_over_selected_hold(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        grasp = np.zeros((8, 8), dtype=np.float32)
        grasp[:, 7] = 0.9
        controller.apply(grasp, current_gripper=0.20, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.40, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.45, raw_force=8.0)

        selected_hold = grasp.copy()
        selected_hold[:, 7] = 0.80
        nominal_release = grasp.copy()
        nominal_release[:, 7] = [0.70, 0.65, 0.55, 0.30, 0.10, 0.0, 0.0, 0.0]
        opened, metadata = controller.apply(
            selected_hold,
            0.45,
            7.0,
            policy_actions=nominal_release,
        )

        np.testing.assert_allclose(opened[:, 7], 0.0)
        self.assertEqual(metadata['gripper_safety_mode'], 'policy_release')
        self.assertEqual(metadata['gentle_policy_release_index'], 3)

    def test_gentle_controller_rejects_transient_policy_open_dip(self):
        controller = self.make_gentle_controller(max_force_rate=None)
        grasp = np.zeros((8, 8), dtype=np.float32)
        grasp[:, 7] = 0.9
        controller.apply(grasp, current_gripper=0.20, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.40, raw_force=2.0)
        controller.apply(grasp, current_gripper=0.45, raw_force=8.0)

        noisy = grasp.copy()
        noisy[:, 7] = [0.70, 0.20, 0.65, 0.60, 0.58, 0.55, 0.50, 0.45]
        held, metadata = controller.apply(noisy, 0.45, 7.0)

        np.testing.assert_allclose(held[:, 7], 0.40)
        self.assertEqual(metadata['gripper_safety_mode'], 'hold_target_band')
        self.assertFalse(metadata['gentle_policy_release_latched'])

    def test_absolute_gripper_safety_holds_above_target_band(self):
        actions = np.zeros((4, 8), dtype=np.float32)
        actions[:, 7] = [0.6, 0.7, 0.8, 0.9]

        safe, latched, metadata = _apply_absolute_gripper_safety(
            actions,
            current_gripper=0.4,
            control_force=2.7,
            desired_force=2.5,
            deadband=0.1,
            stop_margin=0.75,
            release_step=0.05,
            command_min=0.0,
            command_max=1.0,
        )

        np.testing.assert_allclose(safe[:, 7], 0.4)
        np.testing.assert_array_equal(safe[:, :7], actions[:, :7])
        self.assertTrue(latched)
        self.assertEqual(metadata['gripper_safety_mode'], 'hold')
        self.assertTrue(metadata['preempt_action_queue'])

    def test_absolute_gripper_safety_releases_by_bounded_step(self):
        actions = np.zeros((4, 8), dtype=np.float32)
        actions[:, 7] = 0.95

        safe, latched, metadata = _apply_absolute_gripper_safety(
            actions,
            current_gripper=0.8,
            control_force=3.5,
            desired_force=2.5,
            deadband=0.1,
            stop_margin=0.75,
            release_step=0.05,
            command_min=0.0,
            command_max=1.0,
        )

        np.testing.assert_allclose(safe[:, 7], 0.75)
        self.assertTrue(latched)
        self.assertEqual(metadata['gripper_safety_mode'], 'release')
        self.assertAlmostEqual(metadata['gripper_safety_command'], 0.75)

    def test_absolute_gripper_safety_latches_until_force_is_below_target(self):
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.9

        held, latched, metadata = _apply_absolute_gripper_safety(
            actions,
            current_gripper=0.7,
            control_force=2.55,
            desired_force=2.5,
            deadband=0.1,
            stop_margin=0.75,
            release_step=0.05,
            latched=True,
        )
        np.testing.assert_allclose(held[:, 7], 0.7)
        self.assertTrue(latched)
        self.assertEqual(metadata['gripper_safety_mode'], 'hold')

        resumed, latched, metadata = _apply_absolute_gripper_safety(
            actions,
            current_gripper=0.7,
            control_force=2.4,
            desired_force=2.5,
            deadband=0.1,
            stop_margin=0.75,
            release_step=0.05,
            latched=latched,
        )
        np.testing.assert_array_equal(resumed, actions)
        self.assertFalse(latched)
        self.assertEqual(metadata['gripper_safety_mode'], 'none')

    def test_absolute_gripper_safety_waits_for_minimum_closure(self):
        actions = np.zeros((2, 8), dtype=np.float32)
        actions[:, 7] = 0.8

        unchanged, latched, metadata = _apply_absolute_gripper_safety(
            actions,
            current_gripper=0.4,
            control_force=5.0,
            desired_force=2.5,
            deadband=0.1,
            stop_margin=0.75,
            release_step=0.02,
            command_min=0.0,
            command_max=1.0,
            min_closure_position=0.55,
        )
        np.testing.assert_array_equal(unchanged, actions)
        self.assertFalse(latched)
        self.assertFalse(metadata['gripper_safety_closure_ready'])

        released, latched, metadata = _apply_absolute_gripper_safety(
            actions,
            current_gripper=0.55,
            control_force=5.0,
            desired_force=2.5,
            deadband=0.1,
            stop_margin=0.75,
            release_step=0.02,
            command_min=0.0,
            command_max=1.0,
            min_closure_position=0.55,
        )
        np.testing.assert_allclose(released[:, 7], 0.53)
        self.assertTrue(latched)
        self.assertTrue(metadata['gripper_safety_closure_ready'])
        self.assertEqual(metadata['gripper_safety_mode'], 'release')

    def test_gripper_only_selection_keeps_nominal_arm_trajectory(self):
        candidates = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)

        selected = _compose_selected_action(
            candidates,
            selected_idx=2,
            selection_scope='gripper',
            gripper_index=7,
        )

        np.testing.assert_array_equal(selected[:, :7], candidates[0, :, :7])
        np.testing.assert_array_equal(selected[:, 7], candidates[2, :, 7])

    def test_full_selection_returns_complete_selected_candidate(self):
        candidates = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
        selected = _compose_selected_action(
            candidates,
            selected_idx=2,
            selection_scope='full',
            gripper_index=7,
        )
        np.testing.assert_array_equal(selected, candidates[2])

    def test_scalar_and_band_cost(self):
        self.assertAlmostEqual(_force_target_score(3.0, 2.0), 1.0)
        self.assertAlmostEqual(
            _force_target_score(2.5, 2.0, force_min=2.0, force_max=3.0),
            0.0,
        )

    def test_absolute_force_target_preserves_requested_value(self):
        target, baseline = _resolve_tts_force_target(
            'absolute',
            requested_force=2.5,
            current_force=7.0,
        )
        self.assertAlmostEqual(target, 2.5)
        self.assertIsNone(baseline)

    def test_baseline_delta_target_removes_visual_force_offset(self):
        target, baseline = _resolve_tts_force_target(
            'baseline_delta',
            requested_force=0.2,
            current_force=2.8,
            baseline_force=2.5,
        )
        self.assertAlmostEqual(target, 2.7)
        self.assertAlmostEqual(baseline, 2.5)

    def test_action_refill_caps_queue_and_skips_overlapping_predictions(self):
        prediction = np.arange(8, dtype=np.float32)
        refill = _select_action_refill(
            prediction,
            queued_steps=2,
            buffer_steps=6,
            skip_steps=2,
            start_of_episode=False,
        )
        np.testing.assert_array_equal(refill, [4.0, 5.0, 6.0, 7.0])

    def test_first_action_chunk_fills_without_latency_skip(self):
        prediction = np.arange(8, dtype=np.float32)
        refill = _select_action_refill(
            prediction,
            queued_steps=0,
            buffer_steps=6,
            skip_steps=2,
            start_of_episode=True,
        )
        np.testing.assert_array_equal(refill, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(
            _force_target_score(
                4.0,
                2.0,
                force_min=2.0,
                force_max=3.0,
                high_force_weight=4.0,
            ),
            4.0,
        )

    def test_relative_closure_uses_current_gripper(self):
        action = np.zeros((4, 8), dtype=np.float32)
        action[:, 7] = [0.4, 0.5, 0.6, 0.3]
        positive, signed = _gripper_proxy_features(
            action,
            gripper_index=7,
            steps=4,
            closing_sign=1.0,
            current_gripper=0.4,
            mode='relative',
        )
        self.assertAlmostEqual(positive, 0.075, places=6)
        self.assertAlmostEqual(signed, 0.05, places=6)

    def test_safe_selection_rejects_lower_cost_unsafe_candidate(self):
        best, safe, all_unsafe = _select_force_candidate(
            scores=[1.0, 0.0, 0.25],
            proxy_forces=[2.0, 4.0, 2.5],
            safety_limit=3.0,
        )
        self.assertEqual(best, 2)
        self.assertEqual(safe.tolist(), [True, False, True])
        self.assertFalse(all_unsafe)

    def test_policy_force_trajectory_aggregation(self):
        trajectories = np.asarray([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ], dtype=np.float32)
        np.testing.assert_allclose(
            _policy_force_proxies(trajectories, 3, 'last'),
            [3.0, 2.0],
        )
        np.testing.assert_allclose(
            _policy_force_proxies(trajectories, 2, 'mean'),
            [1.5, 3.5],
        )
        np.testing.assert_allclose(
            _policy_force_proxies(trajectories, 4, 'max'),
            [4.0, 4.0],
        )

    def test_policy_force_dynamic_target_uses_candidate_base_and_cap(self):
        trajectories = np.asarray([
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0, 9.0],
        ], dtype=np.float32)
        target, base = _resolve_policy_force_dynamic_target(
            trajectories,
            rise=1.23,
            cap=12.0,
        )
        self.assertAlmostEqual(base, 4.0)
        self.assertAlmostEqual(target, 5.23)

        target, base = _resolve_policy_force_dynamic_target(
            trajectories,
            rise=10.0,
            cap=12.0,
        )
        self.assertAlmostEqual(base, 4.0)
        self.assertAlmostEqual(target, 12.0)

    def test_min_close_fallback_when_all_candidates_are_unsafe(self):
        best, safe, all_unsafe = _select_force_candidate(
            scores=[1.0, 0.0, 0.25],
            proxy_forces=[4.0, 5.0, 6.0],
            safety_limit=3.0,
            unsafe_fallback='min_close',
            signed_closure_means=[0.2, -0.1, 0.0],
        )
        self.assertEqual(best, 1)
        self.assertFalse(safe.any())
        self.assertTrue(all_unsafe)


if __name__ == '__main__':
    unittest.main()
