import unittest

import numpy as np

from src.current_force_baseline import (
    RidgeRegressor,
    current_features,
    exponential_moving_average,
    regression_metrics,
    torque_to_current,
)


class CurrentForceBaselineTest(unittest.TestCase):
    def test_arx_gripper_torque_recovers_current(self):
        torque = np.array([-0.848, 0.0, 0.424])
        np.testing.assert_allclose(torque_to_current(torque), [-2.0, 0.0, 1.0])

    def test_current_only_ridge_recovers_affine_force(self):
        current = np.linspace(-2.0, 3.0, 100)
        target = 1.7 * current - 0.4
        features, names = current_features(current)
        model = RidgeRegressor.fit(features, target, names, alpha=0.0)
        np.testing.assert_allclose(model.predict(features), target, atol=1e-10)

    def test_state_aware_features_capture_width_dependence(self):
        current = np.linspace(0.1, 2.0, 50)
        position = np.linspace(0.2, 0.8, 50)
        velocity = np.where(np.arange(50) % 2, -0.1, 0.1)
        features, names = current_features(
            current, position, velocity, state_aware=True
        )
        self.assertEqual(features.shape, (50, 8))
        self.assertEqual(len(names), 8)
        self.assertTrue(np.allclose(features[:, 4], current * position))

    def test_ema_is_causal_and_identity_at_zero(self):
        values = np.array([0.0, 1.0, 1.0])
        timestamps = np.array([0.0, 0.1, 0.2])
        np.testing.assert_array_equal(
            exponential_moving_average(values, timestamps, 0.0), values
        )
        filtered = exponential_moving_average(values, timestamps, 0.1)
        self.assertEqual(filtered[0], 0.0)
        self.assertGreater(filtered[1], 0.0)
        self.assertLess(filtered[1], 1.0)
        self.assertGreater(filtered[2], filtered[1])

    def test_metrics_identify_perfect_prediction(self):
        target = np.array([0.0, 1.0, 2.0])
        metrics = regression_metrics(target, target, np.array([0.0, 0.1, 0.2]))
        self.assertEqual(metrics["mae_n"], 0.0)
        self.assertEqual(metrics["rmse_n"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)
        self.assertEqual(metrics["delta_residual_rms_n"], 0.0)


if __name__ == "__main__":
    unittest.main()
