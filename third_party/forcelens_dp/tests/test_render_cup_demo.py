import unittest

from render_cup_demo import ARX_OPEN_WIDTH_MM, GRAVITY_M_S2, DashboardData


class RenderCupDemoTest(unittest.TestCase):
    def test_dashboard_data_uses_calibrated_baselines_and_forward_fill(self):
        rows = [
            {
                'calibrated': 'False',
                'measured_closure': '0.40',
            },
            {
                'calibrated': 'True',
                'measured_closure': '0.42',
                'command_closure': '0.42',
                'baseline_force_n': '2.0',
                'filtered_force_n': '2.0',
                'filtered_added_load_n': '0.0',
                'force_limit_n': '5.0',
                'emergency_force_n': '5.5',
            },
            {
                'calibrated': 'True',
                'command_closure': '0.43',
                'filtered_force_n': '3.0',
                'filtered_added_load_n': '1.0',
            },
        ]

        values = DashboardData(rows).frame_values(2)

        self.assertAlmostEqual(values['load_n'], 1.0)
        self.assertAlmostEqual(values['water_g'], 1000.0 / GRAVITY_M_S2)
        self.assertAlmostEqual(values['measured_mm'], 0.0)
        self.assertAlmostEqual(values['command_mm'], 0.01 * ARX_OPEN_WIDTH_MM)
        self.assertAlmostEqual(values['force_rise'], 1.0)
        self.assertAlmostEqual(values['force_limit_rise'], 3.0)
        self.assertAlmostEqual(values['emergency_rise'], 3.5)


if __name__ == '__main__':
    unittest.main()
