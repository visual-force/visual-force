import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from tools.coke.calibrate_coke_tts import build_report


class TestCokeTtsCalibration(unittest.TestCase):
    def make_episode(self, root, name, source, max_gripper, plateau_force):
        episode = root / "episodes" / name
        episode.mkdir(parents=True)
        gripper = np.asarray(
            [0.02] * 10 + [0.10, 0.20, max_gripper, max_gripper, max_gripper]
        )
        observations = [
            {"gripper_pos": np.asarray([position], dtype=np.float32)}
            for position in gripper
        ]
        joblib.dump({"observations": observations}, episode / "data.pkl")
        force = np.asarray(
            [2.0] * 10 + [2.0, 2.2, plateau_force, plateau_force, plateau_force],
            dtype=np.float32,
        )[:, None]
        np.savez(
            episode / "visualforce_pseudo_force_fz.npz",
            force=force,
            force_keys=np.asarray(["Fz"]),
            visualforce_ckpt=np.asarray("checkpoint.pt"),
            input_mode=np.asarray("edge"),
            mask_mode=np.asarray("sam2"),
        )
        return {
            "episode": name,
            "source_tag": source,
            "selected": "true",
        }

    def test_build_report_derives_baseline_relative_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            rows = [
                self.make_episode(dataset, "soft__one", "soft", 0.30, 3.0),
                self.make_episode(dataset, "soft__two", "soft", 0.40, 3.2),
                self.make_episode(dataset, "hard__one", "hard", 0.90, 12.0),
            ]
            with (dataset / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("episode", "source_tag", "selected"),
                )
                writer.writeheader()
                writer.writerows(rows)

            report = build_report(
                argparse.Namespace(
                    dataset=str(dataset),
                    label_name="visualforce_pseudo_force_fz.npz",
                    force_key="Fz",
                    baseline_samples=10,
                    baseline_max_position=0.10,
                    filter_window=3,
                    plateau_tolerance=0.05,
                    target_noise_clearance=0.10,
                    force_quantum=0.10,
                    position_quantum=0.05,
                )
            )

        recommended = report["recommended"]
        self.assertAlmostEqual(recommended["desired_force_delta_n"], 1.1)
        self.assertAlmostEqual(recommended["deadband_n"], 0.1)
        self.assertAlmostEqual(recommended["contact_ready_position"], 0.25)
        self.assertAlmostEqual(recommended["maximum_gripper_position"], 0.40)
        self.assertEqual(recommended["baseline_samples"], 10)
        self.assertEqual(recommended["sampling_candidates"], 32)
        self.assertEqual(report["label_metadata"]["input_mode"], "edge")


if __name__ == "__main__":
    unittest.main()
