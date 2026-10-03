import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from tools.coke.diagnose_coke_gripper import (
    analyze_rollout,
    training_gripper_summary,
)


class TestCokeGripperDiagnosis(unittest.TestCase):
    def _episode(self, root, name, source, command_max, observed_max):
        episode = root / "episodes" / name
        episode.mkdir(parents=True)
        commands = np.linspace(0.0, command_max, 5)
        observations = np.linspace(0.0, observed_max, 5)
        joblib.dump(
            {
                "actions": [
                    {"gripper_pos": np.asarray([value])} for value in commands
                ],
                "observations": [
                    {"gripper_pos": np.asarray([value])} for value in observations
                ],
            },
            episode / "data.pkl",
        )
        return {"episode": name, "source_tag": source, "selected": "true"}

    def test_training_threshold_is_derived_from_hard_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            rows = [
                self._episode(dataset, "soft_one", "soft", 0.3, 0.25),
                self._episode(dataset, "hard_one", "hard", 0.8, 0.7),
                self._episode(dataset, "hard_two", "hard", 1.0, 0.9),
            ]
            with (dataset / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("episode", "source_tag", "selected")
                )
                writer.writeheader()
                writer.writerows(rows)
            summary = training_gripper_summary(dataset)

        self.assertAlmostEqual(
            summary["sources"]["hard"]["episode_action_max"]["q25"], 0.85
        )

    def test_rollout_separates_weak_policy_from_tracking_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weak = root / "weak.csv"
            with weak.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "base_gripper_cmd",
                        "steered_gripper_cmd",
                        "obs_gripper",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "base_gripper_cmd": 0.1,
                            "steered_gripper_cmd": 0.1,
                            "obs_gripper": 0.09,
                        }
                    ]
                )
            result = analyze_rollout(weak, 0.25, 0.75, 0.1)
            self.assertIn(
                "raw_policy_never_reached_contact_closure", result["findings"]
            )
            self.assertNotIn(
                "gripper_did_not_track_executed_command", result["findings"]
            )


if __name__ == "__main__":
    unittest.main()
