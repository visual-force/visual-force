import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "visualforce"


def run_cli(*args, check=True, env=None):
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    return subprocess.run(
        [str(CLI), *args],
        cwd=REPO_ROOT,
        env=command_env,
        check=check,
        capture_output=True,
        text=True,
    )


class VisualForceCliTests(unittest.TestCase):
    def test_help_exposes_estimator_and_critic_workflows(self):
        result = run_cli("--help")
        self.assertIn("estimator train", result.stdout)
        self.assertIn("critic berry", result.stdout)
        self.assertIn("critic flip", result.stdout)
        self.assertIn("critic coke", result.stdout)

    def test_estimator_command_forwards_arguments(self):
        result = run_cli(
            "estimator",
            "train",
            "--data-dir",
            "example-data",
            env={"PYTHON_BIN": "/bin/echo"},
        )
        self.assertIn("scripts/train.py --data-dir example-data", result.stdout)

    def test_flip_profiles_resolve_distinct_training_configuration(self):
        stage2 = run_cli("critic", "flip", "stage2", "dry-run", "max_delta")
        self.assertIn("--target-mode max_delta", stage2.stdout)
        self.assertIn("--cum-loss-weight 0.0", stage2.stdout)
        self.assertIn("flip_chunk_force_obj4_stage2_max_delta_v1", stage2.stdout)

        delta = run_cli("critic", "flip", "stage2_delta", "dry-run")
        self.assertIn("--target-mode delta_trajectory", delta.stdout)
        self.assertIn("--cum-loss-weight 1.0", delta.stdout)
        self.assertIn("flip_delta_force_obj4_stage2_v2_dp_obs", delta.stdout)

        dynamics = run_cli("critic", "flip", "dynamics", "dry-run")
        self.assertIn("--device cuda:0", dynamics.stdout)
        self.assertIn("--no-wandb", dynamics.stdout)

    def test_berry_dry_run_does_not_build_or_label(self):
        result = run_cli("critic", "berry", "action_only", "dry-run")
        self.assertIn("--critic-image-mode none", result.stdout)
        self.assertIn("--target-mode future_peak", result.stdout)
        self.assertIn("--include-current-force", result.stdout)

    def test_unknown_profile_fails_with_actionable_error(self):
        result = run_cli("critic", "flip", "unknown", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown Flip critic profile", result.stderr)


if __name__ == "__main__":
    unittest.main()
