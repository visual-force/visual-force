import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "plug"
    / "build_plug_insertion_aug_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("plug_dataset_builder", SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def make_stage(stage_dir: Path, frames: int = 2) -> None:
    stage_dir.mkdir(parents=True)
    record = {
        "arm_pos": np.zeros(3),
        "arm_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "gripper_pos": np.zeros(1),
    }
    joblib.dump(
        {
            "timestamps": list(range(frames)),
            "observations": [record.copy() for _ in range(frames)],
            "actions": [record.copy() for _ in range(frames)],
        },
        stage_dir / "data.pkl",
    )
    for key in BUILDER.VIDEO_KEYS:
        (stage_dir / f"{key}.mp4").touch()


class TestPlugDatasetBuilder(unittest.TestCase):
    def test_default_view_keeps_valid_success_and_failure_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            for demonstration in ("demo", "demo_fail"):
                for stage in ("stage1", "stage2"):
                    make_stage(source / demonstration / stage)

            output = root / "processed"
            with mock.patch.object(BUILDER, "frame_count", return_value=2):
                rows = BUILDER.build_dataset(source, output, overwrite=False)

            self.assertEqual(sum(row["selected"] == "true" for row in rows), 4)
            self.assertEqual(len(list((output / "episodes").iterdir())), 4)
            self.assertTrue(
                (output / "episodes" / "demo_fail__stage2" / "data.pkl").is_symlink()
            )
            with (output / "manifest.csv").open() as stream:
                manifest = list(csv.DictReader(stream))
            self.assertEqual(
                {row["demonstration_status"] for row in manifest},
                {"success", "failure"},
            )

    def test_success_only_view_records_excluded_failure_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            make_stage(source / "demo" / "stage1")
            make_stage(source / "demo_fail" / "stage1")

            output = root / "processed"
            with mock.patch.object(BUILDER, "frame_count", return_value=2):
                rows = BUILDER.build_dataset(
                    source,
                    output,
                    overwrite=False,
                    success_only=True,
                )

            failure = next(row for row in rows if row["demonstration_status"] == "failure")
            self.assertEqual(failure["selected"], "false")
            self.assertEqual(failure["reason"], "excluded by --success-only")
            self.assertEqual(len(list((output / "episodes").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
