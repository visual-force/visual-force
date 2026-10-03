import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module(
    "flip_force_output_builder",
    ROOT / "tools" / "flip" / "build_flip_force_output_dataset.py",
)
VALIDATOR = load_module(
    "flip_force_output_validator",
    ROOT / "tools" / "flip" / "validate_flip_force_output_dataset.py",
)


def make_stage(
    stage_dir: Path,
    checkpoint: Path,
    *,
    frames: int = 2,
    with_label: bool = True,
) -> None:
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
    if with_label:
        np.savez_compressed(
            stage_dir / BUILDER.DEFAULT_LABEL_NAME,
            force=np.arange(frames, dtype=np.float32)[:, None] - 1.0,
            force_keys=np.asarray(["Fz"]),
            mask_frac=np.full(frames, 0.1, dtype=np.float32),
            visualforce_ckpt=str(checkpoint.resolve()),
            input_mode="edge",
            mask_mode="sam2",
        )


class TestFlipForceOutputDataset(unittest.TestCase):
    def test_builder_selects_valid_stage2_and_copies_cached_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            checkpoint = root / "visualforce.pt"
            checkpoint.touch()
            make_stage(source / "success" / "stage2", checkpoint)
            make_stage(source / "failure_fail" / "stage2", checkpoint)
            make_stage(source / "stage1_only" / "stage1", checkpoint)
            make_stage(source / "broken" / "stage2", checkpoint)
            (source / "broken" / "stage2" / "wrist_image.mp4").unlink()

            output = root / "processed"
            count = lambda path: 2 if path.is_file() else -1
            with mock.patch.object(BUILDER, "frame_count", side_effect=count):
                rows = BUILDER.build_dataset(source, output, overwrite=False)

            selected = [row for row in rows if row["selected"] == "true"]
            self.assertEqual({row["episode"] for row in selected}, {"success", "failure_fail"})
            copied = output / "episodes" / "success" / BUILDER.DEFAULT_LABEL_NAME
            self.assertTrue(copied.is_file())
            self.assertFalse(copied.is_symlink())
            self.assertTrue((output / "episodes" / "success" / "data.pkl").is_symlink())

            with (output / "manifest.csv").open() as stream:
                manifest = list(csv.DictReader(stream))
            missing_stage = next(row for row in manifest if row["episode"] == "stage1_only")
            broken = next(row for row in manifest if row["episode"] == "broken")
            self.assertEqual(missing_stage["reason"], "missing stage2")
            self.assertIn("wrist_image.mp4 has -1 frames", broken["reason"])

            summary = VALIDATOR.validate_dataset(
                output,
                label_name=BUILDER.DEFAULT_LABEL_NAME,
                force_key="Fz",
                expected_checkpoint=checkpoint,
                expected_mask_mode="sam2",
            )
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["frames"], 4)
            self.assertEqual(summary["status_counts"], {"failure": 1, "success": 1})

    def test_builder_keeps_unlabelled_valid_stage_for_later_labelling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            checkpoint = root / "visualforce.pt"
            make_stage(
                source / "demo" / "stage2",
                checkpoint,
                with_label=False,
            )

            output = root / "processed"
            with mock.patch.object(BUILDER, "frame_count", return_value=2):
                rows = BUILDER.build_dataset(source, output, overwrite=False)

            self.assertEqual(rows[0]["selected"], "true")
            self.assertEqual(rows[0]["cached_label"], "false")
            self.assertFalse(
                (output / "episodes" / "demo" / BUILDER.DEFAULT_LABEL_NAME).exists()
            )
            with self.assertRaises(FileNotFoundError):
                VALIDATOR.validate_dataset(
                    output,
                    label_name=BUILDER.DEFAULT_LABEL_NAME,
                    force_key="Fz",
                )

    def test_builder_physically_copies_and_remaps_only_gripper_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            checkpoint = root / "visualforce.pt"
            checkpoint.touch()
            make_stage(source / "demo" / "stage2", checkpoint)
            source_data_path = source / "demo" / "stage2" / "data.pkl"
            source_data = joblib.load(source_data_path)
            source_data["observations"][0]["gripper_pos"] = np.asarray([0.538])
            source_data["actions"][0]["gripper_pos"] = np.asarray([0.538])
            source_data["actions"][1]["gripper_pos"] = np.asarray([0.570])
            joblib.dump(source_data, source_data_path)

            output = root / "processed"
            with mock.patch.object(BUILDER, "frame_count", return_value=2):
                rows = BUILDER.build_dataset(
                    source,
                    output,
                    overwrite=False,
                    materialization="copy",
                    gripper_action_mapping=BUILDER.LEGACY_TO_CURRENT_GRIPPER_MAPPING,
                )

            episode = output / "episodes" / "demo"
            self.assertFalse((episode / "data.pkl").is_symlink())
            self.assertFalse((episode / "base_image.mp4").is_symlink())
            processed = joblib.load(episode / "data.pkl")
            unchanged_source = joblib.load(source_data_path)
            self.assertEqual(
                processed["observations"][0]["gripper_pos"][0],
                unchanged_source["observations"][0]["gripper_pos"][0],
            )
            self.assertAlmostEqual(processed["actions"][0]["gripper_pos"][0], 0.612, places=3)
            self.assertAlmostEqual(processed["actions"][1]["gripper_pos"][0], 0.664, places=3)
            self.assertEqual(
                rows[0]["gripper_action_mapping"],
                BUILDER.LEGACY_TO_CURRENT_GRIPPER_MAPPING,
            )

            summary = VALIDATOR.validate_dataset(
                output,
                label_name=BUILDER.DEFAULT_LABEL_NAME,
                force_key="Fz",
                expected_checkpoint=checkpoint,
                expected_mask_mode="sam2",
                expected_gripper_action_mapping=(
                    BUILDER.LEGACY_TO_CURRENT_GRIPPER_MAPPING
                ),
            )
            self.assertEqual(
                summary["gripper_action_mappings"],
                {BUILDER.LEGACY_TO_CURRENT_GRIPPER_MAPPING: 1},
            )


if __name__ == "__main__":
    unittest.main()
