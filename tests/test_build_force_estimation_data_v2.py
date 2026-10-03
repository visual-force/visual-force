import os
import tempfile
import unittest
from pathlib import Path

from scripts.build_force_estimation_data_v2 import (
    episode_signature,
    parse_source,
    relative_symlink,
)


class BuildForceEstimationDataV2Test(unittest.TestCase):
    def test_parse_source(self):
        self.assertEqual(parse_source("new=/tmp/data"), ("new", Path("/tmp/data")))
        with self.assertRaises(Exception):
            parse_source("missing-separator")

    def test_episode_signature_changes_with_required_content(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            for name in ("video.mp4", "frame_timestamps.csv", "force_timestamps.csv", "meta.json"):
                (episode / name).write_text(name)
            before = episode_signature(episode)
            (episode / "meta.json").write_text("changed")
            self.assertNotEqual(before, episode_signature(episode))

    def test_relative_symlink_is_idempotent_and_rejects_wrong_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("data")
            destination = root / "view" / "source.txt"
            destination.parent.mkdir()

            relative_symlink(source, destination)
            relative_symlink(source, destination)
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_text(), "data")
            self.assertFalse(os.path.isabs(os.readlink(destination)))

            wrong_source = root / "wrong.txt"
            wrong_source.write_text("wrong")
            with self.assertRaises(RuntimeError):
                relative_symlink(wrong_source, destination)


if __name__ == "__main__":
    unittest.main()
