import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.build_real_zero_force_dataset import episode_split_group, stable_split
from src.cached_edge_force_dataset import (
    SOURCE_SENSOR,
    SOURCE_ZERO,
    CachedEdgeForceDataset,
    make_balanced_sampler,
)
from src.steering import normalize_edge_tensor


class CachedEdgeForceDatasetTest(unittest.TestCase):
    def test_manifest_dataset_and_balanced_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "sensor_edges.npy", np.full((4, 8, 8), 128, np.uint8))
            np.save(root / "sensor_labels.npy", np.arange(4, dtype=np.float32)[:, None])
            np.save(root / "zero_edges.npy", np.zeros((2, 8, 8), np.uint8))
            manifest = {
                "format": "visualforce_cached_edge_v1",
                "entries": [
                    {
                        "id": "sensor", "split": "train", "source_kind": "sensor",
                        "edge_path": "sensor_edges.npy", "label_path": "sensor_labels.npy",
                        "sample_count": 4,
                    },
                    {
                        "id": "zero", "split": "train", "source_kind": "zero",
                        "edge_path": "zero_edges.npy", "label_path": None,
                        "constant_label_n": 0.0, "sample_count": 2,
                    },
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            dataset = CachedEdgeForceDataset(path, "train")
            self.assertEqual(len(dataset), 6)
            self.assertEqual(dataset[0]["source_id"].item(), SOURCE_SENSOR)
            self.assertEqual(dataset[4]["source_id"].item(), SOURCE_ZERO)
            self.assertEqual(dataset[3]["force"].item(), 3.0)
            self.assertEqual(dataset[5]["force"].item(), 0.0)
            sampler = make_balanced_sampler(dataset, zero_fraction=0.5, seed=7)
            self.assertEqual(len(list(sampler)), len(dataset))

    def test_stable_split_is_deterministic(self):
        self.assertEqual(stable_split("episode-a", 5), stable_split("episode-a", 5))
        self.assertIn(stable_split("episode-b", 5), {"train", "val"})

    def test_stage_directories_share_episode_split_group(self):
        parent = Path("collection/episode")
        self.assertEqual(episode_split_group(parent / "stage1"), parent)
        self.assertEqual(episode_split_group(parent / "stage2"), parent)
        self.assertEqual(episode_split_group(parent), parent)

    def test_nonzero_p95_edge_normalization_preserves_background(self):
        edge = torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0]]])
        normalized = normalize_edge_tensor(edge, "nonzero_p95")
        self.assertEqual(normalized[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(normalized.max().item(), 1.0)
        self.assertGreater(normalized[0, 0, 1].item(), edge[0, 0, 1].item())

    def test_finger_bbox_normalization_standardizes_scale(self):
        small = torch.zeros((1, 256, 256), dtype=torch.float32)
        small[:, 70:150, 35:55] = 0.4
        small[:, 75:155, 200:220] = 0.4
        large = torch.zeros_like(small)
        large[:, 30:190, 10:50] = 0.8
        large[:, 40:200, 180:220] = 0.8
        small_out = normalize_edge_tensor(small, "finger_bbox_p95")
        large_out = normalize_edge_tensor(large, "finger_bbox_p95")
        self.assertEqual(tuple(small_out.shape), (1, 256, 256))
        self.assertEqual(tuple(large_out.shape), (1, 256, 256))
        self.assertAlmostEqual(float(small_out.max()), 1.0, places=5)
        self.assertAlmostEqual(float(large_out.max()), 1.0, places=5)
        self.assertEqual(int((small_out > 0.5).sum()), int((large_out > 0.5).sum()))


if __name__ == "__main__":
    unittest.main()
