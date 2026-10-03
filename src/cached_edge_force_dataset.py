"""Manifest-backed cached edge datasets for VisualForce force regression.

The cache keeps provenance and split decisions at the episode level while
avoiding repeated video decoding and Sobel filtering during every epoch.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, WeightedRandomSampler

from src.dataset import (
    AUG_BRIGHTNESS_HI,
    AUG_BRIGHTNESS_LO,
    AUG_ROTATE_DEG,
    AUG_SCALE_HI,
    AUG_SCALE_LO,
    AUG_TRANSLATE,
    _random_occlude,
)
from src.steering import EDGE_NORMALIZATIONS, normalize_edge_tensor


SOURCE_SENSOR = 0
SOURCE_ZERO = 1


class CachedEdgeForceDataset(Dataset):
    """Read uint8 edge frames and force targets described by a manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        augment: bool = False,
        occlude: bool = True,
        edge_normalization: str = "none",
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open() as stream:
            manifest = json.load(stream)
        if manifest.get("format") != "visualforce_cached_edge_v1":
            raise ValueError(
                f"Unsupported manifest format: {manifest.get('format')!r}"
            )

        self.root = self.manifest_path.parent
        self.augment = augment
        self.occlude = occlude
        if edge_normalization not in EDGE_NORMALIZATIONS:
            raise ValueError(
                f"edge_normalization must be one of {EDGE_NORMALIZATIONS}, "
                f"got {edge_normalization!r}"
            )
        self.edge_normalization = edge_normalization
        self.entries = [
            entry for entry in manifest["entries"] if entry["split"] == split
        ]
        if not self.entries:
            raise ValueError(f"Manifest has no {split!r} entries")

        self._ends: list[int] = []
        self._source_ids: list[int] = []
        total = 0
        for entry in self.entries:
            count = int(entry["sample_count"])
            if count <= 0:
                raise ValueError(f"Non-positive sample_count in {entry['id']!r}")
            total += count
            self._ends.append(total)
            source_id = (
                SOURCE_SENSOR
                if entry["source_kind"] == "sensor"
                else SOURCE_ZERO
            )
            self._source_ids.extend([source_id] * count)
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

    def __len__(self) -> int:
        return self._ends[-1]

    @property
    def source_ids(self) -> np.ndarray:
        return np.asarray(self._source_ids, dtype=np.int8)

    def _entry_arrays(self, entry_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        arrays = self._arrays.get(entry_index)
        if arrays is not None:
            return arrays
        entry = self.entries[entry_index]
        edge_path = self.root / entry["edge_path"]
        edges = np.load(edge_path, mmap_mode="r")
        labels = None
        if entry.get("label_path"):
            labels = np.load(self.root / entry["label_path"], mmap_mode="r")
        if len(edges) != int(entry["sample_count"]):
            raise ValueError(f"Edge count changed for {entry['id']!r}")
        if labels is not None and len(labels) != len(edges):
            raise ValueError(f"Label count mismatch for {entry['id']!r}")
        arrays = edges, labels
        self._arrays[entry_index] = arrays
        return arrays

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry_index = bisect.bisect_right(self._ends, index)
        start = self._ends[entry_index - 1] if entry_index else 0
        local_index = index - start
        entry = self.entries[entry_index]
        edges, labels = self._entry_arrays(entry_index)

        edge = torch.from_numpy(np.array(edges[local_index], copy=True)).float()
        if edge.ndim == 2:
            edge = edge.unsqueeze(0)
        if edge.shape[0] != 1:
            raise ValueError(f"Expected one edge channel in {entry['id']!r}")
        edge /= 255.0
        edge = normalize_edge_tensor(edge, self.edge_normalization)

        if self.augment:
            height, width = edge.shape[-2:]
            max_tx = int(width * AUG_TRANSLATE)
            max_ty = int(height * AUG_TRANSLATE)
            translate = [
                int(torch.randint(-max_tx, max_tx + 1, ()).item()),
                int(torch.randint(-max_ty, max_ty + 1, ()).item()),
            ]
            angle = (torch.rand(()).item() * 2.0 - 1.0) * AUG_ROTATE_DEG
            scale = AUG_SCALE_LO + torch.rand(()).item() * (AUG_SCALE_HI - AUG_SCALE_LO)
            edge = TF.affine(
                edge,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0,
                fill=0,
            )
            brightness = (
                AUG_BRIGHTNESS_LO
                + torch.rand(()).item() * (AUG_BRIGHTNESS_HI - AUG_BRIGHTNESS_LO)
            )
            edge = (edge * brightness).clamp(0.0, 1.0)
            if self.occlude:
                edge = _random_occlude(edge)

        if labels is None:
            force = torch.tensor([float(entry["constant_label_n"])], dtype=torch.float32)
        else:
            force = torch.as_tensor(
                np.array(labels[local_index], copy=True), dtype=torch.float32
            ).reshape(-1)
        source_id = (
            SOURCE_SENSOR if entry["source_kind"] == "sensor" else SOURCE_ZERO
        )
        return {
            "frame": edge,
            "force": force,
            "source_id": torch.tensor(source_id, dtype=torch.int64),
        }


def make_balanced_sampler(
    dataset: CachedEdgeForceDataset,
    zero_fraction: float,
    seed: int,
) -> WeightedRandomSampler:
    """Sample a requested zero/sensor mixture without duplicating cache files."""
    if not 0.0 < zero_fraction < 1.0:
        raise ValueError("zero_fraction must be strictly between zero and one")
    source_ids = dataset.source_ids
    zero_count = int((source_ids == SOURCE_ZERO).sum())
    sensor_count = int((source_ids == SOURCE_SENSOR).sum())
    if zero_count == 0 or sensor_count == 0:
        raise ValueError("Balanced sampling requires both sensor and zero samples")
    weights = np.where(
        source_ids == SOURCE_ZERO,
        zero_fraction / zero_count,
        (1.0 - zero_fraction) / sensor_count,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
