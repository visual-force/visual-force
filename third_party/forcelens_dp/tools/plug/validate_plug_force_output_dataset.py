#!/usr/bin/env python3
"""Validate per-frame pseudo-force labels in a processed plug dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np


def validate_dataset(dataset: Path, label_name: str, force_key: str) -> None:
    episodes_dir = dataset / "episodes"
    manifest = dataset / "manifest.csv"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Missing episodes directory: {episodes_dir}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    with manifest.open() as stream:
        selected_rows = [
            row
            for row in csv.DictReader(stream)
            if row["selected"] == "true"
        ]
    episode_dirs = sorted(path for path in episodes_dir.iterdir() if path.is_dir())
    manifest_names = sorted(row["episode"] for row in selected_rows)
    episode_names = [path.name for path in episode_dirs]
    if manifest_names != episode_names:
        raise ValueError("Selected manifest rows do not match generated episodes")
    if not episode_dirs:
        raise RuntimeError(f"No episodes found under {episodes_dir}")

    all_force = []
    total_frames = 0
    status_counts: dict[str, int] = {}
    rows_by_name = {row["episode"]: row for row in selected_rows}
    for episode_dir in episode_dirs:
        data = joblib.load(episode_dir / "data.pkl")
        frames = len(data["timestamps"])
        label_path = episode_dir / label_name
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing force label: {label_path}")
        with np.load(label_path, allow_pickle=False) as labels:
            force_keys = [str(key) for key in labels["force_keys"].tolist()]
            if force_key not in force_keys:
                raise KeyError(
                    f"{label_path} does not contain {force_key}; keys={force_keys}"
                )
            force = np.asarray(
                labels["force"][:, force_keys.index(force_key)],
                dtype=np.float32,
            )
            if force.shape != (frames,):
                raise ValueError(
                    f"Force length mismatch in {episode_dir}: "
                    f"{force.shape}, expected ({frames},)"
                )
            if not np.isfinite(force).all():
                raise ValueError(f"Non-finite force labels in {label_path}")
            if "mask_frac" in labels:
                mask_frac = np.asarray(labels["mask_frac"])
                if mask_frac.shape != (frames,) or not np.isfinite(mask_frac).all():
                    raise ValueError(f"Invalid mask fractions in {label_path}")
        all_force.append(force)
        total_frames += frames
        status = rows_by_name[episode_dir.name]["demonstration_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    force = np.concatenate(all_force)
    magnitude = np.abs(force)
    print("Plug force-output dataset validation:")
    print(f"  episodes: {len(episode_dirs)}")
    print(f"  frames:   {total_frames}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(
        "  |Fz|:     "
        f"min={magnitude.min():.4f}, median={np.median(magnitude):.4f}, "
        f"max={magnitude.max():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/insert/plug_insertion_aug_force_output",
    )
    parser.add_argument("--label-name", default="visualforce_pseudo_force_fz.npz")
    parser.add_argument("--force-key", default="Fz")
    args = parser.parse_args()
    validate_dataset(Path(args.dataset), args.label_name, args.force_key)


if __name__ == "__main__":
    main()
