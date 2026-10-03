#!/usr/bin/env python3
"""Build a validated, manifest-backed mixed Coke training dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import joblib
import numpy as np


VIDEO_KEYS = ("base_image", "wrist_image")
STATE_KEYS = {
    "arm_pos": 3,
    "arm_quat": 4,
    "gripper_pos": 1,
}


def frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return -1
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def validate_vector_records(records, record_name: str) -> str | None:
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return f"{record_name}[{index}] is not a dictionary"
        for key, expected_size in STATE_KEYS.items():
            if key not in record:
                return f"{record_name}[{index}] is missing {key}"
            value = np.asarray(record[key])
            if value.size != expected_size:
                return (
                    f"{record_name}[{index}].{key} has {value.size} values; "
                    f"expected {expected_size}"
                )
            if not np.isfinite(value).all():
                return f"{record_name}[{index}].{key} contains non-finite values"
    return None


def validate_episode(episode_dir: Path) -> tuple[bool, str, int]:
    data_path = episode_dir / "data.pkl"
    if not data_path.is_file():
        return False, "missing data.pkl", 0
    try:
        data = joblib.load(data_path)
    except Exception as exc:
        return False, f"cannot load data.pkl: {exc}", 0
    if not isinstance(data, dict):
        return False, "data.pkl is not a dictionary", 0

    required = ("timestamps", "observations", "actions")
    missing = [key for key in required if key not in data]
    if missing:
        return False, f"data.pkl is missing keys: {missing}", 0

    lengths = {key: len(data[key]) for key in required}
    frames = lengths["timestamps"]
    if frames == 0:
        return False, "empty data.pkl", 0
    if set(lengths.values()) != {frames}:
        return False, f"pickle length mismatch: {lengths}", frames

    for key in ("observations", "actions"):
        reason = validate_vector_records(data[key], key)
        if reason is not None:
            return False, reason, frames

    for key in VIDEO_KEYS:
        video_path = episode_dir / f"{key}.mp4"
        count = frame_count(video_path)
        if count != frames:
            return (
                False,
                f"{key}.mp4 has {count} frames; expected {frames}",
                frames,
            )
    return True, "ok", frames


def link_episode(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in ("data.pkl", *[f"{key}.mp4" for key in VIDEO_KEYS]):
        (destination / name).symlink_to((source / name).resolve())


def build_dataset(
    sources: list[tuple[str, Path]],
    output: Path,
    overwrite: bool,
) -> list[dict[str, str | int]]:
    resolved_output = output.resolve()
    for _, source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"Raw dataset not found: {source}")
        if source.resolve() == resolved_output:
            raise ValueError("Output dataset must differ from each raw dataset")

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True)

    rows: list[dict[str, str | int]] = []
    selected_by_source: dict[str, int] = {}
    for source_tag, source_root in sources:
        selected_by_source[source_tag] = 0
        episode_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
        for episode_dir in episode_dirs:
            valid, reason, frames = validate_episode(episode_dir)
            episode_name = f"{source_tag}__{episode_dir.name}"
            if valid:
                link_episode(episode_dir, episodes_dir / episode_name)
                selected_by_source[source_tag] += 1
            rows.append(
                {
                    "episode": episode_name,
                    "source_tag": source_tag,
                    "source_dir": str(episode_dir.resolve()),
                    "frames": frames,
                    "valid": str(valid).lower(),
                    "selected": str(valid).lower(),
                    "reason": reason,
                }
            )

    empty_sources = [tag for tag, count in selected_by_source.items() if count == 0]
    if empty_sources:
        shutil.rmtree(output)
        raise RuntimeError(f"No valid episodes found for sources: {empty_sources}")

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = sum(selected_by_source.values())
    print(f"Built {output} with {selected} episodes")
    for source_tag, count in selected_by_source.items():
        print(f"  {source_tag}: {count}")
    print(f"  skipped invalid: {len(rows) - selected}")
    print(f"  manifest: {manifest}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-root", default="data/pick/coke_soft_0821")
    parser.add_argument("--hard-root", default="data/pick/coke_hard_0821")
    parser.add_argument("--output", default="data/pick/coke_mixed_0821")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_dataset(
        [
            ("soft", Path(args.soft_root)),
            ("hard", Path(args.hard_root)),
        ],
        Path(args.output),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
