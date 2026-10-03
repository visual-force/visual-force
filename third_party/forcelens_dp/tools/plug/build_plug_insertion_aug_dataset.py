#!/usr/bin/env python3
"""Build a validated, manifest-backed plug-insertion training view."""

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


def validate_stage(stage_dir: Path) -> tuple[bool, str, int]:
    data_path = stage_dir / "data.pkl"
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
        video_path = stage_dir / f"{key}.mp4"
        count = frame_count(video_path)
        if count != frames:
            return False, f"{key}.mp4 has {count} frames; expected {frames}", frames
    return True, "ok", frames


def link_stage(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in ("data.pkl", *[f"{key}.mp4" for key in VIDEO_KEYS]):
        (destination / name).symlink_to((source / name).resolve())


def build_dataset(
    source: Path,
    output: Path,
    overwrite: bool,
    success_only: bool = False,
) -> list[dict[str, str | int]]:
    if not source.is_dir():
        raise FileNotFoundError(f"Raw dataset not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Output dataset must differ from the raw dataset")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)

    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True)
    rows: list[dict[str, str | int]] = []
    selected = 0
    selected_frames = 0
    demonstrations = sorted(path for path in source.iterdir() if path.is_dir())
    for demonstration in demonstrations:
        status = "failure" if demonstration.name.endswith("_fail") else "success"
        for stage_dir in sorted(demonstration.glob("stage*")):
            valid, reason, frames = validate_stage(stage_dir)
            include = valid and (status == "success" or not success_only)
            episode = f"{demonstration.name}__{stage_dir.name}"
            if include:
                link_stage(stage_dir, episodes_dir / episode)
                selected += 1
                selected_frames += frames
            rows.append(
                {
                    "episode": episode,
                    "source_dir": str(stage_dir.resolve()),
                    "demonstration_status": status,
                    "stage": stage_dir.name,
                    "frames": frames,
                    "valid": str(valid).lower(),
                    "selected": str(include).lower(),
                    "reason": (
                        reason
                        if not valid
                        else "ok" if include else "excluded by --success-only"
                    ),
                }
            )

    if not rows:
        shutil.rmtree(output)
        raise RuntimeError(f"No stage directories found under {source}")
    if selected == 0:
        shutil.rmtree(output)
        raise RuntimeError(f"No valid stages selected under {source}")

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    status_counts = {
        status: sum(
            row["selected"] == "true" and row["demonstration_status"] == status
            for row in rows
        )
        for status in ("success", "failure")
    }
    print(f"Built {output} with {selected} stages ({selected_frames} frames)")
    print(f"  successful stages: {status_counts['success']}")
    print(f"  failed-demo stages: {status_counts['failure']}")
    print(f"  skipped: {len(rows) - selected}")
    print(f"  manifest: {manifest}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/plug_insertion_aug")
    parser.add_argument(
        "--output",
        default="data/insert/plug_insertion_aug_force_output",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Exclude demonstrations whose directory name ends in _fail.",
    )
    args = parser.parse_args()
    build_dataset(
        Path(args.source),
        Path(args.output),
        overwrite=args.overwrite,
        success_only=args.success_only,
    )


if __name__ == "__main__":
    main()
