#!/usr/bin/env python3
"""Build a validated success-only training view of pivoting_cheng."""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import joblib


VIDEO_KEYS = ("base_image", "wrist_image")


def frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return -1
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def validate_stage(stage_dir: Path) -> tuple[bool, str, int]:
    data_path = stage_dir / "data.pkl"
    if not data_path.is_file():
        return False, "missing data.pkl", 0
    data = joblib.load(data_path)
    lengths = {
        "timestamps": len(data.get("timestamps", [])),
        "observations": len(data.get("observations", [])),
        "actions": len(data.get("actions", [])),
    }
    frames = lengths["timestamps"]
    if frames == 0:
        return False, "empty data.pkl", 0
    if set(lengths.values()) != {frames}:
        return False, f"pickle length mismatch: {lengths}", frames
    for key in VIDEO_KEYS:
        video_path = stage_dir / f"{key}.mp4"
        count = frame_count(video_path)
        if count != frames:
            return (
                False,
                f"{key}.mp4 has {count} frames; expected {frames}",
                frames,
            )
    return True, "ok", frames


def link_stage(stage_dir: Path, episode_dir: Path) -> None:
    episode_dir.mkdir(parents=True)
    for name in ("data.pkl", *[f"{key}.mp4" for key in VIDEO_KEYS]):
        (episode_dir / name).symlink_to((stage_dir / name).resolve())


def build_dataset(source: Path, output: Path, overwrite: bool) -> list[dict]:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True)

    rows = []
    selected = 0
    for demonstration in sorted(path for path in source.iterdir() if path.is_dir()):
        status = "failure" if demonstration.name.endswith("_fail") else "success"
        for stage_dir in sorted(demonstration.glob("stage*")):
            valid, reason, frames = validate_stage(stage_dir)
            include = status == "success" and valid
            episode = f"{demonstration.name}__{stage_dir.name}"
            if include:
                link_stage(stage_dir, episodes_dir / episode)
                selected += 1
            rows.append(
                {
                    "episode": episode,
                    "source_dir": str(stage_dir.resolve()),
                    "status": status,
                    "frames": frames,
                    "valid": str(valid).lower(),
                    "selected": str(include).lower(),
                    "reason": reason,
                }
            )

    if selected == 0:
        raise RuntimeError(f"No valid successful stages found under {source}")
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {selected} successful stages from {source}")
    print(f"Skipped {len(rows) - selected} failed or invalid stages")
    print(f"Saved {output}/manifest.csv")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/pivot/pivoting_cheng")
    parser.add_argument("--output", default="data/pivot/pivoting_cheng_success")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(Path(args.source), Path(args.output), args.overwrite)


if __name__ == "__main__":
    main()
