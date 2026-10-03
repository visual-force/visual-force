#!/usr/bin/env python3
"""Build the deterministic raw episode view for force-estimation data v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import av


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data/force_estimation_data_v2"
DEFAULT_SOURCES = (
    ("v0", REPO_ROOT / "data/episodes"),
    ("v1", REPO_ROOT / "data/data_ball_260422"),
    ("mustafa_v1", REPO_ROOT / "data/Mustafa_eps_v1"),
)
REQUIRED_FILES = (
    "video.mp4",
    "frame_timestamps.csv",
    "force_timestamps.csv",
    "meta.json",
)
OPTIONAL_FILES = ("mask.mp4", "overlay.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Repeat to override the default ordered source collections.",
    )
    args = parser.parse_args()
    args.sources = tuple(parse_source(value) for value in args.source) if args.source else DEFAULT_SOURCES
    return args


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("source must have the form LABEL=PATH")
    return label, Path(path).expanduser()


def count_csv_rows(path: Path) -> int:
    with path.open(newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def count_video_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def relative_symlink(source: Path, destination: Path) -> None:
    expected = os.path.relpath(source.resolve(), destination.parent.resolve())
    if destination.is_symlink():
        if os.readlink(destination) != expected:
            raise RuntimeError(
                f"Existing symlink has the wrong target: {destination} -> "
                f"{os.readlink(destination)} (expected {expected})"
            )
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(expected)


def discover_sources(sources: tuple[tuple[str, Path], ...]) -> list[tuple[str, Path, Path]]:
    episodes = []
    for version, root in sources:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {version} source directory: {root}")
        for episode in sorted(root.glob("EP*")):
            if episode.is_dir():
                episodes.append((version, root, episode))
    return episodes


def episode_signature(episode: Path) -> str:
    digest = hashlib.sha256()
    for name in REQUIRED_FILES:
        digest.update(name.encode())
        with (episode / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    duplicates = []
    seen_signatures: dict[str, dict] = {}
    for version, source_root, source_episode in discover_sources(args.sources):
        missing = [name for name in REQUIRED_FILES if not (source_episode / name).is_file()]
        if missing:
            raise RuntimeError(f"{source_episode} is missing: {', '.join(missing)}")

        signature = episode_signature(source_episode)
        if signature in seen_signatures:
            duplicates.append(
                {
                    "source_version": version,
                    "source_episode": str(source_episode.resolve()),
                    "duplicate_of": seen_signatures[signature]["episode"],
                    "duplicate_source_episode": seen_signatures[signature]["source_episode"],
                    "sha256": signature,
                }
            )
            continue

        index = len(entries) + 1
        target_name = f"EP{index:06d}"
        target_episode = output_dir / target_name
        target_episode.mkdir(exist_ok=True)
        linked_files = []
        for name in REQUIRED_FILES + OPTIONAL_FILES:
            source_file = source_episode / name
            if source_file.is_file():
                relative_symlink(source_file, target_episode / name)
                linked_files.append(name)

        video_frames = count_video_frames(source_episode / "video.mp4")
        frame_rows = count_csv_rows(source_episode / "frame_timestamps.csv")
        force_rows = count_csv_rows(source_episode / "force_timestamps.csv")
        if video_frames != frame_rows:
            raise RuntimeError(
                f"{source_episode}: video has {video_frames} frames but frame CSV "
                f"has {frame_rows} rows"
            )
        if not video_frames or not force_rows:
            raise RuntimeError(f"{source_episode}: empty video or force CSV")

        entries.append(
            {
                "episode": target_name,
                "source_version": version,
                "source_root": str(source_root.resolve()),
                "source_episode": str(source_episode.resolve()),
                "video_frames": video_frames,
                "frame_timestamp_rows": frame_rows,
                "force_timestamp_rows": force_rows,
                "files": linked_files,
                "sha256": signature,
            }
        )
        seen_signatures[signature] = entries[-1]

    expected_names = {entry["episode"] for entry in entries}
    unexpected = sorted(
        path.name
        for path in output_dir.glob("EP*")
        if path.is_dir() and path.name not in expected_names
    )
    if unexpected:
        raise RuntimeError(
            "Output contains episodes not in the deterministic mapping: "
            + ", ".join(unexpected)
        )

    counts = {
        version: sum(entry["source_version"] == version for entry in entries)
        for version, _ in args.sources
    }
    manifest = {
        "format": "visualforce_raw_episode_view_v1",
        "name": output_dir.name,
        "description": (
            "Collision-safe, exact-duplicate-free symlink view combining "
            "ordered force-estimation sensor recordings."
        ),
        "sources": [
            {"version": version, "path": str(path.resolve())}
            for version, path in args.sources
        ],
        "counts": counts,
        "skipped_exact_duplicates": duplicates,
        "episode_count": len(entries),
        "video_frame_count": sum(entry["video_frames"] for entry in entries),
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(counts, indent=2))
    print(f"exact duplicates skipped: {len(duplicates)}")
    print(f"episodes: {len(entries)}")
    print(f"video frames: {manifest['video_frame_count']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
