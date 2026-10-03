#!/usr/bin/env python3
"""Build staged berry datasets as symlinked episode folders.

Outputs:
  - policy dataset: stage2 only from berry_staged + berry_staged_hard
  - verifier dataset: all video-backed stages from berry_staged,
    berry_staged_hard, and berry_staged_hard_predict
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_LIGHT = "data/pick/berry_staged"
DEFAULT_HARD = "data/pick/berry_staged_hard"
DEFAULT_PREDICT = "data/pick/berry_staged_hard_predict"
VIDEO_KEYS = ("base_image", "wrist_image")


def has_required_files(stage_dir: Path) -> bool:
    if not (stage_dir / "data.pkl").is_file():
        return False
    return all((stage_dir / f"{key}.mp4").is_file() for key in VIDEO_KEYS)


def discover_stages(root: Path, stage_name: Optional[str] = None) -> list[Path]:
    stages = []
    for stage_dir in sorted(root.glob("*/stage*")):
        if not stage_dir.is_dir():
            continue
        if stage_name is not None and stage_dir.name != stage_name:
            continue
        if has_required_files(stage_dir):
            stages.append(stage_dir)
    return stages


def link_episode(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    for name in ("data.pkl", *[f"{key}.mp4" for key in VIDEO_KEYS]):
        target = source / name
        if not target.is_file():
            raise FileNotFoundError(f"Missing {target}")
        (dest / name).symlink_to(target.resolve())


def build_dataset(output: Path, sources: list[Tuple[str, Path, Optional[str]]], overwrite: bool):
    episodes_dir = output / "episodes"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    episodes_dir.mkdir(parents=True)

    rows = []
    for source_tag, root, stage_name in sources:
        for stage_dir in discover_stages(root, stage_name=stage_name):
            parent = stage_dir.parent.name
            episode_name = f"{source_tag}_{parent}_{stage_dir.name}"
            link_episode(stage_dir, episodes_dir / episode_name)
            rows.append((episode_name, source_tag, str(stage_dir)))

    manifest = output / "manifest.csv"
    with manifest.open("w") as f:
        f.write("episode,source_tag,source_dir\n")
        for episode_name, source_tag, source_dir in rows:
            f.write(f"{episode_name},{source_tag},{source_dir}\n")

    print(f"{output}: {len(rows)} episodes")
    print(f"  manifest: {manifest}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-root", default=DEFAULT_LIGHT)
    parser.add_argument("--hard-root", default=DEFAULT_HARD)
    parser.add_argument("--predict-root", default=DEFAULT_PREDICT)
    parser.add_argument("--policy-output", default="data/pick/berry_stage2_policy")
    parser.add_argument("--verifier-output", default="data/pick/berry_force_prediction_all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    light_root = Path(args.light_root)
    hard_root = Path(args.hard_root)
    predict_root = Path(args.predict_root)

    build_dataset(
        Path(args.policy_output),
        [
            ("light", light_root, "stage2"),
            ("hard", hard_root, "stage2"),
        ],
        overwrite=args.overwrite,
    )
    build_dataset(
        Path(args.verifier_output),
        [
            ("light", light_root, None),
            ("hard", hard_root, None),
            ("predict", predict_root, None),
        ],
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
