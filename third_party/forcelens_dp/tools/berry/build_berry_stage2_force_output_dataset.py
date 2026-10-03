#!/usr/bin/env python3
"""Build the light+hard berry stage2 policy dataset with pseudo-force labels."""

import argparse
import csv
import shutil
from pathlib import Path


FILES = (
    "data.pkl",
    "base_image.mp4",
    "wrist_image.mp4",
    "visualforce_pseudo_force_fz.npz",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/pick/berry_force_prediction_all",
    )
    parser.add_argument(
        "--output",
        default="data/pick/berry_stage2_policy_force_output",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / args.source
    output = repo_root / args.output
    rows = [
        row
        for row in csv.DictReader((source / "manifest.csv").open())
        if row["source_tag"] in {"light", "hard"}
        and row["episode"].endswith("_stage2")
    ]

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)

    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True)
    for row in rows:
        source_episode = source / "episodes" / row["episode"]
        output_episode = episodes_dir / row["episode"]
        output_episode.mkdir()
        for filename in FILES:
            source_file = source_episode / filename
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            (output_episode / filename).symlink_to(source_file.resolve())

    with (output / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("episode", "source_tag", "source_dir"),
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        tag: sum(row["source_tag"] == tag for row in rows)
        for tag in ("light", "hard")
    }
    print(
        f"Built {len(rows)} labeled stage2 episodes in {output} "
        f"(light={counts['light']}, hard={counts['hard']})"
    )


if __name__ == "__main__":
    main()
