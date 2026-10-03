"""Build a deduplicated symlink dataset for missing VisualForce estimates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.path_utils import is_within


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-per-dataset",
        type=int,
        default=None,
        help="Keep only the first N missing records from each canonical dataset.",
    )
    args = parser.parse_args()
    if args.sample_per_dataset is not None and args.sample_per_dataset <= 0:
        parser.error("--sample-per-dataset must be positive")
    return args


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == root:
        raise ValueError("--output must not be the same directory as --data-root")
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    aliases: dict[Path, list[Path]] = defaultdict(list)
    for base in root.rglob("base_image.mp4"):
        if base.exists() and not is_within(base, output):
            aliases[base.resolve()].append(base)

    labeled = set()
    for force_file in root.rglob("visualforce_pseudo_force_fz.npz"):
        if is_within(force_file, output):
            continue
        base = force_file.parent / "base_image.mp4"
        if base.exists():
            labeled.add(base.resolve())

    counts: dict[tuple[str, str], int] = defaultdict(int)
    records = []
    for base, _paths in sorted(aliases.items(), key=lambda item: str(item[0])):
        relative = base.parent.relative_to(root)
        parts = relative.parts
        if len(parts) < 2 or "retired" in parts or base in labeled:
            continue
        task, dataset = parts[:2]
        group = (task, dataset)
        if args.sample_per_dataset is not None and counts[group] >= args.sample_per_dataset:
            continue
        counts[group] += 1

        name = "__".join(relative.parts)
        destination = episodes_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("data.pkl", "base_image.mp4", "wrist_image.mp4"):
            source = base.parent / filename
            if not source.exists():
                raise FileNotFoundError(source)
            link = destination / filename
            if not link.exists():
                link.symlink_to(source.resolve())
        records.append(
            {
                "episode": name,
                "task": task,
                "dataset": dataset,
                "source": str(base.parent),
            }
        )

    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Built {len(records)} missing-force records in {output}")
    for group, count in sorted(counts.items()):
        print(f"  {'/'.join(group)}: {count}")


if __name__ == "__main__":
    main()
