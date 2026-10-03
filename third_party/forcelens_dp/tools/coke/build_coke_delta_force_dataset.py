#!/usr/bin/env python3
"""Build and validate the three-source Coke delta-force dataset view."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.coke.build_coke_mixed_dataset import link_episode, validate_episode


DEFAULT_LABEL = "visualforce_pseudo_force_fz.npz"


def _safe_replace_output(output: Path, protected: list[Path], overwrite: bool) -> None:
    resolved = output.resolve()
    if any(resolved == path.resolve() for path in protected):
        raise ValueError("Output must differ from raw datasets and the label cache")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)


def _label_source(
    raw_episode: Path,
    label_cache: Path | None,
    episode_name: str,
    label_name: str,
) -> tuple[Path | None, str]:
    raw_label = raw_episode / label_name
    if raw_label.is_file():
        return raw_label, "raw"
    if label_cache is not None:
        cached = label_cache / "episodes" / episode_name / label_name
        if cached.is_file():
            return cached, "cache"
    return None, "missing"


def build_dataset(args: argparse.Namespace) -> None:
    sources = [
        ("soft", Path(args.soft_root)),
        ("hard", Path(args.hard_root)),
        ("dyn", Path(args.dynamics_root)),
    ]
    output = Path(args.output)
    label_cache = Path(args.label_cache) if args.label_cache else None
    protected = [path for _, path in sources]
    if label_cache is not None:
        protected.append(label_cache)
    for _, source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"Raw dataset not found: {source}")
    if label_cache is not None and not label_cache.is_dir():
        raise FileNotFoundError(f"Label cache not found: {label_cache}")

    _safe_replace_output(output, protected, args.overwrite)
    episodes_out = output / "episodes"
    episodes_out.mkdir(parents=True)

    rows: list[dict[str, str | int]] = []
    selected_by_source: dict[str, int] = {}
    reused_labels = 0
    for source_tag, source_root in sources:
        selected_by_source[source_tag] = 0
        for raw_episode in sorted(path for path in source_root.iterdir() if path.is_dir()):
            valid, reason, frames = validate_episode(raw_episode)
            episode_name = f"{source_tag}__{raw_episode.name}"
            label_path, label_origin = _label_source(
                raw_episode, label_cache, episode_name, args.label_name
            )
            if valid:
                destination = episodes_out / episode_name
                link_episode(raw_episode, destination)
                if label_path is not None:
                    (destination / args.label_name).symlink_to(label_path.resolve())
                    reused_labels += 1
                selected_by_source[source_tag] += 1
            rows.append(
                {
                    "episode": episode_name,
                    "source_tag": source_tag,
                    "source_dir": str(raw_episode.resolve()),
                    "frames": frames,
                    "valid": str(valid).lower(),
                    "selected": str(valid).lower(),
                    "reason": reason,
                    "force_label": label_origin,
                }
            )

    empty = [tag for tag, count in selected_by_source.items() if count == 0]
    if empty:
        shutil.rmtree(output)
        raise RuntimeError(f"No valid episodes found for sources: {empty}")

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = sum(selected_by_source.values())
    print(f"Built {output} with {selected} episodes")
    for source_tag, count in selected_by_source.items():
        print(f"  {source_tag}: {count}")
    print(f"  reused force labels: {reused_labels}")
    print(f"  labels still needed: {selected - reused_labels}")
    print(f"  skipped invalid: {len(rows) - selected}")
    print(f"  manifest: {manifest}")


def validate_labels(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    episodes_dir = dataset / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Dataset episodes directory not found: {episodes_dir}")

    episodes = sorted(path for path in episodes_dir.iterdir() if path.is_dir())
    if not episodes:
        raise RuntimeError(f"No episodes found in {episodes_dir}")
    problems: list[str] = []
    by_source: dict[str, int] = {}
    for episode in episodes:
        source = episode.name.split("__", 1)[0]
        by_source[source] = by_source.get(source, 0) + 1
        label_path = episode / args.label_name
        if not label_path.is_file():
            problems.append(f"{episode.name}: missing {args.label_name}")
            continue
        data = joblib.load(episode / "data.pkl")
        with np.load(label_path) as labels:
            if "force" not in labels or "force_keys" not in labels:
                problems.append(f"{episode.name}: malformed force label archive")
                continue
            if args.force_key not in labels["force_keys"].tolist():
                problems.append(f"{episode.name}: force key {args.force_key!r} absent")
            if len(labels["force"]) != len(data["timestamps"]):
                problems.append(
                    f"{episode.name}: {len(labels['force'])} labels for "
                    f"{len(data['timestamps'])} frames"
                )

    print(f"Validated {len(episodes)} episodes in {dataset}")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")
    if problems:
        preview = "\n".join(f"  - {problem}" for problem in problems[:20])
        raise RuntimeError(f"Force-label validation failed:\n{preview}")
    print(f"  force labels: {len(episodes)} complete ({args.force_key})")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--soft-root", default="data/pick/coke_soft_0821")
    build.add_argument("--hard-root", default="data/pick/coke_hard_0821")
    build.add_argument("--dynamics-root", default="data/pick/coke_dyn_0821")
    build.add_argument(
        "--label-cache", default="data/pick/coke_mixed_0821_force_output"
    )
    build.add_argument("--label-name", default=DEFAULT_LABEL)
    build.add_argument("--output", default="data/pick/coke_delta_force_all_0821")
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(func=build_dataset)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--label-name", default=DEFAULT_LABEL)
    validate.add_argument("--force-key", default="Fz")
    validate.set_defaults(func=validate_labels)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
