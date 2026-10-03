#!/usr/bin/env python3
"""Derive per-episode zero-referenced compression labels from a cached manifest.

The force sensor has a different static Fz preload in each recording.  Raw Fz
therefore does not represent grip force: an unloaded opening can be several
newtons away from zero.  This helper preserves the cached visual inputs and
replaces each sensor target with positive compression relative to the median
of that episode's initial unloaded frames.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=20,
        help="Number of initial frames used to estimate each episode's preload.",
    )
    parser.add_argument(
        "--zero-source",
        choices=("preserve", "sensor_prefix"),
        default="preserve",
        help=(
            "Keep the source manifest's task-domain zero controls, or use only "
            "unloaded prefixes from the force-sensor recordings."
        ),
    )
    parser.add_argument(
        "--sensor-episode-min",
        type=int,
        default=None,
        help="Optionally keep only sensor episode numbers at or above this value.",
    )
    parser.add_argument(
        "--sensor-episode-max",
        type=int,
        default=None,
        help="Optionally keep only sensor episode numbers at or below this value.",
    )
    args = parser.parse_args()
    if args.baseline_samples <= 0:
        parser.error("--baseline-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    source_manifest = args.source_manifest.expanduser().resolve()
    source_root = source_manifest.parent
    output_dir = args.output_dir.expanduser().resolve()
    label_dir = output_dir / "cache"
    label_dir.mkdir(parents=True, exist_ok=True)

    with source_manifest.open() as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "visualforce_cached_edge_v1":
        raise ValueError(f"Unsupported format: {manifest.get('format')!r}")

    baselines: dict[str, float] = {}
    maxima: dict[str, float] = {}
    entries = []
    sensor_prefix_entries = []
    for original in manifest["entries"]:
        if args.zero_source == "sensor_prefix" and original["source_kind"] != "sensor":
            continue
        if original["source_kind"] == "sensor" and (
            args.sensor_episode_min is not None or args.sensor_episode_max is not None
        ):
            episode_number = int(original["id"].rsplit("ep", 1)[1])
            if (
                args.sensor_episode_min is not None
                and episode_number < args.sensor_episode_min
            ) or (
                args.sensor_episode_max is not None
                and episode_number > args.sensor_episode_max
            ):
                continue
        entry = dict(original)
        edge_source = (source_root / entry["edge_path"]).resolve()
        entry["edge_path"] = os.path.relpath(edge_source, output_dir)
        if entry["source_kind"] == "sensor":
            signed_path = (source_root / entry["label_path"]).resolve()
            signed = np.load(signed_path, mmap_mode="r")
            if signed.ndim != 2 or signed.shape[1] != 1:
                raise ValueError(f"Expected [N, 1] labels in {signed_path}, got {signed.shape}")
            count = min(args.baseline_samples, len(signed))
            baseline = float(np.median(np.asarray(signed[:count, 0])))
            compression = np.maximum(baseline - np.asarray(signed[:, 0]), 0.0)
            compression = compression.astype(np.float32)[:, None]
            label_path = label_dir / f"{entry['id']}_zero_ref_labels.npy"
            np.save(label_path, compression, allow_pickle=False)
            entry["label_path"] = str(label_path.relative_to(output_dir))
            entry["label_policy"] = (
                "positive_compression=max(initial_Fz_median-signed_Fz,0)"
            )
            entry["sensor_open_baseline_fz_n"] = baseline
            entry["sensor_baseline_samples"] = count
            baselines[entry["id"]] = baseline
            maxima[entry["id"]] = float(compression.max())
            if args.zero_source == "sensor_prefix":
                edge_prefix = np.asarray(
                    np.load(edge_source, mmap_mode="r")[:count], dtype=np.uint8
                )
                edge_prefix_path = label_dir / f"{entry['id']}_open_prefix_edges.npy"
                np.save(edge_prefix_path, edge_prefix, allow_pickle=False)
                sensor_prefix_entries.append(
                    {
                        "id": f"{entry['id']}_open_prefix",
                        "source_kind": "zero",
                        "source_subtype": "sensor_open_prefix",
                        "source_path": entry["source_path"],
                        "split": entry["split"],
                        "split_group": entry.get("split_group", entry["source_path"]),
                        "edge_path": str(edge_prefix_path.relative_to(output_dir)),
                        "label_path": None,
                        "constant_label_n": 0.0,
                        "sample_count": count,
                        "label_policy": "initial force-sensor episode prefix is unloaded",
                        "mask_mode": entry.get("mask_mode"),
                    }
                )
        entries.append(entry)
    entries.extend(sensor_prefix_entries)

    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket = counts.setdefault(entry["split"], {})
        subtype = entry["source_subtype"]
        bucket[subtype] = bucket.get(subtype, 0) + int(entry["sample_count"])

    derived = dict(manifest)
    derived.update(
        {
            "name": output_dir.name,
            "description": (
                "Per-episode zero-referenced positive compression labels derived "
                f"from {source_manifest}. Visual edge caches are reused unchanged."
            ),
            "sensor_label_policy": (
                "positive compression relative to the median of each sensor "
                f"episode's first {args.baseline_samples} frames"
            ),
            "source_manifest": str(source_manifest),
            "sensor_baseline_samples": args.baseline_samples,
            "zero_source": args.zero_source,
            "zero_label_policy": (
                manifest.get("zero_label_policy")
                if args.zero_source == "preserve"
                else "Only initial unloaded prefixes from force-sensor episodes."
            ),
            "sensor_open_baseline_fz_n": baselines,
            "sensor_max_compression_n": maxima,
            "counts": counts,
            "entries": entries,
        }
    )
    output_path = output_dir / "manifest.json"
    with output_path.open("w") as stream:
        json.dump(derived, stream, indent=2)
        stream.write("\n")

    print(f"sensor episodes: {len(baselines)}")
    print(
        "open preload range: "
        f"{min(baselines.values()):.3f} to {max(baselines.values()):.3f} N"
    )
    print(
        "maximum compression range: "
        f"{min(maxima.values()):.3f} to {max(maxima.values()):.3f} N"
    )
    print(f"Manifest: {output_path}")


if __name__ == "__main__":
    main()
