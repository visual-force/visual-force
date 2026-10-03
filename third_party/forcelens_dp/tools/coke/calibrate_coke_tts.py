#!/usr/bin/env python3
"""Derive conservative Coke TTS settings from cached VisualForce labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np


DEFAULT_DATASET = "data/pick/coke_mixed_0821_force_output"
DEFAULT_LABEL = "visualforce_pseudo_force_fz.npz"
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def causal_median(values: np.ndarray, window: int) -> np.ndarray:
    """Apply the same trailing median convention as the online controller."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected one force value per frame, got {values.shape}")
    if window <= 0:
        raise ValueError("filter_window must be positive")
    return np.asarray(
        [
            np.median(values[max(0, index - window + 1) : index + 1])
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def quantile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        f"q{int(quantile * 100):02d}": float(np.quantile(array, quantile))
        for quantile in QUANTILES
    }


def round_up(value: float, quantum: float) -> float:
    units = value / quantum
    nearest = round(units)
    if abs(units - nearest) < 1e-6:
        units = nearest
    return float(round(math.ceil(units) * quantum, 10))


def round_down(value: float, quantum: float) -> float:
    units = value / quantum
    nearest = round(units)
    if abs(units - nearest) < 1e-6:
        units = nearest
    return float(round(math.floor(units) * quantum, 10))


def selected_manifest_rows(dataset: Path) -> list[dict[str, str]]:
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with manifest.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("selected", "").strip().lower() == "true"
        ]
    if not rows:
        raise RuntimeError(f"No selected episodes in {manifest}")
    return rows


def _label_text(labels: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in labels:
        return ""
    return str(np.asarray(labels[key]).item())


def analyze_episode(
    episode_dir: Path,
    source_tag: str,
    label_name: str,
    force_key: str,
    baseline_samples: int,
    baseline_max_position: float,
    plateau_tolerance: float,
    filter_window: int,
) -> tuple[dict[str, float | str | int], dict[str, str]]:
    data_path = episode_dir / "data.pkl"
    label_path = episode_dir / label_name
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing episode data: {data_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing force labels: {label_path}")

    data = joblib.load(data_path)
    observations = data.get("observations", [])
    gripper = np.asarray(
        [float(np.asarray(item["gripper_pos"]).reshape(-1)[0]) for item in observations],
        dtype=np.float64,
    )
    with np.load(label_path, allow_pickle=False) as labels:
        if "force" not in labels or "force_keys" not in labels:
            raise ValueError(f"Malformed force-label archive: {label_path}")
        force_keys = [str(value) for value in labels["force_keys"].tolist()]
        if force_key not in force_keys:
            raise ValueError(f"{force_key!r} is absent from {label_path}")
        force = np.asarray(labels["force"], dtype=np.float64)
        if force.ndim != 2:
            raise ValueError(f"Expected force matrix in {label_path}, got {force.shape}")
        magnitude = np.abs(force[:, force_keys.index(force_key)])
        label_metadata = {
            "visualforce_checkpoint": _label_text(labels, "visualforce_ckpt"),
            "input_mode": _label_text(labels, "input_mode"),
            "mask_mode": _label_text(labels, "mask_mode"),
        }

    if len(gripper) != len(magnitude):
        raise ValueError(
            f"{episode_dir.name}: {len(gripper)} observations but "
            f"{len(magnitude)} force labels"
        )
    if len(gripper) < baseline_samples:
        raise ValueError(
            f"{episode_dir.name}: needs {baseline_samples} baseline frames, "
            f"has {len(gripper)}"
        )
    if not np.isfinite(gripper).all() or not np.isfinite(magnitude).all():
        raise ValueError(f"{episode_dir.name}: non-finite gripper or force values")

    baseline_gripper = gripper[:baseline_samples]
    if np.any(baseline_gripper > baseline_max_position):
        first_bad = int(np.flatnonzero(baseline_gripper > baseline_max_position)[0])
        raise ValueError(
            f"{episode_dir.name}: baseline frame {first_bad} has gripper "
            f"position {baseline_gripper[first_bad]:.4f} above "
            f"{baseline_max_position:.4f}"
        )

    baseline_force = magnitude[:baseline_samples]
    baseline = float(np.median(baseline_force))
    baseline_mad = float(np.median(np.abs(baseline_force - baseline)))
    filtered = causal_median(magnitude, filter_window)
    maximum_closure = float(np.max(gripper))
    plateau_start = maximum_closure - plateau_tolerance
    plateau = gripper >= plateau_start
    if not np.any(plateau):
        raise RuntimeError(f"{episode_dir.name}: empty grasp plateau")
    plateau_delta = np.maximum(0.0, filtered[plateau] - baseline)

    return (
        {
            "episode": episode_dir.name,
            "source_tag": source_tag,
            "frames": int(len(gripper)),
            "baseline_force_n": baseline,
            "baseline_mad_n": baseline_mad,
            "maximum_closure": maximum_closure,
            "plateau_frames": int(np.sum(plateau)),
            "plateau_delta_median_n": float(np.median(plateau_delta)),
            "plateau_delta_q75_n": float(np.quantile(plateau_delta, 0.75)),
        },
        label_metadata,
    )


def build_report(args: argparse.Namespace) -> dict[str, object]:
    dataset = Path(args.dataset)
    episodes_dir = dataset / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    episode_rows: list[dict[str, float | str | int]] = []
    metadata_rows: list[dict[str, str]] = []
    for manifest_row in selected_manifest_rows(dataset):
        episode_name = manifest_row["episode"]
        source_tag = manifest_row["source_tag"]
        metrics, metadata = analyze_episode(
            episodes_dir / episode_name,
            source_tag=source_tag,
            label_name=args.label_name,
            force_key=args.force_key,
            baseline_samples=args.baseline_samples,
            baseline_max_position=args.baseline_max_position,
            plateau_tolerance=args.plateau_tolerance,
            filter_window=args.filter_window,
        )
        episode_rows.append(metrics)
        metadata_rows.append(metadata)

    metadata_values = {
        key: sorted({row[key] for row in metadata_rows})
        for key in metadata_rows[0]
    }
    inconsistent = {key: values for key, values in metadata_values.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Force-label metadata differs between episodes: {inconsistent}")

    by_source: dict[str, list[dict[str, float | str | int]]] = {}
    for row in episode_rows:
        by_source.setdefault(str(row["source_tag"]), []).append(row)
    if "soft" not in by_source or "hard" not in by_source:
        raise ValueError("Calibration requires both soft and hard source tags")

    source_summary: dict[str, dict[str, object]] = {}
    for source_tag, rows in sorted(by_source.items()):
        source_summary[source_tag] = {
            "episodes": len(rows),
            "baseline_force_n": quantile_summary(
                [float(row["baseline_force_n"]) for row in rows]
            ),
            "baseline_mad_n": quantile_summary(
                [float(row["baseline_mad_n"]) for row in rows]
            ),
            "maximum_closure": quantile_summary(
                [float(row["maximum_closure"]) for row in rows]
            ),
            "plateau_delta_median_n": quantile_summary(
                [float(row["plateau_delta_median_n"]) for row in rows]
            ),
        }

    soft = by_source["soft"]
    soft_signal = float(
        np.median([float(row["plateau_delta_median_n"]) for row in soft])
    )
    baseline_noise = float(
        np.median([float(row["baseline_mad_n"]) for row in soft])
    )
    target = round_up(
        max(soft_signal, baseline_noise + args.target_noise_clearance),
        args.force_quantum,
    )
    deadband = round_down(
        min(baseline_noise, target - args.force_quantum),
        args.force_quantum,
    )
    deadband = max(args.force_quantum, deadband)
    soft_high = float(source_summary["soft"]["plateau_delta_median_n"]["q90"])
    emergency_threshold = round_up(
        max(soft_high, target + deadband),
        args.force_quantum,
    )
    stop_margin = float(round(emergency_threshold - target, 10))

    soft_max_closures = [float(row["maximum_closure"]) for row in soft]
    contact_ready = round_up(
        min(soft_max_closures) - args.plateau_tolerance,
        args.position_quantum,
    )
    contact_ready = max(args.baseline_max_position, contact_ready)
    maximum_position = round_up(max(soft_max_closures), args.position_quantum)
    signal_to_noise = (
        None if baseline_noise == 0.0 else soft_signal / baseline_noise
    )

    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "force_label": args.label_name,
        "force_key": args.force_key,
        "force_mode": "magnitude",
        "label_metadata": {key: values[0] for key, values in metadata_values.items()},
        "method": {
            "baseline_samples": args.baseline_samples,
            "baseline_max_position": args.baseline_max_position,
            "filter_window": args.filter_window,
            "plateau_tolerance": args.plateau_tolerance,
            "target_noise_clearance_n": args.target_noise_clearance,
            "force_quantum_n": args.force_quantum,
            "position_quantum": args.position_quantum,
        },
        "sources": source_summary,
        "quality": {
            "soft_signal_median_n": soft_signal,
            "soft_baseline_mad_median_n": baseline_noise,
            "soft_signal_to_noise_mad": signal_to_noise,
            "low_separation": bool(
                signal_to_noise is not None and signal_to_noise < 2.0
            ),
        },
        "recommended": {
            "desired_force_delta_n": target,
            "deadband_n": deadband,
            "stop_margin_n": stop_margin,
            "emergency_delta_n": emergency_threshold,
            "baseline_samples": args.baseline_samples,
            "baseline_max_position": args.baseline_max_position,
            "contact_ready_position": contact_ready,
            "maximum_gripper_position": maximum_position,
            "sampling_candidates": 32,
        },
        "episodes": episode_rows,
    }


def print_report(report: dict[str, object]) -> None:
    sources = report["sources"]
    quality = report["quality"]
    recommended = report["recommended"]
    print("Coke TTS calibration from cached VisualForce estimates")
    print(f"  dataset: {report['dataset']}")
    print(
        "  episodes: "
        f"soft={sources['soft']['episodes']} hard={sources['hard']['episodes']}"
    )
    print(
        "  soft plateau median delta: "
        f"{quality['soft_signal_median_n']:.3f} N"
    )
    print(
        "  soft open-baseline MAD:    "
        f"{quality['soft_baseline_mad_median_n']:.3f} N"
    )
    print(
        "  hard plateau q10 delta:    "
        f"{sources['hard']['plateau_delta_median_n']['q10']:.3f} N"
    )
    print("Recommended Coke grasp profile:")
    print(f"  DESIRED_FORCE={recommended['desired_force_delta_n']:.1f}")
    print(f"  TTS_GENTLE_DEADBAND={recommended['deadband_n']:.1f}")
    print(f"  TTS_GENTLE_STOP_MARGIN={recommended['stop_margin_n']:.1f}")
    print(
        "  TTS_GRIPPER_SAFETY_MIN_POSITION="
        f"{recommended['contact_ready_position']:.2f}"
    )
    print(
        "  TTS_GRIPPER_MAX_POSITION="
        f"{recommended['maximum_gripper_position']:.2f}"
    )
    print(f"  TTS_FORCE_BASELINE_SAMPLES={recommended['baseline_samples']}")
    print(f"  SAMPLING_CANDIDATES={recommended['sampling_candidates']}")
    if quality["low_separation"]:
        print(
            "Warning: soft-grasp signal is less than 2x baseline MAD; "
            "camera-only validation is required before robot motion."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate baseline-relative Coke TTS settings from existing "
            "per-frame VisualForce labels. This never starts robot inference."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--label-name", default=DEFAULT_LABEL)
    parser.add_argument("--force-key", default="Fz")
    parser.add_argument("--baseline-samples", type=int, default=10)
    parser.add_argument("--baseline-max-position", type=float, default=0.10)
    parser.add_argument("--filter-window", type=int, default=3)
    parser.add_argument("--plateau-tolerance", type=float, default=0.05)
    parser.add_argument("--target-noise-clearance", type=float, default=0.10)
    parser.add_argument("--force-quantum", type=float, default=0.10)
    parser.add_argument("--position-quantum", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    for name in (
        "baseline_samples",
        "filter_window",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "baseline_max_position",
        "plateau_tolerance",
        "target_noise_clearance",
        "force_quantum",
        "position_quantum",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"  report: {output}")


if __name__ == "__main__":
    main()
