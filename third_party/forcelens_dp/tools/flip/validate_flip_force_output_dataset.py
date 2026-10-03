#!/usr/bin/env python3
"""Validate a manifest-backed flip dataset and its pseudo-force labels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

try:
    from build_flip_force_output_dataset import (
        IDENTITY_GRIPPER_MAPPING,
        LEGACY_TO_CURRENT_GRIPPER_MAPPING,
        legacy_close_to_current_linear,
    )
except ModuleNotFoundError:
    # Also support importlib-based unit tests that load this file directly.
    builder_path = Path(__file__).with_name("build_flip_force_output_dataset.py")
    builder_spec = importlib.util.spec_from_file_location(
        "_flip_force_output_builder", builder_path
    )
    if builder_spec is None or builder_spec.loader is None:
        raise ImportError(f"Cannot load flip dataset builder: {builder_path}")
    builder_module = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder_module)
    IDENTITY_GRIPPER_MAPPING = builder_module.IDENTITY_GRIPPER_MAPPING
    LEGACY_TO_CURRENT_GRIPPER_MAPPING = (
        builder_module.LEGACY_TO_CURRENT_GRIPPER_MAPPING
    )
    legacy_close_to_current_linear = builder_module.legacy_close_to_current_linear


def _scalar_text(labels, key: str) -> str:
    return str(labels[key].item()) if key in labels else ""


def _values_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    try:
        return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
    except TypeError:
        return left == right


def validate_action_conversion(
    processed: dict,
    source: dict,
    *,
    episode_dir: Path,
    mapping: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not _values_equal(processed["timestamps"], source["timestamps"]):
        raise ValueError(f"Timestamps changed in {episode_dir}")
    if not _values_equal(processed["observations"], source["observations"]):
        raise ValueError(f"Observations changed in {episode_dir}")

    source_gripper = []
    processed_gripper = []
    for index, (processed_action, source_action) in enumerate(
        zip(processed["actions"], source["actions"])
    ):
        processed_arm = {
            key: value for key, value in processed_action.items() if key != "gripper_pos"
        }
        source_arm = {
            key: value for key, value in source_action.items() if key != "gripper_pos"
        }
        if not _values_equal(processed_arm, source_arm):
            raise ValueError(f"Non-gripper action changed in {episode_dir} frame {index}")
        source_close = np.asarray(source_action["gripper_pos"], dtype=np.float64)
        processed_close = np.asarray(
            processed_action["gripper_pos"], dtype=np.float64
        )
        expected = (
            legacy_close_to_current_linear(source_close)
            if mapping == LEGACY_TO_CURRENT_GRIPPER_MAPPING
            else source_close
        )
        if not np.allclose(processed_close, expected, rtol=0.0, atol=1e-7):
            raise ValueError(
                f"Incorrect gripper conversion in {episode_dir} frame {index}: "
                f"got {processed_close}, expected {expected}"
            )
        source_gripper.extend(source_close.ravel())
        processed_gripper.extend(processed_close.ravel())
    return np.asarray(source_gripper), np.asarray(processed_gripper)


def validate_dataset(
    dataset: Path,
    *,
    label_name: str,
    force_key: str,
    expected_checkpoint: Path | None = None,
    expected_mask_mode: str | None = None,
    expected_gripper_action_mapping: str | None = None,
) -> dict[str, object]:
    episodes_dir = dataset / "episodes"
    manifest = dataset / "manifest.csv"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Missing episodes directory: {episodes_dir}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    selected_rows = [row for row in rows if row["selected"] == "true"]
    episode_dirs = sorted(path for path in episodes_dir.iterdir() if path.is_dir())
    manifest_names = sorted(row["episode"] for row in selected_rows)
    episode_names = [path.name for path in episode_dirs]
    if manifest_names != episode_names:
        raise ValueError("Selected manifest rows do not match generated episodes")
    if not episode_dirs:
        raise RuntimeError(f"No episodes found under {episodes_dir}")

    expected_checkpoint_resolved = (
        expected_checkpoint.expanduser().resolve()
        if expected_checkpoint is not None
        else None
    )
    rows_by_name = {row["episode"]: row for row in selected_rows}
    all_force = []
    total_frames = 0
    provenance = Counter()
    status_counts = Counter()
    mappings = Counter()
    source_gripper_values = []
    processed_gripper_values = []
    for episode_dir in episode_dirs:
        manifest_row = rows_by_name[episode_dir.name]
        mapping = manifest_row.get(
            "gripper_action_mapping", IDENTITY_GRIPPER_MAPPING
        )
        mappings[mapping] += 1
        if (
            expected_gripper_action_mapping is not None
            and mapping != expected_gripper_action_mapping
        ):
            raise ValueError(
                f"Gripper mapping mismatch in {episode_dir}: {mapping!r}; "
                f"expected {expected_gripper_action_mapping!r}"
            )
        data = joblib.load(episode_dir / "data.pkl")
        lengths = {
            key: len(data[key])
            for key in ("timestamps", "observations", "actions")
        }
        frames = lengths["timestamps"]
        if frames <= 0 or set(lengths.values()) != {frames}:
            raise ValueError(f"Invalid pickle lengths in {episode_dir}: {lengths}")

        source_path = Path(manifest_row["source_dir"]) / "data.pkl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source data for provenance: {source_path}")
        source_data = joblib.load(source_path)
        source_values, processed_values = validate_action_conversion(
            data,
            source_data,
            episode_dir=episode_dir,
            mapping=mapping,
        )
        source_gripper_values.append(source_values)
        processed_gripper_values.append(processed_values)
        if manifest_row.get("materialization") == "copy":
            copied_files = (
                episode_dir / "data.pkl",
                episode_dir / "base_image.mp4",
                episode_dir / "wrist_image.mp4",
            )
            if any(path.is_symlink() for path in copied_files):
                raise ValueError(f"Physical-copy dataset contains a symlink: {episode_dir}")

        label_path = episode_dir / label_name
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing force label: {label_path}")
        with np.load(label_path, allow_pickle=False) as labels:
            force_keys = [str(key) for key in labels["force_keys"].tolist()]
            if force_key not in force_keys:
                raise KeyError(
                    f"{label_path} does not contain {force_key}; keys={force_keys}"
                )
            force = np.asarray(
                labels["force"][:, force_keys.index(force_key)],
                dtype=np.float32,
            )
            if force.shape != (frames,):
                raise ValueError(
                    f"Force length mismatch in {episode_dir}: "
                    f"{force.shape}, expected ({frames},)"
                )
            if not np.isfinite(force).all():
                raise ValueError(f"Non-finite force labels in {label_path}")
            if "mask_frac" in labels:
                mask_frac = np.asarray(labels["mask_frac"])
                if mask_frac.shape != (frames,) or not np.isfinite(mask_frac).all():
                    raise ValueError(f"Invalid mask fractions in {label_path}")

            checkpoint = _scalar_text(labels, "visualforce_ckpt")
            input_mode = _scalar_text(labels, "input_mode")
            mask_mode = _scalar_text(labels, "mask_mode")
            if expected_checkpoint_resolved is not None:
                if not checkpoint:
                    raise ValueError(f"Missing VisualForce checkpoint metadata: {label_path}")
                if Path(checkpoint).expanduser().resolve() != expected_checkpoint_resolved:
                    raise ValueError(
                        f"Checkpoint mismatch in {label_path}: {checkpoint}; "
                        f"expected {expected_checkpoint_resolved}"
                    )
            if expected_mask_mode is not None and mask_mode != expected_mask_mode:
                raise ValueError(
                    f"Mask-mode mismatch in {label_path}: {mask_mode!r}; "
                    f"expected {expected_mask_mode!r}"
                )
            provenance[(checkpoint, input_mode, mask_mode)] += 1

        all_force.append(force)
        total_frames += frames
        status_counts[rows_by_name[episode_dir.name]["demonstration_status"]] += 1

    force = np.concatenate(all_force)
    magnitude = np.abs(force)
    source_gripper = np.concatenate(source_gripper_values)
    processed_gripper = np.concatenate(processed_gripper_values)
    print("Flip force-output dataset validation:")
    print(f"  episodes: {len(episode_dirs)}")
    print(f"  frames:   {total_frames}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(
        f"  signed {force_key}: min={force.min():.4f}, "
        f"median={np.median(force):.4f}, max={force.max():.4f}"
    )
    print(
        f"  |{force_key}|:       min={magnitude.min():.4f}, "
        f"median={np.median(magnitude):.4f}, max={magnitude.max():.4f}"
    )
    for (checkpoint, input_mode, mask_mode), count in sorted(provenance.items()):
        print(
            f"  labels: {count} checkpoint={checkpoint} "
            f"input={input_mode} mask={mask_mode}"
        )
    for mapping, count in sorted(mappings.items()):
        print(f"  gripper mapping: {count} episodes {mapping}")
    print(
        "  gripper action: "
        f"July median={np.median(source_gripper):.4f}, "
        f"processed median={np.median(processed_gripper):.4f}"
    )
    return {
        "episodes": len(episode_dirs),
        "frames": total_frames,
        "status_counts": dict(status_counts),
        "provenance": dict(provenance),
        "force_min": float(force.min()),
        "force_median": float(np.median(force)),
        "force_max": float(force.max()),
        "gripper_action_mappings": dict(mappings),
        "source_gripper_median": float(np.median(source_gripper)),
        "processed_gripper_median": float(np.median(processed_gripper)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/flip/flip_obj_4_stage2_force_output",
    )
    parser.add_argument(
        "--label-name",
        default="visualforce_pseudo_force_fz.npz",
    )
    parser.add_argument("--force-key", default="Fz")
    parser.add_argument("--visualforce-ckpt", type=Path)
    parser.add_argument("--mask-mode")
    parser.add_argument("--gripper-action-mapping")
    args = parser.parse_args()
    validate_dataset(
        Path(args.dataset),
        label_name=args.label_name,
        force_key=args.force_key,
        expected_checkpoint=args.visualforce_ckpt,
        expected_mask_mode=args.mask_mode,
        expected_gripper_action_mapping=args.gripper_action_mapping,
    )


if __name__ == "__main__":
    main()
