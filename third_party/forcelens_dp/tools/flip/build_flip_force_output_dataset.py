#!/usr/bin/env python3
"""Build a validated, manifest-backed stage-2 flip force-output view."""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_LABEL_NAME = "visualforce_pseudo_force_fz.npz"
IDENTITY_GRIPPER_MAPPING = "identity"
LEGACY_TO_CURRENT_GRIPPER_MAPPING = "legacy_ease_0p080_to_linear_0p088"
GRIPPER_MAPPINGS = (
    IDENTITY_GRIPPER_MAPPING,
    LEGACY_TO_CURRENT_GRIPPER_MAPPING,
)
LEGACY_GRIPPER_OPEN_WIDTH_M = 0.080
CURRENT_GRIPPER_OPEN_WIDTH_M = 0.088


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


def validate_cached_label(
    label_path: Path,
    frames: int,
    force_key: str,
) -> tuple[bool, str, dict[str, str]]:
    metadata = {
        "label_checkpoint": "",
        "label_input_mode": "",
        "label_mask_mode": "",
    }
    if not label_path.is_file():
        return False, "missing cached label", metadata
    try:
        with np.load(label_path, allow_pickle=False) as labels:
            force_keys = [str(key) for key in labels["force_keys"].tolist()]
            if force_key not in force_keys:
                return (
                    False,
                    f"cached label lacks {force_key}; keys={force_keys}",
                    metadata,
                )
            force = np.asarray(
                labels["force"][:, force_keys.index(force_key)],
                dtype=np.float32,
            )
            if force.shape != (frames,):
                return (
                    False,
                    f"cached label has shape {force.shape}; expected ({frames},)",
                    metadata,
                )
            if not np.isfinite(force).all():
                return False, "cached label contains non-finite force", metadata
            for output_key, npz_key in (
                ("label_checkpoint", "visualforce_ckpt"),
                ("label_input_mode", "input_mode"),
                ("label_mask_mode", "mask_mode"),
            ):
                if npz_key in labels:
                    metadata[output_key] = str(labels[npz_key].item())
    except Exception as exc:
        return False, f"cannot read cached label: {exc}", metadata
    return True, "ok", metadata


def ease_in_out_quad(value):
    """Return the legacy controller's normalized ease-in/out curve."""
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value < 0.5,
        2.0 * value**2,
        1.0 - ((-2.0 * value + 2.0) ** 2) / 2.0,
    )


def legacy_close_to_current_linear(close):
    """Preserve jaw width while converting July actions to today's mapping."""
    close = np.asarray(close, dtype=np.float64)
    legacy_open_fraction = ease_in_out_quad(1.0 - close)
    legacy_width_m = LEGACY_GRIPPER_OPEN_WIDTH_M * legacy_open_fraction
    current_close = 1.0 - legacy_width_m / CURRENT_GRIPPER_OPEN_WIDTH_M
    return np.clip(current_close, 0.0, 1.0)


def remap_gripper_actions(data: dict, mapping: str) -> dict:
    """Copy an episode and remap only its commanded gripper actions."""
    if mapping not in GRIPPER_MAPPINGS:
        raise ValueError(f"Unsupported gripper action mapping: {mapping}")
    transformed = copy.deepcopy(data)
    if mapping == IDENTITY_GRIPPER_MAPPING:
        return transformed
    for action in transformed["actions"]:
        original = np.asarray(action["gripper_pos"])
        converted = legacy_close_to_current_linear(original)
        action["gripper_pos"] = converted.astype(original.dtype, copy=False)
    return transformed


def materialize_episode(
    source: Path,
    destination: Path,
    label_name: str,
    copy_cached_label: bool,
    materialization: str,
    gripper_action_mapping: str,
) -> None:
    destination.mkdir(parents=True)
    if materialization not in ("symlink", "copy"):
        raise ValueError(f"Unsupported materialization: {materialization}")

    source_data = source / "data.pkl"
    destination_data = destination / "data.pkl"
    if gripper_action_mapping == IDENTITY_GRIPPER_MAPPING:
        if materialization == "copy":
            shutil.copy2(source_data, destination_data)
        else:
            destination_data.symlink_to(source_data.resolve())
    else:
        data = joblib.load(source_data)
        joblib.dump(
            remap_gripper_actions(data, gripper_action_mapping),
            destination_data,
        )

    for key in VIDEO_KEYS:
        source_video = source / f"{key}.mp4"
        destination_video = destination / f"{key}.mp4"
        if materialization == "copy":
            shutil.copy2(source_video, destination_video)
        else:
            destination_video.symlink_to(source_video.resolve())
    if copy_cached_label:
        # Labels are small and intentionally copied. A future relabel operation
        # may then overwrite the generated view without mutating the raw data.
        shutil.copy2(source / label_name, destination / label_name)


def build_dataset(
    source: Path,
    output: Path,
    *,
    overwrite: bool,
    label_name: str = DEFAULT_LABEL_NAME,
    force_key: str = "Fz",
    success_only: bool = False,
    materialization: str = "symlink",
    gripper_action_mapping: str = IDENTITY_GRIPPER_MAPPING,
) -> list[dict[str, str | int]]:
    if not source.is_dir():
        raise FileNotFoundError(f"Raw flip dataset not found: {source}")
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
    copied_labels = 0
    demonstrations = sorted(path for path in source.iterdir() if path.is_dir())
    for demonstration in demonstrations:
        status = "failure" if demonstration.name.endswith("_fail") else "success"
        stage_dir = demonstration / "stage2"
        if stage_dir.is_dir():
            valid, reason, frames = validate_stage(stage_dir)
        else:
            valid, reason, frames = False, "missing stage2", 0

        include = valid and (status == "success" or not success_only)
        label_valid = False
        label_reason = "stage not selected"
        label_metadata = {
            "label_checkpoint": "",
            "label_input_mode": "",
            "label_mask_mode": "",
        }
        if valid:
            label_valid, label_reason, label_metadata = validate_cached_label(
                stage_dir / label_name,
                frames,
                force_key,
            )
        if include:
            materialize_episode(
                stage_dir,
                episodes_dir / demonstration.name,
                label_name,
                copy_cached_label=label_valid,
                materialization=materialization,
                gripper_action_mapping=gripper_action_mapping,
            )
            selected += 1
            selected_frames += frames
            copied_labels += int(label_valid)

        rows.append(
            {
                "episode": demonstration.name,
                "source_dir": str(stage_dir.resolve()),
                "demonstration_status": status,
                "stage": "stage2",
                "frames": frames,
                "valid": str(valid).lower(),
                "selected": str(include).lower(),
                "reason": (
                    reason
                    if not valid
                    else "ok" if include else "excluded by --success-only"
                ),
                "cached_label": str(label_valid).lower(),
                "label_reason": label_reason,
                "materialization": materialization,
                "gripper_action_mapping": gripper_action_mapping,
                "legacy_gripper_open_width_m": LEGACY_GRIPPER_OPEN_WIDTH_M,
                "current_gripper_open_width_m": CURRENT_GRIPPER_OPEN_WIDTH_M,
                **label_metadata,
            }
        )

    if not rows:
        shutil.rmtree(output)
        raise RuntimeError(f"No demonstrations found under {source}")
    if selected == 0:
        shutil.rmtree(output)
        raise RuntimeError(f"No valid stage-2 episodes selected under {source}")

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    status_counts = {
        status: sum(
            row["selected"] == "true"
            and row["demonstration_status"] == status
            for row in rows
        )
        for status in ("success", "failure")
    }
    print(f"Built {output} with {selected} stage-2 episodes")
    print(f"  frames:              {selected_frames}")
    print(f"  successful episodes: {status_counts['success']}")
    print(f"  failed episodes:     {status_counts['failure']}")
    print(f"  skipped invalid:     {len(rows) - selected}")
    print(f"  cached labels copied: {copied_labels}")
    print(f"  materialization:      {materialization}")
    print(f"  gripper mapping:      {gripper_action_mapping}")
    print(f"  manifest: {manifest}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/flip/flip_obj_4")
    parser.add_argument(
        "--output",
        default="data/flip/flip_obj_4_stage2_force_output",
    )
    parser.add_argument("--label-name", default=DEFAULT_LABEL_NAME)
    parser.add_argument("--force-key", default="Fz")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--materialization",
        choices=("symlink", "copy"),
        default="symlink",
        help="Symlink source files (default) or make an independent physical copy.",
    )
    parser.add_argument(
        "--gripper-action-mapping",
        choices=GRIPPER_MAPPINGS,
        default=IDENTITY_GRIPPER_MAPPING,
        help="Optional action-only conversion for the current ARX gripper mapping.",
    )
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
        label_name=args.label_name,
        force_key=args.force_key,
        success_only=args.success_only,
        materialization=args.materialization,
        gripper_action_mapping=args.gripper_action_mapping,
    )


if __name__ == "__main__":
    main()
