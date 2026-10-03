#!/usr/bin/env python3
"""Build a cached edge-force dataset from sensor and known-zero recordings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import av
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from scripts.precompute_flip_pseudo_force import (
    build_sam2_video_predictor,
    color_masks,
    color_hull_masks,
    sam2_masks,
)
from src.training_utils import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data/visualforce_fz_edge_real_zero_v1"
DEFAULT_SENSOR = REPO_ROOT / "data/data_ball_260422"
DEFAULT_DP_ROOT = REPO_ROOT / "third_party/forcelens_dp"
DEFAULT_EMPTY_DEMO = DEFAULT_DP_ROOT / "data/pick/berry_empty_gripper_0828"
DEFAULT_ROLLOUTS = REPO_ROOT / "rollouts"
DEFAULT_DEMO_ROOTS = (
    DEFAULT_DP_ROOT / "data/pick/berry_staged",
    DEFAULT_DP_ROOT / "data/pick/berry_staged_hard",
    DEFAULT_DP_ROOT / "data/pick/berry_staged_hard_predict",
    DEFAULT_DP_ROOT / "data/pick/coke_soft_0821",
    DEFAULT_DP_ROOT / "data/pick/coke_hard_0821",
    DEFAULT_DP_ROOT / "data/pick/coke_dyn_0821",
    DEFAULT_DP_ROOT / "data/flip/flip_obj_1",
    DEFAULT_DP_ROOT / "data/flip/flip_obj_2",
    DEFAULT_DP_ROOT / "data/flip/flip_obj_3",
    DEFAULT_DP_ROOT / "data/flip/flip_obj_4",
    DEFAULT_DP_ROOT / "data/flip/obj4_dynamics",
    DEFAULT_DP_ROOT / "data/pivot/pivoting_cheng",
    DEFAULT_DP_ROOT / "data/plug_insertion_aug",
    DEFAULT_DP_ROOT / "data/plug_insertion_0824",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sensor-dir", type=Path, default=DEFAULT_SENSOR)
    parser.add_argument("--empty-demo-dir", type=Path, default=DEFAULT_EMPTY_DEMO)
    parser.add_argument("--rollouts-dir", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument(
        "--demo-root", type=Path, action="append", dest="demo_roots",
        help="Repeat to override the default demonstration roots.",
    )
    parser.add_argument("--trim-seconds", type=float, default=0.1)
    parser.add_argument("--rollout-prefix-frames", type=int, default=10)
    parser.add_argument("--demo-prefix-frames", type=int, default=3)
    parser.add_argument("--max-demo-episodes-per-root", type=int, default=20)
    parser.add_argument("--open-gripper-max", type=float, default=0.15)
    parser.add_argument(
        "--min-rollout-edge-fraction", type=float, default=0.05,
        help="Reject rollout edge prefixes whose recorded mask/edge is nearly empty.",
    )
    parser.add_argument("--val-modulus", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mask-mode",
        choices=("stored_sam2", "color", "color_hull"),
        default="stored_sam2",
        help=(
            "Use the original sensor masks plus SAM2 task masks, or apply the "
            "same deterministic green-material or green-fin hull mask to every "
            "source domain."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.trim_seconds < 0:
        parser.error("--trim-seconds must be non-negative")
    if not 0.0 <= args.min_rollout_edge_fraction <= 1.0:
        parser.error("--min-rollout-edge-fraction must be in [0, 1]")
    for name in (
        "rollout_prefix_frames", "demo_prefix_frames",
        "max_demo_episodes_per_root", "val_modulus", "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.demo_roots = tuple(args.demo_roots or DEFAULT_DEMO_ROOTS)
    return args


def stable_split(key: str, val_modulus: int) -> str:
    digest = hashlib.sha256(key.encode()).digest()
    return "val" if int.from_bytes(digest[:8], "big") % val_modulus == 0 else "train"


def episode_split_group(episode: Path) -> Path:
    """Keep stage1/stage2 from one demonstration in the same split."""
    return episode.parent if episode.name in {"stage1", "stage2"} else episode


def read_video_rgb(path: Path, limit: int | None = None) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(np.asarray(frame.to_image().convert("RGB")))
            if limit is not None and len(frames) >= limit:
                break
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


def read_video_gray(path: Path, limit: int | None = None) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(np.asarray(frame.to_image().convert("L")))
            if limit is not None and len(frames) >= limit:
                break
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


def masked_frames_to_edges(
    frames: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Vectorized equivalent of the deployed masked-RGB Sobel preprocessing."""
    if len(frames) != len(masks):
        raise ValueError("frame/mask count mismatch")
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
    ).view(1, 1, 3, 3)
    result = []
    for start in range(0, len(frames), batch_size):
        stop = min(len(frames), start + batch_size)
        rgb = torch.from_numpy(frames[start:stop]).to(device).permute(0, 3, 1, 2).float()
        rgb /= 255.0
        mask = torch.from_numpy(masks[start:stop]).to(device).unsqueeze(1).bool()
        rgb = rgb * mask
        rgb = F.interpolate(
            rgb, size=(256, 256), mode="bilinear", align_corners=False, antialias=True
        )
        gray = 0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]
        visible = (rgb.sum(dim=1, keepdim=True) > 1e-6).float()
        interior = -F.max_pool2d(-visible, kernel_size=3, stride=1, padding=1)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        edge = (torch.sqrt(gx.square() + gy.square() + 1e-8) / 4.0).clamp(0.0, 1.0)
        edge *= interior
        result.append(torch.round(edge[:, 0] * 255.0).byte().cpu().numpy())
    return np.concatenate(result, axis=0)


def save_arrays(
    output_dir: Path,
    entry_id: str,
    edges: np.ndarray,
    labels: np.ndarray | None,
    force: bool,
) -> tuple[str, str | None]:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    edge_path = cache_dir / f"{entry_id}_edges.npy"
    label_path = cache_dir / f"{entry_id}_labels.npy" if labels is not None else None
    if force or not edge_path.exists():
        np.save(edge_path, np.asarray(edges, dtype=np.uint8), allow_pickle=False)
    if labels is not None and (force or not label_path.exists()):
        np.save(label_path, np.asarray(labels, dtype=np.float32), allow_pickle=False)
    return (
        str(edge_path.relative_to(output_dir)),
        str(label_path.relative_to(output_dir)) if label_path else None,
    )


def sensor_entries(args: argparse.Namespace, device: torch.device) -> list[dict]:
    entries = []
    for episode in sorted(args.sensor_dir.glob("EP*")):
        video_path = episode / "video.mp4"
        mask_path = episode / "mask.mp4"
        frame_timestamps_path = episode / "frame_timestamps.csv"
        force_timestamps_path = episode / "force_timestamps.csv"
        required = (video_path, frame_timestamps_path, force_timestamps_path)
        if args.mask_mode == "stored_sam2":
            required += (mask_path,)
        if not all(path.is_file() for path in required):
            continue
        entry_id = f"sensor_{episode.name.lower()}"
        edge_cache = args.output_dir / "cache" / f"{entry_id}_edges.npy"
        label_cache = args.output_dir / "cache" / f"{entry_id}_labels.npy"
        if args.force or not edge_cache.exists() or not label_cache.exists():
            frames = read_video_rgb(video_path)
            if args.mask_mode == "stored_sam2":
                masks = read_video_gray(mask_path) > 127
            elif args.mask_mode == "color":
                masks = color_masks(frames)
            else:
                masks = color_hull_masks(frames)
            with frame_timestamps_path.open(newline="") as stream:
                frame_rows = list(csv.DictReader(stream))
            with force_timestamps_path.open(newline="") as stream:
                force_rows = list(csv.DictReader(stream))
            n = min(len(frames), len(masks), len(frame_rows))
            frame_rel = np.asarray([float(row["t_rel_s"]) for row in frame_rows[:n]])
            frame_wall = np.asarray([float(row["t_wall_s"]) for row in frame_rows[:n]])
            force_wall = np.asarray([float(row["t_wall_s"]) for row in force_rows])
            force_fz = np.asarray([float(row["Fz"]) for row in force_rows])
            keep = (
                (frame_rel >= args.trim_seconds)
                & (frame_rel <= frame_rel.max() - args.trim_seconds)
            )
            frames = frames[:n][keep]
            masks = masks[:n][keep]
            labels = np.interp(frame_wall[keep], force_wall, force_fz).astype(np.float32)[:, None]
            edges = masked_frames_to_edges(frames, masks, args.batch_size, device)
            del frames, masks
            edge_rel, label_rel = save_arrays(
                args.output_dir, entry_id, edges, labels, args.force
            )
        else:
            edge_rel = str(edge_cache.relative_to(args.output_dir))
            label_rel = str(label_cache.relative_to(args.output_dir))
        count = int(np.load(args.output_dir / edge_rel, mmap_mode="r").shape[0])
        number = int(episode.name[2:])
        entries.append({
            "id": entry_id,
            "source_kind": "sensor",
            "source_subtype": "physical_fz",
            "source_path": str(episode.resolve()),
            "split": "val" if number % 10 == 0 else "train",
            "edge_path": edge_rel,
            "label_path": label_rel,
            "sample_count": count,
            "label_policy": "wall_clock_interpolated_signed_Fz",
            "mask_mode": args.mask_mode,
        })
        print(f"sensor {episode.name}: {count}")
    return entries


def discover_demo_episodes(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("wrist_image.mp4") if (path.parent / "data.pkl").is_file()})


def observed_gripper_positions(episode: Path) -> np.ndarray:
    data = joblib.load(episode / "data.pkl")
    return np.asarray([
        float(np.asarray(obs["gripper_pos"]).reshape(-1)[0])
        for obs in data["observations"]
    ])


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[index] for index in indices]


def zero_entry(
    args: argparse.Namespace,
    episode: Path,
    subtype: str,
    frames: np.ndarray,
    predictor,
    sam2_device: str | None,
    device: torch.device,
) -> dict:
    key = f"{subtype}:{episode.resolve()}"
    split_group = episode_split_group(episode).resolve()
    token = hashlib.sha256(key.encode()).hexdigest()[:16]
    entry_id = f"{subtype}_{token}"
    edge_cache = args.output_dir / "cache" / f"{entry_id}_edges.npy"
    if args.force or not edge_cache.exists():
        if args.mask_mode == "color":
            masks = color_masks(frames)
        elif args.mask_mode == "color_hull":
            masks = color_hull_masks(frames)
        else:
            masks = sam2_masks(
                frames,
                predictor=predictor,
                device=sam2_device,
                fallback_color=True,
                separate_objects=True,
            )
        edges = masked_frames_to_edges(frames, masks, args.batch_size, device)
        edge_rel, _ = save_arrays(args.output_dir, entry_id, edges, None, args.force)
    else:
        edge_rel = str(edge_cache.relative_to(args.output_dir))
    count = int(np.load(args.output_dir / edge_rel, mmap_mode="r").shape[0])
    return {
        "id": entry_id,
        "source_kind": "zero",
        "source_subtype": subtype,
        "source_path": str(episode.resolve()),
        "split": stable_split(f"{subtype}:{split_group}", args.val_modulus),
        "split_group": str(split_group),
        "edge_path": edge_rel,
        "label_path": None,
        "constant_label_n": 0.0,
        "sample_count": count,
        "label_policy": "known_or_initial_open_no_contact_zero",
        "mask_mode": args.mask_mode,
    }


def demo_entries(args: argparse.Namespace, device: torch.device) -> list[dict]:
    candidates: list[tuple[Path, str, np.ndarray]] = []
    for episode in discover_demo_episodes(args.empty_demo_dir):
        frames = read_video_rgb(episode / "wrist_image.mp4")
        candidates.append((episode, "empty_demo", frames))

    for root in args.demo_roots:
        episodes = evenly_spaced(
            discover_demo_episodes(root), args.max_demo_episodes_per_root
        )
        for episode in episodes:
            try:
                gripper = observed_gripper_positions(episode)
                frames = read_video_rgb(
                    episode / "wrist_image.mp4", limit=args.demo_prefix_frames
                )
            except (KeyError, ValueError, OSError) as exc:
                print(f"skip demo {episode}: {exc}")
                continue
            n = min(len(frames), len(gripper), args.demo_prefix_frames)
            keep = np.flatnonzero(gripper[:n] <= args.open_gripper_max)
            if not len(keep) or not np.array_equal(keep, np.arange(len(keep))):
                continue
            candidates.append((episode, "demo_open_prefix", frames[: len(keep)]))

    if not candidates:
        return []
    predictor = None
    sam2_device = None
    if args.mask_mode == "stored_sam2":
        predictor, sam2_device = build_sam2_video_predictor(
            "small",
            REPO_ROOT / "third_party/sam2",
            REPO_ROOT / "third_party/sam2/checkpoints/sam2.1_hiera_small.pt",
            str(device),
        )
    entries = []
    for episode, subtype, frames in candidates:
        entry = zero_entry(
            args, episode, subtype, frames, predictor, sam2_device, device
        )
        entries.append(entry)
        print(f"{subtype} {episode.name}: {entry['sample_count']}")
    return entries


def rollout_entries(args: argparse.Namespace, device: torch.device) -> list[dict]:
    entries = []
    for log_path in sorted(args.rollouts_dir.rglob("force_log.csv")):
        episode = log_path.parent
        edge_video = episode / "edge_h264.mp4"
        original_video = episode / "original_h264.mp4"
        if not edge_video.is_file() or (
            args.mask_mode != "stored_sam2" and not original_video.is_file()
        ):
            continue
        with log_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        selected = []
        for row in rows[: args.rollout_prefix_frames]:
            try:
                frame_index = int(row["frame_idx"])
                gripper = float(row["obs_gripper"])
                mask_fraction = float(row["mask_frac"])
            except (KeyError, TypeError, ValueError):
                break
            if gripper > args.open_gripper_max or not 0.005 <= mask_fraction <= 0.40:
                break
            selected.append(frame_index)
        if not selected or selected != list(range(len(selected))):
            continue
        key = f"rollout:{episode.resolve()}"
        token = hashlib.sha256(key.encode()).hexdigest()[:16]
        entry_id = f"rollout_open_prefix_{token}"
        edge_cache = args.output_dir / "cache" / f"{entry_id}_edges.npy"
        if args.force or not edge_cache.exists():
            if args.mask_mode != "stored_sam2":
                frames = read_video_rgb(original_video, limit=len(selected))
                masks = (
                    color_masks(frames)
                    if args.mask_mode == "color"
                    else color_hull_masks(frames)
                )
                edges = masked_frames_to_edges(frames, masks, args.batch_size, device)
            else:
                edges = read_video_gray(edge_video, limit=len(selected))
                if edges.shape[1:] != (256, 256):
                    edge_tensor = torch.from_numpy(edges).float().unsqueeze(1)
                    edge_tensor = F.interpolate(
                        edge_tensor,
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=False,
                    )
                    edges = torch.round(edge_tensor[:, 0]).byte().numpy()
            edge_rel, _ = save_arrays(args.output_dir, entry_id, edges, None, args.force)
        else:
            edge_rel = str(edge_cache.relative_to(args.output_dir))
        cached_edges = np.load(args.output_dir / edge_rel, mmap_mode="r")
        edge_fraction = float(np.mean(cached_edges > 2))
        if edge_fraction < args.min_rollout_edge_fraction:
            print(
                f"skip rollout {episode.name}: edge fraction {edge_fraction:.4f} "
                f"< {args.min_rollout_edge_fraction:.4f}"
            )
            continue
        count = int(cached_edges.shape[0])
        entries.append({
            "id": entry_id,
            "source_kind": "zero",
            "source_subtype": "rollout_open_prefix",
            "source_path": str(episode.resolve()),
            "split": stable_split(key, args.val_modulus),
            "split_group": str(episode.resolve()),
            "edge_path": edge_rel,
            "label_path": None,
            "constant_label_n": 0.0,
            "sample_count": count,
            "edge_fraction": edge_fraction,
            "label_policy": "initial_open_gripper_prefix_zero",
            "mask_mode": args.mask_mode,
        })
        print(f"rollout {episode.name}: {count}")
    return entries


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    entries = []
    entries.extend(sensor_entries(args, device))
    entries.extend(demo_entries(args, device))
    entries.extend(rollout_entries(args, device))
    if not entries:
        raise RuntimeError("No dataset entries were built")

    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket = counts.setdefault(entry["split"], {})
        subtype = entry["source_subtype"]
        bucket[subtype] = bucket.get(subtype, 0) + int(entry["sample_count"])
    manifest = {
        "format": "visualforce_cached_edge_v1",
        "name": args.output_dir.name,
        "description": (
            "Original data_ball_260422 physical Fz plus known-empty demo controls "
            "and conservative initial open-gripper demo/rollout prefixes."
        ),
        "input_mode": "edge",
        "mask_mode": args.mask_mode,
        "force_keys": ["Fz"],
        "sensor_label_policy": "original signed Fz; no estimator pseudo-labels",
        "zero_label_policy": (
            "Only explicit unloaded controls and initial open-gripper prefixes are zero."
        ),
        "build_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "demo_roots"
        } | {"demo_roots": [str(path) for path in args.demo_roots]},
        "counts": counts,
        "entries": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(counts, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
