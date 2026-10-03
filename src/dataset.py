"""
src/dataset.py  --  ForceDataset

Loads an episode directory and yields (masked_frame, force_label) pairs.

Pipeline per sample:
  1. Decode frame from video.mp4 via PyAV
  2. Load corresponding binary mask from mask.mp4
  3. Apply mask: zero-out pixels outside the gripper region
  4. Normalize to [0, 1] and resize to OUTPUT_SIZE
  5. Augment: random mild affine, brightness, and black occluders

Force label is linearly interpolated from the ~20 Hz force CSV to the
frame wall-clock timestamp.  The first and last `trim_seconds` are dropped.
"""

import os
import csv
import re
from typing import Optional

import av
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

OUTPUT_SIZE = (256, 256)
INPUT_MODES = ("rgb", "rgb_edge", "edge")

# Mild augmentation bounds
AUG_TRANSLATE = 0.08   # ±8% of image size
AUG_ROTATE_DEG = 5.0   # ±5 degrees
AUG_SCALE_LO  = 0.85
AUG_SCALE_HI  = 1.30
AUG_BRIGHTNESS_LO = 0.85
AUG_BRIGHTNESS_HI = 1.15
AUG_OCCLUDE_PROB = 0.50
AUG_OCCLUDE_MIN_FRAC = 0.05
AUG_OCCLUDE_MAX_FRAC = 0.25
AUG_OCCLUDE_LARGE_PROB = 0.35
AUG_OCCLUDE_LARGE_MIN_FRAC = 0.18
AUG_OCCLUDE_LARGE_MAX_FRAC = 0.48
AUG_OCCLUDE_MIN_ASPECT = 0.45
AUG_OCCLUDE_MAX_ASPECT = 2.00
AUG_OCCLUDE_LARGE_MIN_ASPECT = 0.70
AUG_OCCLUDE_LARGE_MAX_ASPECT = 1.45
AUG_OCCLUDE_CENTER_X_BAND = (0.35, 0.65)
AUG_OCCLUDE_CENTER_Y_BAND = (0.12, 0.62)
AUG_OCCLUDE_NUM_MAX = 2
AUG_OCCLUDE_SHAPES = ("rect", "ellipse")
AUG_OCCLUDE_MAX_ATTEMPTS = 12


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _decode_video(video_path: str) -> list[np.ndarray]:
    """Decode every frame into a list of uint8 RGB arrays."""
    frames = []
    container = av.open(video_path)
    for frame in container.decode(video=0):
        frames.append(np.array(frame.to_image().convert("RGB")))
    container.close()
    return frames


def _decode_mask(mask_path: str) -> list[np.ndarray]:
    """Decode mask video into a list of bool arrays (H, W)."""
    masks = []
    container = av.open(mask_path)
    for frame in container.decode(video=0):
        gray = np.array(frame.to_image().convert("L"))
        masks.append(gray > 127)
    container.close()
    return masks


def _episode_number(path: str) -> Optional[int]:
    match = re.search(r"EP0*(\d+)$", os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else None


def _random_occlude(
    frame_t: torch.Tensor,
    shapes: tuple[str, ...] = AUG_OCCLUDE_SHAPES,
) -> torch.Tensor:
    """Mask out random patches on the already masked 256x256 gripper input."""
    if torch.rand(1).item() >= AUG_OCCLUDE_PROB:
        return frame_t
    if not shapes:
        return frame_t

    nonzero = (frame_t.sum(dim=0) > 1e-6)
    ys, xs = torch.where(nonzero)
    if len(xs) == 0:
        return frame_t

    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    bbox_area = bbox_w * bbox_h

    out = frame_t.clone()
    n_occ = int(torch.randint(1, AUG_OCCLUDE_NUM_MAX + 1, (1,)).item())
    for _ in range(n_occ):
        if torch.rand(1).item() < AUG_OCCLUDE_LARGE_PROB:
            area_frac = (
                AUG_OCCLUDE_LARGE_MIN_FRAC
                + torch.rand(1).item()
                * (AUG_OCCLUDE_LARGE_MAX_FRAC - AUG_OCCLUDE_LARGE_MIN_FRAC)
            )
            aspect = (
                AUG_OCCLUDE_LARGE_MIN_ASPECT
                + torch.rand(1).item()
                * (AUG_OCCLUDE_LARGE_MAX_ASPECT - AUG_OCCLUDE_LARGE_MIN_ASPECT)
            )
        else:
            area_frac = (
                AUG_OCCLUDE_MIN_FRAC
                + torch.rand(1).item() * (AUG_OCCLUDE_MAX_FRAC - AUG_OCCLUDE_MIN_FRAC)
            )
            aspect = (
                AUG_OCCLUDE_MIN_ASPECT
                + torch.rand(1).item() * (AUG_OCCLUDE_MAX_ASPECT - AUG_OCCLUDE_MIN_ASPECT)
            )
        occ_area = max(1.0, bbox_area * area_frac / n_occ)
        occ_w = int(round((occ_area * aspect) ** 0.5))
        occ_h = int(round((occ_area / aspect) ** 0.5))
        occ_w = max(4, min(occ_w, bbox_w))
        occ_h = max(4, min(occ_h, bbox_h))

        cx_lo = x0 + int(bbox_w * AUG_OCCLUDE_CENTER_X_BAND[0])
        cx_hi = x0 + int(bbox_w * AUG_OCCLUDE_CENTER_X_BAND[1])
        cy_lo = y0 + int(bbox_h * AUG_OCCLUDE_CENTER_Y_BAND[0])
        cy_hi = y0 + int(bbox_h * AUG_OCCLUDE_CENTER_Y_BAND[1])
        if cx_hi <= cx_lo:
            cx_lo, cx_hi = x0, x1
        if cy_hi <= cy_lo:
            cy_lo, cy_hi = y0, y1

        for _attempt in range(AUG_OCCLUDE_MAX_ATTEMPTS):
            cx = int(torch.randint(cx_lo, max(cx_lo + 1, cx_hi), (1,)).item())
            cy = int(torch.randint(cy_lo, max(cy_lo + 1, cy_hi), (1,)).item())

            ox0 = max(0, cx - occ_w // 2)
            ox1 = min(out.shape[2], ox0 + occ_w)
            oy0 = max(0, cy - occ_h // 2)
            oy1 = min(out.shape[1], oy0 + occ_h)
            shape = shapes[int(torch.randint(0, len(shapes), (1,)).item())]

            if shape == "ellipse":
                yy, xx = torch.meshgrid(
                    torch.arange(oy0, oy1, device=out.device),
                    torch.arange(ox0, ox1, device=out.device),
                    indexing="ij",
                )
                rx = max(1.0, (ox1 - ox0) / 2.0)
                ry = max(1.0, (oy1 - oy0) / 2.0)
                occ_mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
            elif shape == "rect":
                occ_mask = torch.ones((oy1 - oy0, ox1 - ox0), dtype=torch.bool, device=out.device)
            else:
                raise ValueError(f"Unknown occlusion shape: {shape}")

            visible_hit = nonzero[oy0:oy1, ox0:ox1] & occ_mask
            if visible_hit.any() or _attempt == AUG_OCCLUDE_MAX_ATTEMPTS - 1:
                patch = out[:, oy0:oy1, ox0:ox1]
                patch[:, occ_mask] = 0.0
                break

    return out


def _append_edge_channel(frame_t: torch.Tensor) -> torch.Tensor:
    """Append a Sobel edge channel computed inside the visible masked region."""
    edge = _edge_channel(frame_t)
    return torch.cat([frame_t, edge], dim=0)


def _edge_channel(frame_t: torch.Tensor) -> torch.Tensor:
    """Return a Sobel edge channel computed inside the visible masked region."""
    gray = (
        0.2989 * frame_t[0:1]
        + 0.5870 * frame_t[1:2]
        + 0.1140 * frame_t[2:3]
    )
    visible = (frame_t.sum(dim=0, keepdim=True) > 1e-6).float()
    # Suppress the one-pixel mask silhouette so the edge channel emphasizes
    # texture/stripe deformation inside the gripper mask.
    interior = -F.max_pool2d(-visible.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=frame_t.dtype,
        device=frame_t.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=frame_t.dtype,
        device=frame_t.device,
    ).view(1, 1, 3, 3)
    gray_b = gray.unsqueeze(0)
    gx = F.conv2d(gray_b, sobel_x, padding=1)
    gy = F.conv2d(gray_b, sobel_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8)[0]
    edge = (edge / 4.0).clamp(0.0, 1.0) * interior
    return edge


class ForceDataset(Dataset):
    """
    Args:
        episode_dir:    path to episode directory containing video.mp4, mask.mp4,
        episode_dir:  path to episode directory containing video.mp4, mask.mp4,
                      frame_timestamps.csv, force_timestamps.csv
        force_keys:   which force columns to predict
        augment:      apply random affine/brightness augmentation
        trim_seconds: seconds to drop from the start and end of the episode
    """

    def __init__(
        self,
        episode_dir: str,
        force_keys: list[str] = ("Fy",),
        augment: bool = False,
        trim_seconds: float = 2.0,
        occluder_shapes: tuple[str, ...] = AUG_OCCLUDE_SHAPES,
        input_mode: str = "rgb",
    ):
        self.episode_dir  = episode_dir
        self.force_keys   = list(force_keys)
        self.augment      = augment
        self.occluder_shapes = tuple(occluder_shapes)
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
        self.input_mode = input_mode

        video_path    = os.path.join(episode_dir, "video.mp4")
        mask_path     = os.path.join(episode_dir, "mask.mp4")
        frame_ts_path = os.path.join(episode_dir, "frame_timestamps.csv")
        force_ts_path = os.path.join(episode_dir, "force_timestamps.csv")

        print(f"  Decoding {episode_dir} ...")
        video_frames = _decode_video(video_path)
        mask_frames  = _decode_mask(mask_path)

        frame_rows = _load_csv(frame_ts_path)
        frame_t    = np.array([float(r["t_rel_s"]) for r in frame_rows])

        force_rows = _load_csv(force_ts_path)
        force_t    = np.array([float(r["t_rel_s"]) for r in force_rows])
        force_wall = np.array([float(r["t_wall_s"]) for r in force_rows])
        frame_wall = np.array([float(r["t_wall_s"]) for r in frame_rows])
        force_vals = {
            k: np.array([float(r[k]) for r in force_rows])
            for k in self.force_keys
        }

        t_max = frame_t.max() if len(frame_t) > 0 else 0.0
        t_lo  = trim_seconds
        t_hi  = t_max - trim_seconds

        self.samples = []
        n = min(len(video_frames), len(mask_frames), len(frame_t))
        for i in range(n):
            t = frame_t[i]
            if t < t_lo or t > t_hi:
                continue

            # Interpolate force to this frame's wall-clock time
            t_wall = frame_wall[i]
            label = np.array(
                [np.interp(t_wall, force_wall, force_vals[k]) for k in self.force_keys],
                dtype=np.float32,
            )
            self.samples.append((video_frames[i], mask_frames[i], label))

        print(f"  {len(self.samples)} frames kept  (trimmed {n - len(self.samples)}, total {n})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        frame_np, mask_np, label = self.samples[idx]

        # Apply gripper mask: zero outside
        masked = frame_np.astype(np.float32) * mask_np[:, :, None]   # (H, W, 3)

        # To tensor (C, H, W) in [0, 1]
        frame_t = torch.from_numpy(masked / 255.0).permute(2, 0, 1).float()

        # Resize to network input size
        frame_t = TF.resize(frame_t, list(OUTPUT_SIZE), antialias=True)

        # Augmentation: random mild affine + brightness jitter + occlusion.
        if self.augment:
            h, w = OUTPUT_SIZE
            max_tx = int(w * AUG_TRANSLATE)
            max_ty = int(h * AUG_TRANSLATE)
            tx = torch.randint(-max_tx, max_tx + 1, (1,)).item()
            ty = torch.randint(-max_ty, max_ty + 1, (1,)).item()
            angle = (torch.rand(1).item() * 2.0 - 1.0) * AUG_ROTATE_DEG
            scale = (AUG_SCALE_LO + torch.rand(1).item() * (AUG_SCALE_HI - AUG_SCALE_LO))
            frame_t = TF.affine(
                frame_t,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=0,
                fill=0,
            )
            brightness = (
                AUG_BRIGHTNESS_LO
                + torch.rand(1).item() * (AUG_BRIGHTNESS_HI - AUG_BRIGHTNESS_LO)
            )
            frame_t = TF.adjust_brightness(frame_t, brightness)
            frame_t = frame_t.clamp(0.0, 1.0)
            frame_t = _random_occlude(frame_t, self.occluder_shapes)

        if self.input_mode == "rgb_edge":
            frame_t = _append_edge_channel(frame_t)
        elif self.input_mode == "edge":
            frame_t = _edge_channel(frame_t)

        return {
            "frame": frame_t,                       # (C, H, W)
            "force": torch.from_numpy(label),       # (force_dim,)
        }


def make_datasets(
    episode_dirs: list[str],
    val_episode: Optional[str] = None,
    val_count: int = 5,
    val_multiple: Optional[int] = None,
    force_keys: tuple = ("Fy",),
    trim_seconds: float = 2.0,
    input_mode: str = "rgb",
) -> tuple[Dataset, Dataset]:
    """
    Build train and validation datasets by holding out validation episodes.
    """
    from torch.utils.data import ConcatDataset

    if val_episode is not None:
        val_dirs = [val_episode]
    elif val_multiple is not None:
        if val_multiple <= 0:
            raise ValueError("val_multiple must be positive")
        val_dirs = [
            d for d in episode_dirs
            if (num := _episode_number(d)) is not None and num % val_multiple == 0
        ]
        if not val_dirs:
            raise ValueError(f"No validation episodes matched EP number multiple of {val_multiple}")
        if len(val_dirs) >= len(episode_dirs):
            raise ValueError("val_multiple selected every episode; no training episodes remain")
    else:
        if val_count <= 0:
            raise ValueError("val_count must be positive")
        if val_count >= len(episode_dirs):
            raise ValueError("val_count must be smaller than the number of episodes")
        val_dirs = list(episode_dirs[-val_count:])

    val_set = set(val_dirs)
    train_dirs = [d for d in episode_dirs if d not in val_set]
    print(f"Validation episodes: {[os.path.basename(d) for d in val_dirs]}")
    print(f"Training episodes: {[os.path.basename(d) for d in train_dirs]}")

    print("Building training datasets:")
    train_ds = ConcatDataset([
        ForceDataset(
            d,
            force_keys=force_keys,
            augment=True,
            trim_seconds=trim_seconds,
            input_mode=input_mode,
        )
        for d in train_dirs
    ])
    print("Building validation datasets:")
    val_ds = ConcatDataset([
        ForceDataset(
            d,
            force_keys=force_keys,
            augment=False,
            trim_seconds=trim_seconds,
            input_mode=input_mode,
        )
        for d in val_dirs
    ])

    return train_ds, val_ds
