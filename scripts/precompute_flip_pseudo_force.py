"""
Precompute pseudo force labels for flip episodes.

The flip datasets do not contain force-torque sensor readings. This script uses
a frozen VisualForce checkpoint to assign a per-frame pseudo force to each
episode and stores the result inside the episode directory. These labels are
then used to train an action-conditioned delta-force critic.
"""

import argparse
import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.flip_delta_force_dataset import discover_flip_episodes, read_video_rgb
from src.steering import VisualForceEstimator, masked_frame_to_tensor


SAM2_MODEL_CFGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
H_MIN, H_MAX = 130.0, 185.0
S_MIN = 0.30
V_MIN = 0.20
MIN_BLOB_AREA = 300


def color_gripper_mask(frame_rgb: np.ndarray) -> np.ndarray:
    hsv = np.array(Image.fromarray(frame_rgb).convert("HSV")).astype(np.float32)
    hue = hsv[:, :, 0] / 255.0 * 360.0
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    mask = (
        (hue >= H_MIN)
        & (hue <= H_MAX)
        & (sat >= S_MIN)
        & (val >= V_MIN)
    ).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _centroids = cv.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n_labels):
        if stats[label, cv.CC_STAT_AREA] >= MIN_BLOB_AREA:
            keep |= labels == label
    return keep


def color_gripper_hull_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Fill one convex fin hull per image half from the color segmentation."""
    color_mask = color_gripper_mask(frame_rgb)
    height, width = color_mask.shape
    mid = width // 2
    hull_mask = np.zeros((height, width), dtype=np.uint8)
    for region, x_offset in ((color_mask[:, :mid], 0), (color_mask[:, mid:], mid)):
        ys, xs = np.where(region)
        if len(xs) < MIN_BLOB_AREA:
            continue
        points = np.column_stack((xs + x_offset, ys)).astype(np.int32)
        cv.fillConvexPoly(hull_mask, cv.convexHull(points), 1)
    return hull_mask.astype(bool)


def predict_episode_force(
    estimator: VisualForceEstimator,
    frames: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) == 0:
        raise ValueError("Cannot predict force for an empty video")
    tensors = []
    preds = []
    for frame, mask in zip(frames, masks):
        mask = mask.astype(bool)
        tensors.append(
            masked_frame_to_tensor(
                frame,
                mask,
                input_mode=estimator.input_mode,
                edge_normalization=estimator.edge_normalization,
            )
        )
        if len(tensors) >= batch_size:
            preds.append(_predict_batch(estimator, tensors))
            tensors = []
    if tensors:
        preds.append(_predict_batch(estimator, tensors))
    force = np.concatenate(preds, axis=0).astype(np.float32)
    mask_fracs = masks.reshape(len(masks), -1).mean(axis=1, dtype=np.float32)
    return force, mask_fracs


def make_edge_observations(
    frames: np.ndarray,
    masks: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    edges = np.empty((len(frames), *image_size), dtype=np.uint8)
    for index, (frame, mask) in enumerate(zip(frames, masks)):
        edge = masked_frame_to_tensor(
            frame,
            mask.astype(bool),
            output_size=image_size,
            input_mode="edge",
        )[0]
        edge_np = edge.detach().cpu().numpy()
        edges[index] = np.rint(edge_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return edges


def color_masks(frames: np.ndarray) -> np.ndarray:
    return np.stack([color_gripper_mask(frame) for frame in frames], axis=0)


def color_hull_masks(frames: np.ndarray) -> np.ndarray:
    return np.stack([color_gripper_hull_mask(frame) for frame in frames], axis=0)


def build_sam2_video_predictor(
    model_key: str,
    sam2_repo: Union[str, Path],
    sam2_ckpt: Union[str, Path],
    device: Optional[str] = None,
):
    sam2_repo = Path(sam2_repo).expanduser().resolve()
    sam2_ckpt = Path(sam2_ckpt).expanduser().resolve()
    if not sam2_repo.is_dir():
        raise FileNotFoundError(f"SAM2 repo not found: {sam2_repo}")
    if not sam2_ckpt.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_ckpt}")
    if model_key not in SAM2_MODEL_CFGS:
        raise ValueError(f"Unsupported SAM2 model: {model_key!r}")

    sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(
        SAM2_MODEL_CFGS[model_key],
        str(sam2_ckpt),
        device=device,
    )
    return predictor, device


def detect_prompt_points(frame_rgb: np.ndarray):
    import segment_gripper

    return segment_gripper.detect_orange_points(frame_rgb)


def sam2_masks(
    frames: np.ndarray,
    predictor,
    device: str,
    fallback_color: bool = False,
    separate_objects: bool = False,
) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("Cannot segment an empty video")
    with tempfile.TemporaryDirectory(prefix="visualforce_flip_sam2_") as frame_dir:
        frame_dir = Path(frame_dir)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"{i:06d}.jpg", quality=95)

        point_coords, point_labels = detect_prompt_points(frames[0])
        device_type = torch.device(device).type
        autocast = (
            torch.autocast(device_type, dtype=torch.bfloat16)
            if device_type == "cuda"
            else contextlib.nullcontext()
        )
        segments = {}
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(frame_dir))
            predictor.reset_state(state)
            if separate_objects:
                for obj_id, (point, label) in enumerate(
                    zip(point_coords, point_labels), start=1
                ):
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=point[None, :],
                        labels=np.asarray([label], dtype=np.int32),
                    )
            else:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=point_coords,
                    labels=point_labels,
                )
            for out_idx, _obj_ids, out_logits in predictor.propagate_in_video(state):
                logits = (out_logits > 0.0).cpu().numpy()
                segments[out_idx] = np.any(logits, axis=0).squeeze()

    masks = []
    for i, frame in enumerate(frames):
        mask = segments.get(i)
        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=bool)
        mask = mask.astype(bool)
        if fallback_color and (mask.mean() < 0.01 or mask.mean() > 0.40):
            mask = color_gripper_mask(frame)
        masks.append(mask)
    return np.stack(masks, axis=0)


def make_masks(
    frames: np.ndarray,
    mask_mode: str,
    predictor=None,
    sam2_device: Optional[str] = None,
    fallback_color: bool = False,
) -> np.ndarray:
    if mask_mode == "color":
        return color_masks(frames)
    if mask_mode == "color_hull":
        return color_hull_masks(frames)
    if mask_mode == "sam2":
        if predictor is None or sam2_device is None:
            raise ValueError("SAM2 predictor is required when mask_mode='sam2'")
        return sam2_masks(
            frames,
            predictor=predictor,
            device=sam2_device,
            fallback_color=fallback_color,
        )
    if mask_mode == "sam2_jaws":
        if predictor is None or sam2_device is None:
            raise ValueError("SAM2 predictor is required when mask_mode='sam2_jaws'")
        return sam2_masks(
            frames,
            predictor=predictor,
            device=sam2_device,
            fallback_color=fallback_color,
            separate_objects=True,
        )
    raise ValueError(f"Unsupported mask_mode: {mask_mode!r}")


def write_mask_preview(
    ep_dir: Path,
    frames: np.ndarray,
    masks: np.ndarray,
    preview_dir: Path,
    preview_frames: int,
    mask_mode: str,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    if n <= 0:
        return
    idxs = np.linspace(0, n - 1, num=min(preview_frames, n), dtype=int).tolist()
    cols = []
    for idx in idxs:
        frame = frames[idx]
        mask = masks[idx].astype(bool)
        masked = frame.copy()
        masked[~mask] = 0
        overlay = frame.copy()
        overlay[mask] = (
            0.4 * overlay[mask].astype(np.float32)
            + 0.6 * np.array([255, 40, 40], dtype=np.float32)
        ).astype(np.uint8)
        mask_rgb = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8) * 255

        rows = []
        for name, img in [
            ("rgb", frame),
            ("overlay", overlay),
            ("masked", masked),
            ("mask", mask_rgb),
        ]:
            tile = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            tile = cv.resize(tile, (180, 135), interpolation=cv.INTER_AREA)
            cv.rectangle(tile, (0, 0), (179, 20), (0, 0, 0), -1)
            cv.putText(
                tile,
                f"{name} f{idx}",
                (4, 14),
                cv.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            rows.append(tile)
        cols.append(np.vstack(rows))
    out_path = preview_dir / f"{ep_dir.name}_{mask_mode}_mask_preview.jpg"
    cv.imwrite(str(out_path), np.hstack(cols))


def _predict_batch(estimator: VisualForceEstimator, tensors: list[torch.Tensor]) -> np.ndarray:
    use_cuda = torch.device(estimator.device).type == "cuda"
    x = torch.stack(tensors, dim=0).to(estimator.device, non_blocking=use_cuda)
    with torch.inference_mode():
        return estimator.model(x).detach().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Flip dataset folder containing episodes/ or episode directories.",
    )
    parser.add_argument(
        "--visualforce-ckpt",
        default="",
    )
    parser.add_argument("--output-name", default="visualforce_pseudo_force_fz.npz")
    parser.add_argument(
        "--edge-output-name",
        default=None,
        help=(
            "Optional per-episode npz name for segmented Sobel edge observations "
            "used by action-conditioned force critics."
        ),
    )
    parser.add_argument("--force-keys", nargs="+", default=["Fz"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", nargs=2, type=int, default=[240, 320])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mask-mode",
        choices=["sam2", "sam2_jaws", "color", "color_hull"],
        default="sam2",
        help=(
            "Mask source for pseudo labels. sam2 matches the ForceLens pipeline; "
            "sam2_jaws tracks the two gripper jaws as separate objects; color_hull "
            "fills one deterministic color-seeded fin hull per image half."
        ),
    )
    parser.add_argument("--sam2-model", choices=SAM2_MODEL_CFGS.keys(), default="small")
    parser.add_argument(
        "--sam2-repo",
        default="third_party/sam2",
        help="Local SAM2 repo path.",
    )
    parser.add_argument(
        "--sam2-ckpt",
        default="third_party/sam2/checkpoints/sam2.1_hiera_small.pt",
        help="Local SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--sam2-fallback-color",
        action="store_true",
        help="Use color mask only if SAM2 returns an obviously bad mask.",
    )
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="Optional directory for mask preview contact sheets.",
    )
    parser.add_argument("--preview-frames", type=int, default=6)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write mask previews and exit without running ForceLens inference.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit processed episodes, useful with --preview-only.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if any(size <= 0 for size in args.image_size):
        parser.error("--image-size values must be positive")
    if args.preview_frames <= 0:
        parser.error("--preview-frames must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    return args


def main():
    args = parse_args()
    episodes = discover_flip_episodes(args.dataset_path)
    if not episodes:
        raise RuntimeError(f"No flip episodes found under {args.dataset_path}")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if args.preview_only and preview_dir is None:
        raise ValueError("--preview-only requires --preview-dir")

    if args.preview_only:
        predictor = None
        sam2_device = None
        if args.mask_mode in {"sam2", "sam2_jaws"}:
            predictor, sam2_device = build_sam2_video_predictor(
                args.sam2_model,
                args.sam2_repo,
                args.sam2_ckpt,
                args.device,
            )
        for ep_dir in tqdm(episodes, desc="preview episodes"):
            frames = read_video_rgb(ep_dir / "wrist_image.mp4", tuple(args.image_size))
            masks = make_masks(
                frames,
                args.mask_mode,
                predictor=predictor,
                sam2_device=sam2_device,
                fallback_color=args.sam2_fallback_color,
            )
            write_mask_preview(
                ep_dir,
                frames,
                masks,
                preview_dir,
                args.preview_frames,
                args.mask_mode,
            )
        print(f"Preview sheets saved under {preview_dir}")
        return

    estimator = VisualForceEstimator(
        args.visualforce_ckpt,
        device=args.device,
        force_keys=args.force_keys,
    )
    predictor = None
    sam2_device = None
    if args.mask_mode in {"sam2", "sam2_jaws"}:
        predictor, sam2_device = build_sam2_video_predictor(
            args.sam2_model,
            args.sam2_repo,
            args.sam2_ckpt,
            args.device,
        )

    for ep_dir in tqdm(episodes, desc="episodes"):
        out_path = ep_dir / args.output_name
        edge_path = ep_dir / args.edge_output_name if args.edge_output_name else None
        need_force = args.overwrite or not out_path.exists()
        need_edge = (
            edge_path is not None
            and (args.overwrite or not edge_path.exists())
        )
        if not need_force and not need_edge:
            continue
        frames = read_video_rgb(ep_dir / "wrist_image.mp4", tuple(args.image_size))
        masks = make_masks(
            frames,
            args.mask_mode,
            predictor=predictor,
            sam2_device=sam2_device,
            fallback_color=args.sam2_fallback_color,
        )
        if preview_dir is not None:
            write_mask_preview(
                ep_dir,
                frames,
                masks,
                preview_dir,
                args.preview_frames,
                args.mask_mode,
            )
        if need_edge:
            edge = make_edge_observations(
                frames,
                masks,
                tuple(args.image_size),
            )
            np.savez_compressed(
                edge_path,
                edge=edge,
                image_size=np.asarray(args.image_size),
                input_mode="edge",
                mask_mode=args.mask_mode,
                edge_normalization="none",
            )
            print(f"{ep_dir}: wrote {edge_path.name}, frames={len(edge)}")
        if not need_force:
            continue
        force, mask_frac = predict_episode_force(
            estimator,
            frames,
            masks,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            out_path,
            force=force,
            force_keys=np.asarray(args.force_keys),
            mask_frac=mask_frac,
            visualforce_ckpt=str(Path(args.visualforce_ckpt).resolve()),
            input_mode=estimator.input_mode,
            mask_mode=args.mask_mode,
            edge_normalization=estimator.edge_normalization,
        )
        print(
            f"{ep_dir}: wrote {out_path.name}, "
            f"frames={len(force)}, mask_frac median={np.median(mask_frac):.4f}"
        )


if __name__ == "__main__":
    main()
