"""
Segment gripper from video using SAM2.

Auto-detects the orange gripper on frame 0 via HSV thresholding, then propagates
the mask with SAM2 across all frames.  Writes two files alongside the input video:
  - mask.mp4     : binary mask (white = gripper, black = background)
  - overlay.mp4  : original frames blended with a coloured mask

Usage:
    python segment_gripper.py <video.mp4>
    python segment_gripper.py <video.mp4> --point 320 240   # override auto-detect
    python segment_gripper.py <video.mp4> --model small
"""

import argparse
import os
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# SAM2 paths
# ---------------------------------------------------------------------------
# SAM2 is loaded from the bundled submodule by default; set SAM2_ROOT to use a
# checkout elsewhere. Weights are read from SAM2_CKPT when set, otherwise from
# the submodule's checkpoints/ directory (populate it with
# `third_party/sam2/checkpoints/download_ckpts.sh`). If no local checkpoint is
# found the weights are fetched from Hugging Face instead.
REPO_ROOT = Path(__file__).resolve().parent
SAM2_REPO = os.environ.get("SAM2_ROOT") or str(REPO_ROOT / "third_party" / "sam2")

MODEL_CFGS = {
    "tiny":  "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

CKPT_FILENAMES = {
    "tiny":  "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base":  "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}

# Fallback HuggingFace IDs for models without a local checkpoint
HF_MODEL_IDS = {
    "tiny":  "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base":  "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def local_checkpoint(model_key: str):
    """Return a local SAM2 checkpoint path for model_key, or None if absent."""
    override = os.environ.get("SAM2_CKPT")
    if override:
        return override
    candidate = Path(SAM2_REPO) / "checkpoints" / CKPT_FILENAMES[model_key]
    return str(candidate) if candidate.is_file() else None


# ---------------------------------------------------------------------------
# Green HSV thresholds (gripper colour)
# ---------------------------------------------------------------------------
H_MIN, H_MAX = 130, 185   # hue in degrees (green/teal fin-ray gripper)
S_MIN = 0.30
V_MIN = 0.20
MIN_BLOB_AREA = 300

# Overlay appearance
MASK_COLOR = np.array([255, 50, 50], dtype=np.uint8)
MASK_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_orange_points(img_np: np.ndarray):
    """Return (point_coords, point_labels) by finding orange blobs in frame 0."""
    hsv = np.array(Image.fromarray(img_np).convert("HSV")).astype(np.float32)
    H = hsv[:, :, 0] / 255.0 * 360.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0
    mask = (H >= H_MIN) & (H <= H_MAX) & (S >= S_MIN) & (V >= V_MIN)

    h, w = mask.shape
    mid = w // 2
    coords = []
    for region, x_offset in [(mask[:, :mid], 0), (mask[:, mid:], mid)]:
        if region.sum() < MIN_BLOB_AREA:
            continue
        ys, xs = np.where(region)
        coords.append([float(xs.mean() + x_offset), float(ys.mean())])

    if not coords:
        print("  [warn] Orange gripper not detected — falling back to centre-left / centre-right")
        coords = [[w * 0.2, h * 0.6], [w * 0.8, h * 0.6]]

    coords_arr = np.array(coords, dtype=np.float32)
    labels_arr = np.ones(len(coords_arr), dtype=np.int32)
    print(f"  Prompt points: {coords_arr.tolist()}")
    return coords_arr, labels_arr


def extract_all_frames(video_path: str, frame_dir: str) -> tuple[int, int, float]:
    """Extract every frame as JPEG. Returns (width, height, fps)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    w, h = stream.width, stream.height
    for i, frame in enumerate(container.decode(video=0)):
        frame.to_image().save(os.path.join(frame_dir, f"{i:06d}.jpg"), quality=90)
    container.close()
    n = len([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    print(f"  {n} frames  {w}x{h}  {fps:.1f} fps")
    return w, h, fps


def apply_overlay(img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend coloured mask over image and draw a bright border."""
    out = img_np.copy()
    out[mask] = (
        (1 - MASK_ALPHA) * img_np[mask].astype(np.float32)
        + MASK_ALPHA * MASK_COLOR.astype(np.float32)
    ).astype(np.uint8)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(mask, 1, mode="edge")
    border = mask & ~(sliding_window_view(pad, (3, 3)).all(axis=(-1, -2)))
    out[border] = [255, 80, 0]
    return out


def write_video(frames_iter, fps: float, out_path: str, width: int, height: int):
    """Write an iterable of (H x W x 3) uint8 RGB arrays to MP4."""
    rate = Fraction(fps).limit_denominator(1000)
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    for frame_np in frames_iter:
        av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def build_predictor(model_key: str):
    sys.path.insert(0, SAM2_REPO)
    from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    ckpt = local_checkpoint(model_key)
    if ckpt and os.path.isfile(ckpt):
        print(f"  Loading checkpoint: {ckpt}")
        predictor = build_sam2_video_predictor(MODEL_CFGS[model_key], ckpt, device=device)
    else:
        print(f"  Checkpoint not cached locally, downloading {HF_MODEL_IDS[model_key]} ...")
        predictor = build_sam2_video_predictor_hf(HF_MODEL_IDS[model_key], device=device)

    return predictor, device


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(video_path: str, init_points=None, init_labels=None, model_key: str = "small"):
    out_dir = os.path.dirname(os.path.abspath(video_path))
    mask_path    = os.path.join(out_dir, "mask.mp4")
    overlay_path = os.path.join(out_dir, "overlay.mp4")

    frame_dir = tempfile.mkdtemp(prefix="visualforce_frames_")
    try:
        # 1. Extract frames
        print("[1/4] Extracting frames ...")
        width, height, fps = extract_all_frames(video_path, frame_dir)
        frame_names = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))

        # 2. Auto-detect or use provided prompt
        print("[2/4] Determining prompt points ...")
        if init_points is None:
            first_frame_np = np.array(
                Image.open(os.path.join(frame_dir, frame_names[0])).convert("RGB")
            )
            point_coords, point_labels = detect_orange_points(first_frame_np)
        else:
            point_coords = np.array(init_points, dtype=np.float32)
            point_labels = np.array(init_labels, dtype=np.int32)
            print(f"  Using provided points: {point_coords.tolist()}")

        # 3. SAM2 propagation
        print(f"[3/4] Loading SAM2 ({model_key}) and propagating ...")
        predictor, device = build_predictor(model_key)

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = predictor.init_state(video_path=frame_dir)
            predictor.reset_state(state)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=point_coords,
                labels=point_labels,
            )
            video_segments: dict[int, np.ndarray] = {}
            for out_idx, _obj_ids, out_logits in predictor.propagate_in_video(state):
                video_segments[out_idx] = (out_logits[0] > 0.0).cpu().numpy().squeeze()

        n = len(frame_names)

        # 4a. Write binary mask video
        print(f"[4/4] Writing {mask_path} ...")
        def mask_frames():
            for i in range(n):
                m = video_segments.get(i, np.zeros((height, width), dtype=bool))
                yield np.stack([m.astype(np.uint8) * 255] * 3, axis=-1)
        write_video(mask_frames(), fps, mask_path, width, height)

        # 4b. Write overlay video
        print(f"      Writing {overlay_path} ...")
        def overlay_frames():
            for i, fname in enumerate(frame_names):
                img = np.array(Image.open(os.path.join(frame_dir, fname)).convert("RGB"))
                if i in video_segments:
                    img = apply_overlay(img, video_segments[i])
                yield img
        write_video(overlay_frames(), fps, overlay_path, width, height)

        print(f"Done!\n  {mask_path}\n  {overlay_path}")

    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def find_episode_videos(path: str) -> list[str]:
    """Given a dataset folder, return sorted list of EP*/video.mp4 paths."""
    ep_dirs = sorted(
        d for d in os.listdir(path)
        if d.startswith("EP") and os.path.isdir(os.path.join(path, d))
    )
    videos = []
    for ep in ep_dirs:
        vp = os.path.join(path, ep, "video.mp4")
        if os.path.isfile(vp):
            videos.append(vp)
        else:
            print(f"  [skip] {ep}: no video.mp4 found")
    return videos


def main():
    parser = argparse.ArgumentParser(description="Segment gripper with SAM2")
    parser.add_argument("input",
                        help="Path to a single video.mp4 OR a dataset folder containing EP* episode folders.")
    parser.add_argument("--point", nargs=2, type=int, action="append", metavar=("X", "Y"),
                        help="Override auto-detect: gripper point on frame 0 (repeatable).")
    parser.add_argument("--neg", nargs=2, type=int, action="append", metavar=("X", "Y"),
                        help="Negative (background) point on frame 0 (repeatable).")
    parser.add_argument("--model", choices=HF_MODEL_IDS.keys(), default="small",
                        help="SAM2 model size (default: small)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip episodes that already have mask.mp4.")
    args = parser.parse_args()

    init_points, init_labels = None, None
    if args.point:
        init_points = list(args.point)
        init_labels = [1] * len(init_points)
        if args.neg:
            init_points += list(args.neg)
            init_labels += [0] * len(args.neg)

    # Resolve input: single video or dataset folder
    if os.path.isfile(args.input):
        videos = [args.input]
    elif os.path.isdir(args.input):
        videos = find_episode_videos(args.input)
        if not videos:
            sys.exit(f"No EP*/video.mp4 found under {args.input}")
        print(f"Found {len(videos)} episodes under {args.input}")
    else:
        sys.exit(f"Error: path not found: {args.input}")

    for i, vp in enumerate(videos):
        ep_name = os.path.basename(os.path.dirname(vp))
        print(f"\n=== [{i+1}/{len(videos)}] {ep_name} ===")
        if args.skip_existing and os.path.isfile(os.path.join(os.path.dirname(vp), "mask.mp4")):
            print("  mask.mp4 already exists, skipping.")
            continue
        run(vp, init_points, init_labels, args.model)


if __name__ == "__main__":
    main()
