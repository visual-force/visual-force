"""
scripts/evaluate.py  --  Evaluate a trained VisualForce checkpoint on one episode.

Outputs:
  - Console: MAE, RMSE per force axis
  - Plot: predicted vs ground-truth force over time  ->  <output_dir>/<episode>/force_curve.png

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
                               --episode data/episodes/EP000002

    python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
                               --data-dir data/data_ball_260422 --split val
"""

import argparse
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import ForceDataset
from src.model.unet import build_unet
from src.training_utils import resolve_device


INPUT_CHANNELS = {
    "rgb": 3,
    "rgb_edge": 4,
    "edge": 1,
}


def find_episodes(data_dir: str) -> list[str]:
    return sorted(
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if d.startswith("EP") and os.path.isdir(os.path.join(data_dir, d))
        and os.path.isfile(os.path.join(data_dir, d, "video.mp4"))
        and os.path.isfile(os.path.join(data_dir, d, "mask.mp4"))
    )


def episode_number(path: str) -> Optional[int]:
    match = re.search(r"EP0*(\d+)$", os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", default=None,
                        help="Episode directory to evaluate on.")
    source.add_argument("--episodes", nargs="+", default=None,
                        help="Explicit list of episode directories to evaluate.")
    source.add_argument("--data-dir", default=None,
                        help="Dataset folder containing EP* episode subdirectories.")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all",
                        help="Subset to evaluate when --data-dir is used.")
    parser.add_argument("--val-count", type=int, default=None,
                        help="Number of final sorted episodes used for validation. "
                             "Defaults to checkpoint args, then 5.")
    parser.add_argument("--val-multiple", type=int, default=None,
                        help="Validation split by EP number multiple. Defaults to "
                             "checkpoint args when available.")
    parser.add_argument("--output", default="eval_output",
                        help="Directory for output plots.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size (default: 64).")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (default: 0).")
    parser.add_argument("--device", default=None,
                        help="Torch device. Defaults to CUDA when available, otherwise CPU.")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def resolve_eval_episodes(args, ckpt) -> list[str]:
    if args.episodes is not None:
        return args.episodes
    if args.episode is not None:
        return [args.episode]
    if args.data_dir is None:
        raise ValueError("Pass --episode, --episodes, or --data-dir.")

    episodes = find_episodes(args.data_dir)
    if not episodes:
        raise RuntimeError(f"No EP* folders with video.mp4 + mask.mp4 found under {args.data_dir}")

    ckpt_args = ckpt.get("args", {})
    val_multiple = args.val_multiple
    if val_multiple is None:
        val_multiple = ckpt_args.get("val_multiple")

    if val_multiple is not None:
        if val_multiple <= 0:
            raise ValueError("val-multiple must be positive")
        val_episodes = [
            ep for ep in episodes
            if (num := episode_number(ep)) is not None and num % val_multiple == 0
        ]
        if not val_episodes:
            raise ValueError(f"No validation episodes matched EP number multiple of {val_multiple}")
        if len(val_episodes) >= len(episodes):
            raise ValueError("val-multiple selected every episode; no training episodes remain")
        val_set = set(val_episodes)
        if args.split == "train":
            return [ep for ep in episodes if ep not in val_set]
        if args.split == "val":
            return val_episodes
        return episodes

    val_count = args.val_count
    if val_count is None:
        val_count = ckpt_args.get("val_count", ckpt.get("val_count", 5))
    if val_count <= 0 or val_count >= len(episodes):
        raise ValueError(f"val-count must be in [1, {len(episodes) - 1}], got {val_count}")

    if args.split == "train":
        return episodes[:-val_count]
    if args.split == "val":
        return episodes[-val_count:]
    return episodes


def evaluate_episode(
    model,
    episode: str,
    force_keys: list[str],
    trim_seconds: float,
    device: torch.device,
    output_root: str,
    batch_size: int,
    workers: int,
):
    episode_name = os.path.basename(os.path.normpath(episode))

    # ---- Dataset ----------------------------------------------------------
    print(f"\nLoading episode: {episode}")
    ds = ForceDataset(
        episode,
        force_keys=force_keys,
        augment=False,
        trim_seconds=trim_seconds,
        input_mode=model.input_mode,
    )
    if len(ds) == 0:
        print("No valid frames found.")
        return None

    # ---- Run inference ----------------------------------------------------
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=workers > 0,
    )
    all_pred = []
    all_gt = []

    with torch.inference_mode():
        for batch in loader:
            frame = batch["frame"].to(device, non_blocking=use_cuda)
            all_pred.append(model(frame).cpu().numpy())
            all_gt.append(batch["force"].numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    all_idx = np.arange(len(ds))

    # ---- Metrics ----------------------------------------------------------
    mae  = np.abs(all_pred - all_gt).mean(axis=0)
    rmse = np.sqrt(((all_pred - all_gt) ** 2).mean(axis=0))
    print("--- Results ---")
    for i, k in enumerate(force_keys):
        print(f"  {k}:  MAE={mae[i]:.4f} N   RMSE={rmse[i]:.4f} N")

    # ---- Time-series plot -------------------------------------------------
    n_axes = len(force_keys)
    fig, axes = plt.subplots(n_axes, 1, figsize=(12, 3 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]

    for i, (ax, k) in enumerate(zip(axes, force_keys)):
        ax.plot(all_idx, all_gt[:, i],   label="Ground truth", color="steelblue", linewidth=1.5)
        ax.plot(all_idx, all_pred[:, i], label="Predicted",    color="tomato",    linewidth=1.5, linestyle="--")
        ax.set_ylabel(f"{k} (N)")
        ax.set_title(f"{k}  MAE={mae[i]:.4f} N  RMSE={rmse[i]:.4f} N")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"VisualForce - {episode_name}", fontsize=12)
    fig.tight_layout()
    episode_out = os.path.join(output_root, episode_name)
    os.makedirs(episode_out, exist_ok=True)
    curve_path = os.path.join(episode_out, "force_curve.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Force curve saved: {curve_path}")

    return {
        "episode": episode_name,
        "n": len(ds),
        "mae": mae,
        "rmse": rmse,
    }


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = resolve_device(args.device)

    # ---- Load checkpoint --------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    force_keys = ckpt["force_keys"]
    force_dim  = len(force_keys)
    ckpt_args = ckpt.get("args", {})
    trim_seconds = ckpt.get("trim_seconds", ckpt.get("args", {}).get("trim_seconds", 2.0))
    force_pooling = ckpt_args.get("force_pooling", "avg")
    force_spatial_size = ckpt_args.get("force_spatial_size", 4)
    input_mode = ckpt_args.get("input_mode", "rgb")
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unsupported checkpoint input_mode: {input_mode!r}")
    print(f"Checkpoint from epoch {ckpt['epoch']}  val_MAE={ckpt['val_mae']:.4f} N")
    print(f"Force keys: {force_keys}")
    print(f"Trim seconds: {trim_seconds}")
    print(f"Input mode: {input_mode}")
    print(f"Force pooling: {force_pooling}")
    print(f"Device: {device}")

    model = build_unet(
        in_channels=INPUT_CHANNELS[input_mode],
        encoder_channels=(32, 64, 128, 256),
        force_dim=force_dim,
        force_hidden_dim=256,
        force_dropout=0.0,
        force_pooling=force_pooling,
        force_spatial_size=force_spatial_size,
        encoder_only=True,
    ).to(device)
    model.input_mode = input_mode
    model.load_state_dict(ckpt["model"])
    model.eval()

    episodes = resolve_eval_episodes(args, ckpt)
    print(f"Episodes ({len(episodes)}): {[os.path.basename(e) for e in episodes]}")

    results = []
    for episode in episodes:
        result = evaluate_episode(
            model,
            episode,
            force_keys,
            trim_seconds,
            device,
            args.output,
            args.batch_size,
            args.workers,
        )
        if result is not None:
            results.append(result)

    if len(results) > 1:
        total = sum(r["n"] for r in results)
        weighted_mae = sum(r["mae"] * r["n"] for r in results) / total
        weighted_rmse = np.sqrt(sum((r["rmse"] ** 2) * r["n"] for r in results) / total)

        print("\n--- Aggregate ---")
        print(f"Samples: {total}")
        for i, k in enumerate(force_keys):
            print(f"  {k}:  MAE={weighted_mae[i]:.4f} N   RMSE={weighted_rmse[i]:.4f} N")


if __name__ == "__main__":
    main()
