#!/usr/bin/env python3
"""Train the edge-only VisualForce estimator from a cached mixed manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cached_edge_force_dataset import (
    SOURCE_SENSOR,
    SOURCE_ZERO,
    CachedEdgeForceDataset,
    make_balanced_sampler,
)
from src.model.unet import build_unet
from src.steering import EDGE_NORMALIZATIONS
from src.training_utils import load_wandb, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/visualforce_fz_edge_real_zero_v1/manifest.json"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--zero-fraction", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-occlusion",
        action="store_true",
        help="Disable random black occluders while retaining affine/brightness augmentation.",
    )
    parser.add_argument(
        "--edge-normalization",
        choices=EDGE_NORMALIZATIONS,
        default="none",
        help="Optional per-frame edge-contrast normalization.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument(
        "--output", type=Path,
        default=Path("checkpoints/visualforce_fz_edge_real_zero_v1"),
    )
    parser.add_argument("--wandb-project", default="force_estimation")
    parser.add_argument("--wandb-run", default="visualforce_fz_edge_real_zero_v1")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.workers < 0 or args.weight_decay < 0:
        parser.error("workers and weight decay must be non-negative")
    if not 0.0 < args.zero_fraction < 1.0:
        parser.error("--zero-fraction must be strictly between zero and one")
    return args


def validation_metrics(model, loader, device: torch.device) -> dict[str, float]:
    absolute = {SOURCE_SENSOR: 0.0, SOURCE_ZERO: 0.0}
    counts = {SOURCE_SENSOR: 0, SOURCE_ZERO: 0}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frame"].to(device, non_blocking=device.type == "cuda")
            force = batch["force"].to(device, non_blocking=device.type == "cuda")
            source = batch["source_id"]
            errors = (model(frames) - force).abs().cpu()
            for source_id in (SOURCE_SENSOR, SOURCE_ZERO):
                mask = source == source_id
                absolute[source_id] += errors[mask].sum().item()
                counts[source_id] += int(mask.sum().item())
    if not all(counts.values()):
        raise RuntimeError("Validation requires both sensor and zero samples")
    sensor_mae = absolute[SOURCE_SENSOR] / counts[SOURCE_SENSOR]
    zero_mae = absolute[SOURCE_ZERO] / counts[SOURCE_ZERO]
    return {
        "sensor_mae": sensor_mae,
        "zero_mae": zero_mae,
        "balanced_mae": 0.5 * (sensor_mae + zero_mae),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    wandb = load_wandb(args.no_wandb)

    train_dataset = CachedEdgeForceDataset(
        args.manifest,
        "train",
        augment=True,
        occlude=not args.no_occlusion,
        edge_normalization=args.edge_normalization,
    )
    val_dataset = CachedEdgeForceDataset(
        args.manifest,
        "val",
        augment=False,
        edge_normalization=args.edge_normalization,
    )
    sampler = make_balanced_sampler(train_dataset, args.zero_fraction, args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": use_cuda,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_unet(
        in_channels=1,
        encoder_channels=(32, 64, 128, 256),
        force_dim=1,
        force_hidden_dim=256,
        force_dropout=0.3,
        force_pooling="spatial",
        force_spatial_size=4,
        encoder_only=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    run = None
    if wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run}_{time.strftime('%Y%m%d_%H%M%S')}",
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )

    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    periodic: list[Path] = []
    print(
        f"train={len(train_dataset)} val={len(val_dataset)} "
        f"zero_fraction={args.zero_fraction:.2f} device={device}"
    )
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        model.train()
        mse_sum = 0.0
        examples = 0
        for batch in train_loader:
            frames = batch["frame"].to(device, non_blocking=use_cuda)
            force = batch["force"].to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(frames)
            loss = F.mse_loss(prediction, force)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mse_sum += loss.item() * len(frames)
            examples += len(frames)
        scheduler.step()
        train_mse = mse_sum / examples
        metrics = validation_metrics(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:4d}/{args.epochs} train_MSE={train_mse:.4f} "
            f"val_sensor_MAE={metrics['sensor_mae']:.4f} N "
            f"val_zero_MAE={metrics['zero_mae']:.4f} N "
            f"val_balanced_MAE={metrics['balanced_mae']:.4f} N lr={lr:.2e}"
        )
        if run is not None:
            wandb.log({
                "metrics/train_mse": train_mse,
                "metrics/val_sensor_mae": metrics["sensor_mae"],
                "metrics/val_zero_mae": metrics["zero_mae"],
                "metrics/val_balanced_mae": metrics["balanced_mae"],
                "lr": lr,
            }, step=epoch)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_mse": train_mse,
            "val_mae": metrics["balanced_mae"],
            "val_metrics": metrics,
            "force_keys": ["Fz"],
            "args": {
                **vars(args),
                "manifest": str(args.manifest),
                "output": str(args.output),
                "input_mode": "edge",
                "force_pooling": "spatial",
                "force_spatial_size": 4,
            },
        }
        if metrics["balanced_mae"] < best:
            best = metrics["balanced_mae"]
            torch.save(checkpoint, args.output / "best.pt")
            with (args.output / "best_metrics.json").open("w") as stream:
                json.dump({"epoch": epoch, **metrics}, stream, indent=2)
                stream.write("\n")
        if args.save_every and epoch % args.save_every == 0:
            path = args.output / f"ep{epoch:04d}_balanced_mae{metrics['balanced_mae']:.4f}.pt"
            torch.save(checkpoint, path)
            periodic.append(path)
            while len(periodic) > args.keep_last:
                periodic.pop(0).unlink(missing_ok=True)

    if run is not None:
        wandb.finish()
    print(f"Best balanced validation MAE: {best:.4f} N")
    print(f"Checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
