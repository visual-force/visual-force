#!/usr/bin/env python3
"""Evaluate a VisualForce checkpoint by source subtype on a cached manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.cached_edge_force_dataset import CachedEdgeForceDataset
from src.steering import VisualForceEstimator
from src.training_utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    estimator = VisualForceEstimator(args.checkpoint, device=device)
    dataset = CachedEdgeForceDataset(
        args.manifest,
        args.split,
        augment=False,
        edge_normalization=estimator.edge_normalization,
    )
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    predictions: dict[str, list[float]] = defaultdict(list)

    for entry_index, entry in enumerate(dataset.entries):
        edges, labels = dataset._entry_arrays(entry_index)
        subtype = entry["source_subtype"]
        for start in range(0, len(edges), args.batch_size):
            stop = min(len(edges), start + args.batch_size)
            frames = torch.from_numpy(np.array(edges[start:stop], copy=True)).float()
            frames = (frames / 255.0).unsqueeze(1).to(device)
            if labels is None:
                target = torch.full(
                    (len(frames), 1), float(entry["constant_label_n"]), device=device
                )
            else:
                target = torch.from_numpy(
                    np.array(labels[start:stop], copy=True)
                ).float().reshape(-1, 1).to(device)
            with torch.inference_mode():
                prediction = estimator.model(frames)
            totals[subtype] += (prediction - target).abs().sum().item()
            counts[subtype] += len(frames)
            predictions[subtype].extend(prediction[:, 0].cpu().tolist())

    metrics = {}
    for subtype in sorted(counts):
        values = np.asarray(predictions[subtype])
        metrics[subtype] = {
            "samples": counts[subtype],
            "mae_n": totals[subtype] / counts[subtype],
            "prediction_mean_n": float(values.mean()),
            "prediction_abs_p95_n": float(np.percentile(np.abs(values), 95)),
        }
    sensor = metrics.get("physical_fz")
    zero_types = [key for key in metrics if key != "physical_fz"]
    zero_count = sum(metrics[key]["samples"] for key in zero_types)
    zero_mae = (
        sum(metrics[key]["mae_n"] * metrics[key]["samples"] for key in zero_types)
        / zero_count
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "sensor_mae_n": sensor["mae_n"] if sensor else None,
        "zero_mae_n": zero_mae,
        "balanced_mae_n": 0.5 * (sensor["mae_n"] + zero_mae) if sensor else None,
        "by_source_subtype": metrics,
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
