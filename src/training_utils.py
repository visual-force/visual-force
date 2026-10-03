"""Shared helpers for VisualForce training and evaluation entry points."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


def resolve_device(requested: str | None) -> torch.device:
    """Resolve a requested Torch device and reject unavailable CUDA devices."""
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def load_wandb(disabled: bool) -> Any | None:
    """Load W&B lazily so scripts can run without the optional dependency."""
    if disabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed; install it or pass --no-wandb"
        ) from exc
    return wandb


def batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move every tensor in a flat batch mapping to one device."""
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def fit_standardizers(
    loader: Iterable[Mapping[str, torch.Tensor]],
    keys: Sequence[str],
    eps: float = 1e-6,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute normalization statistics for requested tensors in one pass."""
    totals: dict[str, torch.Tensor] = {}
    totals_sq: dict[str, torch.Tensor] = {}
    counts = {key: 0 for key in keys}
    for batch in loader:
        for key in keys:
            values = batch[key].double().reshape(-1, batch[key].shape[-1])
            batch_sum = values.sum(dim=0)
            batch_sum_sq = values.square().sum(dim=0)
            if key in totals:
                totals[key] += batch_sum
                totals_sq[key] += batch_sum_sq
            else:
                totals[key] = batch_sum
                totals_sq[key] = batch_sum_sq
            counts[key] += values.shape[0]

    stats = {}
    for key in keys:
        if counts[key] == 0:
            raise ValueError(
                f"Cannot fit standardizer for {key!r} on an empty dataset"
            )
        mean = totals[key] / counts[key]
        variance = totals_sq[key] / counts[key] - mean.square()
        stats[key] = {
            "mean": mean.float(),
            "std": variance.clamp_min(eps).sqrt().float(),
        }
    return stats
