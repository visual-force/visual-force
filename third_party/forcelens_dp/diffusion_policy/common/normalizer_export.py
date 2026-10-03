import json
import pathlib

import numpy as np
import torch
from omegaconf import OmegaConf


def _to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return str(value)


def _select_cfg(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def save_normalizer_files(normalizer, fallback_dir, cfg=None):
    """Save dataset normalizer as both torch state and readable JSON.

    Prefer writing next to the dataset so the stats travel with the data. If a
    config does not expose task.dataset_path, fall back to the training output.
    """
    dataset_path = _select_cfg(cfg, "task.dataset_path", default=None)
    if dataset_path is not None:
        stats_root = pathlib.Path(dataset_path)
    else:
        stats_root = pathlib.Path(fallback_dir)
    stats_dir = stats_root / "norm_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    pt_path = stats_dir / "normalizer.pt"
    json_path = stats_dir / "normalizer.json"

    torch.save(normalizer.state_dict(), pt_path)

    dataset_cfg = _select_cfg(cfg, "task.dataset", default=None)
    payload = {
        "task_name": _select_cfg(cfg, "task.name", default=None),
        "dataset_path": _select_cfg(cfg, "task.dataset_path", default=None),
        "dataset": _to_plain(dataset_cfg),
        "normalizer": _to_plain(normalizer.params_dict),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "pt": str(pt_path),
        "json": str(json_path),
    }
