#!/usr/bin/env python3
"""Measure force-output diffusion-policy diversity on held-out episodes."""

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _pair_indices(n):
    return np.triu_indices(n, k=1)


def _quat_angles_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _candidate_metrics(candidates, ground_truth, score_steps):
    robot = np.asarray(candidates[..., :8], dtype=np.float64)
    force = np.asarray(candidates[..., 8], dtype=np.float64)
    robot = robot[:, :score_steps]
    force = force[:, :score_steps]
    ground_truth = np.asarray(ground_truth[:score_steps], dtype=np.float64)

    pi, pj = _pair_indices(len(robot))
    pair_xyz = np.linalg.norm(
        robot[pi, :, :3] - robot[pj, :, :3], axis=-1
    ).mean(axis=-1) * 1000.0
    pair_rot = _quat_angles_deg(
        robot[pi, :, 3:7], robot[pj, :, 3:7]
    ).mean(axis=-1)
    pair_gripper = np.abs(
        robot[pi, :, 7] - robot[pj, :, 7]
    ).mean(axis=-1)

    gt_xyz = np.linalg.norm(
        robot[:, :, :3] - ground_truth[None, :, :3], axis=-1
    ).mean(axis=-1) * 1000.0
    gt_rot = _quat_angles_deg(
        robot[:, :, 3:7], ground_truth[None, :, 3:7]
    ).mean(axis=-1)
    gt_gripper = np.abs(
        robot[:, :, 7] - ground_truth[None, :, 7]
    ).mean(axis=-1)

    force_proxy = force[:, -1]
    gt_force = float(ground_truth[-1, 8])
    nearest_xyz_idx = int(np.argmin(gt_xyz))

    def stats(prefix, values):
        values = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_p95": float(np.quantile(values, 0.95)),
            f"{prefix}_max": float(values.max()),
        }

    result = {}
    result.update(stats("pair_xyz_mm", pair_xyz))
    result.update(stats("pair_rot_deg", pair_rot))
    result.update(stats("pair_gripper", pair_gripper))
    result.update(stats("gt_xyz_mm", gt_xyz))
    result.update(stats("gt_rot_deg", gt_rot))
    result.update(stats("gt_gripper", gt_gripper))
    result.update({
        "gt_xyz_mm_min": float(gt_xyz.min()),
        "gt_rot_deg_min": float(gt_rot.min()),
        "gt_gripper_min": float(gt_gripper.min()),
        "force_proxy_mean_n": float(force_proxy.mean()),
        "force_proxy_std_n": float(force_proxy.std()),
        "force_proxy_min_n": float(force_proxy.min()),
        "force_proxy_max_n": float(force_proxy.max()),
        "force_proxy_range_n": float(np.ptp(force_proxy)),
        "gt_force_n": gt_force,
        "nearest_xyz_force_n": float(force_proxy[nearest_xyz_idx]),
        "nearest_xyz_force_error_n": float(
            abs(force_proxy[nearest_xyz_idx] - gt_force)
        ),
        "pair_xyz_over_1mm_frac": float(np.mean(pair_xyz > 1.0)),
        "pair_rot_over_1deg_frac": float(np.mean(pair_rot > 1.0)),
        "pair_gripper_over_001_frac": float(np.mean(pair_gripper > 0.01)),
    })
    return result


def _choose_validation_samples(dataset, samples_per_episode):
    val_dataset = dataset.get_validation_dataset()
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends)
    grouped = {}
    for sample_idx, index in enumerate(val_dataset.sampler.indices):
        buffer_start = int(index[0])
        episode_idx = int(np.searchsorted(episode_ends, buffer_start, side="right"))
        grouped.setdefault(episode_idx, []).append(sample_idx)

    selected = []
    for episode_idx in np.flatnonzero(dataset.val_mask):
        available = grouped[int(episode_idx)]
        positions = np.linspace(
            0, len(available) - 1, samples_per_episode + 2
        )[1:-1]
        for position in positions:
            selected.append((int(episode_idx), available[int(round(position))]))
    return val_dataset, selected


def _load_policy(checkpoint, device):
    with checkpoint.open("rb") as file:
        payload = torch.load(
            file, pickle_module=dill, map_location="cpu", weights_only=False
        )
    cfg = payload["cfg"]
    if not bool(OmegaConf.select(
        cfg, "task.dataset.append_force_to_action", default=False
    )):
        raise ValueError(f"Checkpoint is not a force-output policy: {checkpoint}")
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.n_action_steps = 8
    policy.eval().to(device)
    return cfg, workspace, policy


def _summarize(rows):
    keys = [
        "pair_xyz_mm_median",
        "pair_rot_deg_median",
        "pair_gripper_median",
        "force_proxy_std_n",
        "force_proxy_range_n",
        "gt_xyz_mm_min",
        "gt_rot_deg_min",
        "gt_gripper_min",
        "nearest_xyz_force_error_n",
        "pair_xyz_over_1mm_frac",
        "pair_rot_over_1deg_frac",
        "pair_gripper_over_001_frac",
    ]
    summary = {"n_validation_points": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--score-steps", type=int, default=4)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("analysis/force_output_candidate_diversity"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    dataset = None
    episode_names = None

    for checkpoint in args.checkpoint:
        checkpoint = checkpoint.resolve()
        print(f"Loading {checkpoint}")
        cfg, workspace, policy = _load_policy(checkpoint, args.device)
        if dataset is None:
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            val_dataset, selections = _choose_validation_samples(
                dataset, args.samples_per_episode
            )
            dataset_paths = cfg.task.dataset.dataset_path
            if OmegaConf.is_config(dataset_paths):
                dataset_paths = OmegaConf.to_container(dataset_paths)
            if isinstance(dataset_paths, str):
                dataset_paths = [dataset_paths]
            episode_dirs = []
            for path in dataset_paths:
                root = Path(path) / "episodes"
                episode_dirs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
            episode_names = [p.name for p in episode_dirs]

        rows = []
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        for point_idx, (episode_idx, sample_idx) in enumerate(selections):
            sample = val_dataset[sample_idx]
            obs = dict_apply(
                sample["obs"],
                lambda x: x.unsqueeze(0).repeat_interleave(
                    args.candidates, dim=0
                ).to(args.device),
            )
            with torch.inference_mode():
                candidates = policy.predict_action(obs)["action"].cpu().numpy()
            if candidates.shape[-1] != 9:
                raise ValueError(f"Expected 9 outputs, got {candidates.shape}")
            anchor = int(cfg.n_obs_steps) - 1
            ground_truth = sample["action"].numpy()[
                anchor:anchor + candidates.shape[1]
            ]
            row = {
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "point_idx": point_idx,
                "episode_idx": episode_idx,
                "episode": episode_names[episode_idx],
                "outcome_group": (
                    "fail_named" if "_fail_" in episode_names[episode_idx]
                    else "non_fail_named"
                ),
                "validation_sample_idx": sample_idx,
                "candidates": args.candidates,
                "score_steps": args.score_steps,
            }
            row.update(_candidate_metrics(
                candidates, ground_truth, args.score_steps
            ))
            rows.append(row)
            print(
                f"  {point_idx + 1}/{len(selections)} "
                f"{row['episode']}: xyz_pair={row['pair_xyz_mm_median']:.2f}mm "
                f"grip_pair={row['pair_gripper_median']:.4f} "
                f"force_std={row['force_proxy_std_n']:.2f}N"
            )

        key = f"{checkpoint.parent.parent.name}__{checkpoint.stem}"
        all_results[key] = {
            "checkpoint": str(checkpoint),
            "summary": _summarize(rows),
            "points": rows,
        }

        csv_path = args.output_dir / f"{key}.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}")

        del policy, workspace, cfg
        gc.collect()
        torch.cuda.empty_cache()

    json_path = args.output_dir / "summary.json"
    with json_path.open("w") as file:
        json.dump(all_results, file, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
