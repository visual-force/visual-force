#!/usr/bin/env python3
"""Diagnose Coke gripper failures from saved data and rollout logs only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import dill
import joblib
import numpy as np
import torch
from omegaconf import OmegaConf


DEFAULT_DATASET = "data/pick/coke_mixed_0821_force_output"
DEFAULT_ROLLOUT_ROOT = "tts_rollouts/grab_coke_empty_dp"


def _gripper_values(items: Iterable[dict]) -> np.ndarray:
    values = [
        float(np.asarray(item["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        for item in items
    ]
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Gripper values are empty or non-finite")
    return array


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def training_gripper_summary(dataset: Path) -> dict[str, object]:
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    with manifest.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("selected", "").strip().lower() == "true"
        ]
    if not rows:
        raise RuntimeError(f"No selected episodes in {manifest}")

    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        episode = dataset / "episodes" / row["episode"]
        data_path = episode / "data.pkl"
        if not data_path.is_file():
            raise FileNotFoundError(f"Episode data not found: {data_path}")
        data = joblib.load(data_path)
        action = _gripper_values(data["actions"])
        observation = _gripper_values(data["observations"])
        grouped.setdefault(row["source_tag"], []).append(
            {
                "action_max": float(np.max(action)),
                "observation_max": float(np.max(observation)),
            }
        )

    sources = {}
    for source, episodes in sorted(grouped.items()):
        sources[source] = {
            "episodes": len(episodes),
            "episode_action_max": _summary(x["action_max"] for x in episodes),
            "episode_observation_max": _summary(
                x["observation_max"] for x in episodes
            ),
        }
    if "hard" not in sources or "soft" not in sources:
        raise ValueError("Diagnostic dataset must contain soft and hard episodes")
    return {"dataset": str(dataset), "sources": sources}


def checkpoint_gripper_summary(checkpoint: Path) -> dict[str, object]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    action_shape = list(OmegaConf.select(cfg, "task.shape_meta.action.shape"))
    state = payload["state_dicts"].get("ema_model", payload["state_dicts"]["model"])
    prefix = "normalizer.params_dict.action.input_stats."

    stats = {}
    for name in ("min", "q01", "mean", "q99", "max"):
        key = prefix + name
        if key not in state:
            continue
        array = state[key].detach().cpu().numpy().reshape(-1)
        if array.size <= 7:
            raise ValueError(f"Checkpoint action normalizer has shape {array.shape}")
        stats[name] = float(array[7])

    return {
        "checkpoint": str(checkpoint),
        "action_shape": action_shape,
        "append_force_to_action": bool(
            OmegaConf.select(
                cfg, "task.dataset.append_force_to_action", default=False
            )
        ),
        "gripper_index": 7,
        "force_output_index": 8 if action_shape == [9] else None,
        "training_gripper_stats": stats,
    }


def _numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(name, "")
        if value in ("", None):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if np.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def analyze_rollout(
    force_log: Path,
    contact_threshold: float,
    hard_threshold: float,
    tracking_tolerance: float,
) -> dict[str, object]:
    if not force_log.is_file():
        raise FileNotFoundError(f"Rollout log not found: {force_log}")
    with force_log.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Rollout log is empty: {force_log}")

    raw = _numeric_column(rows, "base_gripper_cmd")
    executed = _numeric_column(rows, "steered_gripper_cmd")
    observed = _numeric_column(rows, "obs_gripper")
    if raw.size == 0 or observed.size == 0:
        raise ValueError(
            f"{force_log} needs base_gripper_cmd and obs_gripper columns"
        )
    if executed.size == 0:
        executed = raw

    raw_max = float(np.max(raw))
    executed_max = float(np.max(executed))
    observed_max = float(np.max(observed))
    findings = []
    if raw_max < contact_threshold:
        findings.append("raw_policy_never_reached_contact_closure")
    elif raw_max < hard_threshold:
        findings.append("raw_policy_produced_soft_but_not_hard_closure")
    else:
        findings.append("raw_policy_produced_hard_closure")

    if executed_max + 1e-6 < raw_max - tracking_tolerance:
        findings.append("postprocessing_reduced_policy_closure")
    if observed_max + tracking_tolerance < executed_max:
        findings.append("gripper_did_not_track_executed_command")

    candidate_path = force_log.with_name("candidate_scores.csv")
    candidate = None
    if candidate_path.is_file():
        with candidate_path.open(newline="") as stream:
            candidate_rows = [
                row
                for row in csv.DictReader(stream)
                if row.get("candidate_kind", "policy") == "policy"
            ]
        available = _numeric_column(candidate_rows, "max_gripper_cmd")
        selected = _numeric_column(
            [row for row in candidate_rows if row.get("selected") == "1"],
            "max_gripper_cmd",
        )
        candidate = {
            "path": str(candidate_path),
            "available_max": None if available.size == 0 else float(np.max(available)),
            "selected_max": None if selected.size == 0 else float(np.max(selected)),
        }
        if available.size and np.max(available) < hard_threshold:
            findings.append("no_sampled_candidate_contained_hard_closure")
        elif (
            available.size
            and selected.size
            and np.max(available) >= hard_threshold
            and np.max(selected) < hard_threshold
        ):
            findings.append("tts_rejected_available_hard_closure")

    return {
        "force_log": str(force_log),
        "frames": len(rows),
        "policy_checkpoint": rows[0].get("policy_checkpoint", ""),
        "raw_policy_command_max": raw_max,
        "executed_command_max": executed_max,
        "observed_gripper_max": observed_max,
        "candidate_sampling": candidate,
        "findings": findings,
    }


def discover_rollouts(root: Path, latest: int) -> list[Path]:
    if latest <= 0:
        raise ValueError("latest must be positive")
    paths = sorted(
        root.glob("tts_rollout_*/force_log.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    compatible = []
    for path in paths:
        with path.open(newline="") as stream:
            fields = set(csv.DictReader(stream).fieldnames or ())
        if {"base_gripper_cmd", "obs_gripper"}.issubset(fields):
            compatible.append(path)
        if len(compatible) == latest:
            break
    return list(reversed(compatible))


def build_report(args: argparse.Namespace) -> dict[str, object]:
    training = training_gripper_summary(Path(args.dataset))
    hard_threshold = float(
        training["sources"]["hard"]["episode_action_max"]["q25"]
    )
    rollout_paths = [Path(path) for path in args.rollout]
    if not rollout_paths:
        rollout_paths = discover_rollouts(Path(args.rollout_root), args.latest)
    if not rollout_paths:
        raise FileNotFoundError(f"No rollout logs found under {args.rollout_root}")

    return {
        "contact_threshold": float(args.contact_threshold),
        "hard_threshold": hard_threshold,
        "tracking_tolerance": float(args.tracking_tolerance),
        "training": training,
        "checkpoint": checkpoint_gripper_summary(Path(args.checkpoint)),
        "rollouts": [
            analyze_rollout(
                path,
                contact_threshold=args.contact_threshold,
                hard_threshold=hard_threshold,
                tracking_tolerance=args.tracking_tolerance,
            )
            for path in rollout_paths
        ],
    }


def print_report(report: dict[str, object]) -> None:
    soft = report["training"]["sources"]["soft"]["episode_action_max"]
    hard = report["training"]["sources"]["hard"]["episode_action_max"]
    checkpoint = report["checkpoint"]
    print("Coke gripper diagnostic (offline; no robot commands)")
    print(
        f"Training episode max command: soft median={soft['median']:.3f}, "
        f"hard q25={hard['q25']:.3f}, hard median={hard['median']:.3f}"
    )
    print(
        f"Checkpoint: action_shape={checkpoint['action_shape']} "
        f"gripper_index={checkpoint['gripper_index']} "
        f"force_output_index={checkpoint['force_output_index']}"
    )
    for rollout in report["rollouts"]:
        print(f"\n{rollout['force_log']}")
        print(
            "  max gripper: "
            f"raw={rollout['raw_policy_command_max']:.3f} "
            f"executed={rollout['executed_command_max']:.3f} "
            f"observed={rollout['observed_gripper_max']:.3f}"
        )
        candidate = rollout["candidate_sampling"]
        if candidate is not None:
            print(
                "  candidates: "
                f"available_max={candidate['available_max']} "
                f"selected_max={candidate['selected_max']}"
            )
        print("  findings: " + ", ".join(rollout["findings"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Coke gripper policy output without robot motion."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--rollout", action="append", default=[])
    parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--latest", type=int, default=2)
    parser.add_argument("--contact-threshold", type=float, default=0.25)
    parser.add_argument("--tracking-tolerance", type=float, default=0.10)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
