import os
from pathlib import Path
from typing import Iterable, Union

import cv2 as cv
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


def discover_flip_episodes(dataset_path: Union[str, Path]) -> list[Path]:
    root = Path(dataset_path)
    episodes_dir = root / "episodes"
    if episodes_dir.is_dir():
        candidates = sorted(p for p in episodes_dir.iterdir() if p.is_dir())
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())

    episodes = [
        p
        for p in candidates
        if (p / "data.pkl").is_file() and (p / "wrist_image.mp4").is_file()
    ]
    if episodes:
        return episodes

    return sorted(
        p
        for p in root.rglob("*")
        if p.is_dir()
        and (p / "data.pkl").is_file()
        and (p / "wrist_image.mp4").is_file()
    )


def read_video_rgb(video_path: Union[str, Path], image_size: tuple[int, int]) -> np.ndarray:
    out_h, out_w = image_size
    cap = cv.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def read_edge_obs(edge_path: Union[str, Path], image_size: tuple[int, int]) -> np.ndarray:
    data = np.load(edge_path)
    if "edge" not in data:
        raise ValueError(f"{edge_path} does not contain an 'edge' array")
    edge = data["edge"]
    if edge.ndim != 3:
        raise ValueError(f"Expected edge obs with shape (T,H,W), got {edge.shape}")
    out_h, out_w = image_size
    frames = []
    for frame in edge:
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _stack_agent_pos(items: Iterable[dict]) -> np.ndarray:
    return np.stack(
        [
            np.concatenate([o["arm_pos"], o["arm_quat"], o["gripper_pos"]])
            for o in items
        ],
        axis=0,
    ).astype(np.float32)


def _stack_actions(items: Iterable[dict]) -> np.ndarray:
    return np.stack(
        [
            np.concatenate([a["arm_pos"], a["arm_quat"], a["gripper_pos"]])
            for a in items
        ],
        axis=0,
    ).astype(np.float32)


class FlipDeltaForceDataset(Dataset):
    """
    Flip-only dataset for learning action-conditioned visual force dynamics.

    Each sample contains:
      image: stacked wrist-camera RGB history, (obs_steps * 3, H, W)
      action_delta: candidate action trajectory relative to current observation
      target_delta_force: pseudo-label force deltas from a frozen ForceLens model
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        pseudo_label_name: str = "visualforce_pseudo_force_fz.npz",
        obs_steps: int = 2,
        pred_horizon: int = 16,
        image_size: tuple[int, int] = (240, 320),
        force_key: str = "Fz",
        force_mode: str = "magnitude",
        action_mode: str = "obs_delta",
        image_normalization: str = "zero_one",
        critic_image_mode: str = "rgb",
        edge_obs_name: str = "visualforce_edge_obs.npz",
        target_mode: str = "delta_trajectory",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.pseudo_label_name = pseudo_label_name
        self.obs_steps = int(obs_steps)
        self.pred_horizon = int(pred_horizon)
        self.image_size = tuple(int(x) for x in image_size)
        self.force_key = force_key
        self.force_mode = force_mode
        self.action_mode = action_mode
        self.image_normalization = image_normalization
        self.critic_image_mode = critic_image_mode
        self.edge_obs_name = edge_obs_name
        self.target_mode = target_mode
        if self.force_mode not in {"magnitude", "signed"}:
            raise ValueError("force_mode must be 'magnitude' or 'signed'")
        if self.action_mode not in {"obs_delta", "obs_delta_pos_gripper"}:
            raise ValueError(
                "action_mode must be 'obs_delta' or 'obs_delta_pos_gripper'"
            )
        if self.image_normalization not in {"zero_one", "minus_one_one"}:
            raise ValueError(
                "image_normalization must be 'zero_one' or 'minus_one_one'"
            )
        if self.critic_image_mode not in {"rgb", "edge", "none"}:
            raise ValueError("critic_image_mode must be 'rgb', 'edge', or 'none'")
        if self.target_mode not in {
            "delta_trajectory",
            "final_delta",
            "max_delta",
            "mean_delta",
            "future_peak",
        }:
            raise ValueError(
                "target_mode must be one of: delta_trajectory, final_delta, "
                "max_delta, mean_delta, future_peak"
            )

        self.episodes = []
        self.index = []
        for ep_dir in discover_flip_episodes(self.dataset_path):
            label_path = ep_dir / self.pseudo_label_name
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing pseudo force labels: {label_path}. "
                    "Run scripts/precompute_flip_pseudo_force.py first."
                )
            data = joblib.load(ep_dir / "data.pkl")
            if self.critic_image_mode == "none":
                frames = None
            elif self.critic_image_mode == "edge":
                edge_path = ep_dir / self.edge_obs_name
                if not edge_path.is_file():
                    raise FileNotFoundError(
                        f"Missing edge observations: {edge_path}. "
                        "Run scripts/precompute_flip_pseudo_force.py with "
                        "--edge-output-name first."
                    )
                frames = read_edge_obs(edge_path, self.image_size)
            else:
                frames = read_video_rgb(ep_dir / "wrist_image.mp4", self.image_size)
            agent_pos = _stack_agent_pos(data["observations"])
            actions = _stack_actions(data["actions"])
            labels = np.load(label_path)
            if self.force_key not in labels["force_keys"].tolist():
                raise ValueError(
                    f"{label_path} does not contain force key {self.force_key!r}"
                )
            key_idx = labels["force_keys"].tolist().index(self.force_key)
            force = labels["force"][:, key_idx].astype(np.float32)
            if self.force_mode == "magnitude":
                force = np.abs(force)

            if frames is None:
                n = min(len(agent_pos), len(actions), len(force))
            else:
                n = min(len(frames), len(agent_pos), len(actions), len(force))
                frames = frames[:n]
            agent_pos = agent_pos[:n]
            actions = actions[:n]
            force = force[:n]
            if n <= self.pred_horizon:
                continue

            ep_idx = len(self.episodes)
            self.episodes.append(
                {
                    "dir": ep_dir,
                    "frames": frames,
                    "agent_pos": agent_pos,
                    "actions": actions,
                    "force": force,
                }
            )
            for t in range(n - self.pred_horizon):
                self.index.append((ep_idx, t))

        if not self.index:
            raise RuntimeError(f"No delta-force samples found in {self.dataset_path}")

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def force_dim(self) -> int:
        return 1

    @property
    def image_channels(self) -> int:
        if self.critic_image_mode == "none":
            return 0
        channels_per_frame = 1 if self.critic_image_mode == "edge" else 3
        return self.obs_steps * channels_per_frame

    @property
    def low_dim_dim(self) -> int:
        return self.obs_steps * 8

    @property
    def target_horizon(self) -> int:
        return self.pred_horizon if self.target_mode == "delta_trajectory" else 1

    def _image_history(self, frames: np.ndarray, t: int) -> torch.Tensor:
        if self.critic_image_mode == "none":
            return torch.empty((0, self.image_size[0], self.image_size[1])).float()
        idxs = [max(0, t - self.obs_steps + 1 + i) for i in range(self.obs_steps)]
        hist = frames[idxs].astype(np.float32) / 255.0
        if self.image_normalization == "minus_one_one":
            hist = hist * 2.0 - 1.0
        if self.critic_image_mode == "edge":
            hist = torch.from_numpy(hist[:, None, :, :]).reshape(
                self.image_channels, self.image_size[0], self.image_size[1]
            )
        else:
            hist = torch.from_numpy(hist).permute(0, 3, 1, 2).reshape(
                self.image_channels, self.image_size[0], self.image_size[1]
            )
        return hist

    def _agent_pos_history(self, agent_pos: np.ndarray, t: int) -> torch.Tensor:
        idxs = [max(0, t - self.obs_steps + 1 + i) for i in range(self.obs_steps)]
        hist = agent_pos[idxs].astype(np.float32)
        return torch.from_numpy(hist.reshape(-1)).float()

    def _action_delta(
        self,
        actions: np.ndarray,
        agent_pos: np.ndarray,
        t: int,
    ) -> np.ndarray:
        seq = actions[t : t + self.pred_horizon].astype(np.float32).copy()
        anchor = agent_pos[t].astype(np.float32)
        if self.action_mode == "obs_delta":
            return seq - anchor[None, :]

        out = seq.copy()
        out[:, :3] = seq[:, :3] - anchor[None, :3]
        out[:, 7:8] = seq[:, 7:8] - anchor[None, 7:8]
        return out

    def __len__(self) -> int:
        return len(self.index)

    def _target_delta_force(self, force: np.ndarray, t: int) -> np.ndarray:
        if self.target_mode == "delta_trajectory":
            target = force[t + 1 : t + self.pred_horizon + 1] - force[
                t : t + self.pred_horizon
            ]
            return target[:, None].astype(np.float32)

        current = float(force[t])
        future = force[t + 1 : t + self.pred_horizon + 1]
        if self.target_mode == "final_delta":
            target = float(force[t + self.pred_horizon] - current)
        elif self.target_mode == "max_delta":
            target = float(np.max(future) - current)
        elif self.target_mode == "mean_delta":
            target = float(np.mean(future) - current)
        elif self.target_mode == "future_peak":
            target = float(np.max(future))
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode!r}")
        return np.asarray([[target]], dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]
        force = ep["force"]
        target = self._target_delta_force(force, t)
        return {
            "image": self._image_history(ep["frames"], t),
            "agent_pos": self._agent_pos_history(ep["agent_pos"], t),
            "current_force": torch.tensor([force[t]], dtype=torch.float32),
            "action_delta": torch.from_numpy(
                self._action_delta(ep["actions"], ep["agent_pos"], t)
            ).float(),
            "target_delta_force": torch.from_numpy(target).float(),
        }


def split_train_val(
    dataset: FlipDeltaForceDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if not train_idx:
        train_idx, val_idx = val_idx, train_idx
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_train_val_by_episode(
    dataset: FlipDeltaForceDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    n_eps = len(dataset.episodes)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_eps, generator=gen).tolist()
    n_val = max(1, int(round(n_eps * val_ratio))) if val_ratio > 0 else 0
    val_eps = set(perm[:n_val])
    train_idx = []
    val_idx = []
    for idx, (ep_idx, _t) in enumerate(dataset.index):
        if ep_idx in val_eps:
            val_idx.append(idx)
        else:
            train_idx.append(idx)
    if not train_idx:
        train_idx, val_idx = val_idx, train_idx
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def standardize(x: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["mean"].to(x.device)
    std = stats["std"].to(x.device)
    return (x - mean) / std


def unstandardize(x: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["mean"].to(x.device)
    std = stats["std"].to(x.device)
    return x * std + mean
