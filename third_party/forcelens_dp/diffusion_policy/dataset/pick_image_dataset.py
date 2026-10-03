from typing import Dict, List, Optional
import torch
import numpy as np
import os
import copy
import joblib
import cv2
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


RELATIVE_POSITION_ACTION_MODES = ['action_diff', 'obs_delta', 'obs_anchor']
RELATIVE_GRIPPER_ACTION_MODES = ['same_frame', 'obs_anchor']
FORCE_ACTION_MODES = ['signed', 'magnitude']


def _read_video_frames(video_path: str, out_h: int, out_w: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = cv2.resize(frame, (out_w, out_h))
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def _load_episode(
        episode_dir: str,
        image_keys: List[str],
        out_h: int,
        out_w: int,
        relative_position_action: bool = False,
        relative_position_action_mode: str = 'action_diff',
        relative_gripper_action: bool = False,
        relative_gripper_action_mode: str = 'same_frame',
        append_force_to_action: bool = False,
        force_label_name: str = 'visualforce_pseudo_force_fz.npz',
        force_key: str = 'Fz',
        force_mode: str = 'magnitude',
        ):
    data = joblib.load(os.path.join(episode_dir, 'data.pkl'))
    T = len(data['timestamps'])

    # state: arm_pos(3) + arm_quat(4) + gripper_pos(1) = 8
    obs_list = data['observations']
    agent_pos = np.stack([
        np.concatenate([o['arm_pos'], o['arm_quat'], o['gripper_pos']])
        for o in obs_list
    ], axis=0).astype(np.float32)  # (T, 8)

    # action: arm_pos(3) + arm_quat(4) + gripper_pos(1) = 8
    action = np.stack([
        np.concatenate([a['arm_pos'], a['arm_quat'], a['gripper_pos']])
        for a in data['actions']
    ], axis=0).astype(np.float32)  # (T, 8)
    if relative_position_action:
        if relative_position_action_mode == 'action_diff':
            action_pos = np.zeros_like(action[:, :3])
            action_pos[1:] = np.diff(action[:, :3], axis=0)
        elif relative_position_action_mode == 'obs_delta':
            action_pos = action[:, :3] - agent_pos[:, :3]
        elif relative_position_action_mode == 'obs_anchor':
            action_pos = action[:, :3]
        else:
            raise ValueError(
                'Unsupported relative_position_action_mode='
                f'{relative_position_action_mode!r}'
            )
        action[:, :3] = action_pos
    if relative_gripper_action and relative_gripper_action_mode == 'same_frame':
        action[:, 7:8] = action[:, 7:8] - agent_pos[:, 7:8]

    if append_force_to_action:
        force_path = os.path.join(episode_dir, force_label_name)
        if not os.path.isfile(force_path):
            raise FileNotFoundError(
                f'Missing force label for force-output policy: {force_path}'
            )
        with np.load(force_path, allow_pickle=False) as force_data:
            force_keys = [str(key) for key in force_data['force_keys'].tolist()]
            if force_key not in force_keys:
                raise KeyError(
                    f'Force key {force_key!r} not found in {force_path}; '
                    f'available keys: {force_keys}'
                )
            force = np.asarray(
                force_data['force'][:, force_keys.index(force_key)],
                dtype=np.float32,
            )
        if force.shape != (T,):
            raise ValueError(
                f'Force/action length mismatch in {episode_dir}: '
                f'force={force.shape}, expected=({T},)'
            )
        if not np.isfinite(force).all():
            raise ValueError(f'Non-finite force labels in {force_path}')
        if force_mode == 'magnitude':
            force = np.abs(force)
        elif force_mode != 'signed':
            raise ValueError(
                f'Unsupported force_mode={force_mode!r}; '
                f'available modes: {FORCE_ACTION_MODES}'
            )
        action = np.concatenate([action, force[:, None]], axis=-1)

    episode = {'agent_pos': agent_pos, 'action': action}

    for key in image_keys:
        video_path = os.path.join(episode_dir, f'{key}.mp4')
        episode[key] = _read_video_frames(video_path, out_h, out_w)  # (T, H, W, 3)

    return episode, T


class PickImageDataset(BaseImageDataset):
    def __init__(self,
            dataset_path,  # str or list of str
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            image_keys: List[str] = None,
            image_size: List[int] = None,  # [H, W]
            relative_position_action: bool = False,
            relative_position_action_mode: str = 'action_diff',
            relative_position_anchor_idx: Optional[int] = None,
            relative_gripper_action: bool = False,
            relative_gripper_action_mode: str = 'same_frame',
            action_normalizer_mode: str = 'limits',
            gripper_action_normalizer_mode: Optional[str] = None,
            agent_pos_normalizer_mode: str = 'limits',
            normalizer_range_eps: float = 1e-4,
            append_force_to_action: bool = False,
            force_label_name: str = 'visualforce_pseudo_force_fz.npz',
            force_key: str = 'Fz',
            force_mode: str = 'magnitude',
            ):
        if image_keys is None:
            image_keys = ['base_image', 'wrist_image']
        if image_size is None:
            image_size = [240, 320]
        self.relative_position_action = bool(relative_position_action)
        self.relative_gripper_action = bool(relative_gripper_action)
        self.relative_position_action_mode = relative_position_action_mode
        self.relative_gripper_action_mode = relative_gripper_action_mode
        if self.relative_position_action_mode not in RELATIVE_POSITION_ACTION_MODES:
            raise ValueError(
                'Unsupported relative_position_action_mode='
                f'{self.relative_position_action_mode!r}. '
                f'Available modes: {RELATIVE_POSITION_ACTION_MODES}'
            )
        if self.relative_gripper_action_mode not in RELATIVE_GRIPPER_ACTION_MODES:
            raise ValueError(
                'Unsupported relative_gripper_action_mode='
                f'{self.relative_gripper_action_mode!r}. '
                f'Available modes: {RELATIVE_GRIPPER_ACTION_MODES}'
            )
        if relative_position_anchor_idx is None:
            relative_position_anchor_idx = max(0, int(pad_before))
        self.relative_position_anchor_idx = int(relative_position_anchor_idx)
        self.action_normalizer_mode = action_normalizer_mode
        self.gripper_action_normalizer_mode = gripper_action_normalizer_mode
        self.agent_pos_normalizer_mode = agent_pos_normalizer_mode
        self.normalizer_range_eps = float(normalizer_range_eps)
        self.append_force_to_action = bool(append_force_to_action)
        self.force_label_name = str(force_label_name)
        self.force_key = str(force_key)
        self.force_mode = str(force_mode)
        if self.force_mode not in FORCE_ACTION_MODES:
            raise ValueError(
                f'Unsupported force_mode={self.force_mode!r}. '
                f'Available modes: {FORCE_ACTION_MODES}'
            )
        normalizer_modes = SingleFieldLinearNormalizer.avaliable_modes
        if self.action_normalizer_mode not in normalizer_modes:
            raise ValueError(
                f'Unsupported action_normalizer_mode={self.action_normalizer_mode!r}. '
                f'Available modes: {normalizer_modes}'
            )
        if (
            self.gripper_action_normalizer_mode is not None
            and self.gripper_action_normalizer_mode not in normalizer_modes
        ):
            raise ValueError(
                'Unsupported gripper_action_normalizer_mode='
                f'{self.gripper_action_normalizer_mode!r}. '
                f'Available modes: {normalizer_modes}'
            )
        if self.agent_pos_normalizer_mode not in normalizer_modes:
            raise ValueError(
                f'Unsupported agent_pos_normalizer_mode={self.agent_pos_normalizer_mode!r}. '
                f'Available modes: {normalizer_modes}'
            )

        out_h, out_w = image_size

        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]

        episode_dirs = []
        for path in dataset_path:
            episodes_dir = os.path.join(path, 'episodes')
            episode_dirs += sorted([
                os.path.join(episodes_dir, d)
                for d in os.listdir(episodes_dir)
                if os.path.isdir(os.path.join(episodes_dir, d))
            ])

        replay_buffer = ReplayBuffer.create_empty_numpy()
        for ep_dir in episode_dirs:
            episode, T = _load_episode(
                ep_dir,
                image_keys,
                out_h,
                out_w,
                relative_position_action=self.relative_position_action,
                relative_position_action_mode=self.relative_position_action_mode,
                relative_gripper_action=self.relative_gripper_action,
                relative_gripper_action_mode=self.relative_gripper_action_mode,
                append_force_to_action=self.append_force_to_action,
                force_label_name=self.force_label_name,
                force_key=self.force_key,
                force_mode=self.force_mode,
            )
            replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.image_keys = image_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
        self.train_mask = train_mask

    def _get_anchor_idx(self, sequence_length):
        return min(max(self.relative_position_anchor_idx, 0), sequence_length - 1)

    def _transform_obs_anchor_action(self, action, agent_pos):
        if not (
            self.relative_position_action
            and self.relative_position_action_mode == 'obs_anchor'
        ) and not (
            self.relative_gripper_action
            and self.relative_gripper_action_mode == 'obs_anchor'
        ):
            return action
        action = np.asarray(action, dtype=np.float32).copy()
        anchor_idx = self._get_anchor_idx(action.shape[0])
        if (
            self.relative_position_action
            and self.relative_position_action_mode == 'obs_anchor'
        ):
            anchor_pos = np.asarray(agent_pos[anchor_idx, :3], dtype=np.float32)
            action[:, :3] = action[:, :3] - anchor_pos[None, :]
        if (
            self.relative_gripper_action
            and self.relative_gripper_action_mode == 'obs_anchor'
        ):
            anchor_gripper = np.asarray(agent_pos[anchor_idx, 7:8], dtype=np.float32)
            action[:, 7:8] = action[:, 7:8] - anchor_gripper[None, :]
        return action

    def _get_action_data_for_normalizer(self):
        needs_sequence_transform = (
            self.relative_position_action
            and self.relative_position_action_mode == 'obs_anchor'
        ) or (
            self.relative_gripper_action
            and self.relative_gripper_action_mode == 'obs_anchor'
        )
        if not needs_sequence_transform:
            return self.replay_buffer['action']

        actions = []
        for idx in range(len(self.sampler)):
            sample = self.sampler.sample_sequence(idx)
            action = self._transform_obs_anchor_action(
                sample['action'],
                sample['agent_pos'],
            )
            actions.append(action)
        if len(actions) == 0:
            return np.zeros((0, self.horizon, self.replay_buffer['action'].shape[-1]))
        return np.stack(actions, axis=0)

    def _fit_action_normalizer(self):
        action_data = self._get_action_data_for_normalizer()
        action_normalizer = SingleFieldLinearNormalizer.create_fit(
            action_data,
            mode=self.action_normalizer_mode,
            range_eps=self.normalizer_range_eps,
        )
        if self.gripper_action_normalizer_mode is None:
            return action_normalizer

        gripper_normalizer = SingleFieldLinearNormalizer.create_fit(
            action_data[..., 7:8],
            mode=self.gripper_action_normalizer_mode,
            range_eps=self.normalizer_range_eps,
        )

        scale = action_normalizer.params_dict['scale'].detach().clone()
        offset = action_normalizer.params_dict['offset'].detach().clone()
        scale[7] = gripper_normalizer.params_dict['scale'][0]
        offset[7] = gripper_normalizer.params_dict['offset'][0]

        input_stats = {}
        for key, value in action_normalizer.params_dict['input_stats'].items():
            merged = value.detach().clone()
            merged[7] = gripper_normalizer.params_dict['input_stats'][key][0]
            input_stats[key] = merged

        return SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=input_stats,
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = self._fit_action_normalizer()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['agent_pos'],
            mode=self.agent_pos_normalizer_mode,
            range_eps=self.normalizer_range_eps,
            **kwargs)
        for key in self.image_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        obs_dict = {}
        for key in self.image_keys:
            # (T, H, W, 3) -> (T, 3, H, W), float32 [0, 1]
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255.0
        obs_dict['agent_pos'] = sample['agent_pos'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        action = self._transform_obs_anchor_action(action, obs_dict['agent_pos'])
        return dict_apply({
            'obs': obs_dict,
            'action': action,
        }, torch.from_numpy)
