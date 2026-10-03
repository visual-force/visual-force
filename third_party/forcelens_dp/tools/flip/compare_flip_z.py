#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np


def load_dataset_z(dataset_path):
    rows = []
    for pkl_path in sorted((dataset_path / 'episodes').glob('*/data.pkl')):
        episode = pkl_path.parent.name
        data = joblib.load(pkl_path)
        action_z = np.asarray(
            [row['arm_pos'][2] for row in data['actions']], dtype=np.float64
        )
        obs_z = np.asarray(
            [row['arm_pos'][2] for row in data['observations']], dtype=np.float64
        )
        rows.append(
            {
                'episode': episode,
                'fail': 'fail' in episode.lower(),
                'action_z': action_z,
                'obs_z': obs_z,
            }
        )
    return rows


def summarize_z(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        print(f'{name}: empty')
        return
    print(
        f'{name}: min={values.min():.6f}, '
        f'p01={np.quantile(values, 0.01):.6f}, '
        f'p05={np.quantile(values, 0.05):.6f}, '
        f'median={np.median(values):.6f}, '
        f'p95={np.quantile(values, 0.95):.6f}, '
        f'max={values.max():.6f}'
    )


def load_rollout_z(force_log_path):
    values = {'obs_z': [], 'base_action_z': [], 'steered_action_z': []}
    with force_log_path.open(newline='') as f:
        reader = csv.DictReader(f)
        missing = [key for key in values if key not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f'{force_log_path} is missing {missing}. '
                'Run a new rollout with the updated policy_server.py first.'
            )
        for row in reader:
            for key in values:
                raw = row.get(key, '')
                if raw == '':
                    continue
                values[key].append(float(raw))
    return {key: np.asarray(val, dtype=np.float64) for key, val in values.items()}


def main():
    parser = argparse.ArgumentParser(
        description='Compare flip training z range against rollout z commands.'
    )
    parser.add_argument(
        '--dataset-path',
        default='data/flip/flip_obj_3_all',
        help='Dataset root relative to forcelens_dp or absolute path.',
    )
    parser.add_argument(
        '--force-log',
        required=True,
        help='Path to a rollout force_log.csv.',
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser()
    force_log_path = Path(args.force_log).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    if not force_log_path.is_absolute():
        force_log_path = Path.cwd() / force_log_path

    rows = load_dataset_z(dataset_path)
    print(f'Dataset: {dataset_path}')
    print(
        f'episodes={len(rows)}, '
        f'success={sum(not row["fail"] for row in rows)}, '
        f'fail={sum(row["fail"] for row in rows)}'
    )

    for label, subset in [
        ('all', rows),
        ('success_only', [row for row in rows if not row['fail']]),
        ('fail_only', [row for row in rows if row['fail']]),
    ]:
        if not subset:
            continue
        print(f'\n[{label}]')
        summarize_z('dataset_action_z', np.concatenate([row['action_z'] for row in subset]))
        summarize_z('dataset_obs_z', np.concatenate([row['obs_z'] for row in subset]))

    print(f'\nRollout: {force_log_path}')
    rollout = load_rollout_z(force_log_path)
    for key, values in rollout.items():
        summarize_z(f'rollout_{key}', values)


if __name__ == '__main__':
    main()
