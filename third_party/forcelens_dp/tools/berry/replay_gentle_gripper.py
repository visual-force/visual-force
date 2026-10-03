#!/usr/bin/env python3
"""Replay gentle gripper decisions over an existing VisualForce force log.

This is a controller regression tool, not a physics simulation: the observations
still come from the recorded rollout. It answers whether the new controller
would have limited, held, or released the commands that were actually proposed.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy_server import GentleGripperController  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('force_log', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument(
        '--contact-force-delta',
        '--desired-force',
        '--desired-force-delta',
        dest='contact_force_delta',
        type=float,
        default=5.0,
        help=(
            'Target VisualForce rise above the open-gripper baseline in '
            'newtons. The desired-force spellings are retained as aliases.'
        ),
    )
    parser.add_argument('--deadband', type=float, default=0.5)
    parser.add_argument('--stop-margin', type=float, default=2.0)
    parser.add_argument('--close-step', type=float, default=0.05)
    parser.add_argument('--maintain-step', type=float, default=0.005)
    parser.add_argument('--release-step', type=float, default=0.05)
    parser.add_argument('--filter-window', type=int, default=3)
    parser.add_argument('--baseline-max-position', type=float, default=0.35)
    parser.add_argument('--min-closure-position', type=float, default=0.35)
    parser.add_argument('--max-closure-position', type=float, default=0.82)
    parser.add_argument('--max-command-lead', type=float, default=0.25)
    parser.add_argument('--max-force-rate', type=float, default=20.0)
    return parser.parse_args()


def first_float(row, keys):
    for key in keys:
        value = row.get(key, '')
        if value not in ('', None):
            return float(value)
    raise ValueError(f'None of {keys} is populated in force-log row {row}')


def main():
    args = parse_args()
    with args.force_log.open(newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'Force log is empty: {args.force_log}')

    controller = GentleGripperController(
        desired_force_delta=args.contact_force_delta,
        deadband=args.deadband,
        stop_margin=args.stop_margin,
        close_step=args.close_step,
        maintain_step=args.maintain_step,
        release_step=args.release_step,
        filter_window=args.filter_window,
        baseline_max_position=args.baseline_max_position,
        min_closure_position=args.min_closure_position,
        max_closure_position=args.max_closure_position,
        max_command_lead=args.max_command_lead,
        max_force_rate=args.max_force_rate,
        command_min=0.0,
        command_max=1.0,
    )

    replay_rows = []
    for row in rows:
        current = first_float(
            row,
            ('obs_gripper', 'gripper_safety_current'),
        )
        proposed = first_float(
            row,
            ('base_gripper_cmd', 'steered_gripper_cmd'),
        )
        raw_force = first_float(row, ('control_force_n',))
        action = np.zeros((1, 8), dtype=np.float32)
        action[0, 7] = proposed
        controlled, metadata = controller.apply(
            action,
            current_gripper=current,
            raw_force=raw_force,
        )
        replay_rows.append({
            'frame_idx': row.get('frame_idx', len(replay_rows)),
            'recorded_force_n': raw_force,
            'recorded_gripper': current,
            'recorded_proposed_gripper': proposed,
            'replayed_gripper_command': float(controlled[0, 7]),
            'replayed_mode': metadata['gripper_safety_mode'],
            'replayed_filtered_force_n': metadata['gentle_filtered_force_n'],
            'replayed_baseline_force_n': metadata['gentle_baseline_force_n'],
            'replayed_force_delta_n': metadata['gentle_force_delta_n'],
            'replayed_force_rate_n_s': metadata['gentle_force_rate_n_s'],
            'replayed_last_safe_closure': metadata['gentle_last_safe_closure'],
            'replayed_max_command_lead': metadata['gentle_max_command_lead'],
            'replayed_preempt_action_queue': metadata['preempt_action_queue'],
        })

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(replay_rows[0]))
            writer.writeheader()
            writer.writerows(replay_rows)

    recorded_force = [row['recorded_force_n'] for row in replay_rows]
    recorded_gripper = [row['recorded_gripper'] for row in replay_rows]
    replayed_command = [row['replayed_gripper_command'] for row in replay_rows]
    closes = [
        row for row in replay_rows
        if row['replayed_mode'] in ('close_below_band', 'maintain_below_band')
    ]
    holds = [
        row for row in replay_rows
        if row['replayed_mode'] == 'hold_target_band'
    ]
    releases = [
        row for row in replay_rows
        if row['replayed_mode'] in (
            'open_above_band',
            'rate_release',
            'emergency_release',
        )
    ]
    preempts = [
        row for row in replay_rows if row['replayed_preempt_action_queue']
    ]
    print(f'log: {args.force_log}')
    print(f'frames: {len(replay_rows)}')
    print(f'recorded max force: {max(recorded_force):.3f} N')
    print(f'recorded max gripper: {max(recorded_gripper):.3f}')
    print(f'replayed max command: {max(replayed_command):.3f}')
    print(f'close frames: {len(closes)}')
    print(f'hold-band frames: {len(holds)}')
    print(f'release frames: {len(releases)}')
    print(f'queue-preempt frames: {len(preempts)}')
    if releases:
        print(f'first release frame: {releases[0]["frame_idx"]}')
    if args.output is not None:
        print(f'wrote: {args.output}')


if __name__ == '__main__':
    main()
