"""
Policy inference server for real robot deployment.

Run this in forcelens_dp repo:
    python policy_server.py --ckpt-path outputs/.../checkpoints/xxx.ckpt

On the robot-controller side, use RemotePolicy with:
    image_width=320, image_height=240
"""

import argparse
import atexit
import csv
import contextlib
import math
import queue
import signal
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from collections import deque
from pathlib import Path

import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf
from PIL import Image

from inference_profiles import apply_profile_defaults

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.video_encoding import (
    VideoEncodingError,
    encode_h264,
    h264_encoder_available,
)

CONTROL_PERIOD = 0.1        # 10 Hz
LATENCY_BUDGET = 0.2        # 200 ms
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / CONTROL_PERIOD)  # 2

IMAGE_H = 240
IMAGE_W = 320
VISUALFORCE_INPUT_SIZE = (256, 256)

# Same gripper-colour prior used by VisualForce's offline mask bootstrapping.
GRIPPER_H_MIN = 130.0
GRIPPER_H_MAX = 185.0
GRIPPER_S_MIN = 0.30
GRIPPER_V_MIN = 0.20
GRIPPER_MIN_BLOB_AREA = 300


def make_hsv_gripper_mask(
    frame,
    *,
    h_min=GRIPPER_H_MIN,
    h_max=GRIPPER_H_MAX,
    s_min=GRIPPER_S_MIN,
    v_min=GRIPPER_V_MIN,
    min_blob_area=GRIPPER_MIN_BLOB_AREA,
):
    """Segment the green gripper prior and remove small components."""

    hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * 2.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    mask = (
        (hue >= h_min)
        & (hue <= h_max)
        & (saturation >= s_min)
        & (value >= v_min)
    ).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _centroids = cv.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for label in range(1, n_labels):
        if stats[label, cv.CC_STAT_AREA] >= min_blob_area:
            cleaned[labels == label] = 255
    return cleaned


def _force_target_score(
    proxy_force,
    desired_force,
    force_min=None,
    force_max=None,
    high_force_weight=1.0,
):
    """Return scalar target or force-band cost for one force prediction."""
    proxy_force = float(proxy_force)
    if force_min is None and force_max is None:
        return float((proxy_force - float(desired_force)) ** 2)
    if force_min is None or force_max is None:
        raise ValueError('force_min and force_max must be provided together')
    if proxy_force < force_min:
        return float((force_min - proxy_force) ** 2)
    if proxy_force > force_max:
        return float(high_force_weight * (proxy_force - force_max) ** 2)
    return 0.0


def _resolve_tts_force_target(
    mode,
    requested_force,
    current_force,
    baseline_force=None,
):
    """Resolve an unambiguous absolute score target for TTS candidates."""
    requested_force = float(requested_force)
    current_force = float(current_force)
    if mode == 'absolute':
        return requested_force, None
    if mode != 'baseline_delta':
        raise ValueError(
            "TTS force target mode must be 'absolute' or 'baseline_delta'"
        )
    baseline = (
        current_force if baseline_force is None else float(baseline_force)
    )
    return baseline + requested_force, baseline


def _select_action_refill(
    act_sequence,
    queued_steps,
    buffer_steps,
    skip_steps,
    start_of_episode,
):
    """Return only the non-overlapping actions needed to refill a queue."""
    sequence = np.asarray(act_sequence)
    queued_steps = max(0, int(queued_steps))
    buffer_steps = max(0, int(buffer_steps))
    skip_steps = max(0, int(skip_steps))
    if start_of_episode:
        return sequence[:buffer_steps]
    prediction = sequence[skip_steps:skip_steps + buffer_steps]
    available_slots = max(0, buffer_steps - queued_steps)
    return prediction[queued_steps:queued_steps + available_slots]


def _gripper_proxy_features(
    action,
    gripper_index,
    steps,
    closing_sign,
    current_gripper,
    mode,
):
    """Return positive and signed mean closure for an absolute action chunk."""
    if gripper_index is None:
        return 0.0, 0.0
    gripper = np.asarray(action)[:steps, gripper_index]
    if mode == 'relative':
        close_signal = closing_sign * (gripper - float(current_gripper))
    elif mode == 'absolute':
        close_signal = closing_sign * gripper
    else:
        raise ValueError(f'Unsupported gripper proxy mode: {mode!r}')
    return (
        float(np.maximum(close_signal, 0.0).mean()),
        float(close_signal.mean()),
    )


def _policy_force_proxies(action_force_trajectories, steps, aggregation):
    """Reduce per-step force outputs to one scoreable force per candidate."""
    force = np.asarray(action_force_trajectories, dtype=np.float32)
    if force.ndim != 2:
        raise ValueError(
            'Policy force trajectories must have shape (N, H), '
            f'got {force.shape}'
        )
    steps = min(max(1, int(steps)), force.shape[1])
    force = force[:, :steps]
    if aggregation == 'last':
        return force[:, -1]
    if aggregation == 'mean':
        return force.mean(axis=1)
    if aggregation == 'max':
        return force.max(axis=1)
    raise ValueError(
        f'Unsupported policy force aggregation: {aggregation!r}'
    )


def _resolve_policy_force_dynamic_target(
    action_force_trajectories,
    rise,
    cap,
):
    """Build a candidate-domain target from the sampled first-step force."""
    force = np.asarray(action_force_trajectories, dtype=np.float32)
    if force.ndim != 2 or force.shape[0] == 0 or force.shape[1] == 0:
        raise ValueError(
            'Policy force trajectories must have non-empty shape (N, H), '
            f'got {force.shape}'
        )
    base = float(np.median(force[:, 0]))
    target = min(float(cap), base + float(rise))
    return target, base


def _select_force_candidate(
    scores,
    proxy_forces,
    safety_limit=None,
    unsafe_fallback='least_bad',
    signed_closure_means=None,
):
    """Select the best safe candidate and report whether all were unsafe."""
    scores = np.asarray(scores, dtype=np.float64)
    proxy_forces = np.asarray(proxy_forces, dtype=np.float64)
    if safety_limit is None:
        return int(np.argmin(scores)), np.ones(len(scores), dtype=bool), False

    safe = proxy_forces <= float(safety_limit)
    if safe.any():
        safe_indices = np.flatnonzero(safe)
        best = safe_indices[int(np.argmin(scores[safe]))]
        return int(best), safe, False

    if unsafe_fallback == 'least_bad':
        best = int(np.argmin(scores))
    elif unsafe_fallback == 'min_close':
        if signed_closure_means is None:
            raise ValueError('min_close fallback needs signed closure values')
        best = int(np.argmin(np.asarray(signed_closure_means, dtype=np.float64)))
    else:
        raise ValueError(f'Unsupported unsafe fallback: {unsafe_fallback!r}')
    return best, safe, True


def _compose_selected_action(
    candidates,
    selected_idx,
    selection_scope='full',
    gripper_index=7,
):
    """Build the executable action chunk from nominal and force-selected samples."""
    candidates = np.asarray(candidates)
    if candidates.ndim != 3:
        raise ValueError(
            'TTS candidates must have shape (N, H, D), '
            f'got {candidates.shape}'
        )
    if not 0 <= int(selected_idx) < candidates.shape[0]:
        raise IndexError(
            f'Selected candidate {selected_idx} is outside '
            f'[0, {candidates.shape[0]})'
        )

    selected_idx = int(selected_idx)
    if selection_scope == 'full':
        return candidates[selected_idx].copy()
    if selection_scope != 'gripper':
        raise ValueError(
            f'Unsupported TTS selection scope: {selection_scope!r}'
        )
    if gripper_index is None:
        raise ValueError('Gripper-only TTS selection needs a gripper index')
    gripper_index = int(gripper_index)
    if not -candidates.shape[-1] <= gripper_index < candidates.shape[-1]:
        raise IndexError(
            f'Gripper index {gripper_index} is invalid for '
            f'action dimension {candidates.shape[-1]}'
        )

    # Candidate 0 is the nominal DP sample. Keep its Cartesian position and
    # orientation trajectory, and splice in only the force-selected gripper
    # trajectory. This prevents force reranking from steering the robot arm.
    action = candidates[0].copy()
    action[..., gripper_index] = candidates[selected_idx, ..., gripper_index]
    return action


def _apply_absolute_gripper_safety(
    actions,
    current_gripper,
    control_force,
    desired_force,
    deadband,
    stop_margin,
    release_step,
    gripper_index=7,
    close_positive=True,
    command_min=None,
    command_max=None,
    min_closure_position=None,
    latched=False,
):
    """Hold or gently release an absolute gripper command under high force."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2:
        raise ValueError(
            'Absolute gripper safety expects actions with shape (T, D), '
            f'got {actions.shape}'
        )
    if gripper_index is None:
        raise ValueError('Absolute gripper safety needs a gripper index')
    gripper_index = int(gripper_index)
    if not -actions.shape[1] <= gripper_index < actions.shape[1]:
        raise IndexError(
            f'Gripper index {gripper_index} is invalid for action dimension '
            f'{actions.shape[1]}'
        )

    current_gripper = float(current_gripper)
    control_force = float(control_force)
    desired_force = float(desired_force)
    deadband = float(deadband)
    stop_margin = float(stop_margin)
    release_step = float(release_step)
    min_closure_position = (
        None
        if min_closure_position is None
        else float(min_closure_position)
    )
    if not np.isfinite([
        current_gripper,
        control_force,
        desired_force,
        deadband,
        stop_margin,
        release_step,
        *([] if min_closure_position is None else [min_closure_position]),
    ]).all():
        raise ValueError('Absolute gripper safety inputs must be finite')
    if deadband < 0.0:
        raise ValueError('Absolute gripper safety deadband must be non-negative')
    if stop_margin < deadband:
        raise ValueError(
            'Absolute gripper safety stop margin must be >= deadband'
        )
    if release_step < 0.0:
        raise ValueError('Absolute gripper safety release step must be non-negative')

    enter_force = desired_force + deadband
    release_force = desired_force + stop_margin
    closing_sign = 1.0 if close_positive else -1.0
    closure_position = closing_sign * current_gripper
    closure_ready = (
        min_closure_position is None
        or closure_position >= min_closure_position
    )
    active = bool(latched)
    if not active and closure_ready and control_force > enter_force:
        active = True
    elif active and control_force <= desired_force:
        active = False

    mode = 'none'
    command = None
    if active:
        mode = 'hold'
        command = current_gripper
        if control_force >= release_force:
            mode = 'release'
            release_direction = -closing_sign
            command += release_direction * release_step

        lo = -np.inf if command_min is None else float(command_min)
        hi = np.inf if command_max is None else float(command_max)
        command = float(np.clip(command, lo, hi))
        actions[:, gripper_index] = command

    return actions, active, {
        'gripper_safety_active': active,
        'gripper_safety_mode': mode,
        'gripper_safety_current': current_gripper,
        'gripper_safety_command': '' if command is None else command,
        'gripper_safety_enter_force': enter_force,
        'gripper_safety_release_force': release_force,
        'gripper_safety_release_step': release_step,
        'gripper_safety_closure_ready': closure_ready,
        'gripper_safety_closure_position': closure_position,
        'gripper_safety_min_closure_position': (
            '' if min_closure_position is None else min_closure_position
        ),
        'preempt_action_queue': active,
    }


def _append_gripper_fallback_candidates(
    candidates,
    current_gripper,
    release_step,
    gripper_index=7,
    close_positive=True,
    command_min=None,
    command_max=None,
):
    """Append hold and bounded-open chunks so TTS always has a safe fallback."""
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3:
        raise ValueError(
            'Gripper fallbacks expect candidates with shape (N, H, D), '
            f'got {candidates.shape}'
        )
    if gripper_index is None:
        raise ValueError('Gripper fallbacks need a gripper index')
    gripper_index = int(gripper_index)
    if not -candidates.shape[-1] <= gripper_index < candidates.shape[-1]:
        raise IndexError(
            f'Gripper index {gripper_index} is invalid for '
            f'action dimension {candidates.shape[-1]}'
        )

    current_gripper = float(current_gripper)
    release_step = float(release_step)
    if release_step < 0.0:
        raise ValueError('Gripper fallback release_step must be non-negative')
    closing_sign = 1.0 if close_positive else -1.0
    lo = -np.inf if command_min is None else float(command_min)
    hi = np.inf if command_max is None else float(command_max)

    hold = candidates[0].copy()
    hold[:, gripper_index] = np.clip(current_gripper, lo, hi)
    opened = candidates[0].copy()
    open_command = current_gripper - closing_sign * release_step
    opened[:, gripper_index] = np.clip(open_command, lo, hi)
    return np.concatenate([candidates, hold[None], opened[None]], axis=0)


class GentleGripperController:
    """Direct VisualForce feedback for an absolute-position gripper.

    The command setpoint is persistent, rather than recomputed from the
    measured position every frame. This lets small bounded increments build
    enough actuator lead to overcome the physical gripper deadband. VisualForce
    is zeroed once from a fixed open-gripper baseline at episode start. Force
    contact decisions are ignored until a task-specific, demonstrated minimum
    closure; invalid measurements still hold immediately. Initial contact must
    reach the target itself; the lower hysteresis edge is used only after
    contact. This prevents pose-dependent baseline drift from latching a false
    grasp while the gripper is open. Optionally, after contact, a stable open
    suffix in the nominal policy chunk can latch a full release so neither the
    force hold nor force-based candidate selection erases a learned drop.
    """

    def __init__(
        self,
        desired_force_delta,
        deadband,
        stop_margin,
        close_step,
        maintain_step,
        release_step,
        filter_window=3,
        baseline_max_position=0.35,
        min_closure_position=0.35,
        max_closure_position=0.82,
        max_command_lead=0.25,
        max_force_rate=20.0,
        policy_release_enabled=True,
        policy_release_contact_delta=None,
        policy_gripper_approach=False,
        policy_release_min_steps=2,
        baseline_min_samples=3,
        gripper_index=7,
        close_positive=True,
        command_min=None,
        command_max=None,
        control_period=CONTROL_PERIOD,
    ):
        self.desired_force_delta = float(desired_force_delta)
        self.deadband = float(deadband)
        self.stop_margin = float(stop_margin)
        self.close_step = float(close_step)
        self.maintain_step = float(maintain_step)
        self.release_step = float(release_step)
        self.filter_window = int(filter_window)
        self.baseline_max_position = float(baseline_max_position)
        self.min_closure_position = float(min_closure_position)
        self.max_closure_position = float(max_closure_position)
        self.max_command_lead = float(max_command_lead)
        self.max_force_rate = (
            None if max_force_rate is None else float(max_force_rate)
        )
        self.policy_release_enabled = bool(policy_release_enabled)
        self.policy_release_contact_delta = (
            self.desired_force_delta
            if policy_release_contact_delta is None
            else float(policy_release_contact_delta)
        )
        self.policy_gripper_approach = bool(policy_gripper_approach)
        self.policy_release_min_steps = int(policy_release_min_steps)
        self.baseline_min_samples = int(baseline_min_samples)
        self.gripper_index = gripper_index
        self.close_positive = bool(close_positive)
        self.command_min = command_min
        self.command_max = command_max
        self.control_period = float(control_period)

        if self.desired_force_delta < 0.0:
            raise ValueError('desired_force_delta must be non-negative')
        if self.deadband < 0.0:
            raise ValueError('gentle-gripper deadband must be non-negative')
        if self.stop_margin < self.deadband:
            raise ValueError('gentle-gripper stop_margin must be >= deadband')
        if self.close_step <= 0.0:
            raise ValueError('gentle-gripper close_step must be positive')
        if not 0.0 < self.maintain_step <= self.close_step:
            raise ValueError(
                'gentle-gripper maintain_step must be in (0, close_step]'
            )
        if self.release_step < 0.0:
            raise ValueError('gentle-gripper release_step must be non-negative')
        if self.filter_window <= 0:
            raise ValueError('gentle-gripper filter_window must be positive')
        if self.baseline_max_position > self.min_closure_position:
            raise ValueError(
                'gentle-gripper baseline_max_position must be <= '
                'min_closure_position'
            )
        if self.max_closure_position < self.min_closure_position:
            raise ValueError(
                'gentle-gripper max_closure_position must be >= '
                'min_closure_position'
            )
        if self.max_command_lead <= 0.0:
            raise ValueError('gentle-gripper max_command_lead must be positive')
        if self.max_force_rate is not None and self.max_force_rate <= 0.0:
            raise ValueError('gentle-gripper max_force_rate must be positive')
        if self.policy_release_min_steps <= 0:
            raise ValueError('policy_release_min_steps must be positive')
        if self.policy_release_contact_delta < 0.0:
            raise ValueError(
                'policy_release_contact_delta must be non-negative'
            )
        if self.baseline_min_samples <= 0:
            raise ValueError('gentle-gripper baseline_min_samples must be positive')
        if self.control_period <= 0.0:
            raise ValueError('gentle-gripper control_period must be positive')
        self.reset()

    def reset(self):
        self.force_history = deque(maxlen=self.filter_window)
        self.baseline_samples = deque(maxlen=self.baseline_min_samples)
        self.baseline_force = None
        self.previous_filtered_force = None
        self.command_closure = None
        self.contact_latched = False
        self.contact_seen = False
        self.last_safe_closure = None
        self.release_closure = None
        self.policy_release_latched = False
        self.policy_release_armed = False
        self.policy_release_index = None
        self.policy_release_horizon = None
        self.measurement_fault_latched = False

    def _find_policy_release(self, original_closures):
        """Find a stable open suffix in a post-contact policy chunk."""
        if not self.policy_release_enabled or not self.policy_release_armed:
            return None

        original_closures = np.asarray(original_closures, dtype=np.float64)
        below_threshold = original_closures <= self.min_closure_position
        stable_suffix = np.logical_and.accumulate(below_threshold[::-1])[::-1]
        release_indices = np.flatnonzero(stable_suffix)
        if len(release_indices) == 0:
            return None

        release_index = int(release_indices[0])
        suffix_steps = len(original_closures) - release_index
        if suffix_steps < self.policy_release_min_steps:
            return None
        return release_index

    def _fully_open_closure(self, original_closures):
        """Return the least-closed command allowed by the configured bounds."""
        closure_bounds = []
        closing_sign = 1.0 if self.close_positive else -1.0
        if self.command_min is not None:
            closure_bounds.append(closing_sign * float(self.command_min))
        if self.command_max is not None:
            closure_bounds.append(closing_sign * float(self.command_max))
        if closure_bounds:
            return min(closure_bounds)
        return float(np.min(original_closures))

    def _finish(
        self,
        actions,
        original_gripper,
        closing_sign,
        current_gripper,
        current_closure,
        proposed_closure,
        command_closure,
        mode,
        measurement_valid,
        filtered_force='',
        force_delta='',
        force_rate='',
        just_latched=False,
        preempt=False,
        raw_force='',
    ):
        command_closure = min(command_closure, self.max_closure_position)
        command = closing_sign * command_closure
        lo = -np.inf if self.command_min is None else float(self.command_min)
        hi = np.inf if self.command_max is None else float(self.command_max)
        command = float(np.clip(command, lo, hi))
        command_closure = closing_sign * command
        self.command_closure = command_closure
        actions[:, self.gripper_index] = command
        modified = not np.allclose(original_gripper, actions[:, self.gripper_index])
        position_capped = command_closure >= self.max_closure_position
        absolute_target = (
            ''
            if self.baseline_force is None
            else self.baseline_force + self.desired_force_delta
        )
        absolute_lower = (
            ''
            if self.baseline_force is None
            else self.baseline_force + max(
                0.0,
                self.desired_force_delta - self.deadband,
            )
        )
        absolute_upper = (
            ''
            if self.baseline_force is None
            else self.baseline_force + self.desired_force_delta + self.deadband
        )
        absolute_emergency = (
            ''
            if self.baseline_force is None
            else self.baseline_force + self.desired_force_delta + self.stop_margin
        )
        return actions, {
            'gripper_safety_active': bool(
                self.contact_latched
                or self.policy_release_latched
                or not measurement_valid
            ),
            'gripper_safety_mode': mode,
            'gripper_safety_current': current_gripper,
            'gripper_safety_command': command,
            'gripper_safety_enter_force': absolute_upper,
            'gripper_safety_release_force': absolute_lower,
            'gripper_safety_release_step': self.release_step,
            'gripper_safety_closure_ready': bool(
                current_closure >= self.min_closure_position
            ),
            'gripper_safety_closure_position': current_closure,
            'gripper_safety_min_closure_position': self.min_closure_position,
            # Routine approach updates must not flush the DP arm trajectory.
            # Preemption is reserved for a newly detected contact, sensor
            # fault, or active force release.
            'preempt_action_queue': bool(preempt),
            'gentle_controller': True,
            'gentle_measurement_valid': bool(measurement_valid),
            'gentle_just_latched': bool(just_latched),
            'gentle_raw_force_n': raw_force,
            'gentle_filtered_force_n': filtered_force,
            'gentle_baseline_force_n': (
                '' if self.baseline_force is None else self.baseline_force
            ),
            'gentle_baseline_calibrated': self.baseline_force is not None,
            'gentle_force_delta_n': force_delta,
            'gentle_force_rate_n_s': force_rate,
            'gentle_desired_force_n': absolute_target,
            'gentle_lower_force_n': absolute_lower,
            'gentle_upper_force_n': absolute_upper,
            'gentle_emergency_force_n': absolute_emergency,
            'gentle_desired_delta_n': self.desired_force_delta,
            'gentle_last_safe_closure': (
                '' if self.last_safe_closure is None else self.last_safe_closure
            ),
            'gentle_hold_closure': self.command_closure,
            'gentle_command_closure': command_closure,
            'gentle_close_limited': bool(
                modified and proposed_closure > command_closure
            ),
            'gentle_position_capped': position_capped,
            'gentle_max_closure_position': self.max_closure_position,
            'gentle_max_command_lead': self.max_command_lead,
            'gentle_max_force_rate_n_s': (
                '' if self.max_force_rate is None else self.max_force_rate
            ),
            'gentle_policy_release_latched': self.policy_release_latched,
            'gentle_policy_release_enabled': self.policy_release_enabled,
            'gentle_policy_release_armed': self.policy_release_armed,
            'gentle_policy_release_contact_delta_n': (
                self.policy_release_contact_delta
            ),
            'gentle_policy_gripper_approach': self.policy_gripper_approach,
            'gentle_force_control_armed': self.policy_release_armed,
            'gentle_policy_release_index': (
                ''
                if self.policy_release_index is None
                else self.policy_release_index
            ),
            'gentle_policy_release_threshold': self.min_closure_position,
            'gentle_policy_release_horizon': (
                ''
                if self.policy_release_horizon is None
                else self.policy_release_horizon
            ),
            'gentle_policy_release_min_steps': self.policy_release_min_steps,
        }

    def apply(
        self,
        actions,
        current_gripper,
        raw_force,
        measurement_valid=True,
        policy_actions=None,
    ):
        actions = np.asarray(actions, dtype=np.float32).copy()
        if actions.ndim != 2:
            raise ValueError(
                'Direct force gripper control expects actions with shape (T, D), '
                f'got {actions.shape}'
            )
        if self.gripper_index is None:
            raise ValueError('Direct force gripper control needs a gripper index')
        gi = int(self.gripper_index)
        if not -actions.shape[1] <= gi < actions.shape[1]:
            raise IndexError(
                f'Gripper index {gi} is invalid for action dimension '
                f'{actions.shape[1]}'
            )

        current_gripper = float(current_gripper)
        raw_force = float(raw_force)
        if not np.isfinite([current_gripper, raw_force]).all():
            raise ValueError('Gentle gripper inputs must be finite')

        closing_sign = 1.0 if self.close_positive else -1.0
        current_closure = closing_sign * current_gripper
        proposed_closure = closing_sign * float(actions[0, gi])
        original_gripper = actions[:, gi].copy()
        original_closures = closing_sign * original_gripper
        if policy_actions is None:
            policy_actions = actions
        policy_actions = np.asarray(policy_actions)
        if policy_actions.ndim != 2 or policy_actions.shape[1] != actions.shape[1]:
            raise ValueError(
                'Policy release reference must have shape (T, D) matching '
                f'the selected action dimension, got {policy_actions.shape}'
            )
        policy_closures = closing_sign * policy_actions[:, gi]
        if self.command_closure is None:
            self.command_closure = current_closure

        release_index = self._find_policy_release(policy_closures)
        newly_released = (
            not self.policy_release_latched and release_index is not None
        )
        if newly_released:
            self.policy_release_latched = True
            self.policy_release_index = release_index
            self.policy_release_horizon = len(policy_closures)

        if self.policy_release_latched:
            # The force loop deliberately overrides the approach and grasp, but
            # it must yield to a learned post-contact drop. Latch full-open so
            # falling force on the next frame cannot start the maintenance
            # close loop again.
            self.contact_latched = False
            self.release_closure = None
            command_closure = self._fully_open_closure(policy_closures)
            return self._finish(
                actions,
                original_gripper,
                closing_sign,
                current_gripper,
                current_closure,
                proposed_closure,
                command_closure,
                'policy_release',
                measurement_valid,
                preempt=newly_released,
                raw_force=raw_force,
            )

        if not measurement_valid:
            first_fault = not self.measurement_fault_latched
            self.measurement_fault_latched = True
            if proposed_closure < current_closure:
                command_closure = max(
                    proposed_closure,
                    current_closure - self.release_step,
                )
                mode = 'sensor_open'
            else:
                command_closure = current_closure
                mode = 'sensor_hold'
            return self._finish(
                actions,
                original_gripper,
                closing_sign,
                current_gripper,
                current_closure,
                proposed_closure,
                command_closure,
                mode,
                False,
                preempt=first_fault,
                raw_force=raw_force,
            )
        self.measurement_fault_latched = False

        self.force_history.append(raw_force)
        filtered_force = float(np.median(np.asarray(self.force_history)))

        # VisualForce has a pose- and scene-dependent non-contact offset. Latch it
        # once from the known-open start of the episode. Updating it throughout
        # approach can absorb the small Coke contact signal into the baseline.
        # Once sampling has begun from a verified-open pose, tolerate small
        # actuator settling beyond baseline_max_position. The separate
        # min_closure_position remains the hard boundary before possible
        # contact. Without this continuation, crossing 0.10 after only a few
        # samples leaves the controller in calibration_hold forever.
        baseline_in_progress = (
            len(self.baseline_samples) > 0
            and current_closure < self.min_closure_position
        )
        in_baseline_region = (
            current_closure <= self.baseline_max_position
            or baseline_in_progress
        )
        if in_baseline_region and not self.contact_latched:
            if self.baseline_force is None:
                self.baseline_samples.append(raw_force)
                if len(self.baseline_samples) >= self.baseline_min_samples:
                    self.baseline_force = float(
                        np.median(np.asarray(self.baseline_samples))
                    )

        if self.baseline_force is None:
            # Do not close from a force estimate that has not been zeroed. In
            # the normal open start this lasts only baseline_min_samples frames.
            return self._finish(
                actions,
                original_gripper,
                closing_sign,
                current_gripper,
                current_closure,
                proposed_closure,
                current_closure,
                'calibration_hold',
                True,
                filtered_force=filtered_force,
                force_rate=0.0,
                raw_force=raw_force,
            )

        # Control contact force rise relative to the open-gripper estimate.
        # Hysteresis avoids chattering on noisy frame-by-frame predictions.
        force_delta = max(0.0, filtered_force - self.baseline_force)

        # Release is armed by contact evidence independently of the requested
        # holding force. A high target must not permanently suppress the
        # policy's demonstrated drop, but estimator noise while open must not
        # be allowed to trigger it either.
        if (
            current_closure >= self.min_closure_position
            and force_delta >= self.policy_release_contact_delta
        ):
            self.policy_release_armed = True

        force_rate = 0.0
        if self.previous_filtered_force is not None:
            force_rate = (
                filtered_force - self.previous_filtered_force
            ) / self.control_period
        self.previous_filtered_force = filtered_force

        if self.policy_gripper_approach and not self.policy_release_armed:
            # Preserve the learned approach/grasp timing instead of closing by
            # a fixed increment from the end of baseline calibration. The
            # policy command is still bounded by the demonstrated closure cap
            # and by command lead; force control takes ownership after contact
            # evidence reaches policy_release_contact_delta.
            nominal_closure = float(policy_closures[0])
            command_closure = min(
                nominal_closure,
                current_closure + self.max_command_lead,
                self.max_closure_position,
            )
            return self._finish(
                actions,
                original_gripper,
                closing_sign,
                current_gripper,
                current_closure,
                proposed_closure,
                command_closure,
                'policy_approach',
                True,
                filtered_force=filtered_force,
                force_delta=force_delta,
                force_rate=force_rate,
                raw_force=raw_force,
            )

        lower_force = max(0.0, self.desired_force_delta - self.deadband)
        upper_force = self.desired_force_delta + self.deadband
        emergency_force = self.desired_force_delta + self.stop_margin
        projected_force = force_delta + max(0.0, force_rate) * self.control_period
        closure_ready = current_closure >= self.min_closure_position
        rate_brake = (
            closure_ready
            and self.max_force_rate is not None
            and force_rate >= self.max_force_rate
            and force_delta < upper_force
            and projected_force >= upper_force
        )

        was_latched = self.contact_latched
        if closure_ready and force_delta >= emergency_force:
            self.contact_latched = True
            self.contact_seen = True
            if self.release_closure is None:
                self.release_closure = min(self.command_closure, current_closure)
                if self.last_safe_closure is not None:
                    self.release_closure = min(
                        self.release_closure,
                        self.last_safe_closure,
                    )
                else:
                    self.release_closure -= self.release_step
            command_closure = self.release_closure
            mode = 'emergency_release'
            # Flush queued closing commands once. Holding one recovery setpoint
            # lets the actuator settle instead of opening another step for
            # every delayed force frame.
            preempt = not was_latched
        elif closure_ready and force_delta > upper_force:
            self.contact_latched = True
            self.contact_seen = True
            if self.release_closure is None:
                self.release_closure = min(self.command_closure, current_closure)
                if self.last_safe_closure is not None:
                    self.release_closure = min(
                        self.release_closure,
                        self.last_safe_closure,
                    )
                else:
                    self.release_closure -= self.release_step
            command_closure = self.release_closure
            mode = 'open_above_band'
            # Flush queued closing commands once when contact is detected.
            # Repeated preemption resamples the arm path at camera rate and was
            # observed as shaking around the berry.
            preempt = not was_latched
        elif rate_brake:
            self.contact_latched = True
            self.contact_seen = True
            self.release_closure = None
            command_closure = min(self.command_closure, current_closure)
            mode = 'rate_hold'
            # The one-period projection has crossed the upper band. Stop adding
            # closure now, before actuator and median-filter lag cause a large
            # force overshoot.
            preempt = not was_latched
        elif not closure_ready:
            self.contact_latched = False
            self.release_closure = None
            self.last_safe_closure = current_closure
            close_step = (
                self.maintain_step if self.contact_seen else self.close_step
            )
            command_closure = min(
                max(self.command_closure, current_closure) + close_step,
                current_closure + self.max_command_lead,
                self.max_closure_position,
            )
            mode = (
                'maintain_below_band'
                if self.contact_seen
                else 'close_before_contact_region'
            )
            preempt = False
        elif (
            (not self.contact_seen and force_delta < self.desired_force_delta)
            or (self.contact_seen and force_delta < lower_force)
        ):
            self.contact_latched = False
            self.release_closure = None
            self.last_safe_closure = current_closure
            close_step = (
                self.maintain_step if self.contact_seen else self.close_step
            )
            command_closure = min(
                max(self.command_closure, current_closure) + close_step,
                current_closure + self.max_command_lead,
                self.max_closure_position,
            )
            mode = (
                'maintain_below_band'
                if self.contact_seen
                else 'close_below_band'
            )
            preempt = False
        else:
            self.contact_latched = True
            self.contact_seen = True
            command_closure = (
                min(self.command_closure, current_closure)
                if self.release_closure is None
                else self.release_closure
            )
            mode = 'hold_target_band'
            preempt = not was_latched

        just_latched = self.contact_latched and not was_latched

        return self._finish(
            actions,
            original_gripper,
            closing_sign,
            current_gripper,
            current_closure,
            proposed_closure,
            command_closure,
            mode,
            True,
            filtered_force=filtered_force,
            force_delta=force_delta,
            force_rate=force_rate,
            just_latched=just_latched,
            preempt=preempt,
            raw_force=raw_force,
        )


def _load_steering_classes(visualforce_root):
    if visualforce_root is not None:
        root = Path(visualforce_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f'VisualForce root does not exist: {root}')
        sys.path.insert(0, str(root))

    try:
        from src.steering import ForceSteeringConfig, VisualForceSteeringPipeline
    except ImportError as exc:
        raise ImportError(
            'Could not import VisualForce steering. Run from the VisualForce repo, '
            'or pass --tts-visualforce-root /path/to/VisualForce.'
        ) from exc
    return ForceSteeringConfig, VisualForceSteeringPipeline


class DeltaForceCritic:
    """Action-conditioned force-delta critic used to rerank sampled DP chunks."""

    def __init__(self, ckpt_path, visualforce_root, device='cuda'):
        root = Path(visualforce_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f'VisualForce root does not exist: {root}')
        sys.path.insert(0, str(root))

        from src.flip_delta_force_dataset import standardize, unstandardize
        from src.model.action_delta_force import ActionConditionedDeltaForceNet
        from src.steering import masked_frame_to_tensor

        self.standardize = standardize
        self.unstandardize = unstandardize
        self.device = torch.device(device)
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        args = ckpt.get('args', {})
        self.args = args
        self.obs_steps = int(args.get('obs_steps', 2))
        self.pred_horizon = int(args.get('pred_horizon', 16))
        self.action_mode = args.get('action_mode', 'obs_delta_pos_gripper')
        self.image_normalization = args.get('image_normalization', 'minus_one_one')
        self.critic_image_mode = args.get('critic_image_mode', 'rgb')
        self.target_mode = args.get('target_mode', 'delta_trajectory')
        self.image_size = tuple(args.get('image_size', [240, 320]))
        self.include_agent_pos = bool(args.get('include_agent_pos', ckpt.get('low_dim_dim', 0) > 0))
        self.include_current_force = bool(args.get('include_current_force', False))
        self.masked_frame_to_tensor = masked_frame_to_tensor
        if self.critic_image_mode not in ('rgb', 'edge', 'none'):
            raise ValueError(
                'Unsupported delta-force critic_image_mode: '
                f'{self.critic_image_mode!r}'
            )

        self.action_stats = ckpt['action_stats']
        self.target_stats = ckpt['target_stats']
        self.agent_pos_stats = ckpt.get('agent_pos_stats')
        self.current_force_stats = ckpt.get('current_force_stats')

        self.model = ActionConditionedDeltaForceNet(
            image_channels=int(ckpt['image_channels']),
            action_dim=int(ckpt['action_dim']),
            pred_horizon=self.pred_horizon,
            force_dim=int(ckpt.get('force_dim', 1)),
            low_dim_dim=int(ckpt.get('low_dim_dim', 0)),
            output_horizon=int(ckpt.get('output_horizon', 1)),
        ).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self._warned_horizon = False
        print(
            'Delta-force critic enabled: '
            f'{self.ckpt_path} '
            f'(target_mode={args.get("target_mode", "unknown")}, '
            f'pred_horizon={self.pred_horizon}, '
            f'action_mode={self.action_mode}, '
            f'critic_image_mode={self.critic_image_mode}, '
            f'include_current_force={self.include_current_force})'
        )

    @property
    def predicts_future_peak(self):
        return self.target_mode == 'future_peak'

    def _latest_agent_pos(self, latest_obs):
        return np.concatenate([
            np.asarray(latest_obs['arm_pos'], dtype=np.float32),
            np.asarray(latest_obs['arm_quat'], dtype=np.float32),
            np.asarray(latest_obs['gripper_pos'], dtype=np.float32),
        ]).astype(np.float32)

    def _action_features(self, latest_obs, candidates):
        candidates = np.asarray(candidates, dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[-1] != 8:
            raise ValueError(
                'Delta-force critic expects candidates with shape '
                f'(N, H, 8), got {candidates.shape}'
            )

        horizon = candidates.shape[1]
        if horizon < self.pred_horizon:
            if not self._warned_horizon:
                print(
                    'Delta-force critic received shorter action chunks '
                    f'({horizon}) than training horizon ({self.pred_horizon}); '
                    'padding with the last action.'
                )
                self._warned_horizon = True
            pad = np.repeat(candidates[:, -1:, :], self.pred_horizon - horizon, axis=1)
            seq = np.concatenate([candidates, pad], axis=1)
        else:
            seq = candidates[:, :self.pred_horizon, :]

        anchor = self._latest_agent_pos(latest_obs)
        action_delta = seq.copy()
        if self.action_mode == 'obs_delta':
            action_delta = action_delta - anchor[None, None, :]
        elif self.action_mode == 'obs_delta_pos_gripper':
            action_delta[:, :, :3] = seq[:, :, :3] - anchor[None, None, :3]
            action_delta[:, :, 7:8] = seq[:, :, 7:8] - anchor[None, None, 7:8]
        else:
            raise ValueError(f'Unsupported delta-force action_mode: {self.action_mode!r}')
        return torch.from_numpy(action_delta).float().to(self.device)

    def _image_features(self, obs_dict, frame_key, batch_size, mask=None):
        if self.critic_image_mode == 'none':
            return torch.empty(
                batch_size,
                0,
                self.image_size[0],
                self.image_size[1],
                device=self.device,
            )
        if frame_key is None or frame_key not in obs_dict:
            rgb_keys = [
                key for key, value in obs_dict.items()
                if value.ndim == 5 and value.shape[-3] == 3
            ]
            if len(rgb_keys) != 1:
                raise ValueError(
                    'Delta-force critic needs --tts-frame-key to select an '
                    f'RGB obs key; found {rgb_keys}'
                )
            frame_key = rgb_keys[0]

        image_seq = obs_dict[frame_key][0].float()
        if image_seq.shape[0] < self.obs_steps:
            pad = image_seq[:1].repeat(self.obs_steps - image_seq.shape[0], 1, 1, 1)
            image_seq = torch.cat([pad, image_seq], dim=0)
        image_seq = image_seq[-self.obs_steps:]
        if self.critic_image_mode == 'edge':
            if mask is None:
                raise ValueError('Edge delta-force critic requires the current gripper mask.')
            edge_frames = []
            mask_np = np.asarray(mask).astype(bool)
            for frame_t in image_seq:
                frame_np = (
                    frame_t.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
                    * 255.0
                ).astype(np.uint8)
                edge = self.masked_frame_to_tensor(
                    frame_np,
                    mask_np,
                    output_size=self.image_size,
                    input_mode='edge',
                )[0]
                edge_frames.append(edge)
            image_seq = torch.stack(edge_frames, dim=0).unsqueeze(1)
        else:
            if self.image_normalization == 'minus_one_one':
                image_seq = image_seq * 2.0 - 1.0
            elif self.image_normalization != 'zero_one':
                raise ValueError(
                    'Unsupported delta-force image_normalization: '
                    f'{self.image_normalization!r}'
                )
        if self.critic_image_mode == 'edge' and self.image_normalization == 'minus_one_one':
            image_seq = image_seq * 2.0 - 1.0
        elif self.critic_image_mode == 'edge' and self.image_normalization != 'zero_one':
            raise ValueError(
                'Unsupported delta-force image_normalization: '
                f'{self.image_normalization!r}'
            )
        image = image_seq.reshape(
            1,
            self.obs_steps * image_seq.shape[1],
            image_seq.shape[2],
            image_seq.shape[3],
        )
        return image.repeat(batch_size, 1, 1, 1).to(self.device)

    def _low_dim_features(self, obs_dict, batch_size, current_force=None):
        parts = []
        if self.include_agent_pos:
            if 'agent_pos' not in obs_dict:
                raise KeyError('Delta-force critic checkpoint needs agent_pos in obs.')
            agent_seq = obs_dict['agent_pos'][0].float()
            if agent_seq.shape[0] < self.obs_steps:
                pad = agent_seq[:1].repeat(self.obs_steps - agent_seq.shape[0], 1)
                agent_seq = torch.cat([pad, agent_seq], dim=0)
            agent = agent_seq[-self.obs_steps:].reshape(1, -1)
            agent = agent.repeat(batch_size, 1).to(self.device)
            agent = self.standardize(agent, self.agent_pos_stats)
            parts.append(agent)
        if self.include_current_force:
            if current_force is None:
                raise ValueError('Delta-force critic checkpoint needs current force.')
            force = torch.full(
                (batch_size, 1),
                float(current_force),
                dtype=torch.float32,
                device=self.device,
            )
            force = self.standardize(force, self.current_force_stats)
            parts.append(force)
        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def predict_delta(self, obs_dict, latest_obs, candidates, frame_key=None, mask=None, current_force=None):
        batch_size = int(np.asarray(candidates).shape[0])
        image = self._image_features(obs_dict, frame_key, batch_size, mask=mask)
        action = self._action_features(latest_obs, candidates)
        action = self.standardize(action, self.action_stats)
        low_dim = self._low_dim_features(
            obs_dict,
            batch_size,
            current_force=current_force,
        )

        with torch.no_grad():
            pred_n = self.model(image, action, low_dim)
            pred = self.unstandardize(pred_n, self.target_stats)
        return pred[:, 0, 0].detach().cpu().numpy().astype(np.float32)


class Sam2FrameMasker:
    """Run the same prompt+SAM2 mask path used by VisualForce offline rollouts."""

    def __init__(self, visualforce_root, model_key='small', sam2_repo=None, sam2_ckpt=None):
        root = Path(visualforce_root).expanduser().resolve()
        sys.path.insert(0, str(root))
        import segment_gripper

        sam2_repo_path = (
            Path(sam2_repo).expanduser().resolve()
            if sam2_repo is not None
            else root / 'third_party' / 'sam2'
        )
        if sam2_repo_path.exists():
            segment_gripper.SAM2_REPO = str(sam2_repo_path)
        if sam2_ckpt is not None:
            segment_gripper.LOCAL_CKPTS[model_key] = str(
                Path(sam2_ckpt).expanduser().resolve()
            )

        self.detect_orange_points = segment_gripper.detect_orange_points
        self.predictor, self.device = segment_gripper.build_predictor(model_key)
        self.lock = threading.Lock()
        self.prev_frame = None
        self.prev_mask = None
        self.reject_count = 0

    def reset(self):
        self.prev_frame = None
        self.prev_mask = None
        self.reject_count = 0

    def predict(self, frame):
        frame = np.asarray(frame).astype(np.uint8)
        if self.prev_frame is not None and self.prev_mask is not None:
            return self._predict_two_frame(frame)

        mask = self._predict_single_frame(frame)
        self._accept(frame, mask)
        return mask

    def _predict_single_frame(self, frame):
        point_coords, point_labels = self.detect_orange_points(frame)

        with tempfile.TemporaryDirectory(prefix='visualforce_tts_sam2_') as frame_dir:
            Image.fromarray(frame).save(
                str(Path(frame_dir) / '000000.jpg'),
                quality=95,
            )
            autocast = (
                torch.autocast(self.device, dtype=torch.bfloat16)
                if self.device == 'cuda'
                else contextlib.nullcontext()
            )
            with self.lock, torch.inference_mode(), autocast:
                state = self.predictor.init_state(video_path=frame_dir)
                self.predictor.reset_state(state)
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=point_coords,
                    labels=point_labels,
                )
                for _out_idx, _obj_ids, out_logits in self.predictor.propagate_in_video(state):
                    mask = (out_logits[0] > 0.0).cpu().numpy().squeeze()
                    mask = mask.astype(np.uint8) * 255
                    if not self._is_sane_first(mask):
                        prior = self._green_prior_mask(frame)
                        if self._is_sane_first(prior):
                            print(
                                'TTS SAM2 rejected first-frame mask; '
                                'using HSV gripper prior'
                            )
                            return prior
                    return mask

        return np.zeros(frame.shape[:2], dtype=np.uint8)

    def _predict_two_frame(self, frame):
        prev_frame = self.prev_frame
        prev_mask = self.prev_mask
        if prev_frame.shape[:2] != frame.shape[:2]:
            prev_mask = cv.resize(
                prev_mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv.INTER_NEAREST,
            )
            prev_frame = cv.resize(prev_frame, (frame.shape[1], frame.shape[0]))

        with tempfile.TemporaryDirectory(prefix='visualforce_tts_sam2_pair_') as frame_dir:
            frame_dir = Path(frame_dir)
            Image.fromarray(prev_frame).save(str(frame_dir / '000000.jpg'), quality=95)
            Image.fromarray(frame).save(str(frame_dir / '000001.jpg'), quality=95)
            autocast = (
                torch.autocast(self.device, dtype=torch.bfloat16)
                if self.device == 'cuda'
                else contextlib.nullcontext()
            )
            with self.lock, torch.inference_mode(), autocast:
                state = self.predictor.init_state(video_path=str(frame_dir))
                self.predictor.reset_state(state)
                self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    mask=prev_mask.astype(bool),
                )
                candidate = None
                for out_idx, _obj_ids, out_logits in self.predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=2,
                ):
                    if out_idx == 1:
                        candidate = (out_logits[0] > 0.0).cpu().numpy().squeeze()
                        break

        if candidate is None:
            candidate_mask = prev_mask
        else:
            candidate_mask = candidate.astype(np.uint8) * 255

        if self._is_sane(candidate_mask, prev_mask):
            self._accept(frame, candidate_mask)
            return candidate_mask

        self.reject_count += 1
        print(
            'TTS SAM2 rejected propagated mask; '
            f'using previous mask (reject_count={self.reject_count})'
        )
        return prev_mask

    def _accept(self, frame, mask):
        self.prev_frame = frame.copy()
        self.prev_mask = np.asarray(mask).astype(np.uint8).copy()
        self.reject_count = 0

    def _green_prior_mask(self, frame):
        return make_hsv_gripper_mask(frame)

    def _is_sane_first(self, mask):
        mask_bool = np.asarray(mask).astype(bool)
        area = int(mask_bool.sum())
        total = mask_bool.size
        if area / float(total) < 0.01:
            return False
        if area / float(total) > 0.40:
            return False
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return False
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        img_h, img_w = mask_bool.shape
        if width > 0.95 * img_w and height > 0.95 * img_h:
            return False
        return True

    def _is_sane(self, mask, prev_mask):
        mask_bool = np.asarray(mask).astype(bool)
        prev_bool = np.asarray(prev_mask).astype(bool)
        area = int(mask_bool.sum())
        prev_area = int(prev_bool.sum())
        total = mask_bool.size
        if area == 0:
            return False
        if area / float(total) > 0.40:
            return False
        if prev_area > 0 and area / float(prev_area) > 2.5:
            return False
        if prev_area > 0 and area / float(prev_area) < 0.25:
            return False

        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return False
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        img_h, img_w = mask_bool.shape
        if width > 0.95 * img_w and height > 0.95 * img_h:
            return False
        return True


class DiffusionPolicy:
    def __init__(
        self,
        ckpt_path,
        device='cuda',
        steering_pipeline=None,
        tts_frame_key=None,
        tts_side_frame_key=None,
        tts_frame_color_space='rgb',
        tts_mask_key=None,
        tts_auto_mask=False,
        tts_masker=None,
        tts_mask_mode='sam2',
        tts_steering_mode='scale',
        tts_sampling_candidates=1,
        tts_sampling_score_steps=4,
        tts_action_force_gain=1.0,
        tts_proxy_base_force='visualforce',
        tts_gripper_proxy_mode='absolute',
        tts_force_min=None,
        tts_force_max=None,
        tts_high_force_weight=1.0,
        tts_force_safety_limit=None,
        tts_unsafe_fallback='least_bad',
        tts_activation_force=0.0,
        tts_delta_force_critic=None,
        tts_delta_force_critic_weight=1.0,
        tts_policy_force_output=False,
        tts_auto_policy_force_output=False,
        tts_policy_force_aggregation='last',
        tts_policy_force_dynamic_target_rise=None,
        tts_force_target_mode='absolute',
        tts_selection_scope='full',
        tts_log_candidates=False,
        tts_absolute_gripper_safety=False,
        tts_gentle_gripper_control=False,
        tts_policy_release_enabled=True,
        tts_policy_release_contact_delta=None,
        tts_policy_gripper_approach=False,
        tts_add_gripper_fallback_candidates=False,
        tts_gripper_release_step=0.05,
        tts_gripper_safety_min_position=None,
        tts_contact_force_delta=5.0,
        tts_force_filter_window=3,
        tts_force_baseline_samples=3,
        tts_force_baseline_max_position=0.35,
        tts_gripper_close_step=0.05,
        tts_gripper_maintain_step=0.005,
        tts_gripper_max_position=0.82,
        tts_gripper_max_lead=0.25,
        tts_max_force_rate=20.0,
        policy_action_steps=None,
        relative_action_scale=1.0,
        relative_gripper_action_scale=1.0,
        gripper_command_min=None,
        gripper_command_max=None,
        tts_rollout_dir=None,
        tts_rollout_fps=10.0,
        tts_mask_h_min=GRIPPER_H_MIN,
        tts_mask_h_max=GRIPPER_H_MAX,
        tts_mask_s_min=GRIPPER_S_MIN,
        tts_mask_v_min=GRIPPER_V_MIN,
        tts_mask_min_area=GRIPPER_MIN_BLOB_AREA,
    ):
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        print(f'Loading checkpoint: {self.ckpt_path}')
        with open(self.ckpt_path, 'rb') as f:
            payload = torch.load(
                f,
                pickle_module=dill,
                map_location=device,
            )
        cfg = payload['cfg']
        self.checkpoint_has_force_output = bool(
            OmegaConf.select(
                cfg,
                'task.dataset.append_force_to_action',
                default=False,
            )
        )
        self.relative_position_action = bool(
            OmegaConf.select(
                cfg,
                'task.dataset.relative_position_action',
                default=False,
            )
        )
        self.relative_gripper_action = bool(
            OmegaConf.select(
                cfg,
                'task.dataset.relative_gripper_action',
                default=False,
            )
        )
        self.relative_position_action_mode = OmegaConf.select(
            cfg,
            'task.dataset.relative_position_action_mode',
            default='action_diff',
        )
        if self.relative_position_action or self.relative_gripper_action:
            print(
                'Relative action checkpoint detected '
                f'(xyz={self.relative_position_action}, '
                f'xyz_mode={self.relative_position_action_mode}, '
                f'gripper={self.relative_gripper_action}); '
                'converting to absolute targets.'
            )
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        if policy_action_steps is not None:
            requested_steps = int(policy_action_steps)
            if requested_steps <= 0:
                raise ValueError(
                    f'--policy-action-steps must be positive, got {requested_steps}'
                )
            if hasattr(policy, 'horizon') and hasattr(policy, 'n_obs_steps'):
                max_steps = int(policy.horizon - policy.n_obs_steps + 1)
                if requested_steps > max_steps:
                    print(
                        f'Requested policy action steps {requested_steps}, '
                        f'but horizon={policy.horizon} and '
                        f'n_obs_steps={policy.n_obs_steps} allow at most '
                        f'{max_steps}; using {max_steps}.'
                    )
                    requested_steps = max_steps
            if not hasattr(policy, 'n_action_steps'):
                raise AttributeError(
                    'Loaded policy does not expose n_action_steps; '
                    'cannot override action chunk length.'
                )
            policy.n_action_steps = requested_steps
            print(f'Policy action chunk length set to {requested_steps}.')
        policy.eval().to(device)

        self.policy = policy
        self.device = torch.device(device)
        self.obs_shape_meta = cfg.shape_meta['obs']
        self.warmed_up = False
        self.steering_pipeline = steering_pipeline
        self.tts_frame_key = tts_frame_key
        self.tts_side_frame_key = tts_side_frame_key
        if tts_frame_color_space not in ('rgb', 'bgr'):
            raise ValueError(
                "--tts-frame-color-space must be 'rgb' or 'bgr', "
                f"got {tts_frame_color_space!r}"
            )
        self.tts_frame_color_space = tts_frame_color_space
        self.tts_mask_key = tts_mask_key
        self.tts_auto_mask = tts_auto_mask
        self.tts_masker = tts_masker
        self.tts_mask_mode = tts_mask_mode
        self.tts_steering_mode = tts_steering_mode
        self.tts_sampling_candidates = max(1, int(tts_sampling_candidates))
        self.tts_sampling_score_steps = max(1, int(tts_sampling_score_steps))
        self.tts_action_force_gain = float(tts_action_force_gain)
        if tts_proxy_base_force not in ('visualforce', 'zero'):
            raise ValueError(
                "--tts-proxy-base-force must be 'visualforce' or 'zero'"
            )
        if tts_gripper_proxy_mode not in ('absolute', 'relative'):
            raise ValueError(
                "--tts-gripper-proxy-mode must be 'absolute' or 'relative'"
            )
        if (tts_force_min is None) != (tts_force_max is None):
            raise ValueError(
                '--tts-force-min and --tts-force-max must be provided together'
            )
        self.tts_proxy_base_force = tts_proxy_base_force
        self.tts_gripper_proxy_mode = tts_gripper_proxy_mode
        self.tts_force_min = None if tts_force_min is None else float(tts_force_min)
        self.tts_force_max = None if tts_force_max is None else float(tts_force_max)
        if (
            self.tts_force_min is not None
            and self.tts_force_min > self.tts_force_max
        ):
            raise ValueError('--tts-force-min must be <= --tts-force-max')
        self.tts_high_force_weight = float(tts_high_force_weight)
        if self.tts_high_force_weight <= 0.0:
            raise ValueError('--tts-high-force-weight must be positive')
        self.tts_force_safety_limit = (
            None
            if tts_force_safety_limit is None
            else float(tts_force_safety_limit)
        )
        if tts_unsafe_fallback not in ('least_bad', 'min_close'):
            raise ValueError(
                "--tts-unsafe-fallback must be 'least_bad' or 'min_close'"
            )
        self.tts_unsafe_fallback = tts_unsafe_fallback
        self.tts_activation_force = max(0.0, float(tts_activation_force))
        self.tts_delta_force_critic = tts_delta_force_critic
        self.tts_delta_force_critic_weight = float(tts_delta_force_critic_weight)
        self.tts_policy_force_output = bool(
            tts_policy_force_output
            or (
                tts_auto_policy_force_output
                and self.checkpoint_has_force_output
            )
        )
        self.tts_auto_policy_force_output = bool(tts_auto_policy_force_output)
        self.tts_policy_force_aggregation = str(tts_policy_force_aggregation)
        if self.tts_policy_force_aggregation not in ('last', 'mean', 'max'):
            raise ValueError(
                '--tts-policy-force-aggregation must be last, mean, or max'
            )
        self.tts_policy_force_dynamic_target_rise = (
            None
            if tts_policy_force_dynamic_target_rise is None
            else float(tts_policy_force_dynamic_target_rise)
        )
        if (
            self.tts_policy_force_dynamic_target_rise is not None
            and self.tts_policy_force_dynamic_target_rise < 0.0
        ):
            raise ValueError(
                '--tts-policy-force-dynamic-target-rise must be non-negative'
            )
        self.tts_selection_scope = str(tts_selection_scope)
        if self.tts_selection_scope not in ('full', 'gripper'):
            raise ValueError(
                '--tts-selection-scope must be full or gripper'
            )
        if self.tts_policy_force_output and not self.checkpoint_has_force_output:
            raise ValueError(
                '--tts-policy-force-output requires a checkpoint trained with '
                'task.dataset.append_force_to_action=true'
            )
        if (
            self.tts_policy_force_dynamic_target_rise is not None
            and not self.tts_policy_force_output
        ):
            raise ValueError(
                '--tts-policy-force-dynamic-target-rise requires '
                '--tts-policy-force-output'
            )
        if self.tts_policy_force_output and self.tts_delta_force_critic is not None:
            raise ValueError(
                'Choose either --tts-policy-force-output or '
                '--tts-delta-force-critic-ckpt, not both'
            )
        self.tts_force_target_mode = str(tts_force_target_mode)
        if self.tts_force_target_mode not in ('absolute', 'baseline_delta'):
            raise ValueError(
                '--tts-force-target-mode must be absolute or baseline_delta'
            )
        self.tts_force_target_baseline = None
        self.tts_log_candidates = bool(tts_log_candidates)
        self.tts_absolute_gripper_safety = bool(tts_absolute_gripper_safety)
        self.tts_gentle_gripper_control = bool(tts_gentle_gripper_control)
        self.tts_policy_release_enabled = bool(tts_policy_release_enabled)
        self.tts_add_gripper_fallback_candidates = bool(
            tts_add_gripper_fallback_candidates
        )
        self.tts_gripper_release_step = float(tts_gripper_release_step)
        self.tts_gripper_safety_min_position = (
            None
            if tts_gripper_safety_min_position is None
            else float(tts_gripper_safety_min_position)
        )
        if self.tts_gripper_release_step < 0.0:
            raise ValueError('--tts-gripper-release-step must be non-negative')
        if (
            self.tts_absolute_gripper_safety
            and self.tts_gentle_gripper_control
        ):
            raise ValueError(
                'Choose either --tts-absolute-gripper-safety or '
                '--tts-gentle-gripper-control, not both'
            )
        if (
            self.tts_absolute_gripper_safety
            or self.tts_gentle_gripper_control
        ) and self.steering_pipeline is None:
            raise ValueError(
                'Gripper force control requires VisualForce steering'
            )
        if (
            self.tts_absolute_gripper_safety
            and self.steering_pipeline.config.stop_margin
            < self.steering_pipeline.config.deadband
        ):
            raise ValueError(
                '--tts-stop-margin must be >= --tts-deadband for absolute '
                'gripper safety'
            )
        self.tts_gripper_safety_latched = False
        self._preempt_action_queue = False
        if self.tts_delta_force_critic is not None:
            print(
                'TTS sample reranking will use the delta-force critic '
                f'(weight={self.tts_delta_force_critic_weight:g}).'
            )
        if self.tts_policy_force_output:
            print(
                'TTS sample reranking will use the policy force output '
                f'(aggregation={self.tts_policy_force_aggregation}).'
            )
            if self.tts_policy_force_dynamic_target_rise is not None:
                print(
                    'TTS policy-force dynamic target enabled: '
                    'median sampled first-step force + '
                    f'{self.tts_policy_force_dynamic_target_rise:g} N, capped '
                    'by --tts-desired-force.'
                )
        elif self.tts_auto_policy_force_output:
            print(
                'Checkpoint has no learned force output; TTS will use its '
                'configured critic/proxy fallback.'
            )
        print(f'TTS force target mode: {self.tts_force_target_mode}.')
        if self.tts_log_candidates:
            print('TTS candidate score logging enabled.')
        if self.tts_absolute_gripper_safety:
            print(
                'Absolute gripper safety enabled '
                f'(release_step={self.tts_gripper_release_step:g}, '
                f'min_position={self.tts_gripper_safety_min_position}).'
            )
        if self.tts_add_gripper_fallback_candidates:
            print('TTS hold/open gripper fallback candidates enabled.')
        print(f'TTS candidate selection scope: {self.tts_selection_scope}.')
        print(
            'TTS sampling proxy configuration: '
            f'base={self.tts_proxy_base_force}, '
            f'gripper={self.tts_gripper_proxy_mode}, '
            f'force_band=({self.tts_force_min}, {self.tts_force_max}), '
            f'safety_limit={self.tts_force_safety_limit}, '
            f'unsafe_fallback={self.tts_unsafe_fallback}.'
        )
        if self.tts_activation_force > 0.0:
            print(
                'TTS activation gate enabled: steering only when '
                f'control force >= {self.tts_activation_force:.3f} N.'
            )
        self.relative_action_scale = float(relative_action_scale)
        if self.relative_position_action and self.relative_action_scale != 1.0:
            print(f'Relative-position xyz deltas scaled by {self.relative_action_scale:g}.')
        self.relative_gripper_action_scale = float(relative_gripper_action_scale)
        if self.relative_gripper_action_scale != 1.0:
            print(
                'Gripper command deltas from current observation scaled by '
                f'{self.relative_gripper_action_scale:g}.'
            )
        self.gripper_command_min = (
            None if gripper_command_min is None else float(gripper_command_min)
        )
        self.gripper_command_max = (
            None if gripper_command_max is None else float(gripper_command_max)
        )
        if (
            self.gripper_command_min is not None
            or self.gripper_command_max is not None
        ):
            print(
                'Gripper post-processing enabled '
                f'(min={self.gripper_command_min}, '
                f'max={self.gripper_command_max}).'
            )
        self.gentle_gripper_controller = None
        if self.tts_gentle_gripper_control:
            cfg = self.steering_pipeline.config
            self.gentle_gripper_controller = GentleGripperController(
                desired_force_delta=tts_contact_force_delta,
                deadband=cfg.deadband,
                stop_margin=cfg.stop_margin,
                close_step=tts_gripper_close_step,
                maintain_step=tts_gripper_maintain_step,
                release_step=self.tts_gripper_release_step,
                filter_window=tts_force_filter_window,
                baseline_min_samples=tts_force_baseline_samples,
                baseline_max_position=tts_force_baseline_max_position,
                min_closure_position=(
                    tts_force_baseline_max_position
                    if tts_gripper_safety_min_position is None
                    else tts_gripper_safety_min_position
                ),
                max_closure_position=tts_gripper_max_position,
                max_command_lead=tts_gripper_max_lead,
                max_force_rate=tts_max_force_rate,
                policy_release_enabled=self.tts_policy_release_enabled,
                policy_release_contact_delta=tts_policy_release_contact_delta,
                policy_gripper_approach=tts_policy_gripper_approach,
                gripper_index=cfg.gripper_index,
                close_positive=cfg.close_positive,
                command_min=self.gripper_command_min,
                command_max=self.gripper_command_max,
            )
            print(
                'Direct VisualForce gripper control enabled '
                f'(contact rise={float(tts_contact_force_delta):g} N, '
                f'baseline_samples={int(tts_force_baseline_samples)}, '
                'policy_release_contact_rise='
                f'{float(tts_policy_release_contact_delta if tts_policy_release_contact_delta is not None else tts_contact_force_delta):g} N, '
                f'policy_gripper_approach={bool(tts_policy_gripper_approach)}, '
                f'close_step={float(tts_gripper_close_step):g}, '
                f'open_step={self.tts_gripper_release_step:g}, '
                f'max_lead={float(tts_gripper_max_lead):g}, '
                f'max_position={float(tts_gripper_max_position):g}).'
            )
        self.requires_fresh_observation = self.gentle_gripper_controller is not None
        self.tts_rollout_dir = Path(tts_rollout_dir) if tts_rollout_dir else None
        self.tts_rollout_fps = tts_rollout_fps
        self.tts_rollout = None
        self.tts_rollout_frames = 0
        self.tts_rollout_lock = threading.Lock()
        self.tts_mask_h_min = tts_mask_h_min
        self.tts_mask_h_max = tts_mask_h_max
        self.tts_mask_s_min = tts_mask_s_min
        self.tts_mask_v_min = tts_mask_v_min
        self.tts_mask_min_area = tts_mask_min_area

    def reset(self):
        self.flush_tts_rollout()
        self.tts_gripper_safety_latched = False
        self._preempt_action_queue = False
        self.tts_force_target_baseline = None
        if self.gentle_gripper_controller is not None:
            self.gentle_gripper_controller.reset()
        if self.tts_masker is not None and hasattr(self.tts_masker, 'reset'):
            self.tts_masker.reset()
        self.policy.reset()

    def consume_action_queue_preempt(self):
        preempt = self._preempt_action_queue
        self._preempt_action_queue = False
        return preempt

    def step(self, obs_sequence):
        """obs_sequence: list of obs dicts, length == n_obs_steps"""
        obs_dict = self._build_obs(obs_sequence)
        with torch.no_grad():
            if not self.warmed_up:
                print('Warming up...')
                self.policy.predict_action(obs_dict)
                self.warmed_up = True
        if self.steering_pipeline is not None:
            if self.tts_steering_mode == 'sample':
                actions = self._sample_steer_actions(obs_sequence[-1], obs_dict)
            else:
                with torch.no_grad():
                    result = self.policy.predict_action(obs_dict)
                # (1, horizon, 8) -> (horizon, 8)
                actions = result['action'][0].cpu().numpy()
                actions = self._actions_to_absolute(obs_sequence[-1], actions)
                actions = self._postprocess_gripper_actions(obs_sequence[-1], actions)
                if self.tts_steering_mode == 'monitor':
                    actions = self._monitor_actions(obs_sequence[-1], actions)
                else:
                    actions = self._steer_actions(obs_sequence[-1], actions)
        else:
            with torch.no_grad():
                result = self.policy.predict_action(obs_dict)
            # (1, horizon, 8) -> (horizon, 8)
            actions = result['action'][0].cpu().numpy()
            actions = self._actions_to_absolute(obs_sequence[-1], actions)
            actions = self._postprocess_gripper_actions(obs_sequence[-1], actions)
        return self._split_action(actions)

    def _actions_to_absolute(self, latest_obs, actions):
        if not self.relative_position_action and not self.relative_gripper_action:
            return actions

        actions = np.asarray(actions, dtype=np.float32).copy()
        if self.relative_position_action and 'arm_pos' not in latest_obs:
            raise KeyError(
                'Relative-position action checkpoint needs arm_pos in latest obs.'
            )
        if actions.ndim not in (2, 3):
            raise ValueError(
                f'Expected action array with 2 or 3 dims, got {actions.shape}'
            )
        if self.relative_position_action:
            origin = np.asarray(latest_obs['arm_pos'], dtype=np.float32)
            actions[..., :3] *= self.relative_action_scale
            if self.relative_position_action_mode == 'action_diff':
                if actions.ndim == 2:
                    actions[:, :3] = origin[None, :] + np.cumsum(actions[:, :3], axis=0)
                else:
                    actions[:, :, :3] = (
                        origin[None, None, :]
                        + np.cumsum(actions[:, :, :3], axis=1)
                    )
            elif self.relative_position_action_mode in ('obs_delta', 'obs_anchor'):
                if actions.ndim == 2:
                    actions[:, :3] = origin[None, :] + actions[:, :3]
                else:
                    actions[:, :, :3] = origin[None, None, :] + actions[:, :, :3]
            else:
                raise ValueError(
                    'Unsupported relative_position_action_mode='
                    f'{self.relative_position_action_mode!r}'
                )
        if self.relative_gripper_action:
            if 'gripper_pos' not in latest_obs:
                raise KeyError(
                    'Relative-gripper action checkpoint needs gripper_pos in latest obs.'
                )
            gripper_origin = float(
                np.asarray(latest_obs['gripper_pos'], dtype=np.float32).reshape(-1)[0]
            )
            if actions.ndim == 2:
                actions[:, 7] = gripper_origin + actions[:, 7]
            else:
                actions[:, :, 7] = gripper_origin + actions[:, :, 7]
        return actions

    def _scale_gripper_delta(self, latest_obs, actions):
        """Optionally shrink commanded gripper movement around current gripper."""
        if self.relative_gripper_action_scale == 1.0:
            return actions
        if 'gripper_pos' not in latest_obs:
            raise KeyError(
                'Gripper delta scaling needs gripper_pos in latest obs.'
            )
        actions = np.asarray(actions, dtype=np.float32).copy()
        if actions.ndim not in (2, 3) or actions.shape[-1] < 8:
            raise ValueError(
                f'Expected action array with shape (..., 8), got {actions.shape}'
            )
        origin = float(np.asarray(latest_obs['gripper_pos'], dtype=np.float32).reshape(-1)[0])
        actions[..., 7] = origin + self.relative_gripper_action_scale * (
            actions[..., 7] - origin
        )
        return actions

    def _postprocess_gripper_actions(self, latest_obs, actions):
        actions = self._scale_gripper_delta(latest_obs, actions)
        needs_postprocess = (
            self.gripper_command_min is not None
            or self.gripper_command_max is not None
        )
        if not needs_postprocess:
            return actions
        actions = np.asarray(actions, dtype=np.float32).copy()
        if self.gripper_command_min is not None or self.gripper_command_max is not None:
            lo = -np.inf if self.gripper_command_min is None else self.gripper_command_min
            hi = np.inf if self.gripper_command_max is None else self.gripper_command_max
            actions[..., 7] = np.clip(actions[..., 7], lo, hi)
        return actions

    def _get_tts_frame_mask(self, latest_obs):
        frame_key = self.tts_frame_key
        if frame_key is None:
            image_keys = [
                key for key, meta in self.obs_shape_meta.items()
                if meta.get('type') == 'rgb'
            ]
            if len(image_keys) != 1:
                raise ValueError(
                    'TTS needs --tts-frame-key because the policy has '
                    f'{len(image_keys)} rgb observation keys: {image_keys}'
                )
            frame_key = image_keys[0]

        if frame_key not in latest_obs:
            raise KeyError(
                f'TTS frame key {frame_key!r} not found in obs. '
                f'Available obs keys: {sorted(latest_obs.keys())}'
            )
        frame = self._obs_image_to_rgb(latest_obs[frame_key])
        if self.tts_mask_key is not None and self.tts_mask_key in latest_obs:
            mask = latest_obs[self.tts_mask_key]
        elif self.tts_auto_mask:
            mask = self._make_gripper_mask(frame)
        elif self.tts_mask_key is None:
            raise ValueError(
                'TTS is enabled but no mask source was provided. Pass '
                '--tts-mask-key KEY or --tts-auto-mask.'
            )
        else:
            raise KeyError(
                f'TTS mask key {self.tts_mask_key!r} not found in obs. '
                f'Available obs keys: {sorted(latest_obs.keys())}'
            )
        return frame, mask

    def _obs_image_to_rgb(self, frame):
        frame = np.asarray(frame)
        if self.tts_frame_color_space == 'bgr':
            return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame

    def _obs_image_to_video_bgr(self, frame):
        frame = np.asarray(frame)
        if self.tts_frame_color_space == 'bgr':
            return frame
        return cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    def _get_tts_stats(self, frame, mask):
        mask_area = int(np.asarray(mask).astype(bool).sum())
        mask_frac = mask_area / float(mask.shape[0] * mask.shape[1])
        frame_mean = float(np.asarray(frame).mean())
        hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV).astype(np.float32)
        sat_mean = float((hsv[:, :, 1] / 255.0).mean())
        val_mean = float((hsv[:, :, 2] / 255.0).mean())
        return mask_area, mask_frac, frame_mean, sat_mean, val_mean

    def _steer_actions(self, latest_obs, actions):
        frame, mask = self._get_tts_frame_mask(latest_obs)
        mask_area, mask_frac, frame_mean, sat_mean, val_mean = self._get_tts_stats(frame, mask)
        rollout_frame_idx = self._record_tts_rollout_frame(
            frame, mask, latest_obs
        )
        if self.tts_activation_force > 0.0:
            cfg = self.steering_pipeline.config
            pred = self.steering_pipeline.estimator.predict(
                frame,
                mask,
                force_key=cfg.force_key,
                force_mode=cfg.force_mode,
            )
            if pred.control_force < self.tts_activation_force:
                metadata = {
                    'desired_force': cfg.desired_force,
                    'deadband': cfg.deadband,
                    'over_target': '',
                    'under_target': '',
                    'stopped_or_opened': False,
                    'gripper_index': cfg.gripper_index,
                    'motion_indices': list(cfg.motion_indices),
                    'sampling_mode': 'inactive',
                    'activation_force': self.tts_activation_force,
                }
                inactive_result = self._make_sample_result(
                    actions,
                    pred,
                    force_error=float(cfg.desired_force - pred.control_force),
                    metadata=metadata,
                )
                self._record_tts_force_row(
                    rollout_frame_idx,
                    inactive_result,
                    mask_area=mask_area,
                    mask_frac=mask_frac,
                    frame_mean=frame_mean,
                    sat_mean=sat_mean,
                    val_mean=val_mean,
                    latest_obs=latest_obs,
                )
                print(
                    'TTS inactive '
                    f'{pred.selected_key}={pred.selected_force:.3f}, '
                    f'control={pred.control_force:.3f} < '
                    f'activation={self.tts_activation_force:.3f}; '
                    'returning DP action'
                )
                return np.asarray(actions, dtype=np.float32)

        result = self.steering_pipeline.steer_action_chunk(
            frame,
            mask,
            actions,
        )
        self._record_tts_force_row(
            rollout_frame_idx,
            result,
            mask_area=mask_area,
            mask_frac=mask_frac,
            frame_mean=frame_mean,
            sat_mean=sat_mean,
            val_mean=val_mean,
            latest_obs=latest_obs,
        )
        print(
            'TTS force '
            f'{result.predicted_force.selected_key}='
            f'{result.predicted_force.selected_force:.3f}, '
            f'control={result.predicted_force.control_force:.3f}, '
            f'err={result.force_error:.3f}, '
            f'close_scale={result.close_scale:.3f}, '
            f'motion_scale={result.motion_scale:.3f}, '
            f'mask_px={mask_area}, '
            f'mask_frac={mask_frac:.4f}, '
            f'frame_mean={frame_mean:.1f}, '
            f'sat_mean={sat_mean:.3f}, '
            f'val_mean={val_mean:.3f}'
        )
        return result.action

    def _monitor_actions(self, latest_obs, actions):
        """Estimate and log force without modifying the DP action chunk."""
        frame, mask = self._get_tts_frame_mask(latest_obs)
        mask_area, mask_frac, frame_mean, sat_mean, val_mean = self._get_tts_stats(
            frame, mask
        )
        rollout_frame_idx = self._record_tts_rollout_frame(
            frame, mask, latest_obs
        )
        cfg = self.steering_pipeline.config
        pred = self.steering_pipeline.estimator.predict(
            frame,
            mask,
            force_key=cfg.force_key,
            force_mode=cfg.force_mode,
        )
        metadata = {
            'desired_force': cfg.desired_force,
            'deadband': cfg.deadband,
            'over_target': '',
            'under_target': '',
            'stopped_or_opened': False,
            'gripper_index': cfg.gripper_index,
            'motion_indices': [],
            'sampling_mode': 'monitor',
        }
        monitor_result = self._make_sample_result(
            actions,
            pred,
            force_error=float(cfg.desired_force - pred.control_force),
            metadata=metadata,
        )
        self._record_tts_force_row(
            rollout_frame_idx,
            monitor_result,
            mask_area=mask_area,
            mask_frac=mask_frac,
            frame_mean=frame_mean,
            sat_mean=sat_mean,
            val_mean=val_mean,
            latest_obs=latest_obs,
        )
        print(
            'DP monitor force '
            f'{pred.selected_key}={pred.selected_force:.3f}, '
            f'control={pred.control_force:.3f}, '
            f'mask_px={mask_area}, '
            f'mask_frac={mask_frac:.4f}'
        )
        return np.asarray(actions, dtype=np.float32)

    def _empty_gripper_safety_metadata(self, current_gripper=''):
        return {
            'gripper_safety_active': False,
            'gripper_safety_mode': 'disabled',
            'gripper_safety_current': current_gripper,
            'gripper_safety_command': '',
            'gripper_safety_enter_force': '',
            'gripper_safety_release_force': '',
            'gripper_safety_release_step': '',
            'gripper_safety_closure_ready': '',
            'gripper_safety_closure_position': '',
            'gripper_safety_min_closure_position': '',
            'preempt_action_queue': False,
            'gentle_controller': False,
        }

    def _apply_sample_gripper_control(
        self,
        selected,
        current_gripper,
        predicted_force,
        measurement_valid=True,
        policy_actions=None,
        desired_force=None,
    ):
        cfg = self.steering_pipeline.config
        if desired_force is None:
            desired_force = cfg.desired_force
        metadata = self._empty_gripper_safety_metadata(current_gripper)
        if self.gentle_gripper_controller is not None:
            selected, metadata = self.gentle_gripper_controller.apply(
                selected,
                current_gripper=current_gripper,
                raw_force=predicted_force.control_force,
                measurement_valid=measurement_valid,
                policy_actions=policy_actions,
            )
        elif self.tts_absolute_gripper_safety:
            selected, self.tts_gripper_safety_latched, metadata = (
                _apply_absolute_gripper_safety(
                    selected,
                    current_gripper=current_gripper,
                    control_force=predicted_force.control_force,
                    desired_force=desired_force,
                    deadband=cfg.deadband,
                    stop_margin=cfg.stop_margin,
                    release_step=self.tts_gripper_release_step,
                    gripper_index=cfg.gripper_index,
                    close_positive=cfg.close_positive,
                    command_min=self.gripper_command_min,
                    command_max=self.gripper_command_max,
                    min_closure_position=self.tts_gripper_safety_min_position,
                    latched=self.tts_gripper_safety_latched,
                )
            )
        if metadata['preempt_action_queue']:
            self._preempt_action_queue = True
        return selected, metadata

    def _sample_steer_actions(self, latest_obs, obs_dict):
        frame, mask = self._get_tts_frame_mask(latest_obs)
        mask_area, mask_frac, frame_mean, sat_mean, val_mean = self._get_tts_stats(frame, mask)
        measurement_valid = 0.01 <= mask_frac <= 0.40
        if self.tts_masker is not None and self.tts_masker.reject_count > 0:
            measurement_valid = False
        rollout_frame_idx = self._record_tts_rollout_frame(
            frame, mask, latest_obs
        )

        pred = self.steering_pipeline.estimator.predict(
            frame,
            mask,
            force_key=self.steering_pipeline.config.force_key,
            force_mode=self.steering_pipeline.config.force_mode,
        )
        cfg = self.steering_pipeline.config
        controller_baseline = None
        if self.gentle_gripper_controller is not None:
            controller_baseline = self.gentle_gripper_controller.baseline_force
        if (
            self.tts_force_target_mode == 'baseline_delta'
            and controller_baseline is None
            and self.tts_force_target_baseline is None
        ):
            # Task adapters must start baseline-relative TTS from their known
            # non-contact state. Latch that episode baseline once so estimator
            # offset cannot turn a small force rise into an opening command.
            self.tts_force_target_baseline = float(pred.control_force)
        target_force, target_baseline = _resolve_tts_force_target(
            self.tts_force_target_mode,
            cfg.desired_force,
            pred.control_force,
            baseline_force=(
                controller_baseline
                if controller_baseline is not None
                else self.tts_force_target_baseline
            ),
        )

        with torch.no_grad():
            batched_obs = dict_apply(
                obs_dict,
                lambda x: x.repeat_interleave(self.tts_sampling_candidates, dim=0),
            )
            result = self.policy.predict_action(batched_obs)
            raw_candidates = result['action'].detach().cpu().numpy()
            steps = min(self.tts_sampling_score_steps, raw_candidates.shape[1])
            policy_force_proxies = None
            policy_dynamic_target_base = None
            if self.tts_policy_force_output:
                if raw_candidates.shape[-1] != 9:
                    raise ValueError(
                        'Force-output TTS expects policy actions with 9 dimensions '
                        f'(8 robot + force), got {raw_candidates.shape}'
                    )
                policy_force_proxies = _policy_force_proxies(
                    raw_candidates[..., 8],
                    steps,
                    self.tts_policy_force_aggregation,
                )
                if self.tts_policy_force_dynamic_target_rise is not None:
                    target_force, policy_dynamic_target_base = (
                        _resolve_policy_force_dynamic_target(
                            raw_candidates[..., 8],
                            rise=self.tts_policy_force_dynamic_target_rise,
                            cap=cfg.desired_force,
                        )
                    )
                candidates = raw_candidates[..., :8]
            else:
                candidates = raw_candidates
            candidates = self._actions_to_absolute(latest_obs, candidates)
            candidates = self._postprocess_gripper_actions(latest_obs, candidates)

        gi = cfg.gripper_index
        current_gripper = 0.0
        if gi is not None:
            if 'gripper_pos' not in latest_obs:
                raise KeyError('TTS gripper control needs gripper_pos in latest obs.')
            current_gripper = float(
                np.asarray(latest_obs['gripper_pos']).reshape(-1)[0]
            )
        if self.tts_add_gripper_fallback_candidates:
            candidates = _append_gripper_fallback_candidates(
                candidates,
                current_gripper=current_gripper,
                release_step=self.tts_gripper_release_step,
                gripper_index=gi,
                close_positive=cfg.close_positive,
                command_min=self.gripper_command_min,
                command_max=self.gripper_command_max,
            )
            if policy_force_proxies is not None:
                policy_force_proxies = np.concatenate([
                    np.asarray(policy_force_proxies, dtype=np.float32),
                    np.asarray(
                        [pred.control_force, pred.control_force],
                        dtype=np.float32,
                    ),
                ])

        if (
            self.tts_activation_force > 0.0
            and pred.control_force < self.tts_activation_force
        ):
            selected = candidates[0]
            selected, gripper_safety_metadata = self._apply_sample_gripper_control(
                selected,
                current_gripper,
                pred,
                measurement_valid=measurement_valid,
                policy_actions=candidates[0],
                desired_force=target_force,
            )
            metadata = {
                'desired_force': target_force,
                'requested_force': cfg.desired_force,
                'force_target_mode': self.tts_force_target_mode,
                'force_target_baseline': target_baseline,
                'deadband': cfg.deadband,
                'over_target': '',
                'under_target': '',
                'stopped_or_opened': gripper_safety_metadata['gripper_safety_active'],
                'gripper_index': cfg.gripper_index,
                'motion_indices': list(cfg.motion_indices),
                'sampling_mode': 'inactive',
                'sampling_selection_scope': self.tts_selection_scope,
                'sampling_candidates': len(candidates),
                'sampling_best_idx': 0,
                'sampling_best_score': '',
                'sampling_best_proxy_force': '',
                'sampling_score_steps': '',
                'sampling_action_force_gain': self.tts_action_force_gain,
                'activation_force': self.tts_activation_force,
                **gripper_safety_metadata,
            }
            inactive_result = self._make_sample_result(
                selected,
                pred,
                force_error=float(target_force - pred.control_force),
                metadata=metadata,
            )
            self._record_tts_force_row(
                rollout_frame_idx,
                inactive_result,
                mask_area=mask_area,
                mask_frac=mask_frac,
                frame_mean=frame_mean,
                sat_mean=sat_mean,
                val_mean=val_mean,
                latest_obs=latest_obs,
            )
            print(
                'TTS sample inactive '
                f'{pred.selected_key}={pred.selected_force:.3f}, '
                f'control={pred.control_force:.3f} < '
                f'activation={self.tts_activation_force:.3f}; '
                'returning DP sample '
                f'(gripper_safety={gripper_safety_metadata["gripper_safety_mode"]})'
            )
            return selected

        scores = []
        proxy_forces = []
        critic_deltas = None
        proxy_source = 'closing_proxy'
        closing_sign = 1.0 if cfg.close_positive else -1.0
        closing_means = []
        signed_closure_means = []
        for action in candidates:
            closing_mean, signed_mean = _gripper_proxy_features(
                action,
                gi,
                steps,
                closing_sign,
                current_gripper,
                self.tts_gripper_proxy_mode,
            )
            closing_means.append(closing_mean)
            signed_closure_means.append(signed_mean)

        base_force = (
            pred.control_force
            if self.tts_proxy_base_force == 'visualforce'
            else 0.0
        )
        if policy_force_proxies is not None:
            proxy_source = (
                f'policy_force_output_{self.tts_policy_force_aggregation}'
            )
            for predicted_force in policy_force_proxies:
                proxy_force = float(predicted_force)
                score = _force_target_score(
                    proxy_force,
                    target_force,
                    self.tts_force_min,
                    self.tts_force_max,
                    self.tts_high_force_weight,
                )
                scores.append(score)
                proxy_forces.append(proxy_force)
        elif self.tts_delta_force_critic is not None:
            critic_deltas = self.tts_delta_force_critic.predict_delta(
                obs_dict,
                latest_obs,
                candidates,
                frame_key=self.tts_frame_key,
                mask=mask,
                current_force=pred.control_force,
            )
            proxy_source = 'delta_force_critic'
            if self.tts_delta_force_critic.predicts_future_peak:
                proxy_source = 'future_peak_critic'
            for critic_value in critic_deltas:
                if self.tts_delta_force_critic.predicts_future_peak:
                    proxy_force = float(
                        pred.control_force
                        + self.tts_delta_force_critic_weight
                        * (float(critic_value) - pred.control_force)
                    )
                else:
                    proxy_force = float(
                        base_force
                        + self.tts_delta_force_critic_weight * float(critic_value)
                    )
                score = _force_target_score(
                    proxy_force,
                    target_force,
                    self.tts_force_min,
                    self.tts_force_max,
                    self.tts_high_force_weight,
                )
                scores.append(score)
                proxy_forces.append(proxy_force)
        else:
            proxy_source = (
                f'{self.tts_gripper_proxy_mode}_closing_proxy_'
                f'{self.tts_proxy_base_force}'
            )
            for closing_mean in closing_means:
                proxy_force = float(
                    base_force + self.tts_action_force_gain * closing_mean
                )
                score = _force_target_score(
                    proxy_force,
                    target_force,
                    self.tts_force_min,
                    self.tts_force_max,
                    self.tts_high_force_weight,
                )
                scores.append(score)
                proxy_forces.append(proxy_force)

        best_idx, safe_mask, all_unsafe = _select_force_candidate(
            scores,
            proxy_forces,
            safety_limit=self.tts_force_safety_limit,
            unsafe_fallback=self.tts_unsafe_fallback,
            signed_closure_means=signed_closure_means,
        )
        nominal = candidates[0]
        selected = _compose_selected_action(
            candidates,
            best_idx,
            selection_scope=self.tts_selection_scope,
            gripper_index=gi,
        )
        selected, gripper_safety_metadata = self._apply_sample_gripper_control(
            selected,
            current_gripper,
            pred,
            measurement_valid=measurement_valid,
            policy_actions=nominal,
            desired_force=target_force,
        )
        best_critic_delta = (
            float(critic_deltas[best_idx]) if critic_deltas is not None else ''
        )
        self._record_tts_candidate_rows(
            rollout_frame_idx,
            candidates,
            scores,
            proxy_forces,
            critic_deltas,
            best_idx,
            pred.control_force,
            target_force,
            latest_obs,
            steps,
            proxy_source,
            policy_dynamic_target_base,
            closing_means,
            signed_closure_means,
            safe_mask,
            all_unsafe,
        )

        metadata = {
            'desired_force': target_force,
            'requested_force': cfg.desired_force,
            'force_target_mode': self.tts_force_target_mode,
            'force_target_baseline': target_baseline,
            'deadband': cfg.deadband,
            'over_target': max(0.0, pred.control_force - target_force - cfg.deadband),
            'under_target': max(0.0, target_force - pred.control_force - cfg.deadband),
            'stopped_or_opened': gripper_safety_metadata['gripper_safety_active'],
            'gripper_index': cfg.gripper_index,
            'motion_indices': list(cfg.motion_indices),
            'sampling_mode': 'sample',
            'sampling_candidates': len(candidates),
            'sampling_best_idx': best_idx,
            'sampling_best_score': scores[best_idx],
            'sampling_best_proxy_force': proxy_forces[best_idx],
            'sampling_selection_scope': self.tts_selection_scope,
            'sampling_proxy_source': proxy_source,
            'sampling_critic_delta_force': best_critic_delta,
            'sampling_score_steps': steps,
            'sampling_action_force_gain': self.tts_action_force_gain,
            'sampling_policy_dynamic_target_base': policy_dynamic_target_base,
            'sampling_policy_dynamic_target_rise': (
                self.tts_policy_force_dynamic_target_rise
            ),
            'sampling_policy_dynamic_target_cap': (
                cfg.desired_force
                if self.tts_policy_force_dynamic_target_rise is not None
                else None
            ),
            'sampling_proxy_base_force': base_force,
            'sampling_proxy_base_mode': self.tts_proxy_base_force,
            'sampling_gripper_proxy_mode': self.tts_gripper_proxy_mode,
            'sampling_force_min': self.tts_force_min,
            'sampling_force_max': self.tts_force_max,
            'sampling_force_safety_limit': self.tts_force_safety_limit,
            'sampling_all_unsafe': all_unsafe,
            'sampling_unsafe_fallback': self.tts_unsafe_fallback,
            **gripper_safety_metadata,
        }
        sample_result = self._make_sample_result(
            selected,
            pred,
            force_error=float(target_force - pred.control_force),
            metadata=metadata,
            base_action=nominal,
        )
        self._record_tts_force_row(
            rollout_frame_idx,
            sample_result,
            mask_area=mask_area,
            mask_frac=mask_frac,
            frame_mean=frame_mean,
            sat_mean=sat_mean,
            val_mean=val_mean,
            latest_obs=latest_obs,
        )

        delta_msg = (
            f', delta={best_critic_delta:.3f}'
            if critic_deltas is not None
            else ''
        )
        print(
            'TTS sample '
            f'{pred.selected_key}={pred.selected_force:.3f}, '
            f'control={pred.control_force:.3f}, '
            f'best={best_idx}/{len(candidates)}, '
            f'scope={self.tts_selection_scope}, '
            f'source={proxy_source}, '
            f'proxy={proxy_forces[best_idx]:.3f}'
            f'{delta_msg}, '
            f'score={scores[best_idx]:.4f}, '
            f'all_unsafe={all_unsafe}, '
            f'gripper_safety={gripper_safety_metadata["gripper_safety_mode"]}, '
            f'mask_px={mask_area}, '
            f'mask_frac={mask_frac:.4f}'
        )
        return selected

    def _make_sample_result(
        self,
        action,
        predicted_force,
        force_error,
        metadata,
        base_action=None,
    ):
        class Result:
            pass

        result = Result()
        result.action = np.asarray(action, dtype=np.float32)
        if base_action is None:
            base_action = action
        result.base_action = np.asarray(base_action, dtype=np.float32)
        result.predicted_force = predicted_force
        result.force_error = float(force_error)
        result.close_scale = 1.0
        result.motion_scale = 1.0
        result.metadata = metadata
        return result

    def _make_gripper_mask(self, frame):
        if self.tts_mask_mode == 'sam2':
            if self.tts_masker is None:
                raise ValueError('TTS SAM2 mask mode requested but masker was not initialized')
            return self.tts_masker.predict(frame)

        return make_hsv_gripper_mask(
            frame,
            h_min=self.tts_mask_h_min,
            h_max=self.tts_mask_h_max,
            s_min=self.tts_mask_s_min,
            v_min=self.tts_mask_v_min,
            min_blob_area=self.tts_mask_min_area,
        )

    def _record_tts_rollout_frame(self, frame, mask, latest_obs=None):
        if self.tts_rollout_dir is None:
            return

        frame = np.asarray(frame)
        side_frame = None
        if (
            latest_obs is not None
            and self.tts_side_frame_key
            and self.tts_side_frame_key in latest_obs
        ):
            candidate = np.asarray(latest_obs[self.tts_side_frame_key])
            if candidate.ndim == 3 and candidate.shape[2] == 3:
                side_frame = candidate
        mask_bool = np.asarray(mask).astype(bool)
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        if mask_bool.shape != frame.shape[:2]:
            mask_bool = cv.resize(
                mask_bool.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv.INTER_NEAREST,
            ).astype(bool)

        with self.tts_rollout_lock:
            if self.tts_rollout_dir is None:
                return
            if self.tts_rollout is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_dir = self.tts_rollout_dir / f'tts_rollout_{timestamp}'
                out_dir.mkdir(parents=True, exist_ok=True)
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                size = (frame.shape[1], frame.shape[0])
                self.tts_rollout = {
                    'dir': out_dir,
                    'original': cv.VideoWriter(
                        str(out_dir / '.original_recording.mp4'),
                        fourcc,
                        self.tts_rollout_fps,
                        size,
                    ),
                    'masked_original': cv.VideoWriter(
                        str(out_dir / '.masked_original_recording.mp4'),
                        fourcc,
                        self.tts_rollout_fps,
                        (VISUALFORCE_INPUT_SIZE[1], VISUALFORCE_INPUT_SIZE[0]),
                    ),
                    'edge': cv.VideoWriter(
                        str(out_dir / '.edge_recording.mp4'),
                        fourcc,
                        self.tts_rollout_fps,
                        (VISUALFORCE_INPUT_SIZE[1], VISUALFORCE_INPUT_SIZE[0]),
                    ),
                }
                if side_frame is not None:
                    side_size = (side_frame.shape[1], side_frame.shape[0])
                    self.tts_rollout['side_view'] = cv.VideoWriter(
                        str(out_dir / '.side_view_recording.mp4'),
                        fourcc,
                        self.tts_rollout_fps,
                        side_size,
                    )
                    self.tts_rollout['side_view_size'] = side_size
                self.tts_rollout_frames = 0
                print(f'TTS rollout recording: {out_dir}')

            model_input = self._make_visualforce_input_preview(frame, mask_bool)
            edge_input = self._make_visualforce_edge_preview(frame, mask_bool)

            self.tts_rollout['original'].write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            self.tts_rollout['masked_original'].write(
                cv.cvtColor(model_input, cv.COLOR_RGB2BGR)
            )
            self.tts_rollout['edge'].write(cv.cvtColor(edge_input, cv.COLOR_RGB2BGR))
            if side_frame is not None and 'side_view' in self.tts_rollout:
                side_size = self.tts_rollout['side_view_size']
                if (side_frame.shape[1], side_frame.shape[0]) != side_size:
                    side_frame = cv.resize(
                        side_frame, side_size, interpolation=cv.INTER_AREA
                    )
                self.tts_rollout['side_view'].write(
                    self._obs_image_to_video_bgr(side_frame)
                )
            frame_idx = self.tts_rollout_frames
            self.tts_rollout_frames += 1
            return frame_idx

    def _record_tts_candidate_rows(
        self,
        frame_idx,
        candidates,
        scores,
        proxy_forces,
        critic_deltas,
        best_idx,
        current_force,
        desired_force,
        latest_obs,
        score_steps,
        proxy_source,
        policy_dynamic_target_base,
        closing_means,
        signed_closure_means,
        safe_mask,
        all_unsafe,
    ):
        if not self.tts_log_candidates:
            return
        if self.tts_rollout_dir is None or frame_idx is None:
            return

        with self.tts_rollout_lock:
            if self.tts_rollout is None:
                return
            csv_path = self.tts_rollout['dir'] / 'candidate_scores.csv'
            exists = csv_path.is_file()
            obs_gripper = ''
            if latest_obs is not None and 'gripper_pos' in latest_obs:
                obs_gripper = float(np.asarray(latest_obs['gripper_pos'])[0])
            obs_xyz = ['', '', '']
            if latest_obs is not None and 'arm_pos' in latest_obs:
                obs_xyz = [float(v) for v in np.asarray(latest_obs['arm_pos'])[:3]]

            fieldnames = [
                'frame_idx',
                'candidate_idx',
                'candidate_kind',
                'selected',
                'selection_scope',
                'arm_source',
                'gripper_source',
                'score',
                'proxy_force_n',
                'critic_delta_force_n',
                'current_force_n',
                'desired_force_n',
                'requested_force_n',
                'force_target_mode',
                'force_target_baseline_n',
                'proxy_source',
                'policy_dynamic_target_base_n',
                'policy_dynamic_target_rise_n',
                'policy_dynamic_target_cap_n',
                'proxy_base_mode',
                'gripper_proxy_mode',
                'safe_candidate',
                'all_candidates_unsafe',
                'force_safety_limit_n',
                'score_steps',
                'obs_x',
                'obs_y',
                'obs_z',
                'obs_gripper_cmd',
                'first_x',
                'first_y',
                'first_z',
                'first_gripper_cmd',
                'first_gripper_delta',
                'mean_positive_closure',
                'mean_signed_closure',
                'mean_gripper_cmd',
                'last_gripper_cmd',
                'min_gripper_cmd',
                'max_gripper_cmd',
            ]
            with csv_path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                for i, action in enumerate(np.asarray(candidates)):
                    gripper = action[:, 7] if action.ndim == 2 and action.shape[1] > 7 else np.asarray([])
                    first_gripper = float(gripper[0]) if gripper.size else ''
                    first_delta = (
                        float(first_gripper - obs_gripper)
                        if gripper.size and obs_gripper != ''
                        else ''
                    )
                    row = {
                        'frame_idx': frame_idx,
                        'candidate_idx': i,
                        'candidate_kind': (
                            'policy'
                            if i < self.tts_sampling_candidates
                            else (
                                'hold'
                                if i == self.tts_sampling_candidates
                                else 'open'
                            )
                        ),
                        'selected': int(i == best_idx),
                        'selection_scope': self.tts_selection_scope,
                        'arm_source': int(
                            i == (
                                0
                                if self.tts_selection_scope == 'gripper'
                                else best_idx
                            )
                        ),
                        'gripper_source': int(i == best_idx),
                        'score': float(scores[i]),
                        'proxy_force_n': float(proxy_forces[i]),
                        'critic_delta_force_n': (
                            float(critic_deltas[i])
                            if critic_deltas is not None
                            else ''
                        ),
                        'current_force_n': float(current_force),
                        'desired_force_n': float(desired_force),
                        'requested_force_n': float(
                            self.steering_pipeline.config.desired_force
                        ),
                        'force_target_mode': self.tts_force_target_mode,
                        'force_target_baseline_n': (
                            float(
                                desired_force
                                - self.steering_pipeline.config.desired_force
                            )
                            if self.tts_force_target_mode == 'baseline_delta'
                            else ''
                        ),
                        'proxy_source': proxy_source,
                        'policy_dynamic_target_base_n': (
                            policy_dynamic_target_base
                            if policy_dynamic_target_base is not None
                            else ''
                        ),
                        'policy_dynamic_target_rise_n': (
                            self.tts_policy_force_dynamic_target_rise
                            if self.tts_policy_force_dynamic_target_rise is not None
                            else ''
                        ),
                        'policy_dynamic_target_cap_n': (
                            self.steering_pipeline.config.desired_force
                            if self.tts_policy_force_dynamic_target_rise is not None
                            else ''
                        ),
                        'proxy_base_mode': self.tts_proxy_base_force,
                        'gripper_proxy_mode': self.tts_gripper_proxy_mode,
                        'safe_candidate': int(bool(safe_mask[i])),
                        'all_candidates_unsafe': int(bool(all_unsafe)),
                        'force_safety_limit_n': (
                            self.tts_force_safety_limit
                            if self.tts_force_safety_limit is not None
                            else ''
                        ),
                        'score_steps': int(score_steps),
                        'obs_x': obs_xyz[0],
                        'obs_y': obs_xyz[1],
                        'obs_z': obs_xyz[2],
                        'obs_gripper_cmd': obs_gripper,
                        'first_x': float(action[0, 0]) if action.ndim == 2 else '',
                        'first_y': float(action[0, 1]) if action.ndim == 2 else '',
                        'first_z': float(action[0, 2]) if action.ndim == 2 else '',
                        'first_gripper_cmd': first_gripper,
                        'first_gripper_delta': first_delta,
                        'mean_positive_closure': float(closing_means[i]),
                        'mean_signed_closure': float(signed_closure_means[i]),
                        'mean_gripper_cmd': float(np.mean(gripper)) if gripper.size else '',
                        'last_gripper_cmd': float(gripper[-1]) if gripper.size else '',
                        'min_gripper_cmd': float(np.min(gripper)) if gripper.size else '',
                        'max_gripper_cmd': float(np.max(gripper)) if gripper.size else '',
                    }
                    writer.writerow(row)

    def _make_visualforce_input_preview(self, frame, mask_bool):
        masked = frame.astype(np.float32) * mask_bool[:, :, None]
        masked_u8 = np.clip(masked, 0, 255).astype(np.uint8)
        return np.asarray(
            Image.fromarray(masked_u8).resize(
                (VISUALFORCE_INPUT_SIZE[1], VISUALFORCE_INPUT_SIZE[0]),
                Image.BILINEAR,
            )
        )

    def _make_visualforce_edge_preview(self, frame, mask_bool):
        from src.steering import masked_frame_to_tensor

        edge = masked_frame_to_tensor(
            frame,
            mask_bool,
            output_size=VISUALFORCE_INPUT_SIZE,
            input_mode='edge',
        )[0]
        edge_u8 = (
            edge.detach().cpu().numpy().clip(0.0, 1.0) * 255.0
        ).astype(np.uint8)
        return np.repeat(edge_u8[:, :, None], 3, axis=2)

    def _record_tts_force_row(
        self,
        frame_idx,
        result,
        mask_area,
        mask_frac,
        frame_mean,
        sat_mean,
        val_mean,
        latest_obs=None,
    ):
        if self.tts_rollout_dir is None or frame_idx is None:
            return

        with self.tts_rollout_lock:
            if self.tts_rollout is None:
                return

            out_dir = self.tts_rollout['dir']
            csv_path = out_dir / 'force_log.csv'
            pred = result.predicted_force
            metadata = result.metadata
            gripper_index = metadata.get('gripper_index')

            base_gripper = ''
            action_gripper = ''
            base_xyz = ['', '', '']
            action_xyz = ['', '', '']
            if gripper_index is not None:
                base = np.asarray(result.base_action)
                action = np.asarray(result.action)
                if base.ndim == 2 and -base.shape[1] <= gripper_index < base.shape[1]:
                    base_gripper = float(base[0, gripper_index])
                if action.ndim == 2 and -action.shape[1] <= gripper_index < action.shape[1]:
                    action_gripper = float(action[0, gripper_index])
            base = np.asarray(result.base_action)
            action = np.asarray(result.action)
            if base.ndim == 2 and base.shape[1] >= 3:
                base_xyz = [float(v) for v in base[0, :3]]
            if action.ndim == 2 and action.shape[1] >= 3:
                action_xyz = [float(v) for v in action[0, :3]]

            obs_xyz = ['', '', '']
            obs_gripper = ''
            if latest_obs is not None and 'arm_pos' in latest_obs:
                obs_xyz = [
                    float(v) for v in np.asarray(latest_obs['arm_pos'])[:3]
                ]
            if latest_obs is not None and 'gripper_pos' in latest_obs:
                obs_gripper = float(
                    np.asarray(latest_obs['gripper_pos']).reshape(-1)[0]
                )

            force_values = ';'.join(
                f'{key}={value:.6f}' for key, value in pred.values.items()
            )
            row = {
                'policy_checkpoint': self.ckpt_path,
                'tts_steering_mode': self.tts_steering_mode,
                'frame_idx': frame_idx,
                'time_s': frame_idx / float(self.tts_rollout_fps),
                'force_key': pred.selected_key,
                'selected_force_n': pred.selected_force,
                'control_force_n': pred.control_force,
                'force_mode': pred.force_mode,
                'force_values': force_values,
                'desired_force_n': metadata.get('desired_force', ''),
                'requested_force_n': metadata.get('requested_force', ''),
                'force_target_mode': metadata.get('force_target_mode', 'absolute'),
                'force_target_baseline_n': metadata.get('force_target_baseline', ''),
                'force_error_n': result.force_error,
                'close_scale': result.close_scale,
                'motion_scale': result.motion_scale,
                'over_target_n': metadata.get('over_target', ''),
                'under_target_n': metadata.get('under_target', ''),
                'stopped_or_opened': metadata.get('stopped_or_opened', ''),
                'sampling_mode': metadata.get('sampling_mode', 'scale'),
                'sampling_candidates': metadata.get('sampling_candidates', ''),
                'sampling_best_idx': metadata.get('sampling_best_idx', ''),
                'sampling_best_score': metadata.get('sampling_best_score', ''),
                'sampling_best_proxy_force': metadata.get('sampling_best_proxy_force', ''),
                'sampling_selection_scope': metadata.get('sampling_selection_scope', ''),
                'sampling_proxy_source': metadata.get('sampling_proxy_source', ''),
                'sampling_critic_delta_force': metadata.get('sampling_critic_delta_force', ''),
                'sampling_score_steps': metadata.get('sampling_score_steps', ''),
                'sampling_action_force_gain': metadata.get('sampling_action_force_gain', ''),
                'sampling_policy_dynamic_target_base_n': metadata.get(
                    'sampling_policy_dynamic_target_base', ''
                ),
                'sampling_policy_dynamic_target_rise_n': metadata.get(
                    'sampling_policy_dynamic_target_rise', ''
                ),
                'sampling_policy_dynamic_target_cap_n': metadata.get(
                    'sampling_policy_dynamic_target_cap', ''
                ),
                'sampling_proxy_base_force': metadata.get('sampling_proxy_base_force', ''),
                'sampling_proxy_base_mode': metadata.get('sampling_proxy_base_mode', ''),
                'sampling_gripper_proxy_mode': metadata.get('sampling_gripper_proxy_mode', ''),
                'sampling_force_min': metadata.get('sampling_force_min', ''),
                'sampling_force_max': metadata.get('sampling_force_max', ''),
                'sampling_force_safety_limit': metadata.get('sampling_force_safety_limit', ''),
                'sampling_all_unsafe': metadata.get('sampling_all_unsafe', ''),
                'sampling_unsafe_fallback': metadata.get('sampling_unsafe_fallback', ''),
                'activation_force_n': metadata.get('activation_force', ''),
                'gripper_safety_active': metadata.get('gripper_safety_active', ''),
                'gripper_safety_mode': metadata.get('gripper_safety_mode', ''),
                'gripper_safety_current': metadata.get('gripper_safety_current', ''),
                'gripper_safety_command': metadata.get('gripper_safety_command', ''),
                'gripper_safety_enter_force': metadata.get('gripper_safety_enter_force', ''),
                'gripper_safety_release_force': metadata.get('gripper_safety_release_force', ''),
                'gripper_safety_release_step': metadata.get('gripper_safety_release_step', ''),
                'gripper_safety_closure_ready': metadata.get('gripper_safety_closure_ready', ''),
                'gripper_safety_closure_position': metadata.get('gripper_safety_closure_position', ''),
                'gripper_safety_min_closure_position': metadata.get('gripper_safety_min_closure_position', ''),
                'preempt_action_queue': metadata.get('preempt_action_queue', ''),
                'gentle_controller': metadata.get('gentle_controller', False),
                'gentle_measurement_valid': metadata.get('gentle_measurement_valid', ''),
                'gentle_just_latched': metadata.get('gentle_just_latched', ''),
                'gentle_raw_force_n': metadata.get('gentle_raw_force_n', ''),
                'gentle_filtered_force_n': metadata.get('gentle_filtered_force_n', ''),
                'gentle_baseline_force_n': metadata.get('gentle_baseline_force_n', ''),
                'gentle_baseline_calibrated': metadata.get('gentle_baseline_calibrated', ''),
                'gentle_force_delta_n': metadata.get('gentle_force_delta_n', ''),
                'gentle_force_rate_n_s': metadata.get('gentle_force_rate_n_s', ''),
                'gentle_desired_force_n': metadata.get('gentle_desired_force_n', ''),
                'gentle_lower_force_n': metadata.get('gentle_lower_force_n', ''),
                'gentle_upper_force_n': metadata.get('gentle_upper_force_n', ''),
                'gentle_emergency_force_n': metadata.get('gentle_emergency_force_n', ''),
                'gentle_desired_delta_n': metadata.get('gentle_desired_delta_n', ''),
                'gentle_last_safe_closure': metadata.get('gentle_last_safe_closure', ''),
                'gentle_hold_closure': metadata.get('gentle_hold_closure', ''),
                'gentle_command_closure': metadata.get('gentle_command_closure', ''),
                'gentle_close_limited': metadata.get('gentle_close_limited', ''),
                'gentle_position_capped': metadata.get('gentle_position_capped', ''),
                'gentle_max_closure_position': metadata.get('gentle_max_closure_position', ''),
                'gentle_max_command_lead': metadata.get('gentle_max_command_lead', ''),
                'gentle_max_force_rate_n_s': metadata.get('gentle_max_force_rate_n_s', ''),
                'gentle_policy_release_enabled': metadata.get('gentle_policy_release_enabled', ''),
                'gentle_policy_release_armed': metadata.get('gentle_policy_release_armed', ''),
                'gentle_policy_release_contact_delta_n': metadata.get(
                    'gentle_policy_release_contact_delta_n', ''
                ),
                'gentle_policy_gripper_approach': metadata.get(
                    'gentle_policy_gripper_approach', ''
                ),
                'gentle_force_control_armed': metadata.get(
                    'gentle_force_control_armed', ''
                ),
                'gentle_policy_release_latched': metadata.get('gentle_policy_release_latched', ''),
                'gentle_policy_release_index': metadata.get('gentle_policy_release_index', ''),
                'gentle_policy_release_threshold': metadata.get('gentle_policy_release_threshold', ''),
                'gentle_policy_release_horizon': metadata.get('gentle_policy_release_horizon', ''),
                'gentle_policy_release_min_steps': metadata.get('gentle_policy_release_min_steps', ''),
                'gripper_index': gripper_index if gripper_index is not None else '',
                'obs_gripper': obs_gripper,
                'obs_x': obs_xyz[0],
                'obs_y': obs_xyz[1],
                'obs_z': obs_xyz[2],
                'base_action_x': base_xyz[0],
                'base_action_y': base_xyz[1],
                'base_action_z': base_xyz[2],
                'steered_action_x': action_xyz[0],
                'steered_action_y': action_xyz[1],
                'steered_action_z': action_xyz[2],
                'base_gripper_cmd': base_gripper,
                'steered_gripper_cmd': action_gripper,
                'mask_px': mask_area,
                'mask_frac': mask_frac,
                'frame_mean': frame_mean,
                'sat_mean': sat_mean,
                'val_mean': val_mean,
            }

            write_header = not csv_path.exists()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    def flush_tts_rollout(self):
        with self.tts_rollout_lock:
            if self.tts_rollout is None:
                return
            out_dir = self.tts_rollout['dir']
            self.tts_rollout['original'].release()
            self.tts_rollout['masked_original'].release()
            self.tts_rollout['edge'].release()
            if 'side_view' in self.tts_rollout:
                self.tts_rollout['side_view'].release()
            print(
                f'TTS rollout captured: {self.tts_rollout_frames} frames'
            )
            print(f'TTS force CSV saved: {out_dir / "force_log.csv"}')
            self._encode_h264_rollout(out_dir)
            self.tts_rollout = None
            self.tts_rollout_frames = 0

    def stop_tts_rollout_recording(self):
        with self.tts_rollout_lock:
            self.tts_rollout_dir = None

    def _encode_h264_rollout(self, out_dir):
        if not h264_encoder_available():
            print('ffmpeg not found; skipping H.264 rollout encode')
            return

        jobs = [
            ('.original_recording.mp4', 'original_h264.mp4'),
            (
                '.masked_original_recording.mp4',
                'masked_original_h264.mp4',
            ),
            ('.edge_recording.mp4', 'edge_h264.mp4'),
            ('.side_view_recording.mp4', 'side_view_h264.mp4'),
        ]
        for source_name, filename in jobs:
            source_path = out_dir / source_name
            if not source_path.exists():
                continue
            target_path = out_dir / filename
            try:
                encode_h264(source_path, target_path)
                print(f'TTS H.264 video saved: {target_path}')
            except VideoEncodingError as exc:
                print(f'ffmpeg failed while writing {filename}:')
                print(exc)

    def _build_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, meta in self.obs_shape_meta.items():
            if meta.get('type') == 'rgb':
                # Keep TTS/VisualForce at source resolution, but feed the DP policy
                # the original 320x240 shape it was deployed with.
                imgs = []
                for obs in obs_sequence:
                    img = obs[key]
                    if img.shape[:2] != (IMAGE_H, IMAGE_W):
                        img = cv.resize(img, (IMAGE_W, IMAGE_H))
                    imgs.append(img)
                # (T, H, W, 3) uint8 -> (T, 3, H, W) float32
                imgs = np.stack(imgs, axis=0)
                imgs = imgs.astype(np.float32) / 255.0
                imgs = np.transpose(imgs, (0, 3, 1, 2))  # THWC -> TCHW
                obs_dict_np[key] = imgs
            elif key == 'agent_pos':
                # concatenate arm_pos(3) + arm_quat(4) + gripper_pos(1) per step
                agent_pos = np.stack([
                    np.concatenate([
                        obs['arm_pos'], obs['arm_quat'], obs['gripper_pos']
                    ]) for obs in obs_sequence
                ], axis=0).astype(np.float32)
                obs_dict_np['agent_pos'] = agent_pos
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )
        return obs_dict

    def _split_action(self, actions):
        """(horizon, 8) -> list of action dicts"""
        result = []
        for act in actions:
            result.append({
                'arm_pos':     act[0:3],
                'arm_quat':    act[3:7],
                'gripper_pos': act[7:8],
            })
        return result


class PolicyWrapper:
    """Handles obs history buffering, action chunking, and latency compensation."""

    def __init__(
        self,
        policy,
        n_obs_steps=2,
        n_action_steps=8,
        latency_compensation=True,
        action_refill_steps=None,
    ):
        self.policy = policy
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_skip_steps = LATENCY_STEPS if latency_compensation else 0
        self.action_refill_steps = (
            LATENCY_STEPS
            if action_refill_steps is None
            else int(action_refill_steps)
        )
        if self.action_refill_steps <= 0:
            raise ValueError('action_refill_steps must be positive')
        self.action_buffer_steps = self.n_action_steps - self.action_skip_steps
        if self.action_refill_steps > self.action_buffer_steps:
            raise ValueError(
                'action_refill_steps cannot exceed the number of queued '
                f'actions ({self.action_buffer_steps})'
            )
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        self.has_produced_action = False
        self.empty_queue_warning_active = False
        print(
            'Action buffer: bounded non-overlapping refill '
            f'(capacity={self.action_buffer_steps}, '
            f'refill_below={self.action_refill_steps}).'
        )
        if not latency_compensation:
            print('Action latency compensation disabled; predicted steps are not skipped.')
        threading.Thread(
            target=self._inference_loop, args=(policy,), daemon=True
        ).start()

    def reset(self):
        self.obs_queue.put('reset')

    def flush_tts_rollout(self):
        if hasattr(self.policy, 'flush_tts_rollout'):
            self.policy.flush_tts_rollout()

    def stop_tts_rollout_recording(self):
        if hasattr(self.policy, 'stop_tts_rollout_recording'):
            self.policy.stop_tts_rollout_recording()

    def step(self, obs):
        self.obs_queue.put(obs)
        action = None if self.act_queue.empty() else self.act_queue.get()
        if action is None:
            if not self.empty_queue_warning_active:
                if self.has_produced_action:
                    print(
                        'Warning: action queue starved after startup — '
                        'inference is slower than the buffered horizon'
                    )
                else:
                    print('TTS warmup: waiting for the first action chunk')
                self.empty_queue_warning_active = True
        else:
            self.empty_queue_warning_active = False
        return action

    def _inference_loop(self, policy):
        obs_history = deque(maxlen=self.n_obs_steps)
        start_of_episode = True
        obs_revision = 0
        last_inference_obs_revision = -1
        while True:
            try:
                if not self.obs_queue.empty():
                    item = self.obs_queue.get()
                    if item == 'reset':
                        policy.reset()
                        obs_history.clear()
                        start_of_episode = True
                        obs_revision = 0
                        last_inference_obs_revision = -1
                        self.has_produced_action = False
                        self.empty_queue_warning_active = False
                        while not self.act_queue.empty():
                            self.act_queue.get()
                        continue
                    obs_history.append(item)
                    obs_revision += 1

                requires_fresh = bool(
                    getattr(policy, 'requires_fresh_observation', False)
                )
                has_fresh_observation = (
                    not requires_fresh
                    or obs_revision != last_inference_obs_revision
                )
                if (
                    self.act_queue.qsize() < self.action_refill_steps
                    and len(obs_history) == self.n_obs_steps
                    and has_fresh_observation
                ):
                    act_sequence = policy.step(list(obs_history))
                    last_inference_obs_revision = obs_revision
                    preempt = (
                        policy.consume_action_queue_preempt()
                        if hasattr(policy, 'consume_action_queue_preempt')
                        else False
                    )
                    if preempt:
                        cleared = 0
                        while True:
                            try:
                                self.act_queue.get_nowait()
                                cleared += 1
                            except queue.Empty:
                                break
                        print(
                            'TTS gripper safety preempted '
                            f'{cleared} queued action(s).'
                        )
                    queued_steps = self.act_queue.qsize()
                    act_sequence = _select_action_refill(
                        act_sequence,
                        queued_steps=queued_steps,
                        buffer_steps=self.action_buffer_steps,
                        skip_steps=self.action_skip_steps,
                        start_of_episode=start_of_episode,
                    )
                    start_of_episode = False
                    for action in act_sequence:
                        self.act_queue.put(action)
                        self.has_produced_action = True

                time.sleep(0.001)
            except Exception:
                print('Policy inference thread crashed:')
                traceback.print_exc()
                raise


class PolicyServer:
    def __init__(self, policy, port=5555):
        self.policy = policy
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f'tcp://*:{port}')
        print(f'Policy server listening on port {port}')

    def _decode_obs(self, obs):
        decoded = {}
        for k, v in obs.items():
            if k.endswith('image'):
                # Decode JPEG and preserve source resolution for TTS/VisualForce.
                # _build_obs handles policy-specific resizing later.
                img = cv.imdecode(v, cv.IMREAD_COLOR)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # fix channel order
                decoded[k] = img
            else:
                decoded[k] = v
        return decoded

    def run(self):
        try:
            while True:
                req = self.socket.recv_pyobj()
                rep = {}
                try:
                    if 'reset' in req:
                        self.policy.reset()
                        print('Policy reset')
                    elif 'obs' in req:
                        obs = self._decode_obs(req['obs'])
                        action = self.policy.step(obs)
                        rep['action'] = action
                except Exception as exc:
                    print('Policy server request failed:')
                    traceback.print_exc()
                    rep['error'] = repr(exc)
                self.socket.send_pyobj(rep)
        finally:
            if hasattr(self.policy, 'flush_tts_rollout'):
                self.policy.flush_tts_rollout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--n-obs-steps', type=int, default=2)
    parser.add_argument('--n-action-steps', type=int, default=8)
    parser.add_argument(
        '--action-refill-steps',
        type=int,
        default=None,
        help=(
            'Start the next inference when fewer than this many actions remain. '
            'Defaults to the two-step latency budget.'
        ),
    )
    parser.add_argument(
        '--policy-action-steps',
        type=int,
        default=None,
        help=(
            'Override the loaded policy n_action_steps before inference. '
            'Useful for testing longer chunks up to horizon - n_obs_steps + 1.'
        ),
    )
    parser.add_argument(
        '--relative-action-scale',
        type=float,
        default=1.0,
        help='Scale xyz deltas before accumulating relative-position checkpoints.',
    )
    parser.add_argument(
        '--relative-gripper-action-scale',
        type=float,
        default=1.0,
        help=(
            'Scale gripper command delta around the current observed gripper '
            'position before execution/TTS scoring. 1.0 preserves old behavior.'
        ),
    )
    parser.add_argument(
        '--gripper-command-min',
        type=float,
        default=None,
        help='Optional lower clamp for final gripper commands.',
    )
    parser.add_argument(
        '--gripper-command-max',
        type=float,
        default=None,
        help='Optional upper clamp for final gripper commands.',
    )
    parser.add_argument(
        '--disable-latency-compensation',
        action='store_true',
        help='Do not skip the first 200 ms of predicted actions.',
    )
    parser.add_argument('--tts-visualforce-ckpt', default=None)
    parser.add_argument(
        '--tts-visualforce-root',
        default='',
        help='Repository root containing the external force-estimator code.',
    )
    parser.add_argument(
        '--tts-experiment-profile',
        choices=['coke', 'berry', 'reorientation', 'flip', 'plug_insertion', 'plug'],
        default=None,
        help='Load the canonical final controller contract for one experiment.',
    )
    parser.add_argument('--tts-desired-force', type=float, default=None)
    parser.add_argument(
        '--tts-force-target-mode',
        choices=['absolute', 'baseline_delta'],
        default='absolute',
        help=(
            'Interpret --tts-desired-force as an absolute force or as a rise '
            'above the force latched at episode start.'
        ),
    )
    parser.add_argument('--tts-force-key', default='Fz')
    parser.add_argument(
        '--tts-force-mode',
        choices=['magnitude', 'signed'],
        default='magnitude',
    )
    parser.add_argument('--tts-frame-key', default=None)
    parser.add_argument(
        '--tts-side-frame-key',
        default=None,
        help='Observation key recorded as side_view_h264.mp4; empty disables it.',
    )
    parser.add_argument(
        '--tts-frame-color-space',
        choices=['rgb', 'bgr'],
        default='rgb',
        help=(
            'Color order of camera arrays received by policy_server for TTS. '
            'Use bgr when frames come from cv2/imencode paths.'
        ),
    )
    parser.add_argument('--tts-mask-key', default=None)
    parser.add_argument(
        '--tts-auto-mask',
        action='store_true',
        help='Generate the gripper mask from --tts-frame-key',
    )
    parser.add_argument(
        '--tts-mask-mode',
        choices=['sam2', 'hsv'],
        default='sam2',
        help='Mask generator for --tts-auto-mask. sam2 matches the VisualForce offline chain.',
    )
    parser.add_argument(
        '--tts-steering-mode',
        choices=['scale', 'sample', 'monitor'],
        default='scale',
        help=(
            'scale = feedback scaling; sample = sample DP chunks and rerank; '
            'monitor = log force while returning unmodified DP actions.'
        ),
    )
    parser.add_argument(
        '--tts-sampling-candidates',
        type=int,
        default=8,
        help='Number of DP action chunks to sample when --tts-steering-mode sample.',
    )
    parser.add_argument(
        '--tts-selection-scope',
        choices=['full', 'gripper'],
        default='full',
        help=(
            'full executes the entire force-selected chunk; gripper keeps '
            'candidate 0 arm poses and takes only the selected gripper commands.'
        ),
    )
    parser.add_argument(
        '--tts-sampling-score-steps',
        type=int,
        default=4,
        help='Number of leading action steps used by the sampling force proxy.',
    )
    parser.add_argument(
        '--tts-action-force-gain',
        type=float,
        default=1.0,
        help='Proxy gain: predicted next force = current force + gain * mean closing action.',
    )
    parser.add_argument(
        '--tts-proxy-base-force',
        choices=['visualforce', 'zero'],
        default='visualforce',
        help=(
            'Base term for the linear/delta proxy. visualforce uses the current '
            'visual estimate; zero is the estimator-free paper ablation.'
        ),
    )
    parser.add_argument(
        '--tts-gripper-proxy-mode',
        choices=['absolute', 'relative'],
        default='absolute',
        help=(
            'absolute preserves the legacy mean-command proxy; relative scores '
            'commanded closure from the current observed gripper position.'
        ),
    )
    parser.add_argument(
        '--tts-force-min',
        type=float,
        default=None,
        help='Optional lower edge of a zero-cost demonstrated force band.',
    )
    parser.add_argument(
        '--tts-force-max',
        type=float,
        default=None,
        help='Optional upper edge of a zero-cost demonstrated force band.',
    )
    parser.add_argument(
        '--tts-high-force-weight',
        type=float,
        default=1.0,
        help='Penalty multiplier above --tts-force-max.',
    )
    parser.add_argument(
        '--tts-force-safety-limit',
        type=float,
        default=None,
        help='Reject candidate chunks whose predicted proxy force exceeds this limit.',
    )
    parser.add_argument(
        '--tts-unsafe-fallback',
        choices=['least_bad', 'min_close'],
        default='least_bad',
        help=(
            'Behavior when every sampled candidate exceeds the safety limit. '
            'min_close selects the chunk with the least signed closure.'
        ),
    )
    parser.add_argument(
        '--tts-activation-force',
        type=float,
        default=0.0,
        help=(
            'Disable TTS action modification until the current estimated '
            'control force reaches this threshold in N. 0 preserves old behavior.'
        ),
    )
    parser.add_argument(
        '--tts-delta-force-critic-ckpt',
        default=None,
        help=(
            'Optional action-conditioned delta-force critic checkpoint. '
            'When set, sample-mode TTS scores candidates with '
            'current_force + predicted_delta_force instead of the gripper '
            'closing proxy.'
        ),
    )
    parser.add_argument(
        '--tts-delta-force-critic-weight',
        type=float,
        default=1.0,
        help='Multiplier applied to the critic-predicted force delta.',
    )
    parser.add_argument(
        '--tts-policy-force-output',
        action='store_true',
        help=(
            'Rank sampled chunks using the ninth force dimension jointly '
            'predicted by a force-output policy.'
        ),
    )
    parser.add_argument(
        '--tts-auto-policy-force-output',
        action='store_true',
        help=(
            'Use the ninth learned force dimension when the selected '
            'checkpoint contains one; otherwise retain the configured proxy.'
        ),
    )
    parser.add_argument(
        '--tts-policy-force-aggregation',
        choices=['last', 'mean', 'max'],
        default='last',
        help='Reduce the scored policy force prefix to one force value.',
    )
    parser.add_argument(
        '--tts-policy-force-dynamic-target-rise',
        type=float,
        default=None,
        help=(
            'Use median sampled first-step policy force plus this rise as the '
            'per-request target, capped by --tts-desired-force.'
        ),
    )
    parser.add_argument(
        '--tts-log-candidates',
        action='store_true',
        help=(
            'Write candidate_scores.csv in the TTS rollout directory with one '
            'row per sampled action chunk. This is diagnostic-only.'
        ),
    )
    parser.add_argument(
        '--tts-absolute-gripper-safety',
        action='store_true',
        help=(
            'After sample-mode TTS, hold an absolute gripper above the target '
            'deadband and release it by a bounded step above the stop margin.'
        ),
    )
    parser.add_argument(
        '--tts-gentle-gripper-control',
        action='store_true',
        help=(
            'Zero VisualForce with the open gripper, then control force rise '
            'using a persistent absolute-gripper setpoint.'
        ),
    )
    parser.add_argument(
        '--tts-disable-policy-release',
        action='store_true',
        help=(
            'Do not interpret a policy open-gripper suffix as a latched release '
            'while direct gripper force control is active.'
        ),
    )
    parser.add_argument(
        '--tts-policy-release-contact-delta',
        type=float,
        default=None,
        help=(
            'Minimum baseline-relative force rise that arms a later stable '
            'policy open suffix. Defaults to --tts-contact-force-delta.'
        ),
    )
    parser.add_argument(
        '--tts-policy-gripper-approach',
        action='store_true',
        help=(
            'Use the nominal policy gripper command before contact evidence, '
            'then hand control to direct force feedback.'
        ),
    )
    parser.add_argument(
        '--tts-add-gripper-fallback-candidates',
        action='store_true',
        help='Append explicit hold and bounded-open chunks to sample-mode TTS.',
    )
    parser.add_argument(
        '--tts-gripper-release-step',
        type=float,
        default=0.05,
        help=(
            'Absolute gripper opening step used by '
            '--tts-absolute-gripper-safety.'
        ),
    )
    parser.add_argument(
        '--tts-gripper-safety-min-position',
        type=float,
        default=None,
        help=(
            'Do not arm gripper-force contact handling until signed closure '
            'reaches this demonstrated contact-ready position.'
        ),
    )
    parser.add_argument(
        '--tts-contact-force-delta',
        type=float,
        default=5.0,
        help=(
            'Target VisualForce rise above the open-gripper baseline for direct '
            'gripper control.'
        ),
    )
    parser.add_argument(
        '--tts-force-filter-window',
        type=int,
        default=3,
        help='Median-filter window for gripper VisualForce feedback.',
    )
    parser.add_argument(
        '--tts-force-baseline-samples',
        type=int,
        default=3,
        help=(
            'Number of open-gripper samples used to latch the fixed episode '
            'force baseline.'
        ),
    )
    parser.add_argument(
        '--tts-force-baseline-max-position',
        type=float,
        default=0.35,
        help=(
            'Largest signed gripper closure at which the non-contact '
            'VisualForce baseline may be calibrated.'
        ),
    )
    parser.add_argument(
        '--tts-gripper-close-step',
        type=float,
        default=0.05,
        help=(
            'Maximum new absolute closure increment per fresh observation. '
            'This also provides enough command lead to clear gripper deadband.'
        ),
    )
    parser.add_argument(
        '--tts-gripper-maintain-step',
        type=float,
        default=0.005,
        help='Persistent closure increment after contact when force is low.',
    )
    parser.add_argument(
        '--tts-gripper-max-position',
        type=float,
        default=0.82,
        help='Hard signed-closure limit for gripper control.',
    )
    parser.add_argument(
        '--tts-gripper-max-lead',
        type=float,
        default=0.25,
        help='Maximum persistent command lead above measured gripper closure.',
    )
    parser.add_argument(
        '--tts-max-force-rate',
        type=float,
        default=20.0,
        help='Filtered force-rise rate that triggers contact release.',
    )
    parser.add_argument(
        '--tts-sam2-model',
        choices=['tiny', 'small', 'base', 'large'],
        default='small',
    )
    parser.add_argument(
        '--tts-sam2-repo',
        default=None,
        help='Path to the local SAM2 repo containing the sam2 Python package.',
    )
    parser.add_argument(
        '--tts-sam2-ckpt',
        default=None,
        help='Path to the local SAM2 checkpoint for --tts-sam2-model.',
    )
    parser.add_argument(
        '--tts-rollout-dir',
        default='tts_rollouts',
        help='Directory for TTS overlay/mask rollout videos. Set empty string to disable.',
    )
    parser.add_argument('--tts-rollout-fps', type=float, default=10.0)
    parser.add_argument('--tts-mask-h-min', type=float, default=GRIPPER_H_MIN)
    parser.add_argument('--tts-mask-h-max', type=float, default=GRIPPER_H_MAX)
    parser.add_argument('--tts-mask-s-min', type=float, default=GRIPPER_S_MIN)
    parser.add_argument('--tts-mask-v-min', type=float, default=GRIPPER_V_MIN)
    parser.add_argument('--tts-mask-min-area', type=int, default=GRIPPER_MIN_BLOB_AREA)
    parser.add_argument('--tts-gripper-index', type=int, default=7)
    parser.add_argument('--tts-close-positive', action='store_true')
    parser.add_argument('--tts-close-negative', action='store_true')
    parser.add_argument('--tts-motion-indices', nargs='*', type=int, default=[])
    parser.add_argument('--tts-deadband', type=float, default=0.10)
    parser.add_argument('--tts-slowdown-band', type=float, default=1.0)
    parser.add_argument('--tts-stop-margin', type=float, default=0.75)
    parser.add_argument('--tts-open-command', type=float, default=0.0)
    parser.add_argument('--tts-close-gain', type=float, default=0.0)
    parser.add_argument('--tts-max-close-command', type=float, default=None)
    args = parser.parse_args()
    apply_profile_defaults(args, sys.argv[1:])

    steering_pipeline = None
    tts_masker = None
    tts_delta_force_critic = None
    if args.tts_visualforce_ckpt is not None:
        if args.tts_desired_force is None:
            raise ValueError('--tts-desired-force is required when TTS is enabled')
        if args.tts_close_positive and args.tts_close_negative:
            raise ValueError('Choose only one of --tts-close-positive/--tts-close-negative')

        ForceSteeringConfig, VisualForceSteeringPipeline = _load_steering_classes(
            args.tts_visualforce_root
        )
        close_positive = not args.tts_close_negative
        if args.tts_close_positive:
            close_positive = True
        steering_config = ForceSteeringConfig(
            desired_force=args.tts_desired_force,
            force_key=args.tts_force_key,
            force_mode=args.tts_force_mode,
            deadband=args.tts_deadband,
            slowdown_band=args.tts_slowdown_band,
            stop_margin=args.tts_stop_margin,
            gripper_index=args.tts_gripper_index,
            close_positive=close_positive,
            open_command=args.tts_open_command,
            close_gain=args.tts_close_gain,
            max_close_command=args.tts_max_close_command,
            motion_indices=tuple(args.tts_motion_indices),
        )
        steering_pipeline = VisualForceSteeringPipeline(
            args.tts_visualforce_ckpt,
            steering_config,
            device=args.device,
        )
        if args.tts_delta_force_critic_ckpt is not None:
            tts_delta_force_critic = DeltaForceCritic(
                args.tts_delta_force_critic_ckpt,
                args.tts_visualforce_root,
                device=args.device,
            )
        if args.tts_auto_mask and args.tts_mask_mode == 'sam2':
            tts_masker = Sam2FrameMasker(
                args.tts_visualforce_root,
                args.tts_sam2_model,
                sam2_repo=args.tts_sam2_repo,
                sam2_ckpt=args.tts_sam2_ckpt,
        )
        print('Test-time force steering enabled')

    policy = DiffusionPolicy(
        args.ckpt_path,
        device=args.device,
        steering_pipeline=steering_pipeline,
        tts_frame_key=args.tts_frame_key,
        tts_side_frame_key=args.tts_side_frame_key,
        tts_frame_color_space=args.tts_frame_color_space,
        tts_mask_key=args.tts_mask_key,
        tts_auto_mask=args.tts_auto_mask,
        tts_masker=tts_masker,
        tts_mask_mode=args.tts_mask_mode,
        tts_steering_mode=args.tts_steering_mode,
        tts_sampling_candidates=args.tts_sampling_candidates,
        tts_sampling_score_steps=args.tts_sampling_score_steps,
        tts_action_force_gain=args.tts_action_force_gain,
        tts_proxy_base_force=args.tts_proxy_base_force,
        tts_gripper_proxy_mode=args.tts_gripper_proxy_mode,
        tts_force_min=args.tts_force_min,
        tts_force_max=args.tts_force_max,
        tts_high_force_weight=args.tts_high_force_weight,
        tts_force_safety_limit=args.tts_force_safety_limit,
        tts_unsafe_fallback=args.tts_unsafe_fallback,
        tts_activation_force=args.tts_activation_force,
        tts_delta_force_critic=tts_delta_force_critic,
        tts_delta_force_critic_weight=args.tts_delta_force_critic_weight,
        tts_policy_force_output=args.tts_policy_force_output,
        tts_auto_policy_force_output=args.tts_auto_policy_force_output,
        tts_policy_force_aggregation=args.tts_policy_force_aggregation,
        tts_policy_force_dynamic_target_rise=(
            args.tts_policy_force_dynamic_target_rise
        ),
        tts_force_target_mode=args.tts_force_target_mode,
        tts_selection_scope=args.tts_selection_scope,
        tts_log_candidates=args.tts_log_candidates,
        tts_absolute_gripper_safety=args.tts_absolute_gripper_safety,
        tts_gentle_gripper_control=args.tts_gentle_gripper_control,
        tts_policy_release_enabled=not args.tts_disable_policy_release,
        tts_policy_release_contact_delta=args.tts_policy_release_contact_delta,
        tts_policy_gripper_approach=args.tts_policy_gripper_approach,
        tts_add_gripper_fallback_candidates=(
            args.tts_add_gripper_fallback_candidates
        ),
        tts_gripper_release_step=args.tts_gripper_release_step,
        tts_gripper_safety_min_position=args.tts_gripper_safety_min_position,
        tts_contact_force_delta=args.tts_contact_force_delta,
        tts_force_filter_window=args.tts_force_filter_window,
        tts_force_baseline_samples=args.tts_force_baseline_samples,
        tts_force_baseline_max_position=args.tts_force_baseline_max_position,
        tts_gripper_close_step=args.tts_gripper_close_step,
        tts_gripper_maintain_step=args.tts_gripper_maintain_step,
        tts_gripper_max_position=args.tts_gripper_max_position,
        tts_gripper_max_lead=args.tts_gripper_max_lead,
        tts_max_force_rate=args.tts_max_force_rate,
        policy_action_steps=args.policy_action_steps,
        relative_action_scale=args.relative_action_scale,
        relative_gripper_action_scale=args.relative_gripper_action_scale,
        gripper_command_min=args.gripper_command_min,
        gripper_command_max=args.gripper_command_max,
        tts_rollout_dir=args.tts_rollout_dir if steering_pipeline is not None else None,
        tts_rollout_fps=args.tts_rollout_fps,
        tts_mask_h_min=args.tts_mask_h_min,
        tts_mask_h_max=args.tts_mask_h_max,
        tts_mask_s_min=args.tts_mask_s_min,
        tts_mask_v_min=args.tts_mask_v_min,
        tts_mask_min_area=args.tts_mask_min_area,
    )
    wrapped = PolicyWrapper(
        policy,
        args.n_obs_steps,
        args.n_action_steps,
        latency_compensation=not args.disable_latency_compensation,
        action_refill_steps=args.action_refill_steps,
    )
    atexit.register(wrapped.flush_tts_rollout)

    def flush_and_exit(signum, _frame):
        print(f'Received signal {signum}, flushing TTS rollout before exit')
        wrapped.stop_tts_rollout_recording()
        wrapped.flush_tts_rollout()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, flush_and_exit)
    signal.signal(signal.SIGTERM, flush_and_exit)
    server = PolicyServer(wrapped, port=args.port)
    server.run()


if __name__ == '__main__':
    main()
