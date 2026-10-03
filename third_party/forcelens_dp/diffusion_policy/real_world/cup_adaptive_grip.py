"""Pure stateful control for the no-policy adaptive paper-cup demo.

The robot-side client supplies measured ARX joint torques together with a
Jacobian-derived generalized-torque basis for a one-newton downward payload.
This module projects the change from an empty-cup baseline onto that basis and
uses the estimated added load to schedule a small, bounded gripper closure.
VisualForce is an independent force-safety limit rather than the signal that
drives routine closure.

No robot, camera, torch, or networking dependencies belong here.  Keeping the
control law pure makes its safety bounds testable without hardware.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CupAdaptiveGripConfig:
    """Configuration for :class:`CupAdaptiveGripController`.

    Gripper positions are normalized closure commands: 0 is open and 1 is
    closed.  Forces and added payload are in newtons.
    """

    calibration_samples: int = 10
    force_filter_window: int = 5
    load_filter_window: int = 5
    load_deadband_n: float = 0.30
    closure_per_load_n: float = 0.025
    force_limit_rise_n: float = 3.0
    emergency_margin_n: float = 0.50
    close_step: float = 0.0025
    release_step: float = 0.005
    min_initial_closure: float = 0.20
    max_closure: float = 0.55
    max_closure_delta: float = 0.03
    max_command_lead: float = 0.03
    max_valid_load_n: float = 12.0
    min_torque_basis_norm: float = 1e-3
    payload_torque_sign: float = 1.0
    monitor_only: bool = False

    def __post_init__(self) -> None:
        positive_integer_fields = (
            'calibration_samples',
            'force_filter_window',
            'load_filter_window',
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be a positive integer')
            if value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        nonnegative_fields = (
            'load_deadband_n',
            'closure_per_load_n',
            'force_limit_rise_n',
            'emergency_margin_n',
            'release_step',
            'min_initial_closure',
            'max_closure_delta',
        )
        for name in nonnegative_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')

        positive_fields = (
            'close_step',
            'max_command_lead',
            'max_valid_load_n',
            'min_torque_basis_norm',
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')

        if not 0.0 <= self.min_initial_closure <= 1.0:
            raise ValueError('min_initial_closure must be in [0, 1]')
        if not 0.0 < self.max_closure <= 1.0:
            raise ValueError('max_closure must be in (0, 1]')
        if self.min_initial_closure >= self.max_closure:
            raise ValueError('min_initial_closure must be below max_closure')
        if self.payload_torque_sign not in (-1.0, 1.0):
            raise ValueError('payload_torque_sign must be -1 or 1')


class CupAdaptiveGripController:
    """Estimate added payload and adjust only a bounded gripper setpoint.

    Calibration is intentionally performed after the operator has already
    grasped the empty cup.  The controller never approaches or initially
    grasps an object.  It holds the initial closure until both the visual-force
    and joint-torque baselines have enough stationary samples.

    Normal operation is monotonic: added load schedules more closure, but a
    falling or noisy load estimate does not automatically open a cup containing
    liquid. VisualForce stops routine closing at a baseline-relative safety limit.
    The only automatic opening is one bounded, latched step after the higher
    emergency threshold is crossed.
    """

    def __init__(self, config: Optional[CupAdaptiveGripConfig] = None):
        self.config = config or CupAdaptiveGripConfig()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.initial_closure = None  # type: Optional[float]
        self.command_closure = None  # type: Optional[float]
        self.force_baseline = None  # type: Optional[float]
        self.torque_baseline = None  # type: Optional[np.ndarray]
        self.startup_fault = None  # type: Optional[str]
        self.emergency_latched = False
        self.emergency_command = None  # type: Optional[float]
        self._force_baseline_samples = []
        self._torque_baseline_samples = []
        self._force_history = deque(maxlen=cfg.force_filter_window)
        self._load_history = deque(maxlen=cfg.load_filter_window)

    @property
    def calibrated(self) -> bool:
        return self.force_baseline is not None and self.torque_baseline is not None

    @staticmethod
    def project_added_load(
        joint_torque: np.ndarray,
        baseline_torque: np.ndarray,
        payload_torque_basis: np.ndarray,
        torque_sign: float = 1.0,
        min_basis_norm: float = 1e-3,
    ) -> float:
        """Project a static torque change onto the one-newton payload basis."""

        torque = np.asarray(joint_torque, dtype=np.float64).reshape(-1)
        baseline = np.asarray(baseline_torque, dtype=np.float64).reshape(-1)
        basis = np.asarray(payload_torque_basis, dtype=np.float64).reshape(-1)
        if torque.shape != baseline.shape or torque.shape != basis.shape:
            raise ValueError(
                'joint torque, baseline, and payload basis must have matching shapes'
            )
        if torque.size == 0 or not np.isfinite(
            np.concatenate([torque, baseline, basis])
        ).all():
            raise ValueError('payload projection inputs must be finite and non-empty')
        basis_norm_sq = float(np.dot(basis, basis))
        if basis_norm_sq < float(min_basis_norm) ** 2:
            raise ValueError('payload torque basis is too small at this arm pose')
        return float(torque_sign) * float(np.dot(torque - baseline, basis)) / basis_norm_sq

    def _safe_command(self, current_closure: float) -> float:
        if self.command_closure is not None:
            return float(self.command_closure)
        if self.initial_closure is not None:
            return float(self.initial_closure)
        return float(current_closure)

    def _hold(
        self,
        current_closure: float,
        mode: str,
        **metadata: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        """Return a mode-specific result without advancing the command."""

        return self._result(
            self._safe_command(current_closure),
            mode,
            **metadata,
        )

    def _force_thresholds(self) -> Tuple[float, float]:
        if self.force_baseline is None:
            raise RuntimeError('force thresholds require a calibrated baseline')
        force_limit = self.force_baseline + self.config.force_limit_rise_n
        return force_limit, force_limit + self.config.emergency_margin_n

    def _result(
        self,
        command: float,
        mode: str,
        *,
        raw_force: Any = '',
        filtered_force: Any = '',
        raw_load: Any = '',
        filtered_load: Any = '',
        effective_load: Any = '',
        scheduled_closure: Any = '',
        scheduled_increment: Any = '',
        force_limit: Any = '',
        emergency_force: Any = '',
        measurement_valid: bool = True,
        load_valid: bool = True,
        stationary: bool = True,
        reason: str = '',
        closure_limited: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        cfg = self.config
        # Monitor mode must be observational.  A startup fault must also keep
        # the operator's manual grip unchanged rather than opening the cup just
        # to bring an already-established command under an adaptive bound.
        preserve_manual_grip = cfg.monitor_only or mode == 'startup_fault_hold'
        upper_command = 1.0 if preserve_manual_grip else cfg.max_closure
        command = float(np.clip(command, 0.0, upper_command))
        self.command_closure = command
        return command, {
            'mode': mode,
            'reason': reason,
            'calibrated': self.calibrated,
            'calibration_count': len(self._force_baseline_samples),
            'calibration_required': cfg.calibration_samples,
            'measurement_valid': bool(measurement_valid),
            'load_valid': bool(load_valid),
            'stationary': bool(stationary),
            'monitor_only': bool(cfg.monitor_only),
            'raw_force_n': raw_force,
            'filtered_force_n': filtered_force,
            'baseline_force_n': (
                '' if self.force_baseline is None else self.force_baseline
            ),
            'raw_added_load_n': raw_load,
            'filtered_added_load_n': filtered_load,
            'effective_added_load_n': effective_load,
            'scheduled_closure': scheduled_closure,
            'scheduled_closure_increment': scheduled_increment,
            'force_limit_n': force_limit,
            'emergency_force_n': emergency_force,
            'initial_closure': (
                '' if self.initial_closure is None else self.initial_closure
            ),
            'command_closure': command,
            'max_closure': cfg.max_closure,
            'max_closure_delta': cfg.max_closure_delta,
            'max_command_lead': cfg.max_command_lead,
            'closure_limited': bool(closure_limited),
            'emergency_latched': bool(self.emergency_latched),
            'startup_fault': self.startup_fault or '',
        }

    def update(
        self,
        *,
        current_closure: float,
        raw_force: float,
        joint_torque: np.ndarray,
        payload_torque_basis: np.ndarray,
        stationary: bool = True,
        measurement_valid: bool = True,
        load_valid: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """Advance the controller by one fresh visual-force observation."""

        cfg = self.config
        try:
            current_closure = float(current_closure)
        except (TypeError, ValueError) as exc:
            raise ValueError('current_closure must be a scalar') from exc
        if not np.isfinite(current_closure):
            raise ValueError('current_closure must be finite')
        current_closure = float(np.clip(current_closure, 0.0, 1.0))

        force_finite = np.isfinite(raw_force)
        measurement_valid = bool(measurement_valid and force_finite)
        torque = np.asarray(joint_torque, dtype=np.float64).reshape(-1)
        basis = np.asarray(payload_torque_basis, dtype=np.float64).reshape(-1)
        arrays_valid = (
            torque.size > 0
            and torque.shape == basis.shape
            and np.isfinite(np.concatenate([torque, basis])).all()
            and float(np.linalg.norm(basis)) >= cfg.min_torque_basis_norm
        )
        load_valid = bool(load_valid and arrays_valid)

        if self.initial_closure is None:
            self.initial_closure = current_closure
            self.command_closure = current_closure
            if current_closure < cfg.min_initial_closure:
                self.startup_fault = (
                    f'initial closure {current_closure:.3f} is below the required '
                    f'{cfg.min_initial_closure:.3f}; manually grip the empty cup first'
                )
            elif current_closure >= cfg.max_closure:
                self.startup_fault = (
                    f'initial closure {current_closure:.3f} leaves no headroom below '
                    f'the {cfg.max_closure:.3f} cup limit'
                )

        if self.startup_fault is not None:
            return self._hold(
                current_closure,
                'startup_fault_hold',
                raw_force=raw_force if force_finite else '',
                measurement_valid=measurement_valid,
                load_valid=load_valid,
                stationary=stationary,
                reason=self.startup_fault,
            )

        if not self.calibrated:
            if not measurement_valid or not load_valid:
                return self._hold(
                    current_closure,
                    'calibration_sensor_hold',
                    raw_force=raw_force if force_finite else '',
                    measurement_valid=measurement_valid,
                    load_valid=load_valid,
                    stationary=stationary,
                    reason='waiting for valid VisualForce and ARX torque telemetry',
                )
            if not stationary:
                return self._hold(
                    current_closure,
                    'calibration_motion_hold',
                    raw_force=float(raw_force),
                    measurement_valid=True,
                    load_valid=True,
                    stationary=False,
                    reason='waiting for the held arm to become stationary',
                )

            self._force_baseline_samples.append(float(raw_force))
            self._torque_baseline_samples.append(torque.copy())
            if len(self._force_baseline_samples) >= cfg.calibration_samples:
                self.force_baseline = float(
                    np.median(np.asarray(self._force_baseline_samples), axis=0)
                )
                self.torque_baseline = np.median(
                    np.stack(self._torque_baseline_samples, axis=0), axis=0
                )
                self._force_history.clear()
                self._force_history.append(self.force_baseline)
                self._load_history.clear()
                force_limit, emergency_force = self._force_thresholds()
                return self._hold(
                    current_closure,
                    'calibration_complete',
                    raw_force=float(raw_force),
                    filtered_force=self.force_baseline,
                    raw_load=0.0,
                    filtered_load=0.0,
                    effective_load=0.0,
                    scheduled_closure=self.initial_closure,
                    scheduled_increment=0.0,
                    force_limit=force_limit,
                    emergency_force=emergency_force,
                )
            return self._hold(
                current_closure,
                'calibrating_empty_cup',
                raw_force=float(raw_force),
            )

        if not measurement_valid or not load_valid:
            return self._hold(
                current_closure,
                'sensor_hold',
                raw_force=raw_force if force_finite else '',
                measurement_valid=measurement_valid,
                load_valid=load_valid,
                stationary=stationary,
                reason='invalid VisualForce or payload-load measurement',
            )
        if not stationary:
            return self._hold(
                current_closure,
                'motion_hold',
                raw_force=float(raw_force),
                stationary=False,
                reason='arm motion invalidates the quasi-static payload estimate',
            )

        try:
            raw_load = self.project_added_load(
                torque,
                self.torque_baseline,
                basis,
                torque_sign=cfg.payload_torque_sign,
                min_basis_norm=cfg.min_torque_basis_norm,
            )
        except ValueError as exc:
            return self._hold(
                current_closure,
                'load_projection_hold',
                raw_force=float(raw_force),
                load_valid=False,
                reason=str(exc),
            )
        if abs(raw_load) > cfg.max_valid_load_n:
            return self._hold(
                current_closure,
                'load_outlier_hold',
                raw_force=float(raw_force),
                raw_load=raw_load,
                load_valid=False,
                reason=(
                    f'projected load {raw_load:.3f} N exceeds the '
                    f'{cfg.max_valid_load_n:.3f} N validity bound'
                ),
            )

        self._load_history.append(raw_load)
        filtered_load = float(np.median(np.asarray(self._load_history)))
        effective_load = max(0.0, filtered_load - cfg.load_deadband_n)
        scheduled_increment = min(
            cfg.max_closure_delta,
            cfg.closure_per_load_n * effective_load,
        )
        scheduled_closure = min(
            cfg.max_closure,
            float(self.initial_closure) + scheduled_increment,
        )

        self._force_history.append(float(raw_force))
        filtered_force = float(np.median(np.asarray(self._force_history)))
        force_limit, emergency_force = self._force_thresholds()
        operating_values = {
            'raw_force': float(raw_force),
            'filtered_force': filtered_force,
            'raw_load': raw_load,
            'filtered_load': filtered_load,
            'effective_load': effective_load,
            'scheduled_closure': scheduled_closure,
            'scheduled_increment': scheduled_increment,
            'force_limit': force_limit,
            'emergency_force': emergency_force,
        }

        if cfg.monitor_only:
            if filtered_force >= emergency_force:
                mode = 'monitor_emergency_force'
            elif filtered_force >= force_limit:
                mode = 'monitor_force_limit'
            else:
                mode = 'monitor_load'
            return self._hold(
                current_closure,
                mode,
                **operating_values,
            )

        if self.emergency_latched:
            return self._result(
                self.emergency_command,
                'emergency_release_hold',
                **operating_values,
            )

        if filtered_force >= emergency_force:
            anchor = min(self._safe_command(current_closure), current_closure)
            self.emergency_command = max(
                float(self.initial_closure),
                anchor - cfg.release_step,
            )
            self.emergency_latched = True
            return self._result(
                self.emergency_command,
                'emergency_release',
                **operating_values,
            )

        if scheduled_increment <= 0.0:
            mode = 'hold_empty_cup'
            command = self._safe_command(current_closure)
            closure_limited = False
        elif filtered_force >= force_limit:
            command = self._safe_command(current_closure)
            closure_limited = False
            mode = 'force_limit_hold'
        else:
            anchor = max(self._safe_command(current_closure), current_closure)
            proposed = min(anchor + cfg.close_step, scheduled_closure)
            hard_limit = min(
                cfg.max_closure,
                float(self.initial_closure) + cfg.max_closure_delta,
                current_closure + cfg.max_command_lead,
            )
            command = min(proposed, hard_limit)
            closure_limited = (
                scheduled_closure > hard_limit + 1e-12
                and command >= hard_limit - 1e-12
            )
            if anchor >= scheduled_closure - 1e-12:
                # The load schedule is monotonic in command space: never open
                # automatically when the estimated load later falls.
                command = anchor
                mode = 'hold_load_schedule'
            elif closure_limited:
                mode = 'closure_limit_hold'
            else:
                mode = 'close_for_load'

        return self._result(
            command,
            mode,
            closure_limited=closure_limited,
            **operating_values,
        )
