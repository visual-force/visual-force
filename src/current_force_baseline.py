"""Utilities for calibrated motor-current force baselines.

The ARX SDK reports gripper motor torque as current multiplied by a fixed
motor constant.  These helpers deliberately keep the regression small and
auditable so it can serve as a paper baseline rather than a second learned
vision model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ARX_DM_J4310_TORQUE_CONSTANT_NM_PER_A = 0.424


def torque_to_current(
    torque_nm: np.ndarray,
    torque_constant_nm_per_a: float = ARX_DM_J4310_TORQUE_CONSTANT_NM_PER_A,
) -> np.ndarray:
    """Recover motor current from the ARX SDK's gripper torque field."""
    if not np.isfinite(torque_constant_nm_per_a) or torque_constant_nm_per_a <= 0:
        raise ValueError("torque_constant_nm_per_a must be finite and positive")
    return np.asarray(torque_nm, dtype=np.float64) / torque_constant_nm_per_a


def current_features(
    current_a: np.ndarray,
    gripper_pos: np.ndarray | None = None,
    gripper_vel: np.ndarray | None = None,
    *,
    state_aware: bool = False,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build current-only or state-aware force-regression features."""
    current = np.asarray(current_a, dtype=np.float64).reshape(-1)
    if not state_aware:
        return current[:, None], ("current_a",)

    if gripper_pos is None or gripper_vel is None:
        raise ValueError("state-aware features require gripper position and velocity")
    pos = np.asarray(gripper_pos, dtype=np.float64).reshape(-1)
    vel = np.asarray(gripper_vel, dtype=np.float64).reshape(-1)
    if not (len(current) == len(pos) == len(vel)):
        raise ValueError("current, position, and velocity must have equal lengths")

    features = np.column_stack(
        [
            current,
            pos,
            vel,
            np.abs(vel),
            current * pos,
            current * vel,
            np.square(current),
            np.square(pos),
        ]
    )
    names = (
        "current_a",
        "gripper_pos",
        "gripper_vel",
        "abs_gripper_vel",
        "current_x_pos",
        "current_x_vel",
        "current_sq",
        "position_sq",
    )
    return features, names


@dataclass(frozen=True)
class RidgeRegressor:
    """Standardized ridge regression with an unregularized intercept."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        target: np.ndarray,
        feature_names: tuple[str, ...],
        *,
        alpha: float = 1e-3,
    ) -> "RidgeRegressor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != len(y):
            raise ValueError("features must be [samples, features] and match target")
        if x.shape[1] != len(feature_names):
            raise ValueError("feature_names does not match the feature matrix")
        if len(y) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training data must be non-empty and finite")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return cls(tuple(feature_names), mean, scale, coefficients, float(alpha))

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match fitted features")
        design = np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
        return design @ self.coefficients

    def as_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "alpha": self.alpha,
        }


def exponential_moving_average(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    time_constant_s: float,
) -> np.ndarray:
    """Causal EMA for irregular timestamps; zero time constant is identity."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have equal lengths")
    if len(values) == 0 or time_constant_s <= 0:
        return values.copy()
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be non-decreasing")

    filtered = values.copy()
    for index in range(1, len(values)):
        dt = max(0.0, timestamps[index] - timestamps[index - 1])
        weight = 1.0 - np.exp(-dt / time_constant_s)
        filtered[index] = filtered[index - 1] + weight * (
            values[index] - filtered[index - 1]
        )
    return filtered


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    timestamps_s: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Accuracy and temporal residual metrics for one prediction stream."""
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("target and prediction must be non-empty and equally sized")
    residual = pred - y
    ss_res = float(np.sum(np.square(residual)))
    centered = y - y.mean()
    ss_tot = float(np.sum(np.square(centered)))
    if np.std(y) > 1e-12 and np.std(pred) > 1e-12:
        pearson = float(np.corrcoef(y, pred)[0, 1])
    else:
        pearson = float("nan")

    result: dict[str, float | int] = {
        "n": int(len(y)),
        "mae_n": float(np.mean(np.abs(residual))),
        "rmse_n": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_n": float(np.mean(residual)),
        "residual_std_n": float(np.std(residual)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "pearson_r": pearson,
    }

    if len(y) > 1:
        delta_residual = np.diff(residual)
        result["delta_residual_rms_n"] = float(
            np.sqrt(np.mean(np.square(delta_residual)))
        )
        result["prediction_delta_std_n"] = float(np.std(np.diff(pred)))
        result["target_delta_std_n"] = float(np.std(np.diff(y)))
        if timestamps_s is not None:
            timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
            if len(timestamps) != len(y):
                raise ValueError("timestamps must match target")
            dt = np.diff(timestamps)
            valid = dt > 1e-9
            result["residual_slew_rms_n_per_s"] = (
                float(np.sqrt(np.mean(np.square(delta_residual[valid] / dt[valid]))))
                if valid.any()
                else float("nan")
            )
    else:
        result.update(
            {
                "delta_residual_rms_n": float("nan"),
                "prediction_delta_std_n": float("nan"),
                "target_delta_std_n": float("nan"),
                "residual_slew_rms_n_per_s": float("nan"),
            }
        )
    return result
