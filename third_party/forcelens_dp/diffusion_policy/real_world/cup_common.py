"""Small numerical helpers shared by the paper-cup workflows."""

from typing import Any, Dict, Tuple

import numpy as np


GRAVITY_M_S2 = 9.80665


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, or ``default`` when unavailable."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    """Convert a value to a one-dimensional finite vector of a fixed size."""

    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return vector


def finite_scalar(value: Any, name: str) -> float:
    """Convert a scalar or one-element array to a finite float."""

    return float(finite_vector(value, 1, name)[0])


def normalized_quaternion(value: Any, name: str = 'arm_quat') -> np.ndarray:
    """Return a normalized xyzw quaternion with a consistent sign."""

    quaternion = finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError(f'{name} must be non-zero')
    quaternion = quaternion / norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def quaternion_distance(a: Any, b: Any) -> float:
    """Return the sign-invariant angular distance between xyzw quaternions."""

    qa = normalized_quaternion(a, 'first quaternion')
    qb = normalized_quaternion(b, 'second quaternion')
    dot = abs(float(np.dot(qa, qb)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def cartesian_pose_error(
    position: Any,
    quaternion: Any,
    target_position: Any,
    target_quaternion: Any,
) -> Tuple[float, float]:
    """Return translation and angular error for two Cartesian poses."""

    position_error = np.linalg.norm(
        finite_vector(position, 3, 'arm_pos')
        - finite_vector(target_position, 3, 'target arm_pos')
    )
    rotation_error = quaternion_distance(quaternion, target_quaternion)
    return float(position_error), rotation_error


def is_stationary(joint_velocity: Any, max_abs_velocity: float) -> bool:
    """Return whether fresh, finite joint velocities are within a limit."""

    velocity = np.asarray(joint_velocity, dtype=np.float64).reshape(-1)
    return bool(
        velocity.size > 0
        and np.isfinite(velocity).all()
        and float(np.max(np.abs(velocity))) <= max_abs_velocity
    )


def measured_hold_action(
    obs: Dict[str, Any],
    *,
    normalize_orientation: bool = False,
) -> Dict[str, np.ndarray]:
    """Build a Cartesian hold action from the latest measured robot state."""

    quaternion = (
        normalized_quaternion(obs['arm_quat'])
        if normalize_orientation
        else finite_vector(obs['arm_quat'], 4, 'arm_quat')
    )
    return {
        'arm_pos': finite_vector(obs['arm_pos'], 3, 'arm_pos').copy(),
        'arm_quat': quaternion.copy(),
        'gripper_pos': finite_vector(
            obs['gripper_pos'], 1, 'gripper_pos'
        ).copy(),
    }
