"""
Test-time force steering utilities for VisualForce.

This module intentionally does not depend on a particular base policy.  The base
policy can live in another repository and call these classes with:

  1. the latest RGB frame,
  2. the latest gripper mask from SAM/SAM2, and
  3. the action or action chunk proposed by the frozen base policy.

The current VisualForce predictor is image-conditioned only, so the implemented
steering is feedback/scaling rather than true action-gradient guidance.  A
future action-conditioned force model can reuse the same outer API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.model.unet import build_unet


ArrayLike = Union[Sequence[float], np.ndarray]
OUTPUT_SIZE = (256, 256)
INPUT_CHANNELS = {
    "rgb": 3,
    "rgb_edge": 4,
    "edge": 1,
}
EDGE_NORMALIZATIONS = ("none", "nonzero_p95", "finger_bbox_p95")


@dataclass
class ForcePrediction:
    """Force prediction returned by the VisualForce estimator."""

    values: Dict[str, float]
    selected_key: str
    selected_force: float
    control_force: float
    force_mode: str


@dataclass
class ForceSteeringConfig:
    """
    Configuration for action scaling from an estimated force signal.

    Args:
        desired_force: Target force used for steering.  If force_mode is
            "magnitude", this is a magnitude in N.
        force_key: Force channel used for control, e.g. "Fz".
        force_mode: "magnitude" uses abs(force); "signed" uses the raw value.
        deadband: No steering is applied inside this force error band.
        slowdown_band: Force above target is mapped linearly to action scaling.
        stop_margin: If force exceeds desired_force + stop_margin, closing is
            stopped and optional opening can be commanded.
        gripper_index: Action dimension controlling gripper open/close.  If
            None, only motion_indices are scaled.
        close_positive: True if positive gripper command closes the gripper.
            False if negative gripper command closes it.
        min_close_scale: Lower bound for scaling a closing gripper command before
            the stop/open regime.
        open_command: Optional gripper command used when force is far above the
            target.  The sign is inferred from close_positive.
        close_gain: Optional proportional boost when force is below target.
            Keep this at 0.0 for conservative safety-first behavior.
        max_close_command: Optional absolute limit for boosted close commands.
        motion_indices: Optional action dimensions to scale down when force is
            above target, e.g. approach axes that can increase contact pressure.
        min_motion_scale: Lower bound for motion scaling.
    """

    desired_force: float
    force_key: str = "Fz"
    force_mode: str = "magnitude"
    deadband: float = 0.10
    slowdown_band: float = 1.0
    stop_margin: float = 0.75
    gripper_index: Optional[int] = None
    close_positive: bool = True
    min_close_scale: float = 0.0
    open_command: float = 0.0
    close_gain: float = 0.0
    max_close_command: Optional[float] = None
    motion_indices: tuple[int, ...] = ()
    min_motion_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.force_mode not in {"magnitude", "signed"}:
            raise ValueError("force_mode must be 'magnitude' or 'signed'")
        if self.desired_force < 0 and self.force_mode == "magnitude":
            raise ValueError("desired_force must be non-negative in magnitude mode")
        if self.slowdown_band <= 0:
            raise ValueError("slowdown_band must be > 0")
        if self.deadband < 0:
            raise ValueError("deadband must be >= 0")
        if not 0.0 <= self.min_close_scale <= 1.0:
            raise ValueError("min_close_scale must be in [0, 1]")
        if not 0.0 <= self.min_motion_scale <= 1.0:
            raise ValueError("min_motion_scale must be in [0, 1]")


@dataclass
class ForceSteeringResult:
    """Output of the steering step."""

    action: np.ndarray
    base_action: np.ndarray
    predicted_force: ForcePrediction
    force_error: float
    close_scale: float
    motion_scale: float
    metadata: Dict[str, Any]


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("frame must be an RGB array with shape (H, W, 3)")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_bool_mask(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    return arr > 127 if arr.dtype == np.uint8 else arr.astype(bool)


def masked_frame_to_tensor(
    frame_rgb: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    input_mode: str = "rgb",
    edge_normalization: str = "none",
) -> torch.Tensor:
    """Apply a binary gripper mask and return a normalized CHW tensor."""
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"input_mode must be one of {tuple(INPUT_CHANNELS)}, got {input_mode!r}")
    if edge_normalization not in EDGE_NORMALIZATIONS:
        raise ValueError(
            f"edge_normalization must be one of {EDGE_NORMALIZATIONS}, "
            f"got {edge_normalization!r}"
        )

    frame_np = _as_rgb_array(frame_rgb)
    mask_np = _as_bool_mask(mask)
    if frame_np.shape[:2] != mask_np.shape:
        mask_img = Image.fromarray(mask_np.astype(np.uint8) * 255)
        mask_img = mask_img.resize((frame_np.shape[1], frame_np.shape[0]), Image.NEAREST)
        mask_np = np.array(mask_img) > 127

    masked = frame_np.astype(np.float32) * mask_np[:, :, None]
    masked_u8 = np.clip(masked, 0, 255).astype(np.uint8)
    resized = Image.fromarray(masked_u8).resize(
        (int(output_size[1]), int(output_size[0])),
        Image.BILINEAR,
    )
    frame_t = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0)
    frame_t = frame_t.permute(2, 0, 1).float()
    if input_mode == "rgb_edge":
        edge = normalize_edge_tensor(_edge_channel(frame_t), edge_normalization)
        frame_t = torch.cat([frame_t, edge], dim=0)
    elif input_mode == "edge":
        frame_t = normalize_edge_tensor(_edge_channel(frame_t), edge_normalization)
    return frame_t


def _append_edge_channel(frame_t: torch.Tensor) -> torch.Tensor:
    edge = _edge_channel(frame_t)
    return torch.cat([frame_t, edge], dim=0)


def _edge_channel(frame_t: torch.Tensor) -> torch.Tensor:
    gray = (
        0.2989 * frame_t[0:1]
        + 0.5870 * frame_t[1:2]
        + 0.1140 * frame_t[2:3]
    )
    visible = (frame_t.sum(dim=0, keepdim=True) > 1e-6).float()
    interior = -F.max_pool2d(-visible.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=frame_t.dtype,
        device=frame_t.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=frame_t.dtype,
        device=frame_t.device,
    ).view(1, 1, 3, 3)
    gray_b = gray.unsqueeze(0)
    gx = F.conv2d(gray_b, sobel_x, padding=1)
    gy = F.conv2d(gray_b, sobel_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8)[0]
    edge = (edge / 4.0).clamp(0.0, 1.0) * interior
    return edge


def _canonicalize_finger_edges(edge: torch.Tensor) -> torch.Tensor:
    """Place each finger at a fixed image scale while preserving its shape."""
    if edge.ndim != 3 or edge.shape[0] != 1:
        raise ValueError(f"Expected a single CHW edge image, got {tuple(edge.shape)}")
    height, width = edge.shape[-2:]
    canvas = torch.zeros_like(edge)
    target_height = max(1, int(round(height * 0.75)))
    target_max_width = max(1, int(round(width * 0.42)))
    for side, (x0, x1) in enumerate(((0, width // 2), (width // 2, width))):
        region = edge[:, :, x0:x1]
        coordinates = torch.nonzero(region[0] > 1e-6, as_tuple=False)
        if not len(coordinates):
            continue
        y_min, x_min = coordinates.min(dim=0).values.tolist()
        y_max, x_max = coordinates.max(dim=0).values.tolist()
        crop = region[:, y_min : y_max + 1, x_min : x_max + 1]
        crop_height, crop_width = crop.shape[-2:]
        scale = min(target_height / crop_height, target_max_width / crop_width)
        resized_height = max(1, int(round(crop_height * scale)))
        resized_width = max(1, int(round(crop_width * scale)))
        resized = F.interpolate(
            crop.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        center_x = width // 4 if side == 0 else 3 * width // 4
        paste_y = max(0, (height - resized_height) // 2)
        paste_x = max(0, min(width - resized_width, center_x - resized_width // 2))
        canvas[
            :, paste_y : paste_y + resized_height, paste_x : paste_x + resized_width
        ] = resized
    return canvas


def normalize_edge_tensor(edge: torch.Tensor, normalization: str) -> torch.Tensor:
    """Canonicalize/normalize one CHW edge image without background pixels."""
    if normalization == "none":
        return edge
    if normalization == "finger_bbox_p95":
        edge = _canonicalize_finger_edges(edge)
    elif normalization != "nonzero_p95":
        raise ValueError(f"Unsupported edge normalization: {normalization!r}")
    positive = edge[edge > 1e-6]
    if not len(positive):
        return edge
    scale = torch.quantile(positive, 0.95).clamp_min(1e-6)
    return (edge / scale).clamp(0.0, 1.0)


class VisualForceEstimator:
    """Loads a VisualForce checkpoint and predicts force from frame + SAM mask."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device | None = None,
        force_keys: Optional[Iterable[str]] = None,
        force_dropout: float = 0.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.force_keys = list(force_keys or ckpt.get("force_keys", ["Fz"]))
        ckpt_args = ckpt.get("args", {})
        force_pooling = ckpt_args.get("force_pooling", "avg")
        force_spatial_size = ckpt_args.get("force_spatial_size", 4)
        self.input_mode = ckpt_args.get("input_mode", "rgb")
        self.edge_normalization = ckpt_args.get("edge_normalization", "none")
        if self.input_mode not in INPUT_CHANNELS:
            raise ValueError(f"Unsupported checkpoint input_mode: {self.input_mode!r}")
        if self.edge_normalization not in EDGE_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported checkpoint edge_normalization: "
                f"{self.edge_normalization!r}"
            )

        self.model = build_unet(
            in_channels=INPUT_CHANNELS[self.input_mode],
            encoder_channels=(32, 64, 128, 256),
            force_dim=len(self.force_keys),
            force_hidden_dim=256,
            force_dropout=force_dropout,
            force_pooling=force_pooling,
            force_spatial_size=force_spatial_size,
            encoder_only=True,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def predict(
        self,
        frame_rgb: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
        force_key: str = "Fz",
        force_mode: str = "magnitude",
    ) -> ForcePrediction:
        if force_key not in self.force_keys:
            raise ValueError(f"force_key {force_key!r} not in checkpoint keys {self.force_keys}")
        if force_mode not in {"magnitude", "signed"}:
            raise ValueError("force_mode must be 'magnitude' or 'signed'")

        frame_t = masked_frame_to_tensor(
            frame_rgb,
            mask,
            input_mode=self.input_mode,
            edge_normalization=self.edge_normalization,
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(frame_t).detach().cpu().numpy()[0]

        values = {k: float(v) for k, v in zip(self.force_keys, pred)}
        selected = values[force_key]
        control = abs(selected) if force_mode == "magnitude" else selected
        return ForcePrediction(
            values=values,
            selected_key=force_key,
            selected_force=selected,
            control_force=control,
            force_mode=force_mode,
        )

    def predict_from_files(
        self,
        frame_path: str | Path,
        mask_path: str | Path,
        force_key: str = "Fz",
        force_mode: str = "magnitude",
    ) -> ForcePrediction:
        frame = Image.open(frame_path)
        mask = Image.open(mask_path)
        return self.predict(frame, mask, force_key=force_key, force_mode=force_mode)


class ForceActionSteerer:
    """
    Conservative test-time steering for a frozen base policy action.

    The base policy is treated as the task prior.  This class only modifies
    configured action dimensions according to the current estimated force.
    """

    def __init__(self, config: ForceSteeringConfig) -> None:
        self.config = config

    def steer(
        self,
        base_action: ArrayLike,
        predicted_force: ForcePrediction,
    ) -> ForceSteeringResult:
        cfg = self.config
        action = np.array(base_action, dtype=np.float32, copy=True)
        base = action.copy()

        force_error = float(cfg.desired_force - predicted_force.control_force)
        over = max(0.0, -force_error - cfg.deadband)
        under = max(0.0, force_error - cfg.deadband)

        close_scale = 1.0
        motion_scale = 1.0
        stopped_or_opened = False

        if over > 0:
            close_scale = max(cfg.min_close_scale, 1.0 - over / cfg.slowdown_band)
            motion_scale = max(cfg.min_motion_scale, 1.0 - over / cfg.slowdown_band)

        if cfg.gripper_index is not None:
            gi = cfg.gripper_index
            if gi < -action.size or gi >= action.size:
                raise IndexError(f"gripper_index {gi} out of bounds for action size {action.size}")

            gripper_cmd = float(action[gi])
            closing_sign = 1.0 if cfg.close_positive else -1.0
            closing_amount = max(0.0, closing_sign * gripper_cmd)

            if over >= cfg.stop_margin:
                if cfg.open_command > 0:
                    action[gi] = -closing_sign * abs(cfg.open_command)
                elif closing_amount > 0:
                    action[gi] = 0.0
                stopped_or_opened = True
            elif closing_amount > 0:
                action[gi] = closing_sign * closing_amount * close_scale

            if under > 0 and cfg.close_gain > 0:
                boosted = float(action[gi]) + closing_sign * cfg.close_gain * under
                if cfg.max_close_command is not None:
                    max_cmd = abs(cfg.max_close_command)
                    boosted = float(np.clip(boosted, -max_cmd, max_cmd))
                action[gi] = boosted

        for idx in cfg.motion_indices:
            if idx < -action.size or idx >= action.size:
                raise IndexError(f"motion index {idx} out of bounds for action size {action.size}")
            if over > 0:
                action[idx] = action[idx] * motion_scale

        return ForceSteeringResult(
            action=action,
            base_action=base,
            predicted_force=predicted_force,
            force_error=force_error,
            close_scale=close_scale,
            motion_scale=motion_scale,
            metadata={
                "desired_force": cfg.desired_force,
                "deadband": cfg.deadband,
                "over_target": over,
                "under_target": under,
                "stopped_or_opened": stopped_or_opened,
                "gripper_index": cfg.gripper_index,
                "motion_indices": list(cfg.motion_indices),
            },
        )

    def steer_chunk(
        self,
        base_actions: np.ndarray,
        predicted_force: ForcePrediction,
    ) -> ForceSteeringResult:
        """
        Apply the same force feedback to every action in an action chunk.

        base_actions must be shaped (T, action_dim).  The return action keeps the
        same shape; metadata and force values are shared across the chunk.
        """

        arr = np.asarray(base_actions, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("base_actions must have shape (T, action_dim)")

        steered = []
        last_result: Optional[ForceSteeringResult] = None
        for row in arr:
            last_result = self.steer(row, predicted_force)
            steered.append(last_result.action)

        assert last_result is not None
        return ForceSteeringResult(
            action=np.stack(steered, axis=0),
            base_action=arr.copy(),
            predicted_force=predicted_force,
            force_error=last_result.force_error,
            close_scale=last_result.close_scale,
            motion_scale=last_result.motion_scale,
            metadata={**last_result.metadata, "chunk_len": int(arr.shape[0])},
        )


class VisualForceSteeringPipeline:
    """Convenience wrapper combining force prediction and action steering."""

    def __init__(
        self,
        checkpoint: str | Path,
        steering_config: ForceSteeringConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.estimator = VisualForceEstimator(checkpoint, device=device)
        self.steerer = ForceActionSteerer(steering_config)
        self.config = steering_config

    def steer_action(
        self,
        frame_rgb: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
        base_action: ArrayLike,
    ) -> ForceSteeringResult:
        pred = self.estimator.predict(
            frame_rgb,
            mask,
            force_key=self.config.force_key,
            force_mode=self.config.force_mode,
        )
        return self.steerer.steer(base_action, pred)

    def steer_action_chunk(
        self,
        frame_rgb: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
        base_actions: np.ndarray,
    ) -> ForceSteeringResult:
        pred = self.estimator.predict(
            frame_rgb,
            mask,
            force_key=self.config.force_key,
            force_mode=self.config.force_mode,
        )
        return self.steerer.steer_chunk(base_actions, pred)
