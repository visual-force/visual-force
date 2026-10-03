"""Render a cup-adaptation recording with a legible telemetry dashboard.

The physical jaw motion in this experiment is only a few millimetres. This
renderer keeps the robot video untouched and adds synchronized, explicitly
labelled estimates beside it so the adaptive response is visible without
increasing the commanded grip force.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2 as cv
import numpy as np

from diffusion_policy.real_world.cup_common import GRAVITY_M_S2, finite_float


DEFAULT_ROLLOUT_ROOT = Path(
    "tts_rollouts/cup_adaptive"
)
ARX_OPEN_WIDTH_MM = 88.0
CANVAS_SIZE = (1280, 720)
VIDEO_AREA_WIDTH = 900


def _first_finite(rows: Iterable[Dict[str, str]], key: str) -> float:
    for row in rows:
        value = finite_float(row.get(key), math.nan)
        if math.isfinite(value):
            return value
    return math.nan


def _first_calibrated_closure(rows: Sequence[Dict[str, str]]) -> float:
    for row in rows:
        if row.get("calibrated") == "True":
            value = finite_float(row.get("measured_closure"), math.nan)
            if math.isfinite(value):
                return value
    return _first_finite(rows, "measured_closure")


def _forward_fill(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    values = []
    last = math.nan
    for row in rows:
        value = finite_float(row.get(key), math.nan)
        if math.isfinite(value):
            last = value
        values.append(last)
    return values


def _latest_run(root: Path) -> Path:
    candidates = [
        path
        for path in root.glob("cup_adaptive_*")
        if (path / "adaptive_grip.csv").is_file()
        and any(
            (path / name).is_file()
            for name in ("demo_h264.mp4", ".demo_recording.mp4")
        )
    ]
    if not candidates:
        raise FileNotFoundError(f"no completed cup recordings found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_run(value: str, root: Path) -> Path:
    run_dir = _latest_run(root) if value == "latest" else Path(value).expanduser()
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"cup run directory does not exist: {run_dir}")
    return run_dir


def _put_text(
    image: np.ndarray,
    text: str,
    origin,
    *,
    scale: float = 0.55,
    color=(236, 241, 247),
    thickness: int = 1,
) -> None:
    cv.putText(
        image,
        str(text),
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv.LINE_AA,
    )


def _bar(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    value: float,
    maximum: float,
    color,
    marker: Optional[float] = None,
) -> None:
    height = 15
    cv.rectangle(image, (x, y), (x + width, y + height), (55, 63, 73), -1)
    if math.isfinite(value) and maximum > 0.0:
        fraction = float(np.clip(value / maximum, 0.0, 1.0))
        cv.rectangle(
            image,
            (x, y),
            (x + int(round(width * fraction)), y + height),
            color,
            -1,
        )
    if marker is not None and math.isfinite(marker) and maximum > 0.0:
        marker_x = x + int(round(width * np.clip(marker / maximum, 0.0, 1.0)))
        cv.line(
            image,
            (marker_x, y - 3),
            (marker_x, y + height + 3),
            (0, 190, 255),
            2,
        )
    cv.rectangle(image, (x, y), (x + width, y + height), (126, 137, 150), 1)


def _status(mode: str):
    if mode.startswith("emergency_release"):
        return "SAFETY RELEASE", (48, 64, 230)
    if mode == "force_limit_hold":
        return "FORCE-LIMIT HOLD", (0, 176, 255)
    if mode == "close_for_load":
        return "TIGHTENING GRIP", (86, 220, 120)
    if mode == "hold_load_schedule":
        return "ADAPTIVE GRIP HELD", (86, 220, 120)
    if mode == "calibration_complete":
        return "EMPTY BASELINE READY", (235, 190, 72)
    if mode.startswith("calibrat"):
        return "CALIBRATING EMPTY CUP", (235, 190, 72)
    if mode in {"hold_empty_cup", "monitor_only"}:
        return "EMPTY CUP / READY", (235, 190, 72)
    if mode.endswith("hold") or "fault" in mode:
        return "CONTROLLER HOLD", (0, 176, 255)
    return mode.replace("_", " ").upper() or "WAITING", (160, 169, 180)


class DashboardData:
    def __init__(self, rows: Sequence[Dict[str, str]]):
        self.rows = rows
        self.initial_closure = _first_calibrated_closure(rows)
        self.force_baseline = _first_finite(rows, "baseline_force_n")
        self.load = _forward_fill(rows, "filtered_added_load_n")
        self.force = _forward_fill(rows, "filtered_force_n")
        self.measured = _forward_fill(rows, "measured_closure")
        self.command = _forward_fill(rows, "command_closure")
        self.force_limit = _forward_fill(rows, "force_limit_n")
        self.emergency_force = _forward_fill(rows, "emergency_force_n")

    def frame_values(self, index: int):
        load_n = self.load[index]
        force_n = self.force[index]
        measured_mm = (
            self.measured[index] - self.initial_closure
        ) * ARX_OPEN_WIDTH_MM
        command_mm = (
            self.command[index] - self.initial_closure
        ) * ARX_OPEN_WIDTH_MM
        force_rise = force_n - self.force_baseline
        force_limit_rise = self.force_limit[index] - self.force_baseline
        emergency_rise = self.emergency_force[index] - self.force_baseline
        return {
            "load_n": load_n,
            "water_g": max(0.0, load_n) * 1000.0 / GRAVITY_M_S2,
            "measured_mm": measured_mm,
            "command_mm": command_mm,
            "force_n": force_n,
            "force_rise": force_rise,
            "force_limit_rise": force_limit_rise,
            "emergency_rise": emergency_rise,
        }


def _format(value: float, pattern: str, unavailable: str = "--") -> str:
    return pattern.format(value) if math.isfinite(value) else unavailable


def _fit_video(frame: np.ndarray, canvas: np.ndarray) -> None:
    source_h, source_w = frame.shape[:2]
    target_h = CANVAS_SIZE[1]
    scale = min(VIDEO_AREA_WIDTH / source_w, target_h / source_h)
    width = max(1, int(round(source_w * scale)))
    height = max(1, int(round(source_h * scale)))
    resized = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
    x = (VIDEO_AREA_WIDTH - width) // 2
    y = (target_h - height) // 2
    canvas[y : y + height, x : x + width] = resized


def _draw_dashboard(
    frame: np.ndarray,
    row: Dict[str, str],
    values: Dict[str, float],
) -> np.ndarray:
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), (20, 25, 31), np.uint8)
    _fit_video(frame, canvas)
    x = VIDEO_AREA_WIDTH + 22
    width = CANVAS_SIZE[0] - x - 22

    _put_text(canvas, "CUP ADAPTATION", (x, 37), scale=0.72, thickness=2)
    status, status_color = _status(row.get("mode", ""))
    cv.rectangle(canvas, (x, 55), (x + width, 93), status_color, -1)
    _put_text(
        canvas,
        status,
        (x + 12, 81),
        scale=0.57,
        color=(12, 18, 22),
        thickness=2,
    )

    _put_text(
        canvas,
        "ARX ADDED-LOAD ESTIMATE",
        (x, 128),
        scale=0.46,
        color=(166, 178, 191),
    )
    _put_text(
        canvas,
        _format(values["load_n"], "{:+.2f} N"),
        (x, 165),
        scale=0.85,
        color=(235, 190, 72),
        thickness=2,
    )
    _put_text(
        canvas,
        _format(values["water_g"], "about {:.0f} g water"),
        (x, 190),
        scale=0.48,
        color=(190, 199, 209),
    )
    _bar(
        image=canvas,
        x=x,
        y=204,
        width=width,
        value=max(0.0, values["load_n"]),
        maximum=3.0,
        color=(235, 190, 72),
    )
    _put_text(canvas, "0", (x, 239), scale=0.38, color=(135, 147, 160))
    _put_text(
        canvas,
        "3 N",
        (x + width - 30, 239),
        scale=0.38,
        color=(135, 147, 160),
    )

    _put_text(
        canvas,
        "GRIP TIGHTENING FROM EMPTY",
        (x, 278),
        scale=0.46,
        color=(166, 178, 191),
    )
    _put_text(
        canvas,
        _format(values["measured_mm"], "{:+.2f} mm measured"),
        (x, 313),
        scale=0.67,
        color=(86, 220, 120),
        thickness=2,
    )
    _bar(
        image=canvas,
        x=x,
        y=328,
        width=width,
        value=max(0.0, values["measured_mm"]),
        maximum=3.0,
        color=(86, 220, 120),
    )
    _put_text(
        canvas,
        _format(values["command_mm"], "target {:+.2f} mm"),
        (x, 374),
        scale=0.52,
        color=(100, 206, 245),
    )
    _bar(
        image=canvas,
        x=x,
        y=388,
        width=width,
        value=max(0.0, values["command_mm"]),
        maximum=3.0,
        color=(100, 206, 245),
    )
    _put_text(canvas, "0", (x, 423), scale=0.38, color=(135, 147, 160))
    _put_text(
        canvas,
        "3 mm",
        (x + width - 45, 423),
        scale=0.38,
        color=(135, 147, 160),
    )

    _put_text(
        canvas,
        "VISUALFORCE ESTIMATE",
        (x, 461),
        scale=0.46,
        color=(166, 178, 191),
    )
    _put_text(
        canvas,
        _format(values["force_n"], "{:.2f} N total"),
        (x, 496),
        scale=0.67,
        color=(105, 164, 255),
        thickness=2,
    )
    _put_text(
        canvas,
        _format(values["force_rise"], "{:+.2f} N from empty"),
        (x, 522),
        scale=0.48,
        color=(190, 199, 209),
    )
    force_scale = values["emergency_rise"]
    if not math.isfinite(force_scale) or force_scale <= 0.0:
        force_scale = 3.5
    _bar(
        image=canvas,
        x=x,
        y=537,
        width=width,
        value=max(0.0, values["force_rise"]),
        maximum=force_scale,
        color=(105, 164, 255),
        marker=values["force_limit_rise"],
    )
    _put_text(canvas, "empty", (x, 572), scale=0.38, color=(135, 147, 160))
    _put_text(
        canvas,
        "limit",
        (x + width - 39, 572),
        scale=0.38,
        color=(0, 190, 255),
    )

    cv.line(canvas, (x, 594), (x + width, 594), (65, 74, 85), 1)
    _put_text(
        canvas,
        "LOAD UP",
        (x + 2, 628),
        scale=0.63,
        color=(235, 190, 72),
        thickness=2,
    )
    _put_text(
        canvas,
        ">",
        (x + 126, 628),
        scale=0.75,
        color=(180, 190, 200),
        thickness=2,
    )
    _put_text(
        canvas,
        "GRIP UP",
        (x + 168, 628),
        scale=0.63,
        color=(86, 220, 120),
        thickness=2,
    )
    _put_text(
        canvas,
        "Estimates, not calibrated force/weight sensors",
        (x, 677),
        scale=0.38,
        color=(135, 147, 160),
    )
    _put_text(
        canvas,
        "Jaw travel uses the ARX 88 mm width mapping",
        (x, 699),
        scale=0.38,
        color=(135, 147, 160),
    )
    return canvas


def render(run_dir: Path, video_path: Path, output_path: Path) -> None:
    csv_path = run_dir / "adaptive_grip.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"telemetry CSV not found: {csv_path}")
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"telemetry CSV contains no rows: {csv_path}")

    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = capture.get(cv.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 10.0
    dashboard = DashboardData(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cup-dashboard-") as temp_dir:
        intermediate = Path(temp_dir) / "dashboard_mp4v.mp4"
        writer = cv.VideoWriter(
            str(intermediate),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            CANVAS_SIZE,
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"could not create intermediate video: {intermediate}")

        written = 0
        try:
            for index, row in enumerate(rows):
                ok, frame = capture.read()
                if not ok:
                    break
                rendered = _draw_dashboard(frame, row, dashboard.frame_values(index))
                writer.write(rendered)
                written += 1
        finally:
            capture.release()
            writer.release()
        if written == 0:
            raise RuntimeError("no synchronized video/telemetry frames were available")
        if written != len(rows):
            print(
                f"Warning: rendered {written} frames for {len(rows)} telemetry "
                "rows; the input video ended first."
            )

        if shutil.which("ffmpeg") is None:
            shutil.copy2(intermediate, output_path)
            print(f"Dashboard video saved without H.264 conversion: {output_path}")
            return
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(intermediate),
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg dashboard conversion failed: {detail}") from exc
    print(f"Cup dashboard video saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add a synchronized telemetry dashboard to a cup demo video."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="latest",
        help="cup run directory, or 'latest' (default)",
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=DEFAULT_ROLLOUT_ROOT,
        help="root searched when run_dir is 'latest'",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="frame-aligned source video (default: the run's side video)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (default: RUN_DIR/demo_dashboard_h264.mp4)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = _resolve_run(args.run_dir, args.rollout_root.expanduser().resolve())
    if args.video is None:
        candidates = (
            run_dir / "demo_h264.mp4",
            run_dir / ".demo_recording.mp4",
        )
        video_path = next((path for path in candidates if path.is_file()), None)
        if video_path is None:
            raise FileNotFoundError(f"no side-view video found in {run_dir}")
    else:
        video_path = args.video.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "demo_dashboard_h264.mp4"
    )
    if output_path.resolve() == video_path.resolve():
        raise ValueError("output path must differ from the source video")
    render(run_dir, video_path, output_path)


if __name__ == "__main__":
    main()
