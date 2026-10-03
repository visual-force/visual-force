"""Shared helpers for finalizing OpenCV recordings with FFmpeg."""

import shutil
import subprocess
from pathlib import Path
from typing import Union


class VideoEncodingError(RuntimeError):
    """Raised when FFmpeg cannot transcode a recording."""


def h264_encoder_available() -> bool:
    return shutil.which('ffmpeg') is not None


def encode_h264(
    source: Union[str, Path],
    target: Union[str, Path],
) -> None:
    """Transcode one video to browser-compatible H.264 and remove its source."""

    source = Path(source)
    target = Path(target)
    command = [
        'ffmpeg',
        '-y',
        '-i',
        str(source),
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        str(target),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        if isinstance(exc.stderr, bytes):
            detail = exc.stderr.decode('utf-8', errors='replace').strip()
        else:
            detail = str(exc.stderr or exc).strip()
        raise VideoEncodingError(detail) from exc
    source.unlink(missing_ok=True)
