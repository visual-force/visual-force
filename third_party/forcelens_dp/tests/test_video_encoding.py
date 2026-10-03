import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diffusion_policy.common.video_encoding import (
    VideoEncodingError,
    encode_h264,
)


class VideoEncodingTest(unittest.TestCase):
    def test_encode_h264_removes_intermediate_after_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'source.mp4'
            target = Path(temp_dir) / 'target.mp4'
            source.write_bytes(b'intermediate')

            with mock.patch(
                'diffusion_policy.common.video_encoding.subprocess.run'
            ) as run:
                encode_h264(source, target)

            self.assertFalse(source.exists())
            command = run.call_args.args[0]
            self.assertEqual(command[0], 'ffmpeg')
            self.assertEqual(command[-1], str(target))

    def test_encode_h264_preserves_intermediate_after_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'source.mp4'
            source.write_bytes(b'intermediate')
            error = subprocess.CalledProcessError(
                1,
                ['ffmpeg'],
                stderr=b'encoder failed',
            )

            with mock.patch(
                'diffusion_policy.common.video_encoding.subprocess.run',
                side_effect=error,
            ):
                with self.assertRaisesRegex(VideoEncodingError, 'encoder failed'):
                    encode_h264(source, Path(temp_dir) / 'target.mp4')

            self.assertTrue(source.exists())


if __name__ == '__main__':
    unittest.main()
