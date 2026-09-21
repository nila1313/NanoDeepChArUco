import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from nanodeepcharuco.manifest import (
    build_input_manifest,
    sha256_file,
)


class TestManifest(unittest.TestCase):
    @patch(
        "nanodeepcharuco.manifest."
        "build_logical_frame_ids"
    )
    @patch(
        "nanodeepcharuco.manifest."
        "inspect_videos"
    )
    def test_two_stage_intrinsics_are_fingerprinted(
        self,
        inspect_videos,
        build_logical_frame_ids,
    ):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            left_video = root / "left.mp4"
            right_video = root / "right.mp4"

            left_video.write_bytes(
                b"left-video"
            )
            right_video.write_bytes(
                b"right-video"
            )

            left_calibration = (
                root / "left_stage1.yml"
            )
            right_calibration = (
                root / "right_stage1.yml"
            )

            left_calibration.write_bytes(
                b"left-intrinsics"
            )
            right_calibration.write_bytes(
                b"right-intrinsics"
            )

            inspect_videos.return_value = [
                SimpleNamespace(
                    path=str(left_video),
                    frame_count=100,
                    width=1920,
                    height=1080,
                    fps=60.0,
                ),
                SimpleNamespace(
                    path=str(right_video),
                    frame_count=100,
                    width=1920,
                    height=1080,
                    fps=60.0,
                ),
            ]

            build_logical_frame_ids.return_value = [
                0,
                20,
                40,
            ]

            config = SimpleNamespace(
                videos=[
                    str(left_video),
                    str(right_video),
                ],
                frames_start=0,
                frames_end=60,
                frames_step=20,
                frames_offsets=[
                    0,
                    0,
                ],
                calibration_single_paths=[
                    str(left_calibration),
                    str(right_calibration),
                ],
            )

            manifest, _ = build_input_manifest(
                config
            )

            entries = manifest[
                "calibration_single_inputs"
            ]

            self.assertEqual(
                len(entries),
                2,
            )

            self.assertEqual(
                [
                    entry["camera_index"]
                    for entry in entries
                ],
                [
                    0,
                    1,
                ],
            )

            self.assertEqual(
                entries[0]["path"],
                str(
                    left_calibration.resolve()
                ),
            )

            self.assertEqual(
                entries[1]["path"],
                str(
                    right_calibration.resolve()
                ),
            )

            self.assertEqual(
                entries[0]["sha256"],
                sha256_file(
                    left_calibration
                ),
            )

            self.assertEqual(
                entries[1]["sha256"],
                sha256_file(
                    right_calibration
                ),
            )

            self.assertEqual(
                entries[0][
                    "file_size_bytes"
                ],
                left_calibration.stat().st_size,
            )

            self.assertEqual(
                entries[1][
                    "file_size_bytes"
                ],
                right_calibration.stat().st_size,
            )


if __name__ == "__main__":
    unittest.main()
