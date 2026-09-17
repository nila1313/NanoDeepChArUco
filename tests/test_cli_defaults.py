import unittest
from argparse import Namespace

from nanodeepcharuco.cli import (
    build_parser,
    resolve_sync_mode,
)


class TestCliDefaults(unittest.TestCase):
    def test_canonical_luma_is_default(self):
        parser = build_parser()
        self.assertTrue(
            parser.get_default("canonical_luma")
        )

    def test_no_canonical_luma_opt_out(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--videos",
                "camera.mp4",
                "--board",
                "board.npy",
                "--data_path",
                "runs/test",
                "--no_canonical_luma",
            ]
        )
        self.assertFalse(args.canonical_luma)

    def test_stereo_defaults_to_auto_sync(self):
        args = Namespace(
            videos=["left.mp4", "right.mp4"],
            frames_offsets=None,
            auto_sync=False,
            fixed_sync=False,
        )

        args = resolve_sync_mode(args)

        self.assertTrue(args.auto_sync)
        self.assertEqual(
            args.frames_offsets,
            [0, 0],
        )

    def test_explicit_offsets_select_fixed_sync(self):
        args = Namespace(
            videos=["left.mp4", "right.mp4"],
            frames_offsets=[0, 1],
            auto_sync=False,
            fixed_sync=False,
        )

        args = resolve_sync_mode(args)

        self.assertFalse(args.auto_sync)
        self.assertEqual(
            args.frames_offsets,
            [0, 1],
        )

    def test_fixed_sync_override(self):
        args = Namespace(
            videos=["left.mp4", "right.mp4"],
            frames_offsets=None,
            auto_sync=False,
            fixed_sync=True,
        )

        args = resolve_sync_mode(args)

        self.assertFalse(args.auto_sync)

    def test_single_camera_does_not_auto_sync(self):
        args = Namespace(
            videos=["camera.mp4"],
            frames_offsets=None,
            auto_sync=False,
            fixed_sync=False,
        )

        args = resolve_sync_mode(args)

        self.assertFalse(args.auto_sync)

    def test_conflicting_sync_modes_fail(self):
        args = Namespace(
            videos=["left.mp4", "right.mp4"],
            frames_offsets=None,
            auto_sync=True,
            fixed_sync=True,
        )

        with self.assertRaises(ValueError):
            resolve_sync_mode(args)


if __name__ == "__main__":
    unittest.main()
