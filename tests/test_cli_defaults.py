import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from argparse import Namespace

from nanodeepcharuco.config import resolve_config

from nanodeepcharuco.cli_args import validate_args

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


    def test_two_stage_calibration_single_paths_parse(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--videos",
                "left.mp4",
                "right.mp4",
                "--board",
                "board.npy",
                "--data_path",
                "runs/test",
                "--calibration_single",
                "left_stage1.yml",
                "right_stage1.yml",
                "--models",
                "omnidir",
            ]
        )

        self.assertEqual(
            args.calibration_single,
            [
                "left_stage1.yml",
                "right_stage1.yml",
            ],
        )

        self.assertEqual(
            args.models,
            ["omnidir"],
        )


    def test_two_stage_paths_survive_config_resolution(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--videos",
                "left.mp4",
                "right.mp4",
                "--board",
                "small_board.npy",
                "--data_path",
                "runs/test",
                "--calibration_single",
                "left_stage1.yml",
                "right_stage1.yml",
                "--models",
                "omnidir",
                "--projection",
                "fisheye_equidistant",
            ]
        )

        # Normally provided by a profile before resolve_config().
        args.nano_executable = "nano"
        args.deepcharuco_root = "deepcharuco"
        args.deep_checkpoint = "detector.ckpt"
        args.refinenet_checkpoint = "refinenet.ckpt"
        args.deep_config = "deep.yaml"

        config = resolve_config(args)

        self.assertEqual(
            config.calibration_single_paths,
            [
                str(
                    Path(
                        "left_stage1.yml"
                    ).resolve()
                ),
                str(
                    Path(
                        "right_stage1.yml"
                    ).resolve()
                ),
            ],
        )

        self.assertEqual(
            config.models,
            ["omnidir"],
        )

        self.assertEqual(
            config.projection,
            "fisheye_equidistant",
        )


    def test_two_stage_validation_accepts_shared_model(self):
        parser = build_parser()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            left_video = root / "left.mp4"
            right_video = root / "right.mp4"
            board = root / "small_board.npy"

            left_calibration = (
                root / "left_stage1.yml"
            )
            right_calibration = (
                root / "right_stage1.yml"
            )

            nano = root / "detect_batch"
            deep_root = root / "deepcharuco"
            deep_checkpoint = (
                root / "detector.ckpt"
            )
            refinenet_checkpoint = (
                root / "refinenet.ckpt"
            )
            deep_config = root / "deep.yaml"

            calibcam_python = (
                root / "calibcam_python"
            )

            for file_path in [
                left_video,
                right_video,
                board,
                left_calibration,
                right_calibration,
                nano,
                deep_checkpoint,
                refinenet_checkpoint,
                deep_config,
                calibcam_python,
            ]:
                file_path.touch()

            deep_root.mkdir()

            nano.chmod(0o755)
            calibcam_python.chmod(0o755)

            args = parser.parse_args(
                [
                    "--videos",
                    str(left_video),
                    str(right_video),
                    "--board",
                    str(board),
                    "--data_path",
                    str(root / "run"),
                    "--calibration_single",
                    str(left_calibration),
                    str(right_calibration),
                    "--models",
                    "omnidir",
                    "--projection",
                    "fisheye_equidistant",
                ]
            )

            args.nano_executable = str(nano)
            args.deepcharuco_root = str(deep_root)
            args.deep_checkpoint = str(
                deep_checkpoint
            )
            args.refinenet_checkpoint = str(
                refinenet_checkpoint
            )
            args.deep_config = str(deep_config)
            args.calibcam_python = str(
                calibcam_python
            )

            args = resolve_sync_mode(args)

            # Must not raise.
            validate_args(args)

    def test_two_stage_validation_rejects_wrong_intrinsic_count(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--videos",
                "left.mp4",
                "right.mp4",
                "--board",
                "small_board.npy",
                "--data_path",
                "runs/test",
                "--calibration_single",
                "left_stage1.yml",
                "--models",
                "omnidir",
            ]
        )

        args.nano_executable = "nano"
        args.deepcharuco_root = "deepcharuco"
        args.deep_checkpoint = "detector.ckpt"
        args.refinenet_checkpoint = "refinenet.ckpt"
        args.deep_config = "deep.yaml"
        args.calibcam_python = "/opt/calibcam/python"

        args = resolve_sync_mode(args)

        with self.assertRaisesRegex(
            ValueError,
            "one calibration file per video",
        ):
            validate_args(args)


    def test_normal_stereo_still_rejects_single_model(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--videos",
                "left.mp4",
                "right.mp4",
                "--board",
                "board.npy",
                "--data_path",
                "runs/test",
                "--models",
                "omnidir",
            ]
        )

        args.nano_executable = "nano"
        args.deepcharuco_root = "deepcharuco"
        args.deep_checkpoint = "detector.ckpt"
        args.refinenet_checkpoint = "refinenet.ckpt"
        args.deep_config = "deep.yaml"
        args.calibcam_python = "/opt/calibcam/python"

        args = resolve_sync_mode(args)

        with self.assertRaisesRegex(
            ValueError,
            "one value per video",
        ):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
