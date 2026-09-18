import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanodeepcharuco.pipeline.run_detection import (
    load_deep_detector,
    run_detection_pipeline,
)


class FakeLayout:
    metadata_dir = Path(
        "/tmp/run/nanodeepcharuco"
    )
    temp_dir = (
        metadata_dir
        / "tmp"
    )

    def detection_path(
        self,
        camera_index,
    ):
        return (
            Path("/tmp/run")
            / f"detection_{camera_index:03d}.yml"
        )


class TestDetectionWorkflow(unittest.TestCase):
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "DeepCharucoDetector"
    )
    def test_load_deep_detector_forwards_configuration(
        self,
        detector_class,
    ):
        config = SimpleNamespace(
            deepcharuco_root="/deep/root",
            deep_checkpoint="/models/deep.ckpt",
            refinenet_checkpoint="/models/refine.ckpt",
            deep_config="/configs/deep.yml",
            device="cpu",
        )

        loaded = object()

        detector_class.return_value.load.return_value = (
            loaded
        )

        result = load_deep_detector(
            config
        )

        self.assertIs(
            result,
            loaded,
        )

        detector_class.assert_called_once_with(
            deepcharuco_root="/deep/root",
            deep_checkpoint="/models/deep.ckpt",
            refinenet_checkpoint="/models/refine.ckpt",
            config_path="/configs/deep.yml",
            device="cpu",
        )

        detector_class.return_value.load.assert_called_once_with()

    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "summarize_calibcam_payload",
        return_value="summary",
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "save_camera_run_payload"
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "run_fixed_sync_detection"
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "expected_charuco_corner_ids",
        return_value=[0, 1, 2],
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "load_deep_detector"
    )
    def test_fixed_detection_returns_written_paths(
        self,
        load_deep,
        expected_ids,
        run_fixed,
        save_payload,
        summarize,
    ):
        config = SimpleNamespace(
            auto_sync=False,
            board="/data/board.npy",
        )

        deep_detector = object()
        left_run = object()
        right_run = object()

        load_deep.return_value = (
            deep_detector
        )

        run_fixed.return_value = [
            left_run,
            right_run,
        ]

        save_payload.side_effect = [
            {"camera": 0},
            {"camera": 1},
        ]

        layout = FakeLayout()

        with redirect_stdout(StringIO()):
            result = run_detection_pipeline(
                config=config,
                layout=layout,
                video_infos=[],
                logical_frame_ids=[
                    10,
                    20,
                ],
            )

        self.assertEqual(
            result,
            [
                Path(
                    "/tmp/run/detection_000.yml"
                ),
                Path(
                    "/tmp/run/detection_001.yml"
                ),
            ],
        )

        run_fixed.assert_called_once_with(
            config=config,
            logical_frame_ids=[
                10,
                20,
            ],
            deep_detector=deep_detector,
            temp_dir=layout.temp_dir,
        )

        self.assertEqual(
            save_payload.call_count,
            2,
        )

    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "summarize_calibcam_payload",
        return_value="summary",
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "save_camera_run_payload"
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "resolve_auto_sync_runs"
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "run_auto_sync_discovery"
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "expected_charuco_corner_ids",
        return_value=[0, 1, 2],
    )
    @patch(
        "nanodeepcharuco.pipeline.run_detection."
        "load_deep_detector"
    )
    def test_auto_sync_detection_returns_written_paths(
        self,
        load_deep,
        expected_ids,
        run_discovery,
        resolve_sync,
        save_payload,
        summarize,
    ):
        config = SimpleNamespace(
            auto_sync=True,
            board="/data/board.npy",
        )

        deep_detector = object()
        left_discovery = object()
        right_discovery = object()
        left_run = object()
        right_run = object()

        load_deep.return_value = deep_detector

        run_discovery.return_value = (
            left_discovery,
            right_discovery,
        )

        resolve_sync.return_value = (
            left_run,
            right_run,
            [],
            [],
        )

        save_payload.side_effect = [
            {"camera": 0},
            {"camera": 1},
        ]

        layout = FakeLayout()
        video_infos = [
            object(),
            object(),
        ]
        logical_ids = [
            10,
            20,
        ]

        with redirect_stdout(StringIO()):
            result = run_detection_pipeline(
                config=config,
                layout=layout,
                video_infos=video_infos,
                logical_frame_ids=logical_ids,
            )

        self.assertEqual(
            result,
            [
                Path(
                    "/tmp/run/detection_000.yml"
                ),
                Path(
                    "/tmp/run/detection_001.yml"
                ),
            ],
        )

        run_discovery.assert_called_once_with(
            config=config,
            run_dir=layout.metadata_dir,
            video_infos=video_infos,
            deep=deep_detector,
        )

        resolve_sync.assert_called_once_with(
            config=config,
            run_dir=layout.metadata_dir,
            calibration_frame_ids=logical_ids,
            left_discovery_run=left_discovery,
            right_discovery_run=right_discovery,
        )

        self.assertEqual(
            save_payload.call_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
