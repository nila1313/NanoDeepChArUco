import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nanodeepcharuco.run_layout import RunLayout


class TestRunLayout(unittest.TestCase):
    def test_calibcam_outputs_live_at_data_path_root(self):
        with TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "calibration"

            layout = RunLayout.from_data_path(
                data_path
            )
            layout.prepare()

            self.assertEqual(
                layout.calibcam_data_path,
                data_path.resolve(),
            )

            self.assertEqual(
                layout.detection_path(0),
                data_path.resolve()
                / "detection_000.yml",
            )

            self.assertEqual(
                layout.detection_path(1),
                data_path.resolve()
                / "detection_001.yml",
            )

    def test_nanodeep_metadata_is_namespaced(self):
        with TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "calibration"

            layout = RunLayout.from_data_path(
                data_path
            )
            layout.prepare()

            expected_metadata = (
                data_path.resolve()
                / "nanodeepcharuco"
            )

            self.assertEqual(
                layout.metadata_dir,
                expected_metadata,
            )

            self.assertEqual(
                layout.resolved_config_path,
                expected_metadata
                / "resolved_config.yml",
            )

            self.assertEqual(
                layout.input_manifest_path,
                expected_metadata
                / "input_manifest.yml",
            )

            self.assertEqual(
                layout.temp_dir,
                expected_metadata / "tmp",
            )

            self.assertTrue(
                layout.metadata_dir.is_dir()
            )

            self.assertTrue(
                layout.temp_dir.is_dir()
            )

    def test_negative_camera_index_is_rejected(self):
        with TemporaryDirectory() as tmp:
            layout = RunLayout.from_data_path(
                tmp
            )

            with self.assertRaises(ValueError):
                layout.detection_path(-1)


if __name__ == "__main__":
    unittest.main()
