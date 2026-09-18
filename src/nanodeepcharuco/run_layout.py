from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Filesystem layout for one NanoDeepChArUco calibration run."""

    root: Path

    @classmethod
    def from_data_path(
        cls,
        data_path: str | Path,
    ) -> "RunLayout":
        root = (
            Path(data_path)
            .expanduser()
            .resolve()
        )
        return cls(root=root)

    @property
    def metadata_dir(self) -> Path:
        """NanoDeepChArUco-specific metadata and temporary files."""
        return self.root / "nanodeepcharuco"

    @property
    def temp_dir(self) -> Path:
        return self.metadata_dir / "tmp"

    @property
    def calibcam_data_path(self) -> Path:
        """Directory exposed to the CalibCam backend."""
        return self.root

    @property
    def resolved_config_path(self) -> Path:
        return self.metadata_dir / "resolved_config.yml"

    @property
    def input_manifest_path(self) -> Path:
        return self.metadata_dir / "input_manifest.yml"

    def detection_path(
        self,
        camera_index: int,
    ) -> Path:
        if camera_index < 0:
            raise ValueError(
                "camera_index must be non-negative."
            )

        return (
            self.root
            / f"detection_{camera_index:03d}.yml"
        )

    def prepare(self) -> None:
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
