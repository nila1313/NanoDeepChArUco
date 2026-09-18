from __future__ import annotations

from pathlib import Path

from nanodeepcharuco.detection.hybrid import (
    NanoDeepCharucoDetector,
)


def build_hybrid_detector(
    *,
    config,
    deep_detector,
    work_dir: str | Path,
) -> NanoDeepCharucoDetector:
    """Build the configured Nano + DeepChArUco detector."""

    return NanoDeepCharucoDetector(
        board_path=config.board,
        nano_executable=config.nano_executable,
        deep_detector=deep_detector,
        work_dir=work_dir,
        gamma=config.gamma,
        deep_self_ransac_px=(
            config.deep_self_ransac_px
        ),
    )
