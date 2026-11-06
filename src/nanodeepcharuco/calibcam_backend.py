import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from calibcam.detection_backends import DetectionBackend
from calibcamlib import Board


def _normalize_nanodeep_config_paths(config):
    """Normalize NanoDeepCharuco path options for serialization."""
    config = dict(config)
    path_keys = (
        "project_root",
        "nano_executable",
        "deep_checkpoint",
        "refinenet_checkpoint",
        "config_path",
        "work_dir",
    )

    for key in path_keys:
        if config.get(key):
            config[key] = str(config[key])

    return config


def _build_nanodeep_detector(board, config):
    """Initialize the NanoDeepCharuco detector when selected."""
    config = _normalize_nanodeep_config_paths(config)
    required = (
        "project_root",
        "nano_executable",
        "deep_checkpoint",
        "refinenet_checkpoint",
        "work_dir",
    )

    missing = [
        key for key in required
        if not config.get(key)
    ]

    if missing:
        raise ValueError(
            "NanoDeepCharuco backend is missing required options: "
            + ", ".join(missing)
        )

    try:
        from nanodeepcharuco.detection.deep_charuco import (
            DeepCharucoDetector,
        )
        from nanodeepcharuco.detection.nanodeepcharuco import (
            NanoDeepCharucoDetector,
        )
    except ImportError as exc:
        raise ImportError(
            "NanoDeepCharuco backend requires the optional "
            "nanodeepcharuco dependencies"
        ) from exc

    deep = DeepCharucoDetector(
        project_root=config["project_root"],
        deep_checkpoint=config["deep_checkpoint"],
        refinenet_checkpoint=config["refinenet_checkpoint"],
        config_path=config.get("config_path") or None,
        device=config.get("device") or None,
    ).load()

    return NanoDeepCharucoDetector(
        board_path=None,
        nano_executable=config["nano_executable"],
        deep_detector=deep,
        work_dir=config["work_dir"],
        keep_work=config.get("keep_work", False),
        board_params=board.get_board_params(),
    )


def _nanodeep_frame_prefix(video):
    """Return a stable per-video prefix for NanoDeepCharuco work files."""
    video_path = Path(video).expanduser().resolve()
    digest = hashlib.sha256(
        str(video_path).encode("utf-8")
    ).hexdigest()[:10]

    return f"{video_path.stem}_{digest}"


def _nanodeep_corners_to_charuco(corners_by_id):
    """Convert NanoDeepCharuco corner mapping to OpenCV ChArUco arrays."""
    if not corners_by_id:
        return None, None

    corner_ids = sorted(corners_by_id)

    charuco_corners = np.asarray(
        [
            np.asarray(corners_by_id[corner_id], dtype=np.float32).reshape(2)
            for corner_id in corner_ids
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    charuco_ids = np.asarray(
        corner_ids,
        dtype=np.int32,
    ).reshape(-1, 1)

    return charuco_corners, charuco_ids


def create_context(video, board, opts):
    """Create per-video state for NanoDeepCharuco detection."""
    return {
        "detector": _build_nanodeep_detector(
            board,
            opts["detection_opts"]["nanodeepcharuco"],
        ),
        "frame_prefix": _nanodeep_frame_prefix(video),
    }


def detect_frame(frame, frame_idx, context):
    """Detect ChArUco corners in one frame with NanoDeepCharuco."""
    frame_tag = (
        f"{context['frame_prefix']}_"
        f"{int(frame_idx):08d}"
    )

    result = context["detector"].process_frame(
        frame,
        frame_tag,
    )

    return _nanodeep_corners_to_charuco(
        result.corners
    )


class NanoDeepCharucoBackend(DetectionBackend):
    """CalibCam adapter for NanoDeepCharuco detection."""

    def create_context(
        self,
        video: str | Path,
        board: Board,
        opts: Mapping[str, Any],
    ) -> Any:
        return create_context(video, board, opts)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        context: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return detect_frame(frame, frame_idx, context)
