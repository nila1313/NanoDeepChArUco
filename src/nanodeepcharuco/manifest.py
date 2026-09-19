from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from nanodeepcharuco.config import RunConfig
from nanodeepcharuco.video import (
    VideoInfo,
    build_logical_frame_ids,
    inspect_videos,
)



HASH_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(
    path: str | Path,
) -> str:
    """
    Return the SHA256 digest of a file.

    The file is streamed in chunks so large calibration
    videos do not need to be loaded into memory.
    """
    file_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    digest = hashlib.sha256()

    with file_path.open("rb") as stream:
        while True:
            chunk = stream.read(
                HASH_CHUNK_BYTES
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def build_input_manifest(
    config: RunConfig,
) -> tuple[dict, list[VideoInfo]]:
    infos = inspect_videos(
        config.videos
    )

    frame_ids = build_logical_frame_ids(
        video_infos=infos,
        frames_start=config.frames_start,
        frames_end=config.frames_end,
        frames_step=config.frames_step,
        frames_offsets=config.frames_offsets,
    )

    manifest = {
        "videos": [
            {
                "camera_index": i,
                "path": info.path,
                "file_size_bytes": int(
                    Path(info.path).stat().st_size
                ),
                "sha256": sha256_file(
                    info.path
                ),
                "frame_count": info.frame_count,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frame_offset": int(
                    config.frames_offsets[i]
                ),
            }
            for i, info in enumerate(infos)
        ],
        "logical_frames": {
            "count": int(
                len(frame_ids)
            ),
            "first": int(
                frame_ids[0]
            ),
            "last": int(
                frame_ids[-1]
            ),
            "step": int(
                config.frames_step
            ),
            "frame_ids": [
                int(x)
                for x in frame_ids
            ],
        },
    }

    calibration_single_paths = getattr(
        config,
        "calibration_single_paths",
        None,
    )

    if calibration_single_paths is not None:
        manifest["calibration_single_inputs"] = [
            {
                "camera_index": camera_index,
                "path": str(
                    Path(calibration_path)
                    .expanduser()
                    .resolve()
                ),
                "file_size_bytes": int(
                    Path(calibration_path)
                    .expanduser()
                    .resolve()
                    .stat()
                    .st_size
                ),
                "sha256": sha256_file(
                    calibration_path
                ),
            }
            for camera_index, calibration_path
            in enumerate(
                calibration_single_paths
            )
        ]

    return manifest, infos


def save_input_manifest(
    manifest: dict,
    run_dir: Path,
) -> Path:
    output = (
        run_dir
        / "input_manifest.yml"
    )

    with open(output, "w") as f:
        yaml.safe_dump(
            manifest,
            f,
            sort_keys=False,
        )

    return output
