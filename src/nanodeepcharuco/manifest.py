from __future__ import annotations

from pathlib import Path

import yaml

from nanodeepcharuco.config import RunConfig
from nanodeepcharuco.video import (
    VideoInfo,
    build_logical_frame_ids,
    inspect_videos,
)


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
