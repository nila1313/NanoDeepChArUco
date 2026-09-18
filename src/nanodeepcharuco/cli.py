from __future__ import annotations

from nanodeepcharuco.cli_args import (
    build_parser,
    resolve_sync_mode,
    validate_args,
)

from nanodeepcharuco.config import (
    apply_profile,
    apply_runtime_backend,
    resolve_config,
    save_resolved_config,
)

from nanodeepcharuco.run_layout import RunLayout
from nanodeepcharuco.pipeline.run_calibration import (
    run_calibration_pipeline,
)
from nanodeepcharuco.manifest import (
    build_input_manifest,
    save_input_manifest,
)

from nanodeepcharuco.video import (
    inspect_videos,
    build_logical_frame_ids,
)

from nanodeepcharuco.pipeline.run_detection import (
    run_detection_pipeline,
)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args = apply_profile(args)
    args = apply_runtime_backend(args)
    args = resolve_sync_mode(args)

    validate_args(args)

    config = resolve_config(args)

    layout = RunLayout.from_data_path(
        config.data_path
    )
    layout.prepare()

    config_path = save_resolved_config(
        config,
        layout.metadata_dir,
    )

    manifest, _ = build_input_manifest(
        config
    )

    manifest_path = save_input_manifest(
        manifest,
        layout.metadata_dir,
    )

    print("NanoDeepChArUco")
    print("================")
    print("videos        :", config.videos)
    print("board         :", config.board)
    print("frames_start  :", config.frames_start)
    print("frames_end    :", config.frames_end)
    print("frames_step   :", config.frames_step)
    print("frames_offsets:", config.frames_offsets)
    print("models        :", config.models)
    print("projection    :", config.projection)
    print("data_path     :", config.data_path)
    print("detect_only   :", config.detect_only)
    print("config saved  :", config_path)
    print("manifest saved:", manifest_path)

    video_infos = inspect_videos(
        config.videos
    )

    logical_ids = build_logical_frame_ids(
        video_infos=video_infos,
        frames_start=config.frames_start,
        frames_end=config.frames_end,
        frames_step=config.frames_step,
        frames_offsets=config.frames_offsets,
    )

    print()
    print(
        "logical frames :",
        len(logical_ids),
    )

    if len(logical_ids) == 0:
        raise RuntimeError(
            "No logical frames were selected."
        )

    print(
        "logical range  :",
        int(logical_ids[0]),
        "to",
        int(logical_ids[-1]),
    )

    detection_paths = run_detection_pipeline(
        config=config,
        layout=layout,
        video_infos=video_infos,
        logical_frame_ids=logical_ids,
    )

    run_calibration_pipeline(
        config=config,
        layout=layout,
        detection_paths=detection_paths,
    )



if __name__ == "__main__":
    main()
