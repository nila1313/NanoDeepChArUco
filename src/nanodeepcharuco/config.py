from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RunConfig:
    videos: list[str]
    board: str
    frames_start: int
    frames_end: Optional[int]
    frames_step: int
    frames_offsets: list[int]
    models: list[str]
    projection: str
    data_path: str
    detect_only: bool
    gamma: float
    deep_self_ransac_px: float

    nano_executable: str
    deepcharuco_root: str
    deep_checkpoint: str
    refinenet_checkpoint: str
    deep_config: str
    device: Optional[str]

    calibcam_python: Optional[str]


def resolve_config(args) -> RunConfig:
    n_cams = len(args.videos)

    frames_offsets = (
        list(args.frames_offsets)
        if args.frames_offsets is not None
        else [0] * n_cams
    )

    models = (
        list(args.models)
        if args.models is not None
        else ["omnidir"] * n_cams
    )

    return RunConfig(
        videos=[
            str(Path(v).expanduser().resolve())
            for v in args.videos
        ],
        board=str(
            Path(args.board).expanduser().resolve()
        ),
        frames_start=args.frames_start,
        frames_end=args.frames_end,
        frames_step=args.frames_step,
        frames_offsets=frames_offsets,
        models=models,
        projection=args.projection,
        data_path=str(
            Path(args.data_path).expanduser().resolve()
        ),
        detect_only=args.detect_only,
        gamma=args.gamma,
        deep_self_ransac_px=args.deep_self_ransac_px,

        nano_executable=str(
            Path(args.nano_executable)
            .expanduser()
            .resolve()
        ),
        deepcharuco_root=str(
            Path(args.deepcharuco_root)
            .expanduser()
            .resolve()
        ),
        deep_checkpoint=str(
            Path(args.deep_checkpoint)
            .expanduser()
            .resolve()
        ),
        refinenet_checkpoint=str(
            Path(args.refinenet_checkpoint)
            .expanduser()
            .resolve()
        ),
        deep_config=str(
            Path(args.deep_config)
            .expanduser()
            .resolve()
        ),
        device=args.device,

        calibcam_python=(
            str(
                Path(args.calibcam_python)
                .expanduser()
                .resolve()
            )
            if args.calibcam_python is not None
            else None
        ),
    )


def prepare_run_directory(config: RunConfig) -> Path:
    run_dir = Path(config.data_path)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir


def save_resolved_config(
    config: RunConfig,
    run_dir: Path,
) -> Path:
    output = run_dir / "resolved_config.yml"

    with open(output, "w") as f:
        yaml.safe_dump(
            asdict(config),
            f,
            sort_keys=False,
        )

    return output
