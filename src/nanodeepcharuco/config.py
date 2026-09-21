from __future__ import annotations

import os

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_BACKEND_CONFIG = (
    PROJECT_ROOT
    / ".nanodeepcharuco"
    / "backend.yml"
)


def apply_runtime_backend(args):
    """
    Resolve the CalibCam Python executable automatically.

    Resolution order:

    1. Explicit --calibcam_python
    2. NANODEEPCHARUCO_CALIBCAM_PYTHON environment variable
    3. Repository-local runtime backend configuration written
       by setup.sh

    Explicit command-line configuration always wins.
    """

    if getattr(args, "calibcam_python", None) is not None:
        return args

    env_python = os.environ.get(
        "NANODEEPCHARUCO_CALIBCAM_PYTHON"
    )

    if env_python:
        args.calibcam_python = str(
            Path(env_python)
            .expanduser()
            .resolve()
        )
        return args

    if not RUNTIME_BACKEND_CONFIG.is_file():
        return args

    with RUNTIME_BACKEND_CONFIG.open("r") as f:
        runtime = yaml.safe_load(f) or {}

    calibcam_python = runtime.get(
        "calibcam_python"
    )

    if calibcam_python:
        args.calibcam_python = str(
            Path(calibcam_python)
            .expanduser()
            .resolve()
        )

    return args


def apply_profile(args):
    """
    Fill external Nano/Deep asset paths from a profile.

    Explicit command-line paths always win over profile values.

    --profile may be either:
      - a built-in profile name such as "large_7x7"
      - a path to a YAML profile file
    """

    profile_name = getattr(
        args,
        "profile",
        None,
    )

    if profile_name is None:
        return args

    requested = Path(
        str(profile_name)
    ).expanduser()

    if requested.suffix.lower() in {
        ".yaml",
        ".yml",
    } or requested.is_absolute():
        profile_path = requested
    else:
        profile_path = (
            PROJECT_ROOT
            / "configs"
            / "profiles"
            / f"{profile_name}.yaml"
        )

    profile_path = (
        profile_path
        .expanduser()
        .resolve()
    )

    if not profile_path.is_file():
        raise FileNotFoundError(
            "NanoDeepChArUco profile not found: "
            f"{profile_path}"
        )

    with profile_path.open("r") as f:
        profile = yaml.safe_load(f) or {}

    required_keys = [
        "board",
        "nano_executable",
        "deepcharuco_root",
        "deep_checkpoint",
        "refinenet_checkpoint",
        "deep_config",
    ]

    missing = [
        key
        for key in required_keys
        if key not in profile
    ]

    if missing:
        raise ValueError(
            "Profile is missing required fields: "
            + ", ".join(missing)
        )

    for key in required_keys:
        # Explicit CLI argument wins.
        if getattr(args, key, None) is not None:
            continue

        value = Path(
            str(profile[key])
        ).expanduser()

        if not value.is_absolute():
            value = (
                PROJECT_ROOT
                / value
            )

        setattr(
            args,
            key,
            str(value.resolve()),
        )

    args.profile = str(
        profile_path
    )

    return args


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
    calibration_single_paths: Optional[list[str]]
    data_path: str
    detect_only: bool
    gamma: float
    deep_self_ransac_px: float

    auto_sync: bool
    sync_offsets: list[int]
    sync_frame_step: int
    sync_window_size: int
    sync_window_step: int
    sync_min_inlier_ratio: float
    sync_min_ratio_gap: float
    sync_min_persistence: int
    sync_max_gap_frames: int

    nano_executable: str
    deepcharuco_root: str
    deep_checkpoint: str
    refinenet_checkpoint: str
    deep_config: str
    device: Optional[str]
    canonical_luma: bool

    calibcam_python: Optional[str]


def resolve_config(args) -> RunConfig:
    n_cams = len(args.videos)

    frames_offsets = (
        list(args.frames_offsets)
        if args.frames_offsets is not None
        else [0] * n_cams
    )

    calibration_single = getattr(
        args,
        "calibration_single",
        None,
    )

    models = (
        list(args.models)
        if args.models is not None
        else (
            ["omnidir"]
            if calibration_single is not None
            else ["omnidir"] * n_cams
        )
    )

    calibration_single_paths = (
        [
            str(
                Path(path)
                .expanduser()
                .resolve()
            )
            for path in calibration_single
        ]
        if calibration_single is not None
        else None
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
        calibration_single_paths=(
            calibration_single_paths
        ),
        data_path=str(
            Path(args.data_path).expanduser().resolve()
        ),
        detect_only=args.detect_only,
        gamma=args.gamma,
        deep_self_ransac_px=args.deep_self_ransac_px,

        auto_sync=bool(
            getattr(
                args,
                "auto_sync",
                False,
            )
        ),
        sync_offsets=list(
            getattr(
                args,
                "sync_offsets",
                None,
            )
            or range(-3, 4)
        ),
        sync_frame_step=int(
            getattr(
                args,
                "sync_frame_step",
                1,
            )
        ),
        sync_window_size=int(
            getattr(
                args,
                "sync_window_size",
                100,
            )
        ),
        sync_window_step=int(
            getattr(
                args,
                "sync_window_step",
                100,
            )
        ),
        sync_min_inlier_ratio=float(
            getattr(
                args,
                "sync_min_inlier_ratio",
                0.20,
            )
        ),
        sync_min_ratio_gap=float(
            getattr(
                args,
                "sync_min_ratio_gap",
                0.05,
            )
        ),
        sync_min_persistence=int(
            getattr(
                args,
                "sync_min_persistence",
                2,
            )
        ),
        sync_max_gap_frames=int(
            getattr(
                args,
                "sync_max_gap_frames",
                400,
            )
        ),

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
        canonical_luma=bool(
            args.canonical_luma
        ),

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
