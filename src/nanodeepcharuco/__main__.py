from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nanodeepcharuco",
        description=(
            "Generate CalibCam-compatible stereo ChArUco detections with "
            "OpenCV, ArUco Nano, or the NanoDeepCharuco hybrid detector."
        ),
    )
    parser.add_argument("--videos", nargs=2, required=True, type=existing_file,
                        metavar=("LEFT", "RIGHT"))
    parser.add_argument("--board", required=True, type=existing_file)
    parser.add_argument("--detector", choices=("opencv", "nano", "hybrid"),
                        default="nano")
    parser.add_argument("--frames_start", type=int, default=0)
    parser.add_argument("--frames_end", type=int, default=None)
    parser.add_argument("--frames_step", type=positive_int, default=20)
    parser.add_argument("--frames_offsets", "--frames_offset", nargs=2, type=int,
                        default=(0, 0), dest="frames_offsets",
                        metavar=("LEFT_OFFSET", "RIGHT_OFFSET"))
    parser.add_argument("--models", "--model", nargs=2,
                        default=("omnidir", "omnidir"), dest="models",
                        metavar=("LEFT_MODEL", "RIGHT_MODEL"))
    parser.add_argument("--projection", default="perspective")
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--nano_executable", type=Path,
                        default=PROJECT_ROOT / "third_party/aruco_nano/build/detect_batch")
    parser.add_argument("--deep_checkpoint", type=Path, default=None)
    parser.add_argument("--refinenet_checkpoint", type=Path,
                        default=PROJECT_ROOT / "models/refinenet/refinenet.ckpt")
    parser.add_argument("--deep_config", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--run_calibcam", action="store_true",
                        help="Run CalibCam after generating detection payloads.")
    parser.add_argument("--calibration_single", action="store_true",
                        help="Backward-compatible alias that enables CalibCam.")
    parser.add_argument("--calibration_multi", action="store_true",
                        help="Backward-compatible alias that enables CalibCam.")
    parser.add_argument(
        "--stage1_intrinsics",
        nargs=2,
        type=existing_file,
        default=None,
        metavar=("LEFT_YML", "RIGHT_YML"),
        help=(
            "Stage-1 single-camera calibration files. "
            "When provided, CalibCam keeps these intrinsics fixed "
            "and optimizes stereo extrinsics only."
        ),
    )
    parser.add_argument("--calibcam_python", type=Path,
                        default=Path(sys.executable))
    return parser.parse_args(argv)


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def make_detector(args: argparse.Namespace, side: str, output_root: Path):
    if args.detector == "opencv":
        from nanodeepcharuco.detection.opencv_charuco import OpenCVCharucoDetector
        return OpenCVCharucoDetector(args.board)

    nano_executable = resolved(args.nano_executable)
    work_dir = output_root / "work" / side
    if args.detector == "nano":
        from nanodeepcharuco.detection.nano_charuco import NanoCharucoDetector
        return NanoCharucoDetector(args.board, nano_executable, work_dir)

    if args.deep_checkpoint is None:
        raise ValueError("--deep_checkpoint is required for the hybrid detector")

    if args.deep_config is None:
        raise ValueError("--deep_config is required for the hybrid detector")
    from nanodeepcharuco.detection.deep_charuco import DeepCharucoDetector
    from nanodeepcharuco.detection.hybrid import NanoDeepCharucoDetector
    deep = DeepCharucoDetector(
        project_root=PROJECT_ROOT,
        deep_checkpoint=resolved(args.deep_checkpoint),
        refinenet_checkpoint=resolved(args.refinenet_checkpoint),
        config_path=resolved(args.deep_config),
        device=args.device,
    ).load()
    return NanoDeepCharucoDetector(args.board, nano_executable, deep, work_dir)


def run_calibcam(args: argparse.Namespace, output_root: Path,
                 detection_paths: tuple[Path, Path]) -> None:
    calibcam_output = output_root / "calibcam_output"
    command = [
        str(resolved(args.calibcam_python)), "-m", "calibcam",
        "--videos", *(str(path) for path in args.videos),
        "--detection", *(str(path) for path in detection_paths),
        "--board", str(args.board), "--models", *args.models,
        "--projection", args.projection,
        "--data_path", str(calibcam_output),
    ]
    if args.stage1_intrinsics is not None:
        command.extend([
            "--calibration_single",
            *(str(path) for path in args.stage1_intrinsics),
            "--calibration_multi",
            "--multi_vars",
            "extrinsics",
            "extrinsics",
        ])
    else:
        if args.calibration_single:
            command.append("--calibration_single")
        if args.calibration_multi:
            command.append("--calibration_multi")

    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from nanodeepcharuco.pipeline.detection_runner import (
        build_and_save_payload, run_detector_on_video,
    )
    output_root = resolved(args.data_path)
    inputs_dir = output_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    detectors = (
        make_detector(args, "left", output_root),
        make_detector(args, "right", output_root),
    )
    camera_data = []
    for side, video, detector, offset in zip(
        ("left", "right"), args.videos, detectors, args.frames_offsets
    ):
        camera_data.append(run_detector_on_video(
            video, detector, args.frames_step, side,
            frames_start=args.frames_start, frames_end=args.frames_end,
            frame_offset=offset,
        ))

    detection_paths = (inputs_dir / "detection_000.npy",
                       inputs_dir / "detection_001.npy")
    for data, path, offset in zip(
        camera_data,
        detection_paths,
        args.frames_offsets,
    ):
        build_and_save_payload(
            data,
            path,
            frame_offset=offset,
            frames_start=args.frames_start,
            frames_step=args.frames_step,
        )

    manifest = {
        "pipeline": "NanoDeepCharuco_Basic", "detector": args.detector,
        "videos": [str(path) for path in args.videos], "board": str(args.board),
        "frames_start": args.frames_start, "frames_end": args.frames_end,
        "frames_step": args.frames_step, "frames_offsets": list(args.frames_offsets),
        "models": list(args.models), "projection": args.projection,
        "calibration_mode": (
            "two_stage_extrinsics"
            if args.stage1_intrinsics is not None
            else "standard"
        ),
        "stage1_intrinsics": (
            [str(path) for path in args.stage1_intrinsics]
            if args.stage1_intrinsics is not None
            else None
        ),
        "detections": [str(path) for path in detection_paths],
        "detected_frames": [len(data) for data in camera_data],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Detection output: {inputs_dir}")
    if (
        args.run_calibcam
        or args.calibration_single
        or args.calibration_multi
        or args.stage1_intrinsics is not None
    ):
        run_calibcam(args, output_root, detection_paths)
        print(f"Calibration output: {output_root / 'calibcam_output'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
