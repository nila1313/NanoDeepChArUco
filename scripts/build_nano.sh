#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_ENV="nanodeepcharuco_nano_build"
OPENCV_VERSION="4.13.0"

SOURCE_DIR="$ROOT/third_party/aruco_nano/source"
BUILD_DIR="$ROOT/third_party/aruco_nano/build"

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: Conda or Miniforge is required."
    exit 1
}

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -qx "$BUILD_ENV"; then

    echo "Updating Nano build environment..."

    conda install -y \
        -n "$BUILD_ENV" \
        --override-channels \
        -c conda-forge \
        "libopencv=$OPENCV_VERSION" \
        cmake \
        pkg-config \
        make \
        cxx-compiler
else
    echo "Creating Nano build environment..."

    conda create -y \
        -n "$BUILD_ENV" \
        --override-channels \
        -c conda-forge \
        "libopencv=$OPENCV_VERSION" \
        cmake \
        pkg-config \
        make \
        cxx-compiler
fi

ACTUAL_OPENCV="$(
    conda run -n "$BUILD_ENV" \
        pkg-config --modversion opencv4 \
        | awk 'NF {last=$0} END {print last}'
)"

if [[ "$ACTUAL_OPENCV" != "$OPENCV_VERSION" ]]; then
    echo "ERROR: unexpected native OpenCV version"
    echo "expected: $OPENCV_VERSION"
    echo "actual:   $ACTUAL_OPENCV"
    exit 1
fi

PREFIX="$(
    conda run -n "$BUILD_ENV" \
        sh -c 'printf "%s\n" "$CONDA_PREFIX"' \
        | awk 'NF {last=$0} END {print last}'
)"

OPENCV_DIR="$PREFIX/lib/cmake/opencv4"

rm -rf "$BUILD_DIR"

conda run -n "$BUILD_ENV" \
    cmake \
    -S "$SOURCE_DIR" \
    -B "$BUILD_DIR" \
    -DOpenCV_DIR="$OPENCV_DIR" \
    -DCMAKE_BUILD_TYPE=Release

conda run -n "$BUILD_ENV" \
    cmake \
    --build "$BUILD_DIR" \
    --config Release

EXE="$BUILD_DIR/detect_batch"

if [[ ! -x "$EXE" ]]; then
    echo "ERROR: Nano executable was not created."
    exit 1
fi

echo
echo "ArUco Nano ready:"
echo "$EXE"
