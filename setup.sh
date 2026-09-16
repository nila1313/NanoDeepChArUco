#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROJECT_ENV="nanodeepcharuco"
CALIBCAM_ENV="calibcam_baseline_420"

NANO_BUILD_ENV="nanodeepcharuco_nano_build"
NANO_OPENCV_VERSION="4.13.0"

DEEPCHARUCO_COMMIT="37d569fc582b790843dce408c14556747927711c"

CALIBCAM_REPO="https://github.com/bbo-lab/calibcam.git"
CALIBCAM_COMMIT="f51aa60961b6a9a7abea8b7c377dd5dfa7599f50"
CALIBCAM_VERSION="4.2.0"

CALIBCAMLIB_REPO="https://github.com/bbo-lab/calibcamlib.git"
CALIBCAMLIB_COMMIT="5e3888b0647a4e20cf54ec54734ec6ed770a2169"
CALIBCAMLIB_VERSION="0.5.2"

die() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

have_conda_env() {
    conda env list \
        | awk 'NF && $1 !~ /^#/ {print $1}' \
        | grep -qx "$1"
}


# ----------------------------------------------------------------------
# Basic tools
# ----------------------------------------------------------------------

section "1/9 Checking basic tools"

command -v git >/dev/null 2>&1 \
    || die "git is required."

command -v conda >/dev/null 2>&1 \
    || die "Conda/Miniforge is required. Install Miniforge first."

OS="$(uname -s)"

command -v c++ >/dev/null 2>&1 \
    || die "A C++ compiler is required."

if ! command -v git-lfs >/dev/null 2>&1; then
    if [[ "$OS" == "Darwin" ]]; then
        command -v brew >/dev/null 2>&1 \
            || die "git-lfs is required. Install it or install Homebrew."

        echo "Installing git-lfs..."
        brew install git-lfs
    else
        die "git-lfs is required."
    fi
fi

# ----------------------------------------------------------------------
# Git LFS
# ----------------------------------------------------------------------

section "2/9 Fetching Git LFS assets"

git lfs install
git lfs pull


# ----------------------------------------------------------------------
# DeepChArUco submodule
# ----------------------------------------------------------------------

section "3/9 Initializing DeepChArUco"

git submodule update --init --recursive

ACTUAL_DEEP_COMMIT="$(
    git -C third_party/deepcharuco/upstream rev-parse HEAD
)"

if [[ "$ACTUAL_DEEP_COMMIT" != "$DEEPCHARUCO_COMMIT" ]]; then
    die "Unexpected DeepChArUco commit:
expected: $DEEPCHARUCO_COMMIT
actual:   $ACTUAL_DEEP_COMMIT"
fi

echo "DeepChArUco commit verified:"
echo "$ACTUAL_DEEP_COMMIT"


# ----------------------------------------------------------------------
# Main NanoDeepChArUco environment
# ----------------------------------------------------------------------

section "4/9 Preparing NanoDeepChArUco environment"

if have_conda_env "$PROJECT_ENV"; then
    echo "Conda environment already exists: $PROJECT_ENV"
else
    echo "Creating Conda environment: $PROJECT_ENV"
    conda env create -f environment.yml
fi

conda run -n "$PROJECT_ENV" \
    python -m pip install -e "$ROOT"


# ----------------------------------------------------------------------
# Build native ArUco Nano detector
# ----------------------------------------------------------------------

section "5/9 Building native ArUco Nano detector"

NANO_SRC="$ROOT/third_party/aruco_nano/source"
NANO_BUILD="$ROOT/third_party/aruco_nano/build"

if have_conda_env "$NANO_BUILD_ENV"; then
    echo "Updating pinned Nano build environment: $NANO_BUILD_ENV"

    conda install -y \
        -n "$NANO_BUILD_ENV" \
        --override-channels \
        -c conda-forge \
        "libopencv=$NANO_OPENCV_VERSION" \
        cmake \
        pkg-config
else
    echo "Creating pinned Nano build environment: $NANO_BUILD_ENV"

    conda create -y \
        -n "$NANO_BUILD_ENV" \
        --override-channels \
        -c conda-forge \
        "libopencv=$NANO_OPENCV_VERSION" \
        cmake \
        pkg-config
fi

NANO_OPENCV_ACTUAL="$(
    conda run -n "$NANO_BUILD_ENV" \
        pkg-config --modversion opencv4 \
        | awk 'NF {last=$0} END {print last}'
)"

if [[ "$NANO_OPENCV_ACTUAL" != "$NANO_OPENCV_VERSION" ]]; then
    die "Unexpected Nano OpenCV version:
expected: $NANO_OPENCV_VERSION
actual:   $NANO_OPENCV_ACTUAL"
fi

NANO_PREFIX="$(
    conda run -n "$NANO_BUILD_ENV" \
        sh -c 'printf "%s\n" "$CONDA_PREFIX"' \
        | awk 'NF {last=$0} END {print last}'
)"

OPENCV_DIR="$NANO_PREFIX/lib/cmake/opencv4"

[[ -f "$OPENCV_DIR/OpenCVConfig.cmake" ]] \
    || die "OpenCV CMake configuration not found: $OPENCV_DIR"

echo "Nano OpenCV version : $NANO_OPENCV_ACTUAL"
echo "Nano OpenCV prefix  : $NANO_PREFIX"

# Remove any old CMake cache so Homebrew/system OpenCV
# cannot leak into the new native build.
rm -rf "$NANO_BUILD"

conda run -n "$NANO_BUILD_ENV" \
    cmake \
    -S "$NANO_SRC" \
    -B "$NANO_BUILD" \
    -DOpenCV_DIR="$OPENCV_DIR" \
    -DCMAKE_BUILD_TYPE=Release

conda run -n "$NANO_BUILD_ENV" \
    cmake \
    --build "$NANO_BUILD" \
    --config Release

NANO_EXE="$NANO_BUILD/detect_batch"

[[ -x "$NANO_EXE" ]] \
    || die "Nano detector was not built successfully."

echo "Nano executable:"
file "$NANO_EXE"


# ----------------------------------------------------------------------
# CalibCam environment
# ----------------------------------------------------------------------

section "6/9 Preparing official CalibCam 4.2 backend"

if have_conda_env "$CALIBCAM_ENV"; then
    echo "Conda environment already exists: $CALIBCAM_ENV"
else
    echo "Creating Conda environment: $CALIBCAM_ENV"

    conda create -y \
        -n "$CALIBCAM_ENV" \
        python=3.10 \
        pip
fi

CALIBCAM_OK="$(
    conda run -n "$CALIBCAM_ENV" \
        python -c "
import importlib.metadata as md

try:
    cc = md.version('bbo-calibcam')
    lib = md.version('bbo-calibcamlib')
except Exception:
    print('NO')
else:
    print(
        'YES'
        if cc == '$CALIBCAM_VERSION'
        and lib == '$CALIBCAMLIB_VERSION'
        else 'NO'
    )
" 2>/dev/null || true
)"

if [[ "$CALIBCAM_OK" != *"YES"* ]]; then
    echo "Installing pinned CalibCamLib..."

    conda run -n "$CALIBCAM_ENV" \
        python -m pip install \
        "git+$CALIBCAMLIB_REPO@$CALIBCAMLIB_COMMIT"

    echo "Installing pinned CalibCam..."

    conda run -n "$CALIBCAM_ENV" \
        python -m pip install \
        "git+$CALIBCAM_REPO@$CALIBCAM_COMMIT"
else
    echo "Verified CalibCam packages already installed."
fi


# ----------------------------------------------------------------------
# Verify CalibCam
# ----------------------------------------------------------------------

section "7/9 Verifying CalibCam backend"

CALIBCAM_PY="$(
    conda run -n "$CALIBCAM_ENV" \
        python -c 'import sys; print(sys.executable)' \
        | awk 'NF {last=$0} END {print last}'
)"

[[ -x "$CALIBCAM_PY" ]] \
    || die "Could not locate CalibCam Python executable."

"$CALIBCAM_PY" - <<PY
import calibcam
import importlib.metadata as md

cc = md.version("bbo-calibcam")
lib = md.version("bbo-calibcamlib")

print("CalibCam module :", calibcam.__file__)
print("CalibCam        :", cc)
print("CalibCamLib     :", lib)

assert cc == "$CALIBCAM_VERSION", cc
assert lib == "$CALIBCAMLIB_VERSION", lib
PY


# ----------------------------------------------------------------------
# Machine-local runtime configuration
# ----------------------------------------------------------------------

section "8/9 Writing runtime backend configuration"

mkdir -p "$ROOT/.nanodeepcharuco"

cat > "$ROOT/.nanodeepcharuco/backend.yml" <<YAML
calibcam_python: $CALIBCAM_PY
calibcam_version: "$CALIBCAM_VERSION"
calibcam_commit: "$CALIBCAM_COMMIT"
calibcamlib_version: "$CALIBCAMLIB_VERSION"
calibcamlib_commit: "$CALIBCAMLIB_COMMIT"
YAML

cat "$ROOT/.nanodeepcharuco/backend.yml"


# ----------------------------------------------------------------------
# Final verification
# ----------------------------------------------------------------------

section "9/9 Final setup verification"

for asset in \
    "configs/boards/small_5x6.npy" \
    "configs/boards/large_7x7.npy" \
    "configs/profiles/small_5x6.yaml" \
    "configs/profiles/large_7x7.yaml" \
    "configs/deep/small_5x6.yaml" \
    "configs/deep/large_7x7.yaml" \
    "models/deepcharuco/small_5x6/detector.ckpt" \
    "models/deepcharuco/large_7x7/detector.ckpt" \
    "models/refinenet/refinenet.ckpt"
do
    [[ -f "$ROOT/$asset" ]] \
        || die "Missing required asset: $asset"
done

for model in \
    "models/deepcharuco/small_5x6/detector.ckpt" \
    "models/deepcharuco/large_7x7/detector.ckpt" \
    "models/refinenet/refinenet.ckpt"
do
    SIZE="$(wc -c < "$ROOT/$model")"

    if (( SIZE < 1000000 )); then
        die "Model looks like an unresolved Git LFS pointer: $model"
    fi
done

conda run -n "$PROJECT_ENV" \
    python -c "
import nanodeepcharuco
print('NanoDeepChArUco:', nanodeepcharuco.__file__)
"

conda run -n "$PROJECT_ENV" \
    python -c "
from argparse import Namespace
from nanodeepcharuco.config import apply_runtime_backend

args = Namespace(calibcam_python=None)
args = apply_runtime_backend(args)

assert args.calibcam_python
print('CalibCam backend:', args.calibcam_python)
"

conda run -n "$PROJECT_ENV" \
    nanodeepcharuco --help >/dev/null

echo
echo "============================================================"
echo "READY"
echo "============================================================"
echo
echo "NanoDeepChArUco environment : $PROJECT_ENV"
echo "CalibCam environment        : $CALIBCAM_ENV"
echo "Nano executable             : $NANO_EXE"
echo "CalibCam Python             : $CALIBCAM_PY"
echo
echo "Activate with:"
echo "  conda activate $PROJECT_ENV"
echo
