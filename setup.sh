#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROJECT_ENV="nanodeepcharuco"
CALIBCAM_ENV="calibcam_baseline_420"

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

section "1/7 Checking tools"

command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is required."
    exit 1
}

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: Conda or Miniforge is required."
    exit 1
}

section "2/7 Preparing Python environment"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -qx "$PROJECT_ENV"; then

    conda env update \
        -n "$PROJECT_ENV" \
        -f environment.yml \
        --prune
else
    conda env create -f environment.yml
fi

section "3/7 Preparing CalibCam environment"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -qx "$CALIBCAM_ENV"; then

    conda env update \
        -n "$CALIBCAM_ENV" \
        -f environment-calibcam.yml \
        --prune
else
    conda env create -f environment-calibcam.yml
fi

section "4/7 Fetching Git LFS assets"

conda run -n "$PROJECT_ENV" git lfs install
conda run -n "$PROJECT_ENV" git lfs pull

section "5/7 Initializing DeepChArUco"

git submodule update --init --recursive

EXPECTED_DEEPCHARUCO_COMMIT="37d569fc582b790843dce408c14556747927711c"

ACTUAL_DEEPCHARUCO_COMMIT="$(
    git -C third_party/deepcharuco/upstream rev-parse HEAD
)"

if [[ "$ACTUAL_DEEPCHARUCO_COMMIT" != "$EXPECTED_DEEPCHARUCO_COMMIT" ]]; then
    echo "ERROR: unexpected DeepChArUco revision"
    echo "expected: $EXPECTED_DEEPCHARUCO_COMMIT"
    echo "actual:   $ACTUAL_DEEPCHARUCO_COMMIT"
    exit 1
fi

section "6/7 Building ArUco Nano"

bash scripts/build_nano.sh

section "7/7 Verifying installation"

conda run -n "$PROJECT_ENV" \
    nanodeepcharuco --help >/dev/null

conda run -n "$PROJECT_ENV" \
    pytest -q

conda run -n "$CALIBCAM_ENV" \
    python -c "import calibcam, calibcamlib, numpy; print('CalibCam', calibcam.__version__, 'calibcamlib', calibcamlib.__version__, 'NumPy', numpy.__version__)"


CALIBCAM_BOARD="configs/boards/small_5x6_dict6x6_250_meters_numpy1_compatible.npy"

conda run -n "$CALIBCAM_ENV" \
    python -c "import numpy as np; p='$CALIBCAM_BOARD'; b=np.load(p, allow_pickle=True)[()]; assert int(b['boardWidth']) == 5; assert int(b['boardHeight']) == 6; assert int(b['dictionary_type']) == 10; assert float(b['square_size_real']) == 0.02; print('CalibCam board serialization OK:', p)"

echo
echo "READY"
echo
echo "Activate with:"
echo "  conda activate $PROJECT_ENV"
