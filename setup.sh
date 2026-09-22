#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROJECT_ENV="nanodeepcharuco"

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

section "1/6 Checking tools"

command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is required."
    exit 1
}

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: Conda or Miniforge is required."
    exit 1
}

section "2/6 Preparing Python environment"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -qx "$PROJECT_ENV"; then

    conda env update \
        -n "$PROJECT_ENV" \
        -f environment.yml \
        --prune
else
    conda env create -f environment.yml
fi

section "3/6 Fetching Git LFS assets"

conda run -n "$PROJECT_ENV" git lfs install
conda run -n "$PROJECT_ENV" git lfs pull

section "4/6 Initializing DeepChArUco"

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

section "5/6 Building ArUco Nano"

bash scripts/build_nano.sh

section "6/6 Verifying installation"

conda run -n "$PROJECT_ENV" \
    nanodeepcharuco --help >/dev/null

conda run -n "$PROJECT_ENV" \
    pytest -q

echo
echo "READY"
echo
echo "Activate with:"
echo "  conda activate $PROJECT_ENV"
