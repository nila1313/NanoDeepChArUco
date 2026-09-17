# NanoDeepChArUco

NanoDeepChArUco is a portable stereo-camera calibration pipeline combining:

- ArUco Nano detection
- gamma-enhanced Nano retry
- DeepChArUco recovery
- automatic stereo synchronization
- CalibCam-compatible ChArUco detections
- official BBO CalibCam calibration

## Pipeline

```text
video pair
   ↓
Raw ArUco Nano
   ↓
Gamma Nano retry
   ↓
DeepChArUco recovery
   ↓
shared stereo detections
   ↓
automatic trusted synchronization
   ↓
CalibCam detection YAML
   ↓
official CalibCam
   ↓
final calibration
```

## Installation

Prerequisites:

- Git
- Git LFS
- Conda / Miniforge
- CMake
- OpenCV development libraries
- Homebrew on macOS

Clone the repository:

```bash
git clone --recurse-submodules \
  https://github.com/nila1313/NanoDeepChArUco.git

cd NanoDeepChArUco
```

Run the complete setup:

```bash
./setup.sh
```

The setup script prepares:

- Git LFS model assets
- the pinned DeepChArUco submodule
- the `nanodeepcharuco` Conda environment
- the native ArUco Nano detector
- the verified CalibCam 4.2 environment
- machine-local CalibCam backend configuration

A successful setup finishes with:

```text
READY
```

Then activate:

```bash
conda activate nanodeepcharuco
```

## Portable profiles

### `small_5x6`

- 5 x 6 ChArUco squares
- `DICT_6X6_250`
- 20 ChArUco corner IDs

### `large_7x7`

- 7 x 7 ChArUco squares
- `DICT_4X4_50`
- 36 ChArUco corner IDs

## Default stereo calibration

For a normal two-camera run, NanoDeepChArUco automatically estimates
temporal synchronization and keeps only trusted stable synchronization
regions.

Canonical native-luminance decoding is enabled by default for
cross-platform reproducibility.

Example:

```bash
RUN="runs/auto_sync_run"
mkdir -p "$RUN"

nanodeepcharuco \
  --videos /path/to/left.MP4 /path/to/right.MP4 \
  --profile small_5x6 \
  --frames_start 0 \
  --frames_end 2640 \
  --frames_step 20 \
  --sync_offsets -3 -2 -1 0 1 2 3 \
  --sync_frame_step 20 \
  --sync_window_size 400 \
  --sync_window_step 200 \
  --sync_min_inlier_ratio 0.20 \
  --sync_min_ratio_gap 0.05 \
  --sync_min_persistence 2 \
  --sync_max_gap_frames 400 \
  --models omnidir omnidir \
  --projection perspective \
  --data_path "$RUN" \
  2>&1 | tee "$RUN/full_pipeline.log"
```

No `--auto_sync` flag is required for a normal two-video run.

The legacy `--auto_sync` flag is still accepted for backward
compatibility.

Automatic synchronization evaluates candidate offsets, keeps trusted
windows, combines persistent evidence into stable segments, and gives
both cameras shared logical detection IDs.

For example:

```text
offset +1: Left(t) ↔ Right(t+1)
offset -1: Left(t) ↔ Right(t-1)
```

## Fixed-offset expert mode

If a validated frame offset is already known and should be forced, use
fixed synchronization.

Example:

```bash
RUN="runs/fixed_offset_run"
mkdir -p "$RUN"

nanodeepcharuco \
  --videos /path/to/left.MP4 /path/to/right.MP4 \
  --profile small_5x6 \
  --frames_start 0 \
  --frames_end 2640 \
  --frames_step 20 \
  --fixed_sync \
  --frames_offsets 0 1 \
  --models omnidir omnidir \
  --projection perspective \
  --data_path "$RUN" \
  2>&1 | tee "$RUN/full_pipeline.log"
```

`--frames_offsets 0 1` means:

```text
Left(t) ↔ Right(t+1)
```

In general:

```text
physical_frame[camera] = logical_frame + frame_offset[camera]
```

Explicit `--frames_offsets` also preserve the historical fixed-offset
behavior even if `--fixed_sync` is omitted.

## Portable video decoding

Canonical luminance decoding is enabled by default for CLI runs.

This reads the native video luminance plane and avoids platform-dependent
YUV-to-BGR conversion before expanding the frame to the three-channel format
expected by the detector.

The default portable behavior can also be requested explicitly with:

```bash
--canonical_luma
```

For legacy comparison only, it can be disabled with:

```bash
--no_canonical_luma
```

The legacy decoding path is not the recommended mode for reproducible
cross-platform runs.

## Detection only

To generate NanoDeepChArUco detections without running CalibCam:

```bash
nanodeepcharuco \
  --videos /path/to/left.MP4 /path/to/right.MP4 \
  --profile small_5x6 \
  --frames_step 20 \
  --detect_only \
  --data_path runs/detection_only
```

For two-camera input, detection-only mode still uses the default automatic
trusted synchronization. Use `--fixed_sync --frames_offsets ...` only when
a fixed alignment is intentionally required.

## Detector routing

The recovery order is:

```text
RAW_NANO
   ↓
GAMMA_NANO
   ↓
DEEP_NANO_H
   ↓
DEEP_SELF
```

`GAMMA_NANO` uses gamma 1.4 and is selected only when it improves the Nano result.

`DEEP_NANO_H` uses DeepChArUco recovery supported by Nano geometry.

`DEEP_SELF` is the final DeepChArUco self-recovery stage.

## Output structure

A normal full run contains:

```text
run/
├── resolved_config.yml
├── input_manifest.yml
├── detection_000.yml
├── detection_001.yml
├── full_pipeline.log
└── calibcam_output/
    ├── calibration_single_000.yml
    ├── calibration_single_001.yml
    ├── joinedsingles_calibraton.yml
    ├── multicam_calibration.yml
    ├── multicam_calibration.npy
    ├── multicam_calibration.mat
    └── multicam_calibration_board_positions.yml
```

Automatic-sync runs additionally contain:

```text
sync_window_results.yml
sync_segments.yml
sync_pairs.yml
```

`full_pipeline.log` is created by the example commands through `tee`.

## Verified dependency revisions

### DeepChArUco

```text
37d569fc582b790843dce408c14556747927711c
```

### BBO CalibCam

```text
version: 4.2.0
commit: f51aa60961b6a9a7abea8b7c377dd5dfa7599f50
```

### BBO CalibCamLib

```text
version: 0.5.2
commit: 5e3888b0647a4e20cf54ec54734ec6ed770a2169
```

CalibCam runs in a separate Python 3.10 Conda environment.

The machine-specific backend path is stored in:

```text
.nanodeepcharuco/backend.yml
```

This file is generated by `setup.sh` and intentionally excluded from Git.

An explicit backend can still be supplied with:

```bash
--calibcam_python /path/to/python
```

or with:

```bash
export NANODEEPCHARUCO_CALIBCAM_PYTHON=/path/to/python
```

## Cross-platform validation

The current portable pipeline has been validated end-to-end on:

```text
macOS ARM
Linux x86-64
```

using both built-in board profiles.

### Pair 01 — `small_5x6`

Dataset:

```text
20230613/4pi/030_checkerboard_1
```

Automatic synchronization selected the trusted `+1` alignment and produced:

```text
111 synchronized stereo pairs
synchronized range: 400 to 2600

final median reprojection residuals:
camera 0: 0.23 px
camera 1: 0.36 px
```

Mac and Linux produced identical synchronization decisions, synchronized
frame mappings, ChArUco IDs, finite-value masks, and valid-corner masks.

The maximum cross-platform detected-corner difference was approximately:

```text
0.000122 px
```

The final stereo calibrations were practically equivalent.

Full report:

```text
reports/PAIR01_PORTABILITY_VALIDATION.md
```

### Pair 02 — `large_7x7`

Dataset:

```text
20230613/4pi/040_checkerboard_2
```

Automatic synchronization produced two trusted `+1` stable regions and:

```text
243 synchronized stereo pairs
synchronized range: 420 to 8680

final median reprojection residuals:
camera 0: 0.88 px
camera 1: 0.77 px
```

Mac and Linux again produced identical synchronization decisions,
synchronized frame mappings, ChArUco IDs, finite-value masks, and
valid-corner masks.

Detected-corner differences remained sub-millipixel and the final stereo
calibrations were practically equivalent.

Full report:

```text
reports/PAIR02_PORTABILITY_VALIDATION.md
```

### Portability statement

The supported wording for the validated pipeline is:

> Numerically reproducible across the tested platforms, with negligible
> architecture-dependent floating-point differences.

The implementation is **not** claimed to be universally bit-for-bit
deterministic across CPU architectures.

Together, Pair 01 and Pair 02 validate both built-in profiles:

```text
small_5x6
large_7x7
```

### Additional synchronization regression

Pair 03 also completed automatic synchronization and final CalibCam
calibration successfully:

```text
stable offset = -1
105 synchronized stereo pairs
```

## Notes

The native ArUco Nano executable is built locally at:

```text
third_party/aruco_nano/build/detect_batch
```

Build outputs, calibration runs, and machine-local runtime configuration are intentionally excluded from version control.

Model checkpoints are stored using Git LFS.
