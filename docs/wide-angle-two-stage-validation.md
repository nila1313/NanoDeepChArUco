# Wide-Angle Two-Stage Calibration Validation

## 1. Purpose

This document describes the validated explicit-offset wide-angle two-stage
calibration workflow on the branch:

    pipeline/wide-angle-two-stage

The workflow separates calibration into two stages:

1. Stage 1 estimates camera intrinsics independently using the large board.
2. Stage 2 freezes those intrinsics and estimates stereo extrinsics using the
   small physical metric board.

Validation was performed on Linux with official CalibCam 4.2.0.

This validates the pipeline behavior for the tested datasets and configuration.
It does not establish absolute physical ground-truth accuracy.

## 2. Validated Architecture

Stage 1:

    Large 7x7 ChArUco board
              |
              v
       Hybrid detection
              |
        +-----+-----+
        |           |
        v           v
      LEFT        RIGHT
        |           |
        v           v
    one-camera   one-camera
     CalibCam     CalibCam
    single+multi single+multi
        |           |
        v           v
    refined      refined
    left         right
    intrinsics   intrinsics

Stage 2:

    refined left/right intrinsics
              |
              | A, k, xi fixed
              v
    Small 5x6 metric ChArUco board
              |
              v
       Hybrid detection
              |
              v
    Left(t) <-> Right(t+1)
              |
              v
     CalibCam extrinsics-only
              |
              v
      final stereo calibration

## 3. Software Configuration

Validation environment:

    Host:       bbo3008
    OS:         Linux
    Repository: NanoDeepChArUco
    Branch:     pipeline/wide-angle-two-stage
    CalibCam:   4.2.0

Stage-1 correction commit:

    0e9dbd1 Fix Stage-1 two-stage intrinsic calibration

Test result after correction:

    12 passed

git diff --check completed without errors.

## 4. Stage 1 - Intrinsic Calibration

### Dataset

    20230613/4pi/040_checkerboard_2

Videos:

    CADDX000014_left.MP4
    CADDX000014_right.MP4

### Board

    configs/boards/large_7x7_dict4x4_50.npy

Properties:

    7 x 7 ChArUco board
    DICT_4X4_50

### Detector

Stage 1 used the Hybrid detector:

    ArUco Nano
        +
    DeepChArUco recovery

Files:

    models/deepcharuco/large_7x7/detector.ckpt
    models/refinenet/refinenet.ckpt
    configs/deepcharuco/pair2_epoch146.yaml

### Sampling

    frames_start   = 0
    frames_step    = 20
    frames_offsets = 0 0

### Correct Stage-1 Procedure

Each camera is calibrated independently.

For the left camera CalibCam receives only:

    left video
    left detection payload

For the right camera CalibCam receives only:

    right video
    right detection payload

Each camera runs:

    --calibration_single
    --calibration_multi
    --models omnidir
    --projection fisheye_equidistant

The resulting files report:

    n_cams = 1
    calibration_single = True
    calibration_multi = True

The refined Stage-1 files are:

    calibcam_output/left/multicam_calibration.yml
    calibcam_output/right/multicam_calibration.yml

### Stage-1 Result

Used frames:

    Left:  283
    Right: 241

Left intrinsic matrix:

    [[827.22256498,   0.0,        977.36790784],
     [  0.0,        828.56659015, 538.58479959],
     [  0.0,          0.0,          1.0       ]]

Left distortion:

    [-0.00392972,
     -0.02102143,
     -0.00017045,
     -0.00129361,
      0.00282049]

Right intrinsic matrix:

    [[832.14017687,   0.0,        964.52041725],
     [  0.0,        830.07215391, 545.85831311],
     [  0.0,          0.0,          1.0       ]]

Right distortion:

    [-0.00972676,
     -0.02321390,
     -0.00052200,
      0.00017645,
      0.00378959]

For both cameras:

    xi = 0
    model = omnidir
    projection = fisheye_equidistant

The corrected CLI Stage-1 output matched the manually reproduced corrected
control exactly:

    max |A difference|  = 0.0
    max |k difference|  = 0.0
    max |xi difference| = 0.0

## 5. Stage-1 Correction

The earlier implementation ran both cameras together and used only:

    --calibration_single

The resulting single-camera files were incorrectly treated as the final
Stage-1 intrinsic calibrations.

Although optimization converged, those intrinsics produced poor Stage-2
geometry and large reprojection residuals.

Inspection of the previously validated calibration showed the intended
Stage-1 configuration:

    n_cams = 1
    calibration_single = True
    calibration_multi = True

for each camera independently.

The implementation was corrected to reproduce this procedure.

## 6. Stage 2 - Fixed-Intrinsics Stereo Calibration

### Dataset

    20230613/4pi/030_checkerboard_1

Videos:

    CADDX000013_left.MP4
    CADDX000013_right.MP4

### Board

    configs/boards/small_5x6_dict6x6_250_meters.npy

Properties:

    5 x 6 ChArUco board
    DICT_6X6_250
    square_size_real = 0.02 m

### Detector

Stage 2 used:

    models/deepcharuco/small_5x6/detector.ckpt
    models/refinenet/refinenet.ckpt
    configs/deepcharuco/small_5x6.yaml

### Temporal Alignment

Validated Pair-01 alignment:

    Left(t) <-> Right(t+1)

CLI:

    --frames_offsets 0 1

Sampling:

    frames_start = 0
    frames_end   = 2640
    frames_step  = 20

Final stereo pairs:

    131

Physical frame ranges:

    Left:  0, 20, 40, ..., 2600
    Right: 1, 21, 41, ..., 2601

Measured physical right-minus-left offset:

    [1]

## 7. Frozen Intrinsics

Stage 2 receives:

    --stage1_intrinsics LEFT.yml RIGHT.yml

CalibCam is invoked with:

    --calibration_single LEFT.yml RIGHT.yml
    --calibration_multi
    --multi_vars extrinsics extrinsics

For both cameras:

    A  = fixed
    k  = fixed
    xi = fixed

Verification:

    CAMERA 0
    A exact:  True
    k exact:  True
    xi exact: True

    CAMERA 1
    A exact:  True
    k exact:  True
    xi exact: True

Maximum absolute differences:

    A  = 0.0
    k  = 0.0
    xi = 0.0

## 8. Final Stereo Geometry

Camera-1 rotation vector:

    [-0.00130167,
     -0.33527816,
     -0.26444972]

Camera-1 translation vector in meters:

    [-0.02818036,
      0.00534644,
     -0.00611911]

Baseline:

    29.32849626924211 mm

Rotation-vector magnitude:

    24.466491423373633 degrees

## 9. Reprojection Results

Final median reprojection residuals:

    Left:  0.31 px
    Right: 0.38 px

Maximum reported residuals:

    Left:  6.72 px
    Right: 5.05 px

Rejection:

    Discarded detections: 0.00%
    Discarded frames:     0.00%

Optimization:

    converged = True
    final cost = 1.7532e+03

## 10. Reference Control

Using the same new Stage-2 Hybrid detections with the previously validated
Stage-1 intrinsic files produced:

    baseline = 29.5107879 mm
    rotation magnitude = 24.4455435 degrees
    median residuals = approximately 0.33 px / 0.42 px

This isolated the earlier problem to the incomplete Stage-1 procedure rather
than Stage-2 synchronization, Hybrid small-board detection, fixed-intrinsics
handling, or CalibCam 4.2 integration.

## 11. Conclusion

For the tested Pair-02 Stage-1 and Pair-01 Stage-2 datasets, the corrected
pipeline:

- performs independent one-camera Stage-1 refinement;
- produces refined left and right intrinsic files;
- freezes A, k, and xi exactly during Stage 2;
- preserves Left(t) <-> Right(t+1) synchronization;
- uses 131 synchronized stereo pairs;
- converges without rejecting detections or frames;
- achieves low sub-pixel median reprojection residuals;
- produces stereo geometry close to the previous validated reference.

The validation applies to the tested software versions, datasets, detector
configuration, board definitions, and synchronization settings.
