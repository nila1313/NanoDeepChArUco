# Two-Stage Calibration Validation — Pair 01

## 1. Purpose

This experiment validates the two-stage wide-angle calibration workflow implemented in NanoDeepChArUco.

The workflow separates camera calibration into:

1. Stage A — Intrinsic calibration
   - Large ChArUco board
   - Independent calibration of the left and right cameras

2. Stage B — Stereo extrinsic calibration
   - Small physical metric ChArUco board
   - Stage-A intrinsics are supplied to CalibCam and kept fixed
   - Only stereo extrinsics and board poses are optimized

The validation also compares NanoDeepChArUco detections against an official/reference detection source on exactly the same physical stereo frames.

## 2. Software Configuration

Calibration backend:

- CalibCam: 4.2.0
- Camera model: `omnidir`
- Projection: `fisheye_equidistant`
- Stage-B optimization:
  - `--calibration_multi`
  - `--multi_vars extrinsics extrinsics`

Official CalibCam remains the calibration optimizer.

NanoDeepChArUco is responsible for detection, synchronization, payload generation, and orchestration.

## 3. Calibration Dataset

Stage-B dataset:

`20230613/4pi/030_checkerboard_1`

Videos:

- `CADDX000013_left.MP4`
- `CADDX000013_right.MP4`

Controlled stereo alignment:

`Left(t) <-> Right(t+1)`

Therefore:

- left frame = t
- right frame = t + 1

## 4. Stage-B Board

Small physical ChArUco board:

`configs/boards/small_5x6_dict6x6_250_meters.npy`

Properties:

- Grid: 5 x 6
- Dictionary: DICT_6X6_250
- Number of ChArUco corners: 20
- Square size: approximately 0.02 m
- Units: meters

Board SHA-256:

`87eec5264b487fae9e92bf29801b5cf7c9fdcb302caad467b150f73591f933e1`

## 5. Frozen Stage-A Intrinsics

Selected Stage-A calibration files:

- `runs/two_stage/stage1_intrinsics/selected_intrinsics/left_multicam_calibration.yml`
- `runs/two_stage/stage1_intrinsics/selected_intrinsics/right_multicam_calibration.yml`

SHA-256:

- LEFT: `dbaf04beb91fdd6756726d0f2b7c015db983513e52c2d7951cada5cb75422ce9`
- RIGHT: `a137bf2fd1b1d5eabc1ce8ceaa6b597a3adff00afd9b39ff3910f83f7b607498`

These files are passed into Stage B through:

`--calibration_single LEFT.yml RIGHT.yml`

with:

`--multi_vars extrinsics extrinsics`

## 6. Full NanoDeepChArUco Stage-B Run

The controlled NanoDeepChArUco Stage-B run used:

- Frame range: 0 to 2640
- Frame step: 20
- Left offset: 0
- Right offset: +1

Results:

- Candidate logical frames: 132
- Shared stereo detections: 131
- Left valid detected corners: 2200
- Right valid detected corners: 2021
- CalibCam convergence: yes
- Rejected detections: 0%
- Rejected frames: 0%

Final median reprojection residuals:

| Camera | Median residual |
| --- | ---: |
| Left | 0.36 px |
| Right | 0.44 px |

Final Stage-B camera-1 rotation vector:

- x: -0.00400939361025635
- y: -0.33554497320150717
- z: -0.26401284223151844

Final Stage-B camera-1 translation vector in meters:

- x: -0.028092722847257577
- y: 0.005538983513281896
- z: -0.006983275546142348

Stereo baseline:

- 0.0294728274 m
- 29.4728 mm

Rotation-vector magnitude:

- 24.464 degrees

## 7. Fixed-Intrinsics Verification

Stage-A and Stage-B intrinsic parameters were compared numerically for both cameras.

Checked intrinsic parameters:

- A
- k
- xi

Results for LEFT camera:

- A exact equal: True
- k exact equal: True
- xi exact equal: True

Results for RIGHT camera:

- A exact equal: True
- k exact equal: True
- xi exact equal: True

Maximum absolute difference for every checked parameter:

- 0.0

Therefore, Stage B preserved the frozen Stage-A intrinsics exactly for both cameras.

## 8. Matched-Frame Detector Comparison

A controlled detector comparison was constructed between:

- NanoDeepChArUco detections
- Official/reference detections

Only physical stereo frame pairs available in both detection sets were retained.

The intersection contained:

- 61 physical stereo pairs

Both matched detection sets were reconstructed with identical:

- left physical frame sequence
- right physical frame sequence
- logical detection IDs
- right-camera temporal offset of +1
- physical board definition
- frozen Stage-A intrinsics
- CalibCam version
- camera model
- projection
- optimization variables

The intended experimental difference was the detected corner coordinates.

Fairness checks:

- Same detection count: True
- LEFT frame mappings identical: True
- RIGHT frame mappings identical: True
- NanoDeep right-camera offset exactly +1: True
- Official right-camera offset exactly +1: True
- NanoDeep logical IDs identical between cameras: True
- Official logical IDs identical between cameras: True

The matched datasets therefore use exactly the same 61 physical stereo frame pairs.

## 9. Matched-61 Detection Coverage

NanoDeepChArUco:

- Stereo frames: 61
- Left valid corners: 949
- Right valid corners: 830
- CalibCam left feature summary: 15 +/- 3 corners/frame
- CalibCam right feature summary: 13 +/- 3 corners/frame

Official/reference:

- Stereo frames: 61
- Left valid corners: 1086
- Right valid corners: 1043
- CalibCam left feature summary: 17 +/- 2 corners/frame
- CalibCam right feature summary: 17 +/- 3 corners/frame

The official/reference detector supplied more corner observations on this matched subset.

## 10. Matched-61 Calibration Results

Both matched calibration runs:

- Produced pose estimates for all 61 frames
- Converged successfully
- Rejected 0% of frames
- Rejected 0% of detections
- Preserved the frozen Stage-A intrinsics exactly

Matched reprojection results:

| Metric | NanoDeepChArUco | Official/reference |
| --- | ---: | ---: |
| Left median reprojection residual | 0.44 px | 0.53 px |
| Right median reprojection residual | 0.58 px | 0.63 px |
| Final optimization cost | 1481.2 | 2532.2 |
| Frames rejected | 0% | 0% |
| Detections rejected | 0% | 0% |
| Converged | Yes | Yes |

For this matched 61-frame experiment, NanoDeepChArUco produced lower median reprojection residuals while the official/reference detector supplied more detected corners.

## 11. Matched-61 Stereo Extrinsics

NanoDeepChArUco camera-1 rotation vector:

- x: -0.004167017938297436
- y: -0.3372935904565835
- z: -0.26381097860124675

NanoDeepChArUco camera-1 translation vector in meters:

- x: -0.027966625594480262
- y: 0.005495243893213641
- z: -0.006929650424533889

NanoDeepChArUco baseline:

- 29.3317218655 mm

NanoDeepChArUco rotation-vector magnitude:

- 24.5357468906 degrees

Official/reference camera-1 rotation vector:

- x: -0.004774305178072996
- y: -0.3351748199557665
- z: -0.264040316635733

Official/reference camera-1 translation vector in meters:

- x: -0.028165863521062307
- y: 0.0054966421868295244
- z: -0.0068263080703733475

Official/reference baseline:

- 29.4979223860 mm

Official/reference rotation-vector magnitude:

- 24.4487370362 degrees

## 12. Extrinsic Agreement

The difference between the NanoDeepChArUco and official/reference stereo solutions is:

- Translation-vector difference: 0.2244489893 mm
- Baseline difference: 0.1662005205 mm
- Relative rotation difference: 0.1264558788 degrees

The relative rotation difference was calculated from the relative rotation matrix, rather than by directly subtracting Rodrigues-vector magnitudes.

This shows that both detection sources produce closely agreeing stereo geometry under the same fixed-intrinsics calibration conditions.

## 13. Interpretation

The controlled matched-frame experiment shows that NanoDeepChArUco and the official/reference detection source produce closely agreeing stereo extrinsics when all other calibration variables are held constant.

For the same 61 physical stereo pairs:

- The official/reference detector produced more corner observations.
- NanoDeepChArUco produced lower median reprojection residuals in this experiment.
- Both methods converged successfully.
- Neither method required frame or detection rejection.
- The translation estimates differed by approximately 0.224 mm.
- The baseline estimates differed by approximately 0.166 mm.
- The relative rotations differed by approximately 0.126 degrees.
- Both methods preserved the frozen Stage-A intrinsic parameters exactly.

These results support compatibility of NanoDeepChArUco detections with the fixed-intrinsics two-stage CalibCam workflow.

This experiment does not establish that one detection method is universally better than the other. Additional datasets are required before making a broader performance claim.

## 14. Pair 01 Validation Status

- PASS: Stage-A independent intrinsic calibration
- PASS: Stage-B physical small-board calibration
- PASS: Fixed-intrinsics CalibCam integration
- PASS: Stage-A A, k, and xi preserved exactly
- PASS: Controlled Left(t) <-> Right(t+1) synchronization
- PASS: NanoDeepChArUco detection payload compatibility
- PASS: Matched physical-frame comparison
- PASS: CalibCam convergence
- PASS: Close extrinsic agreement with the reference detector

Pair 01 therefore serves as the first controlled validation case for the two-stage NanoDeepChArUco calibration workflow.
