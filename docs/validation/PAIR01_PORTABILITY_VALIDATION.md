# Pair 01 Cross-Platform Portability Validation

## Status

**PASS — numerically reproducible across macOS ARM and Linux x86-64 for the tested Pair 01 configuration.**

The pipeline is not bit-for-bit deterministic across CPU architectures.
Remaining differences are negligible floating-point differences and do not
materially change synchronization, detected-corner structure, or final
stereo calibration.

Validation completed: 2026-09-17.

## Repository baseline

Validated repository commit:

    77e9345 — Document Pair 02 portability validation

Branch:

    feature/portable-assets

Production ArUco Nano build:

    OpenCV 4.13.0

CalibCam backend:

    CalibCam version:    4.2.0
    CalibCam commit:     f51aa60961b6a9a7abea8b7c377dd5dfa7599f50
    CalibCamLib version: 0.5.2
    CalibCamLib commit:  5e3888b0647a4e20cf54ec54734ec6ed770a2169

Platforms:

    macOS ARM
    Linux x86-64

The Linux test used a clean consumer checkout and the repository setup
script successfully produced the pinned OpenCV-4.13 Nano build and
CalibCam runtime backend.

## Dataset

Pair 01:

    20230613/4pi/030_checkerboard_1

Videos:

    CADDX000013_left.MP4
    CADDX000013_right.MP4

SHA256:

    left:
    7aa1441c0faf1acacbe69f5133687c5a21a3514fdf58d946b997e516c1bf85f2

    right:
    a4d8d327c3e4784012a368333c08bb3cb68dce12ca3be2dcfc5cb1575db2239d

Both platforms used byte-identical source videos.

Board:

    small_5x6.npy
    5x6 ChArUco
    DICT_6X6_250
    20 ChArUco corners

Physical board dimensions:

    square_size_real = 0.02 m
    marker_size_real = 0.013333333333333332 m

## Detection configuration

    frames_start: 0
    frames_end: 2640
    frames_step: 20
    frames_offsets: [0, 0]

    device: cpu
    gamma: 1.4
    canonical_luma: true

    auto_sync: true
    sync_offsets: [-3, -2, -1, 0, 1, 2, 3]
    sync_frame_step: 20
    sync_window_size: 400
    sync_window_step: 200
    sync_min_inlier_ratio: 0.2
    sync_min_ratio_gap: 0.05
    sync_min_persistence: 2
    sync_max_gap_frames: 400

    deep_self_ransac_px: 5.0

## Production synchronization result

Mac and Linux produced the same automatic synchronization scores and
the same trusted stable segment.

Stable segment:

    400-2799:
      offset=+1
      support=9
      mean_ratio=0.7594
      mean_gap=0.1197

Final sampling:

    requested calibration frames: 132
    available calibration frames: 131
    synchronized stereo pairs:    111
    synchronized range:           400 to 2600

Frame mapping:

    Camera 0: 400 to 2600, step 20
    Camera 1: 401 to 2601, step 20

Both payloads:

    marker_coords_shape = [1, 111, 20, 2]
    n_detection_idxs    = 111
    n_marker_ids        = 20

The synchronization decision, synchronized frame mapping, marker IDs,
finite-value masks, and valid-corner masks were identical across both
platforms.

## Mac vs Linux production detections

The production detection YAML files are not byte-identical because a
small number of floating-point corner coordinates differ across CPU
architectures.

The detection structure is nevertheless identical.

### Camera 0

    valid corners Linux:       1820
    valid corners Mac:         1820
    finite scalar mask exact:  true
    valid corner mask exact:   true

    points differing at all:   18
    max point delta:           0.0001220703125 px
    mean point delta:          0.0000006120284 px
    median point delta:        0.0 px
    99th percentile:           0.0 px

    points > 0.000001 px:      18
    points > 0.00001 px:       18
    points > 0.0001 px:        4
    points > 0.001 px:         0

### Camera 1

    valid corners Linux:       1621
    valid corners Mac:         1621
    finite scalar mask exact:  true
    valid corner mask exact:   true

    points differing at all:   18
    max point delta:           0.00006103515625 px
    mean point delta:          0.0000003765278 px
    median point delta:        0.0 px
    99th percentile:           0.0000152587890625 px

    points > 0.000001 px:      18
    points > 0.00001 px:       18
    points > 0.0001 px:        0
    points > 0.001 px:         0

These differences are well below one thousandth of a pixel.

## Final CalibCam comparison

Each platform independently ran its production detections through the
same CalibCam 4.2.0 / CalibCamLib 0.5.2 backend.

Both runs selected:

    Camera 0 usable single-camera frames: 109
    Camera 1 usable single-camera frames: 92
    shared stereo poses:                  90

Both final robust stages discarded:

    detections: 0%
    frames:     0%

Both converged to approximately:

    final cost:                 1.4972e+03
    median residual Camera 0:   0.23 px
    median residual Camera 1:   0.36 px
    maximum residual Camera 0:  5.59 px
    maximum residual Camera 1:  5.09 px

## End-to-end Mac vs Linux calibration differences

### Camera 0

    |fx difference| = 0.0031562901 px
                      0.00023437%

    |fy difference| = 0.0015256097 px
                      0.00011330%

    |cx difference| = 0.0008792676 px
    |cy difference| = 0.0006078815 px

    |xi difference| = 5.7611663e-07

### Camera 1

    |fx difference| = 0.0011032046 px
                      0.00008204%

    |fy difference| = 0.0008077330 px
                      0.00006009%

    |cx difference| = 0.0008673572 px
    |cy difference| = 0.0000262280 px

    |xi difference| = 2.8937361e-06

### Stereo geometry

Rotation difference:

    2.4421543949e-06 rad
    1.3992513974e-04 deg

Translation difference:

    [ 1.72369910e-07,
     -2.62447206e-08,
     -7.09132460e-08 ]

Translation delta norm:

    1.8822555559e-07

The final Mac and Linux stereo solutions are therefore practically
equivalent.

## Linux reference calibration

SHA256:

    36e69a4cdbe002bd39be36638637f05df47f346cbbfa84c75a803c6155248416

Camera 0:

    fx = 1346.7145718587149
    fy = 1346.5802149785504
    cx = 985.4508351922418
    cy = 534.7897071058602
    xi = 0.6264027123047979

Camera 1:

    fx = 1344.709424901071
    fy = 1344.2677432172295
    cx = 968.33256967122
    cy = 550.5570413243465
    xi = 0.6173853074831506

    rvec_cam1 =
    [0.011481884215567126,
     -0.3267716977421915,
     -0.2634795392113898]

    tvec_cam1 =
    [-0.028427874047340703,
      0.005409444976575321,
     -0.005985695294600867]

## Conclusion

For the tested Pair 01 small-board configuration, NanoDeepChArUco is
numerically reproducible across macOS ARM and Linux x86-64.

The production pipeline gives:

    identical source-video bytes
    identical synchronization decisions
    identical synchronized frame mappings
    identical ChArUco corner validity masks
    identical detected corner IDs
    sub-millipixel coordinate differences only
    practically equivalent final camera intrinsics
    practically equivalent final stereo extrinsics

Recommended wording:

> Numerically reproducible across the tested platforms, with negligible
> architecture-dependent floating-point differences.

The implementation should not be described as universally bit-for-bit
deterministic across CPU architectures.

Together with the Pair 02 validation, the current validation covers both
built-in board profiles:

    small_5x6
    large_7x7
