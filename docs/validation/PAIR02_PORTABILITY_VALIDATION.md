# Pair 02 Cross-Platform Portability Validation

## Status

**PASS — numerically reproducible across macOS ARM and Linux x86-64 for the tested Pair 02 configuration.**

The pipeline is not bit-for-bit deterministic across CPU architectures. Remaining differences are negligible floating-point differences and do not materially change synchronization, detected-corner structure, or final stereo calibration.

Validation performed: 2026-09-16 to 2026-09-17.

## Repository baseline

Validated repository commit:

    4c45b81 — Pin Nano build to OpenCV 4.13

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

## Dataset

Pair 02:

    20230613/4pi/040_checkerboard_2

Videos:

    CADDX000014_left.MP4
    CADDX000014_right.MP4

SHA256:

    left:
    da014d03ee165e3296feaaa199747fa6c06a800be661691fd7000426d872cdd1

    right:
    ab28310f2599fd642d690ef335dcd7e374f24ba9145ee35ef2d1dafbfddf9877

Both platforms used byte-identical source videos.

Board:

    large_7x7.npy
    7x7 ChArUco
    DICT_4X4_50
    36 ChArUco corners

## Detection configuration

    frames_start: 0
    frames_end: 8760
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

Both platforms produced the same stable synchronization solution:

    420-3619:
      offset=+1
      support=12
      mean_ratio=0.5471
      mean_gap=0.1063

    6020-8819:
      offset=+1
      support=13
      mean_ratio=0.6245
      mean_gap=0.2624

Final sampling:

    requested calibration frames: 438
    available calibration frames: 342
    synchronized stereo pairs:    243
    synchronized range:           420 to 8680

Frame mapping:

    Camera 0: 420 to 8680
    Camera 1: 421 to 8681

Both payloads:

    marker_coords_shape = [1, 243, 36, 2]
    n_detection_idxs    = 243
    n_marker_ids        = 36

## Linux production-binary validation

The installer-generated Linux production Nano binary was compared with the previously validated isolated OpenCV-4.13 build.

The following outputs were byte-for-byte identical:

    sync_window_results.yml
    sync_segments.yml
    sync_pairs.yml
    detection_000.yml
    detection_001.yml

Linux detection SHA256:

    detection_000.yml:
    19b9e6f9e76a4bb2b3fbebf56faa2de9001a964ce64a293662881b2822fb765d

    detection_001.yml:
    6d55df0bb001be1885058669fdef17be3b2301d982f5c8b91e4adab5c1979ae4

## Mac vs Linux production detections

The production detection YAML files are not byte-identical because a small number of floating-point corner coordinates differ across architectures.

However, the detection structure is identical.

### Camera 0

    valid corners Linux:       4140
    valid corners Mac:         4140
    finite scalar mask exact:  true
    valid corner mask exact:   true

    corners differing at all:  63
    max point delta:           0.0006573688 px
    mean point delta:          0.0000009769 px
    median point delta:        0.0 px
    99th percentile:           0.0000305176 px
    points > 0.0001 px:        13
    points > 0.001 px:         0

### Camera 1

    valid corners Linux:       4321
    valid corners Mac:         4321
    finite scalar mask exact:  true
    valid corner mask exact:   true

    corners differing at all:  53
    max point delta:           0.0005033088 px
    mean point delta:          0.0000011357 px
    median point delta:        0.0 px
    99th percentile:           0.0000305176 px
    points > 0.0001 px:        16
    points > 0.001 px:         0

Therefore both platforms select the same frames, synchronized pairs, ChArUco IDs, and valid/invalid corner masks.

## Controlled CalibCam platform test

The exact Linux detection YAML files were used by CalibCam on both Mac and Linux.

Controlled inputs:

    same videos
    same detection YAMLs
    same board
    same CalibCam 4.2.0
    same CalibCamLib 0.5.2

Both runs converged to approximately:

    final robust-stage cost: 3.3973e+04
    median residual cam 0:   0.88 px
    median residual cam 1:   0.77 px

The optimizer followed slightly different floating-point paths:

    Linux final robust stage: 39 function evaluations
    Mac final robust stage:   34 function evaluations

Largest focal-length differences:

    Cam 0 fx: 0.2321 px (0.0170%)
    Cam 0 fy: 0.2336 px (0.0171%)
    Cam 1 fx: 0.1190 px (0.0086%)
    Cam 1 fy: 0.1209 px (0.0087%)

Principal-point differences:

    less than 0.006 px

xi differences:

    Cam 0: 0.00027417
    Cam 1: 0.00015292

Stereo geometry:

    rotation difference:
    1.959828455757e-06 rad
    1.122898990845e-04 deg

    translation delta norm:
    1.004170390795e-06

## Full end-to-end comparison

Each platform independently ran:

    video
      -> canonical luminance decoding
      -> Nano / DeepChArUco detection
      -> automatic synchronization
      -> synchronized detection payloads
      -> CalibCam
      -> final stereo calibration

Final residual quality:

    Cam 0 median residual: 0.88 px
    Cam 1 median residual: 0.77 px

    Cam 0 max residual: 44.17 px
    Cam 1 max residual: 30.19 px

End-to-end intrinsic differences:

    Cam 0 fx:
      0.209053 px
      0.015342%

    Cam 0 fy:
      0.209793 px
      0.015366%

    Cam 0 principal point:
      maximum difference approximately 0.0032 px

    Cam 1 fx:
      0.095033 px
      0.006843%

    Cam 1 fy:
      0.096585 px
      0.006951%

    Cam 1 principal point:
      maximum difference approximately 0.0048 px

xi differences:

    Cam 0: 0.000250099
    Cam 1: 0.000118056

End-to-end stereo geometry:

    rotation difference:
    1.582051448821e-06 rad
    9.064487098998e-05 deg

    translation delta:
    [5.1200562220521384e-08,
     -1.3891814596486002e-07,
      6.085232262219298e-07]

    translation delta norm:
    6.262749122403e-07

## Linux reference calibration

SHA256:

    d53335cc9b4bc8162d29024706a3d4fec69d02a71707d07c1c290206e323ca72

Camera 0:

    fx = 1362.5938983040937
    fy = 1365.3156875051586
    cx = 976.2452481016791
    cy = 539.9852303027392
    xi = 0.6446624100711089

Camera 1:

    fx = 1388.739623747452
    fy = 1389.595011613711
    cx = 957.3105629041225
    cy = 548.718946793617
    xi = 0.667747837678141

    rvec_cam1 =
    [0.004864203392448383,
     -0.3282409262735904,
     -0.26175000294477013]

    tvec_cam1 =
    [-0.026348464901948337,
      0.007777764638621594,
     -0.006312281297516686]

## Conclusion

For the tested Pair 02 large-board configuration, NanoDeepChArUco is numerically reproducible across macOS ARM and Linux x86-64.

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

> Numerically reproducible across the tested platforms, with negligible architecture-dependent floating-point differences.

The implementation should not be described as universally bit-for-bit deterministic across CPU architectures.

This validation applies specifically to the tested Pair 02 / large_7x7 configuration. Additional datasets or board profiles should be tested separately before making a broader universal reproducibility claim.
