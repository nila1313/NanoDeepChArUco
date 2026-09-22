# NanoDeepChArUco

NanoDeepChArUco is a stereo ChArUco detection front end that produces
CalibCam-compatible detections from two camera videos.

The pipeline uses explicit frame sampling and offsets, so the temporal
alignment is controlled directly by the user.

Three detector modes are available:

- `opencv` — OpenCV ArUco/ChArUco baseline
- `nano` — ArUco Nano marker detection followed by ChArUco interpolation
- `hybrid` — ArUco Nano with gamma retry and DeepChArUco recovery

The generated detection payloads can be passed directly to CalibCam.

## Pipeline

```text
left/right videos
        │
        ▼
frame sampling + explicit offsets
        │
        ▼
OpenCV / ArUco Nano / Hybrid detector
        │
        ▼
ChArUco corners
        │
        ▼
CalibCam-compatible detection payloads
        │
        ▼
optional CalibCam calibration
```

## Installation

Requirements:

- Git
- Git LFS
- Conda or Miniforge

Clone the repository together with the DeepChArUco submodule:

```bash
git clone --recurse-submodules \
  https://github.com/nila1313/NanoDeepChArUco.git

cd NanoDeepChArUco
```

Run the setup script:

```bash
./setup.sh
```

Then activate the environment:

```bash
conda activate nanodeepcharuco
```

## Board files

Two board definitions are included:

```text
configs/boards/
├── large_7x7_dict4x4_50.npy
└── small_5x6_dict6x6_250_meters.npy
```

The small board uses `DICT_6X6_250`.

The large board uses `DICT_4X4_50`.

## Nano example

```bash
nanodeepcharuco \
  --videos \
  /path/to/left.MP4 \
  /path/to/right.MP4 \
  --board \
  configs/boards/small_5x6_dict6x6_250_meters.npy \
  --detector nano \
  --frames_start 0 \
  --frames_end 2640 \
  --frames_step 20 \
  --frames_offsets 0 1 \
  --data_path runs/example_nano
```

## Frame-offset semantics

Offsets specify which physical video frame is used for each synchronized sample.

For example:

```text
--frames_offsets 0 1
```

means:

```text
Left(t) ↔ Right(t + 1)
```

while:

```text
--frames_offsets 1 0
```

means:

```text
Left(t + 1) ↔ Right(t)
```

The output keeps synchronized sample IDs separate from physical frame numbers:

- `detection_idxs` stores the shared synchronized sample index
- `frame_idxs` stores the actual physical frame used from each video

For example, with:

```text
frames_start = 420
frames_step = 20
frames_offsets = 0 1
```

the first synchronized sample uses:

```text
left frame  = 420
right frame = 421
detection index = 0 for both cameras
```

## Hybrid example

The hybrid detector combines ArUco Nano with DeepChArUco recovery.

Example using the included large-board detector:

```bash
nanodeepcharuco \
  --videos \
  /path/to/left.MP4 \
  /path/to/right.MP4 \
  --board \
  configs/boards/large_7x7_dict4x4_50.npy \
  --detector hybrid \
  --deep_checkpoint \
  models/deepcharuco/large_7x7/detector.ckpt \
  --refinenet_checkpoint \
  models/refinenet/refinenet.ckpt \
  --deep_config \
  configs/deepcharuco/pair2_epoch146.yaml \
  --frames_step 20 \
  --frames_offsets 0 1 \
  --data_path runs/example_hybrid
```

The pinned DeepChArUco source is stored as a Git submodule under:

```text
third_party/deepcharuco/upstream
```

## Output

A detection run creates:

```text
runs/example/
├── inputs/
│   ├── detection_000.npy
│   └── detection_001.npy
├── run_manifest.json
└── work/
```

`work/` contains temporary detector files when required by the selected detector.

Each detection file follows the CalibCam-compatible array schema and contains:

```text
version
storage_method
marker_coords
marker_ids
detection_idxs
frame_idxs
```

## Running CalibCam

CalibCam is optional and can be kept in a separate Python environment.

To continue directly into CalibCam, provide its Python interpreter:

```bash
nanodeepcharuco \
  --videos \
  /path/to/left.MP4 \
  /path/to/right.MP4 \
  --board \
  configs/boards/small_5x6_dict6x6_250_meters.npy \
  --detector nano \
  --frames_step 20 \
  --frames_offsets 0 1 \
  --models omnidir omnidir \
  --projection perspective \
  --data_path runs/example_calibration \
  --run_calibcam \
  --calibration_single \
  --calibration_multi \
  --calibcam_python /path/to/calibcam/environment/bin/python
```

CalibCam results are written under:

```text
runs/example_calibration/calibcam_output/
```

## Models

Runtime checkpoints are stored with Git LFS:

```text
models/
├── deepcharuco/
│   ├── large_7x7/
│   │   └── detector.ckpt
│   └── small_5x6/
│       └── detector.ckpt
└── refinenet/
    └── refinenet.ckpt
```

## Native ArUco Nano

The native ArUco Nano source is stored under:

```text
third_party/aruco_nano/source/
```

Build it independently with:

```bash
bash scripts/build_nano.sh
```

The generated executable is:

```text
third_party/aruco_nano/build/detect_batch
```

The build directory is generated locally and is not committed.

## Tests

Run the unit tests with:

```bash
pytest -q
```

Check the command-line interface with:

```bash
nanodeepcharuco --help
```

## Repository branches

`main`

The clean explicit-offset NanoDeepChArUco pipeline documented here.

`feature/advanced-auto-sync-pipeline`

Preserves the previous advanced pipeline with automatic synchronization and the broader calibration workflow.

`feature/two-stage-calibration`

Preserves the focused two-stage calibration development branch.

## Third-party components

ArUco Nano and DeepChArUco retain their own source provenance and licensing information in their respective third-party directories.

## License

The original NanoDeepChArUco source code in this repository is licensed under the MIT License. See `LICENSE`.

Third-party components retain their own licenses.

The model checkpoint files under `models/` are not covered by the repository MIT License unless separate licensing terms explicitly state otherwise.
