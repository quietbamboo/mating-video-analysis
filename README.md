# Mating Video Analysis

A configurable computer-vision toolkit for analyzing chicken mating videos.
It provides modules for chicken-label detection, rooster detection and
multi-object tracking, and rooster/hen identity matching. The command-line
interfaces can be applied to user-provided videos, models, and data paths;
the published dataset is also available for training and evaluation.

## What you can do

- train a 60-class chicken color/mark detector;
- train a one-class rooster detector on a custom YOLO dataset;
- detect and track roosters in videos with YOLO and BoT-SORT;
- export trajectories in MOTChallenge and CVAT-compatible formats;
- infer rooster and mating-hen identities from videos and rooster logs;
- evaluate tracking and identity-matching results;
- adjust model, tracking, temporal, ROI, and filtering parameters from the
  command line.

The main processing flow is:

```text
Video
  ├─> rooster detection + BoT-SORT ─> trajectories / MOT annotations
  └─> chicken-label detection ──────> visible identity labels
                                           │
Rooster logs + mating intervals ───────────┴─> rooster/hen identity results
```

## Repository structure

| Directory | Application |
| --- | --- |
| [`chicken-label/`](chicken-label/) | Prepare data and train the chicken color/mark detector. |
| [`chicken-motion/`](chicken-motion/) | Prepare leakage-safe splits and train the rooster detector. |
| [`track-code/`](track-code/) | Track roosters and export or evaluate MOT annotations. |
| [`identify-matching/`](identify-matching/) | Run rooster and mating-hen identity attribution and analyze results. |
| [`docs/vest-pilot/`](docs/vest-pilot/) | Preliminary observations of vest wearing, daily counts, and descriptive summaries. |

## Requirements

- Python 3.10 or newer;
- a CUDA-capable GPU is recommended for training and video inference, although
  CPU execution is supported;
- FFmpeg-compatible video decoding through OpenCV/PyAV;
- trained YOLO weights for inference.

Each component has its own dependency file so that users only need to install
the part they intend to run.

## Data and model assets

The public dataset is hosted on Hugging Face:

**[njau-hjx/mating-video-analysis](https://huggingface.co/datasets/njau-hjx/mating-video-analysis)**

Download the required data from the dataset page, or use your own compatible
files through the command-line options described below. For inference, provide
your own trained YOLO weights with `--model` or `--label-model`. Large datasets,
model weights, and generated outputs are intentionally not stored in this Git
repository.

## Preliminary observations of vest wearing

We recorded mating counts across sequential periods with different vest-wearing
conditions under equal observation time. These preliminary descriptive observations
do not establish behavioral neutrality; further controlled experiments are needed
to evaluate potential vest-related effects.

[View the experimental summary, figure, and daily counts](docs/vest-pilot/)

## Quick start: track roosters in your own videos

Install the tracking dependencies:

```bash
cd track-code
python -m pip install -r code/requirements.txt
```

Run tracking with explicit input, model, and output paths:

```bash
python code/track_roosters_to_mot.py \
  --model /path/to/rooster-detector.pt \
  --input-dir /path/to/videos \
  --output-dir /path/to/tracking-results \
  --device 0 \
  --conf 0.5 \
  --iou 0.5 \
  --save-preview
```

Each input video produces MOT annotations, detection confidences, frame-level
quality statistics, a summary file, and optionally an annotated preview video.
See [`track-code/code/README_MOT.md`](track-code/code/README_MOT.md) for the
complete output layout and CVAT export details.

### Common tracking parameters

| Parameter | Purpose | Default |
| --- | --- | --- |
| `--model` | YOLO rooster-detector checkpoint | `track-code/models/chicken-rooster-11-29.pt` |
| `--input-dir` | Directory containing input videos | `track-code/dataset/` |
| `--output-dir` | Directory for MOT and report outputs | `track-code/mot_dataset/` |
| `--tracker` | BoT-SORT configuration file | bundled configuration |
| `--classes` | Model class IDs or names to track | auto-detect rooster class |
| `--conf` | Detection confidence threshold | `0.5` |
| `--iou` | NMS IoU threshold | `0.5` |
| `--imgsz` | YOLO inference image size | `1280` |
| `--device` | `auto`, `cpu`, `0`, or a GPU list such as `0,1` | `auto` |
| `--max-det` | Maximum detections passed to the tracker per frame | `4` |
| `--save-preview` | Save a video with bounding boxes and track IDs | disabled |
| `--extract-frames` | Export a complete MOTChallenge `img1/` directory | disabled |
| `--recursive` | Search for videos recursively | disabled |
| `--overwrite` | Replace existing results | disabled |

Run the following command to view every available option:

```bash
python code/track_roosters_to_mot.py --help
```

## Run identity matching

The identity-matching module combines mating-event metadata, videos, rooster
track logs, and a chicken-label detector.

```bash
cd identify-matching
python -m pip install -e .

python scripts/evaluate.py \
  --csv /path/to/mating_clips.csv \
  --videos-dir /path/to/videos \
  --logs-dir /path/to/rooster_logs \
  --label-model /path/to/chicken-label-detector.pt \
  --device 0 \
  --output /path/to/identity-results.csv \
  --save-video
```

Validate paths and record correspondence without loading the model:

```bash
python scripts/evaluate.py \
  --csv /path/to/mating_clips.csv \
  --videos-dir /path/to/videos \
  --logs-dir /path/to/rooster_logs \
  --validate-only
```

### Common identity-matching parameters

| Parameter | Purpose | Default |
| --- | --- | --- |
| `--label-conf` | Chicken-label detection confidence | `0.25` |
| `--label-iou` | Chicken-label NMS IoU | `0.70` |
| `--label-imgsz` | Label-model input size | `640` |
| `--roi-area-ratio` | Central rooster-box ROI area ratio | `1/9` |
| `--tau-gap-seconds` | Minimum label-disappearance interval | `1.0` s |
| `--trajectory-window-seconds` | Trajectory history duration | configurable |
| `--trajectory-window-fps-multiplier` | History length relative to video FPS | configurable |
| `--outlier-sigma-multiplier` | Trajectory-distance filtering range | `2` |
| `--detections-cache-dir` | Cache detections for parameter sweeps | disabled |
| `--method` | `proposed` or `nearest-baseline` | `proposed` |
| `--save-video` | Save annotated result videos | disabled |

See [`identify-matching/README.md`](identify-matching/README.md) for slicing,
ablation, caching, analysis, and debugging options.

## Train models on your own data

Both training modules accept custom dataset roots, YOLO checkpoints, devices,
batch sizes, epoch counts, and early-stopping patience.

### Chicken-label detector

```bash
cd chicken-label
python -m pip install -r requirements.txt
python code/train.py \
  --data-root /path/to/chicken-label-dataset \
  --model yolo11m.pt \
  --device 0 \
  --batch 24 \
  --epochs 1000
```

### Rooster detector

```bash
cd chicken-motion
python -m pip install -r requirements.txt
python code/train.py \
  --data-root /path/to/rooster-dataset \
  --model yolo11n.pt \
  --device 0 \
  --batch 48 \
  --epochs 500
```

The supplied YAML files define the expected YOLO class names and split layout.
Use `--data-config` to select another compatible configuration.

## Suggested local layout

The default paths expect the following structure. Every major path can be
overridden, so reproducing this layout is optional when using custom data.

```text
chicken-label/data/
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/

chicken-motion/data/
├── images/{train,val,test}/
└── labels/{train,val,test}/

track-code/
├── dataset/
├── models/chicken-rooster-11-29.pt
└── mot_dataset_true/

identify-matching/
├── data/
│   ├── mating_clips.csv
│   ├── rooster_logs/
│   └── videos/
└── models/chicken-label-12-23.pt
```

## Evaluation

- Tracking evaluation: see
  [`track-code/code/README_TRACKING_EVAL.md`](track-code/code/README_TRACKING_EVAL.md).
- Identity-result analysis: see
  [`identify-matching/README.md`](identify-matching/README.md#analyze-generated-results).

These evaluation workflows are optional. Users who only want to process their
own videos can use the tracking and identity-matching commands directly.

## License

See [LICENSE](LICENSE) for the terms of use.
