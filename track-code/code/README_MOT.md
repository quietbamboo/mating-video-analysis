# Rooster multi-object tracking and MOT export

`track_roosters_to_mot.py` runs the trained rooster detector with BoT-SORT and
exports MOT 1.0 annotations that can also be imported into CVAT.

The videos and manual annotations are part of the separately published public
dataset, not the code repository. After downloading them, use this layout:

```text
track-code/
├── dataset/                         # 14 videos
├── mot_dataset_true/                # manual MOT ground truth
├── models/chicken-rooster-11-29.pt
└── code/
```

## Install and run

Run from `track-code/`:

```bash
python -m pip install -r code/requirements.txt
python code/track_roosters_to_mot.py
```

Defaults:

- model: `models/chicken-rooster-11-29.pt`;
- input: `dataset/`;
- output: `mot_dataset/`;
- tracker: `code/botsort_rooster.yaml`;
- maximum and expected detections per frame: 4;
- inference size: 1280;
- MOT/CVAT label: `rooster`.

All paths can be overridden:

```bash
python code/track_roosters_to_mot.py \
  --model /path/to/rooster.pt \
  --input-dir /path/to/videos \
  --output-dir /path/to/predictions \
  --device 0 --save-preview
```

Add `--extract-frames` to create a complete MOTChallenge `img1/` directory.
Add `--overwrite` to regenerate an existing sequence directory.

## Output

Each video produces:

```text
mot_dataset/<sequence>/
├── <source-video>.mp4
├── seqinfo.ini
├── gt/gt.txt
├── gt/labels.txt
├── cvat_mot_annotations.zip
├── detections_with_confidence.csv
├── frame_counts.csv
├── summary.json
├── preview.mp4       # with --save-preview
└── img1/             # with --extract-frames
```

The nine MOT columns are `frame_id, track_id, x, y, width, height,
mark/confidence, class_id, visibility`. Detection confidence is additionally
saved in `detections_with_confidence.csv`.
