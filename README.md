# Reproducibility code

This repository contains the code used for label detection, rooster detection,
multi-object tracking, and rooster/hen identity matching. 

Public dataset: [njau-hjx/mating-video-analysis](https://huggingface.co/datasets/njau-hjx/mating-video-analysis)

## Components

| Directory | Purpose |
| --- | --- |
| `chicken-label/` | Train the 60-class color/mark detector. |
| `chicken-motion/` | Split data without source leakage and train the rooster detector. |
| `track-code/` | Generate BoT-SORT trajectories and evaluate them against manual MOT annotations. |
| `identify-matching/` | Reproduce rooster and mating-hen identity attribution. |

## Place the separately downloaded data

After downloading and extracting the public dataset, arrange or link it as
follows. These paths are ignored by Git.

```text
chicken-label/data/
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/

chicken-motion/data/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
└── labels/test/

track-code/
├── dataset/                 # 14 tracking-evaluation videos
└── mot_dataset_true/        # manual MOT ground truth

identify-matching/data/
├── mating_clips.csv
├── rooster_logs/
└── videos/
```
