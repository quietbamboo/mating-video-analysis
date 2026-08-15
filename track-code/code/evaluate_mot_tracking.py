#!/usr/bin/env python3
"""Evaluate rooster trajectories with the official TrackEval metrics.

The expected input layout is the one produced by ``track_roosters_to_mot.py``:

    predictions/<sequence>/gt/gt.txt
    ground_truth/<sequence>/gt/gt.txt

Although the prediction file is named ``gt.txt`` for CVAT import compatibility,
it is treated as tracker output by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from configparser import ConfigParser, Error as ConfigParserError
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Represent a path without recording a user-specific absolute prefix."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


@dataclass(frozen=True)
class Detection:
    frame: int
    track_id: int
    box: tuple[float, float, float, float]
    mark_or_confidence: float
    class_id: int | None


PERCENT_FIELDS = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "CLR_Re",
    "CLR_Pr",
    "IDF1",
    "IDR",
    "IDP",
)

COUNT_FIELDS = (
    "CLR_TP",
    "CLR_FP",
    "CLR_FN",
    "IDSW",
    "Frag",
    "IDTP",
    "IDFP",
    "IDFN",
)

CSV_COLUMNS = (
    "sequence",
    "frames",
    "gt_detections",
    "tracker_detections",
    "gt_trajectories",
    "tracker_trajectories",
    "HOTA_pct",
    "DetA_pct",
    "AssA_pct",
    "LocA_pct",
    "MOTA_pct",
    "MOTP_pct",
    "IDF1_pct",
    "IDP_pct",
    "IDR_pct",
    "Recall_pct",
    "Precision_pct",
    "TP",
    "FP",
    "FN",
    "IDSW",
    "Frag",
    "IDTP",
    "IDFP",
    "IDFN",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute official HOTA, CLEAR MOT, and Identity metrics for the "
            "shared sequences in two MOT-format directories."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "mot_dataset",
        help="BoT-SORT output root (default: mot_dataset).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REPO_ROOT / "mot_dataset_true",
        help="Manually corrected ground-truth root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "mot_evaluation",
        help="Directory for CSV and JSON results.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for CLEAR and Identity metrics (default: 0.5).",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=1,
        help=(
            "Ground-truth class ID to evaluate; use a negative value to keep "
            "all classes (default: 1)."
        ),
    )
    return parser.parse_args()


def read_mot_file(
    path: Path,
    *,
    is_ground_truth: bool,
    class_id: int | None,
) -> list[Detection]:
    if not path.is_file():
        raise FileNotFoundError(path)

    detections: list[Detection] = []
    seen_frame_ids: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 6:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 6 columns, got {len(row)}"
                )
            try:
                frame = int(float(row[0]))
                track_id = int(float(row[1]))
                box = tuple(float(value) for value in row[2:6])
                mark_or_confidence = float(row[6]) if len(row) > 6 else 1.0
                row_class = int(float(row[7])) if len(row) > 7 else None
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric MOT value") from exc

            if frame < 1:
                raise ValueError(f"{path}:{line_number}: frame IDs must start at 1")
            if track_id < 0:
                raise ValueError(f"{path}:{line_number}: track ID must be non-negative")
            if not all(math.isfinite(value) for value in box):
                raise ValueError(f"{path}:{line_number}: box contains a non-finite value")
            if box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"{path}:{line_number}: width and height must be positive")

            # MOT ground truth column 7 is the zero/one "mark" flag. Tracker
            # output uses this position for confidence and is not filtered.
            if is_ground_truth and mark_or_confidence <= 0:
                continue
            if (
                is_ground_truth
                and class_id is not None
                and row_class is not None
                and row_class != class_id
            ):
                continue

            key = (frame, track_id)
            if key in seen_frame_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate frame/track pair {key}"
                )
            seen_frame_ids.add(key)
            detections.append(
                Detection(frame, track_id, box, mark_or_confidence, row_class)
            )

    return detections


def sequence_length(sequence_dir: Path, detections: Iterable[Detection]) -> int:
    """Use seqinfo.ini when available, otherwise infer the final annotated frame."""
    inferred = max((det.frame for det in detections), default=0)
    seqinfo = sequence_dir / "seqinfo.ini"
    if not seqinfo.is_file():
        return inferred

    parser = ConfigParser()
    parser.read(seqinfo, encoding="utf-8")
    try:
        declared = parser.getint("Sequence", "seqLength")
    except (ValueError, ConfigParserError):
        return inferred
    if inferred > declared:
        raise ValueError(
            f"{seqinfo}: contains detections through frame {inferred}, "
            f"beyond declared seqLength={declared}"
        )
    return declared


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xywh boxes, matching TrackEval's implementation."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    a = boxes_a.astype(float, copy=True)
    b = boxes_b.astype(float, copy=True)
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection_wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(
        a[:, 3] - a[:, 1], 0.0
    )
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(
        b[:, 3] - b[:, 1], 0.0
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > np.finfo(float).eps,
    )


def remap_ids(detections: list[Detection]) -> tuple[dict[int, int], int]:
    ids = sorted({det.track_id for det in detections})
    return {track_id: index for index, track_id in enumerate(ids)}, len(ids)


def build_trackeval_data(
    ground_truth: list[Detection],
    predictions: list[Detection],
    num_timesteps: int,
) -> dict[str, object]:
    gt_by_frame: dict[int, list[Detection]] = defaultdict(list)
    pred_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in ground_truth:
        gt_by_frame[detection.frame].append(detection)
    for detection in predictions:
        pred_by_frame[detection.frame].append(detection)

    gt_id_map, num_gt_ids = remap_ids(ground_truth)
    pred_id_map, num_tracker_ids = remap_ids(predictions)
    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarity_scores: list[np.ndarray] = []

    for frame in range(1, num_timesteps + 1):
        frame_gt = sorted(gt_by_frame[frame], key=lambda item: item.track_id)
        frame_pred = sorted(pred_by_frame[frame], key=lambda item: item.track_id)
        gt_ids.append(
            np.asarray([gt_id_map[item.track_id] for item in frame_gt], dtype=int)
        )
        tracker_ids.append(
            np.asarray(
                [pred_id_map[item.track_id] for item in frame_pred], dtype=int
            )
        )
        gt_boxes = np.asarray([item.box for item in frame_gt], dtype=float).reshape(
            -1, 4
        )
        pred_boxes = np.asarray(
            [item.box for item in frame_pred], dtype=float
        ).reshape(-1, 4)
        similarity_scores.append(box_iou(gt_boxes, pred_boxes))

    return {
        "num_timesteps": num_timesteps,
        "num_gt_ids": num_gt_ids,
        "num_tracker_ids": num_tracker_ids,
        "num_gt_dets": len(ground_truth),
        "num_tracker_dets": len(predictions),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
    }


def scalar_hota_result(result: dict[str, object]) -> dict[str, float]:
    """TrackEval reports HOTA fields over 19 thresholds; report their mean."""
    output: dict[str, float] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            output[key] = float(np.mean(value))
        else:
            output[key] = float(value)
    return output


def merge_results(
    hota: dict[str, object],
    clear: dict[str, object],
    identity: dict[str, object],
) -> dict[str, float]:
    return {
        **scalar_hota_result(hota),
        **{key: float(value) for key, value in clear.items()},
        **{key: float(value) for key, value in identity.items()},
    }


def csv_row(
    sequence: str,
    frames: int,
    gt_detections: int,
    tracker_detections: int,
    gt_trajectories: int,
    tracker_trajectories: int,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "frames": frames,
        "gt_detections": gt_detections,
        "tracker_detections": tracker_detections,
        "gt_trajectories": gt_trajectories,
        "tracker_trajectories": tracker_trajectories,
        "HOTA_pct": 100 * metrics["HOTA"],
        "DetA_pct": 100 * metrics["DetA"],
        "AssA_pct": 100 * metrics["AssA"],
        "LocA_pct": 100 * metrics["LocA"],
        "MOTA_pct": 100 * metrics["MOTA"],
        "MOTP_pct": 100 * metrics["MOTP"],
        "IDF1_pct": 100 * metrics["IDF1"],
        "IDP_pct": 100 * metrics["IDP"],
        "IDR_pct": 100 * metrics["IDR"],
        "Recall_pct": 100 * metrics["CLR_Re"],
        "Precision_pct": 100 * metrics["CLR_Pr"],
        "TP": int(metrics["CLR_TP"]),
        "FP": int(metrics["CLR_FP"]),
        "FN": int(metrics["CLR_FN"]),
        "IDSW": int(metrics["IDSW"]),
        "Frag": int(metrics["Frag"]),
        "IDTP": int(metrics["IDTP"]),
        "IDFP": int(metrics["IDFP"]),
        "IDFN": int(metrics["IDFN"]),
    }


def print_summary(row: dict[str, object]) -> None:
    print(
        f"{row['sequence']}: "
        f"HOTA={row['HOTA_pct']:.2f}, "
        f"MOTA={row['MOTA_pct']:.2f}, "
        f"IDF1={row['IDF1_pct']:.2f}, "
        f"IDSW={row['IDSW']}, "
        f"FP={row['FP']}, FN={row['FN']}"
    )


def find_sequences(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        path.parent.parent.name
        for path in root.glob("*/gt/gt.txt")
        if path.is_file()
    }


def main() -> int:
    args = parse_args()
    try:
        from trackeval.metrics import CLEAR, HOTA, Identity
    except ImportError as exc:  # pragma: no cover - optional until evaluation
        raise SystemExit(
            "TrackEval is required. Install the pinned dependency with:\n"
            "  python -m pip install trackeval==1.3.0\n"
            "or run without modifying the environment:\n"
            "  uv run --with trackeval==1.3.0 python code/evaluate_mot_tracking.py"
        ) from exc
    if not 0 < args.iou_threshold <= 1:
        raise SystemExit("--iou-threshold must be in (0, 1]")
    selected_class = None if args.class_id < 0 else args.class_id

    pred_sequences = find_sequences(args.predictions)
    gt_sequences = find_sequences(args.ground_truth)
    common_sequences = sorted(pred_sequences & gt_sequences)
    if not common_sequences:
        raise SystemExit("No shared sequence containing gt/gt.txt was found.")

    skipped_predictions = sorted(pred_sequences - gt_sequences)
    missing_predictions = sorted(gt_sequences - pred_sequences)
    if skipped_predictions:
        print(
            f"Note: skipping {len(skipped_predictions)} prediction sequence(s) "
            "without manual ground truth.",
            file=sys.stderr,
        )
    if missing_predictions:
        raise SystemExit(
            "Ground-truth sequences without predictions: "
            + ", ".join(missing_predictions)
        )

    metric_config = {
        "THRESHOLD": args.iou_threshold,
        "PRINT_CONFIG": False,
    }
    hota_metric = HOTA({"PRINT_CONFIG": False})
    clear_metric = CLEAR(metric_config)
    identity_metric = Identity(metric_config)

    sequence_hota: dict[str, dict[str, object]] = {}
    sequence_clear: dict[str, dict[str, object]] = {}
    sequence_identity: dict[str, dict[str, object]] = {}
    sequence_data: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for sequence in common_sequences:
        pred_dir = args.predictions / sequence
        gt_dir = args.ground_truth / sequence
        predictions = read_mot_file(
            pred_dir / "gt" / "gt.txt",
            is_ground_truth=False,
            class_id=None,
        )
        ground_truth = read_mot_file(
            gt_dir / "gt" / "gt.txt",
            is_ground_truth=True,
            class_id=selected_class,
        )
        frames = sequence_length(pred_dir, [*ground_truth, *predictions])
        data = build_trackeval_data(ground_truth, predictions, frames)
        sequence_data[sequence] = data

        sequence_hota[sequence] = hota_metric.eval_sequence(data)
        sequence_clear[sequence] = clear_metric.eval_sequence(data)
        sequence_identity[sequence] = identity_metric.eval_sequence(data)
        metrics = merge_results(
            sequence_hota[sequence],
            sequence_clear[sequence],
            sequence_identity[sequence],
        )
        row = csv_row(
            sequence,
            frames,
            len(ground_truth),
            len(predictions),
            int(data["num_gt_ids"]),
            int(data["num_tracker_ids"]),
            metrics,
        )
        rows.append(row)
        print_summary(row)

    combined_hota = hota_metric.combine_sequences(sequence_hota)
    combined_clear = clear_metric.combine_sequences(sequence_clear)
    combined_identity = identity_metric.combine_sequences(sequence_identity)
    combined_metrics = merge_results(
        combined_hota, combined_clear, combined_identity
    )
    overall = csv_row(
        "OVERALL",
        sum(int(data["num_timesteps"]) for data in sequence_data.values()),
        sum(int(data["num_gt_dets"]) for data in sequence_data.values()),
        sum(int(data["num_tracker_dets"]) for data in sequence_data.values()),
        sum(int(data["num_gt_ids"]) for data in sequence_data.values()),
        sum(int(data["num_tracker_ids"]) for data in sequence_data.values()),
        combined_metrics,
    )
    rows.append(overall)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tracking_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    json_path = args.output_dir / "tracking_metrics.json"
    json_payload = {
        "evaluation": {
            "implementation": "TrackEval 1.3.0",
            "clear_and_identity_iou_threshold": args.iou_threshold,
            "hota_iou_thresholds": [
                round(float(value), 2) for value in hota_metric.array_labels
            ],
            "ground_truth_class_id": selected_class,
            "prediction_root": portable_path(args.predictions),
            "ground_truth_root": portable_path(args.ground_truth),
            "evaluated_sequences": common_sequences,
            "skipped_prediction_sequences": skipped_predictions,
        },
        "per_sequence": rows[:-1],
        "overall": overall,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nCombined result")
    print_summary(overall)
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
