#!/usr/bin/env python3
"""Run YOLO + BoT-SORT on videos and export CVAT-compatible MOT annotations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class VideoSummary:
    video: str
    frames: int
    source_frames_reported: int
    fps: float
    width: int
    height: int
    mot_rows: int
    unique_track_ids: int
    raw_track_ids: list[int]
    selected_model_classes: dict[int, str]
    expected_objects_per_frame: int
    frames_with_expected_objects: int
    frames_with_fewer_objects: int
    frames_with_more_objects: int
    mean_tracked_objects_per_frame: float
    min_tracked_objects_per_frame: int
    max_tracked_objects_per_frame: int
    elapsed_seconds: float
    processing_fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用训练好的 YOLO 模型和 BoT-SORT 跟踪视频中的公鸡，"
            "并生成 MOT/CVAT 标注。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "chicken-rooster-11-29.pt",
        help="YOLO .pt 模型路径",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="输入视频目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "mot_dataset",
        help="输出目录",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        default=Path(__file__).with_name("botsort_rooster.yaml"),
        help="BoT-SORT 配置文件",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="ID_OR_NAME",
        help=(
            "要跟踪的模型类别，可传类别 ID 或名称。省略时自动选择名称包含 "
            "rooster/cock/公鸡 的类别；单类别模型则自动选择唯一类别"
        ),
    )
    parser.add_argument("--label", default="rooster", help="CVAT/MOT 中的标签名称")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO 推理图像尺寸")
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备，如 0、0,1、cpu；auto 自动选择",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=4,
        help="每帧送入跟踪器的最大检测数；设为 0 表示不限制（Ultralytics 默认 300）",
    )
    parser.add_argument(
        "--expected-objects",
        type=int,
        default=4,
        help="用于质量报告的每帧预期公鸡数量，不会伪造缺失框",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="额外生成严格 MOTChallenge 的 img1/*.jpg（占用磁盘较大）",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="保存带框和 ID 的预览视频 preview.mp4",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索输入目录中的视频",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的同名视频输出目录",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"模型不存在：{args.model}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{args.input_dir}")
    if not args.tracker.is_file():
        raise FileNotFoundError(f"跟踪器配置不存在：{args.tracker}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf 必须在 [0, 1] 范围内")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou 必须在 [0, 1] 范围内")
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0")
    if args.max_det < 0:
        raise ValueError("--max-det 不能小于 0")
    if args.expected_objects <= 0:
        raise ValueError("--expected-objects 必须大于 0")
    if not args.label.strip() or "\n" in args.label:
        raise ValueError("--label 不能为空或包含换行符")


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def normalize_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {class_id: str(name) for class_id, name in enumerate(names)}
    raise TypeError(f"无法解析模型类别名称：{type(names).__name__}")


def resolve_classes(
    model_names: dict[int, str], requested: list[str] | None
) -> list[int]:
    if requested:
        by_name = {name.casefold(): class_id for class_id, name in model_names.items()}
        selected: list[int] = []
        for value in requested:
            try:
                class_id = int(value)
            except ValueError:
                class_id = by_name.get(value.casefold(), -1)
            if class_id not in model_names:
                available = ", ".join(
                    f"{class_id}:{name}" for class_id, name in model_names.items()
                )
                raise ValueError(f"未知类别 {value!r}；模型类别为：{available}")
            if class_id not in selected:
                selected.append(class_id)
        return selected

    rooster_terms = ("rooster", "cock", "公鸡")
    inferred = [
        class_id
        for class_id, name in model_names.items()
        if any(term in name.casefold() for term in rooster_terms)
    ]
    if inferred:
        return inferred
    if len(model_names) == 1:
        return list(model_names)
    available = ", ".join(
        f"{class_id}:{name}" for class_id, name in model_names.items()
    )
    raise ValueError(
        "模型含有多个类别，无法自动判断公鸡类别。"
        f"请使用 --classes 指定。可用类别：{available}"
    )


def read_video_info(video_path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    info = VideoInfo(
        width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    )
    capture.release()
    if info.width <= 0 or info.height <= 0 or info.fps <= 0:
        raise RuntimeError(f"视频元数据无效：{video_path} ({info})")
    return info


def prepare_sequence_dir(sequence_dir: Path, overwrite: bool) -> None:
    if sequence_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出已存在：{sequence_dir}；如需重做请添加 --overwrite"
            )
        resolved_output = sequence_dir.resolve()
        if resolved_output.parent == resolved_output or len(resolved_output.parts) < 3:
            raise RuntimeError(f"拒绝删除不安全的输出路径：{resolved_output}")
        shutil.rmtree(resolved_output)
    (sequence_dir / "gt").mkdir(parents=True)


def write_seqinfo(
    sequence_dir: Path, sequence_name: str, info: VideoInfo, sequence_length: int
) -> None:
    text = (
        "[Sequence]\n"
        f"name={sequence_name}\n"
        "imDir=img1\n"
        f"frameRate={info.fps:.8g}\n"
        f"seqLength={sequence_length}\n"
        f"imWidth={info.width}\n"
        f"imHeight={info.height}\n"
        "imExt=.jpg\n"
    )
    (sequence_dir / "seqinfo.ini").write_text(text, encoding="utf-8")


def make_cvat_zip(sequence_dir: Path) -> Path:
    zip_path = sequence_dir / "cvat_mot_annotations.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(sequence_dir / "gt" / "gt.txt", "gt/gt.txt")
        archive.write(sequence_dir / "gt" / "labels.txt", "gt/labels.txt")
    return zip_path


def validate_mot_file(mot_path: Path, info: VideoInfo, processed_frames: int) -> int:
    row_count = 0
    with mot_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 9:
                raise ValueError(
                    f"{mot_path}:{line_number} 应有 9 列，实际为 {len(row)} 列"
                )
            frame_id, track_id, x, y, width, height = (
                int(row[0]),
                int(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            )
            if not (1 <= frame_id <= processed_frames):
                raise ValueError(f"{mot_path}:{line_number} 帧号越界：{frame_id}")
            if track_id <= 0:
                raise ValueError(f"{mot_path}:{line_number} track ID 非正数")
            if width <= 0 or height <= 0:
                raise ValueError(f"{mot_path}:{line_number} 检测框尺寸无效")
            tolerance = 1.01
            if (
                x < -tolerance
                or y < -tolerance
                or x + width > info.width + tolerance
                or y + height > info.height + tolerance
            ):
                raise ValueError(f"{mot_path}:{line_number} 检测框超出图像边界")
            row_count += 1
    return row_count


def iter_tracked_boxes(result: Any) -> Iterable[tuple[int, int, float, float, float, float, float]]:
    boxes = result.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return
    xyxy_values = boxes.xyxy.detach().cpu().tolist()
    track_ids = boxes.id.detach().cpu().int().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().int().tolist()
    for xyxy, track_id, confidence, class_id in zip(
        xyxy_values, track_ids, confidences, class_ids
    ):
        yield (
            int(track_id),
            int(class_id),
            float(confidence),
            float(xyxy[0]),
            float(xyxy[1]),
            float(xyxy[2]),
            float(xyxy[3]),
        )


def process_video(
    model: Any,
    video_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    selected_classes: list[int],
    model_names: dict[int, str],
) -> VideoSummary:
    info = read_video_info(video_path)
    sequence_dir = output_root / video_path.stem
    prepare_sequence_dir(sequence_dir, args.overwrite)
    if args.extract_frames:
        (sequence_dir / "img1").mkdir()

    (sequence_dir / "gt" / "labels.txt").write_text(
        args.label.strip() + "\n", encoding="utf-8"
    )
    shutil.copy2(video_path, sequence_dir / video_path.name)

    mot_path = sequence_dir / "gt" / "gt.txt"
    audit_path = sequence_dir / "detections_with_confidence.csv"
    counts_path = sequence_dir / "frame_counts.csv"
    preview_writer: cv2.VideoWriter | None = None
    raw_to_mot_id: dict[int, int] = {}
    frame_counts: list[int] = []
    started = time.perf_counter()

    track_kwargs: dict[str, Any] = {
        "source": str(video_path),
        "stream": True,
        "tracker": str(args.tracker),
        "persist": False,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "classes": selected_classes,
        "verbose": False,
    }
    if args.max_det:
        track_kwargs["max_det"] = args.max_det
    if args.device != "auto":
        track_kwargs["device"] = args.device

    with (
        mot_path.open("w", encoding="utf-8", newline="") as mot_handle,
        audit_path.open("w", encoding="utf-8-sig", newline="") as audit_handle,
        counts_path.open("w", encoding="utf-8-sig", newline="") as counts_handle,
    ):
        mot_writer = csv.writer(mot_handle, lineterminator="\n")
        audit_writer = csv.writer(audit_handle, lineterminator="\n")
        counts_writer = csv.writer(counts_handle, lineterminator="\n")
        audit_writer.writerow(
            [
                "frame_id",
                "mot_track_id",
                "raw_botsort_id",
                "model_class_id",
                "model_class_name",
                "confidence",
                "x",
                "y",
                "width",
                "height",
            ]
        )
        counts_writer.writerow(["frame_id", "tracked_objects", "expected_objects", "status"])

        results = model.track(**track_kwargs)
        for frame_id, result in enumerate(results, start=1):
            frame = result.orig_img
            if frame is None:
                raise RuntimeError(f"{video_path} 第 {frame_id} 帧缺少原始图像")

            if args.extract_frames:
                frame_path = sequence_dir / "img1" / f"{frame_id:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"写入帧失败：{frame_path}")

            if args.save_preview:
                if preview_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    preview_writer = cv2.VideoWriter(
                        str(sequence_dir / "preview.mp4"),
                        fourcc,
                        info.fps,
                        (info.width, info.height),
                    )
                    if not preview_writer.isOpened():
                        raise RuntimeError("无法创建 preview.mp4")
                preview_writer.write(result.plot())

            tracked_count = 0
            for raw_id, class_id, confidence, x1, y1, x2, y2 in iter_tracked_boxes(result):
                x1 = min(max(x1, 0.0), float(info.width))
                y1 = min(max(y1, 0.0), float(info.height))
                x2 = min(max(x2, x1), float(info.width))
                y2 = min(max(y2, y1), float(info.height))
                width = x2 - x1
                height = y2 - y1
                if width <= 0.0 or height <= 0.0:
                    continue

                mot_track_id = raw_to_mot_id.setdefault(raw_id, len(raw_to_mot_id) + 1)
                # CVAT MOT: frame, track, x, y, w, h, not_ignored, class, visibility
                mot_writer.writerow(
                    [
                        frame_id,
                        mot_track_id,
                        f"{x1:.3f}",
                        f"{y1:.3f}",
                        f"{width:.3f}",
                        f"{height:.3f}",
                        1,
                        1,
                        "1.0",
                    ]
                )
                audit_writer.writerow(
                    [
                        frame_id,
                        mot_track_id,
                        raw_id,
                        class_id,
                        model_names[class_id],
                        f"{confidence:.6f}",
                        f"{x1:.3f}",
                        f"{y1:.3f}",
                        f"{width:.3f}",
                        f"{height:.3f}",
                    ]
                )
                tracked_count += 1

            frame_counts.append(tracked_count)
            status = (
                "ok"
                if tracked_count == args.expected_objects
                else ("fewer" if tracked_count < args.expected_objects else "more")
            )
            counts_writer.writerow(
                [frame_id, tracked_count, args.expected_objects, status]
            )
            if frame_id == 1 or frame_id % 50 == 0:
                print(
                    f"  {video_path.name}: {frame_id}/{info.frame_count or '?'} 帧",
                    flush=True,
                )

    if preview_writer is not None:
        preview_writer.release()

    processed_frames = len(frame_counts)
    if processed_frames == 0:
        raise RuntimeError(f"视频未产生任何帧：{video_path}")
    mot_rows = validate_mot_file(mot_path, info, processed_frames)
    write_seqinfo(sequence_dir, video_path.stem, info, processed_frames)
    zip_path = make_cvat_zip(sequence_dir)
    with zipfile.ZipFile(zip_path) as archive:
        if set(archive.namelist()) != {"gt/gt.txt", "gt/labels.txt"}:
            raise RuntimeError(f"CVAT ZIP 内容异常：{zip_path}")

    elapsed = time.perf_counter() - started
    summary = VideoSummary(
        video=video_path.name,
        frames=processed_frames,
        source_frames_reported=info.frame_count,
        fps=info.fps,
        width=info.width,
        height=info.height,
        mot_rows=mot_rows,
        unique_track_ids=len(raw_to_mot_id),
        raw_track_ids=sorted(raw_to_mot_id),
        selected_model_classes={
            class_id: model_names[class_id] for class_id in selected_classes
        },
        expected_objects_per_frame=args.expected_objects,
        frames_with_expected_objects=sum(
            count == args.expected_objects for count in frame_counts
        ),
        frames_with_fewer_objects=sum(
            count < args.expected_objects for count in frame_counts
        ),
        frames_with_more_objects=sum(
            count > args.expected_objects for count in frame_counts
        ),
        mean_tracked_objects_per_frame=sum(frame_counts) / processed_frames,
        min_tracked_objects_per_frame=min(frame_counts),
        max_tracked_objects_per_frame=max(frame_counts),
        elapsed_seconds=elapsed,
        processing_fps=processed_frames / elapsed if elapsed else 0.0,
    )
    (sequence_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_dataset_summary(output_dir: Path, summaries: list[VideoSummary]) -> None:
    payload = {
        "format": "MOT 1.0 / CVAT-compatible",
        "videos": [asdict(summary) for summary in summaries],
        "totals": {
            "videos": len(summaries),
            "frames": sum(summary.frames for summary in summaries),
            "mot_rows": sum(summary.mot_rows for summary in summaries),
            "frames_with_expected_objects": sum(
                summary.frames_with_expected_objects for summary in summaries
            ),
            "frames_with_fewer_objects": sum(
                summary.frames_with_fewer_objects for summary in summaries
            ),
            "frames_with_more_objects": sum(
                summary.frames_with_more_objects for summary in summaries
            ),
        },
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "缺少 ultralytics。请先运行：python -m pip install -r code/requirements.txt"
        ) from exc
    videos = find_videos(args.input_dir, args.recursive)
    if not videos:
        raise FileNotFoundError(f"未在 {args.input_dir} 中找到视频文件")

    print(f"加载模型：{args.model}", flush=True)
    model = YOLO(str(args.model))
    model_names = normalize_model_names(model.names)
    selected_classes = resolve_classes(model_names, args.classes)
    selected_text = ", ".join(
        f"{class_id}:{model_names[class_id]}" for class_id in selected_classes
    )
    print(f"模型类别：{model_names}")
    print(f"跟踪类别：{selected_text}")
    print(f"视频数量：{len(videos)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[VideoSummary] = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] 处理 {video_path.name}", flush=True)
        summary = process_video(
            model,
            video_path,
            args.output_dir,
            args,
            selected_classes,
            model_names,
        )
        summaries.append(summary)
        print(
            f"  完成：{summary.frames} 帧，{summary.mot_rows} 个框，"
            f"{summary.unique_track_ids} 个轨迹 ID，"
            f"{summary.frames_with_expected_objects}/{summary.frames} 帧检测到"
            f"预期的 {args.expected_objects} 只",
            flush=True,
        )

    write_dataset_summary(args.output_dir, summaries)
    print(f"全部完成。输出目录：{args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
