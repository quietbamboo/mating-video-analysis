#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在公开的本地数据集上复现公鸡和交配母鸡的身份识别流程。

本脚本不请求线上 API，也不回写线上结果。它只负责：

1. 从 mating_clips.csv 读取视频名及公鸡/母鸡真值；
2. 从 logs/<clip_name>.csv 读取公鸡框；
3. 从 videos/<clip_name> 读取已经下载的视频；
4. 调用 chicken_identify.matching 中的离线识别算法；
5. 将逐视频预测和汇总准确率写入 local_results.csv。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import logging
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CSV = PROJECT_ROOT / "data" / "mating_clips.csv"
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "data" / "rooster_logs"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "generated" / "local_results.csv"
DEFAULT_LABEL_MODEL = PROJECT_ROOT / "models" / "chicken-label-12-23.pt"
DEFAULT_ANNOTATED_DIR = PROJECT_ROOT / "output" / "annotated-videos"
DEFAULT_ROI_AREA_RATIO = 1.0 / 9.0
DEFAULT_TAU_GAP_SECONDS = 1.0
DEFAULT_OUTLIER_SIGMA_MULTIPLIER = 2.0
DETECTION_CACHE_VERSION = 1

RESULT_FIELDS = [
    "clip_name",
    "expected_rooster",
    "predicted_rooster",
    "rooster_correct",
    "expected_hen",
    "predicted_hen",
    "hen_correct",
    "hen_exact",
    "mate_start_pos",
    "mate_end_pos",
    "range_source",
    "status",
    "error",
]


@dataclass(frozen=True)
class ClipCase:
    row_number: int
    clip_name: str
    expected_rooster: str
    expected_hen: str
    mate_start_pos: int | None
    mate_end_pos: int | None


def parse_optional_nonnegative_int(
    value: str | None, field: str, row_number: int
) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"第 {row_number} 行的 {field} 不是整数: {value!r}") from exc
    if number < 0:
        raise ValueError(f"第 {row_number} 行的 {field} 不能小于 0")
    return number


def parse_roi_area_ratio(value: str) -> float:
    """解析 0.25 或 1/9 形式的 ROI 面积比例。"""

    try:
        ratio = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            f"ROI 面积比例应为小数或分数，例如 0.25 或 1/9: {value!r}"
        ) from exc
    if not 0 < ratio <= 1:
        raise argparse.ArgumentTypeError("ROI 面积比例必须在 (0, 1] 内")
    return ratio


def parse_positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"秒数应为有效数字: {value!r}") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("秒数必须是大于 0 的有限数值")
    return seconds


def parse_positive_number(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"参数应为有效数字: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("参数必须是大于 0 的有限数值")
    return number


def load_cases(csv_path: Path) -> list[ClipCase]:
    cases: list[ClipCase] = []
    seen: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"clip_name", "rooster", "hen"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("数据集 CSV 缺少字段: " + ", ".join(sorted(missing)))

        for row_number, row in enumerate(reader, start=2):
            clip_name = (row.get("clip_name") or "").strip()
            if not clip_name:
                raise ValueError(f"第 {row_number} 行的 clip_name 为空")
            if Path(clip_name).name != clip_name:
                raise ValueError(f"第 {row_number} 行的 clip_name 不是纯文件名")
            key = clip_name.casefold()
            if key in seen:
                raise ValueError(f"第 {row_number} 行出现重复视频: {clip_name}")
            seen.add(key)

            cases.append(
                ClipCase(
                    row_number=row_number,
                    clip_name=clip_name,
                    expected_rooster=(row.get("rooster") or "").strip(),
                    expected_hen=(row.get("hen") or "").strip(),
                    mate_start_pos=parse_optional_nonnegative_int(
                        row.get("mate_start_pos"), "mate_start_pos", row_number
                    ),
                    mate_end_pos=parse_optional_nonnegative_int(
                        row.get("mate_end_pos"), "mate_end_pos", row_number
                    ),
                )
            )
    if not cases:
        raise ValueError("数据集 CSV 中没有视频记录")
    return cases


def load_rooster_boxes(log_path: Path) -> dict[int, list[float]]:
    boxes_by_frame: dict[int, list[float]] = {}
    with log_path.open("r", encoding="utf-8-sig") as log_file:
        header = log_file.readline().strip()
        if header != "frame_pos,boxes":
            raise ValueError(
                f"日志 CSV 表头应为 'frame_pos,boxes'，实际为 {header!r}"
            )

        # 现有日志中的 boxes 是未加 CSV 引号的 JSON，例如：
        # 65,[[2586.0,811.0,3386.0,1611.0]]
        # 因而不能用 csv.DictReader，必须只在第一个逗号处分割。
        for row_number, line in enumerate(log_file, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                frame_text, boxes_text = line.split(",", maxsplit=1)
                frame_pos = int(frame_text)
                all_boxes = json.loads(boxes_text)
                box = all_boxes[0]
                if len(box) < 4:
                    raise ValueError("box 坐标少于 4 个")
                box = [float(value) for value in box]
            except (TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{log_path.name} 第 {row_number} 行格式错误: {exc}"
                ) from exc
            if frame_pos in boxes_by_frame:
                raise ValueError(f"{log_path.name} 中 frame_pos={frame_pos} 重复")
            boxes_by_frame[frame_pos] = box

    if not boxes_by_frame:
        raise ValueError(f"{log_path.name} 中没有有效的公鸡框")
    return dict(sorted(boxes_by_frame.items()))


def load_detection_cache(
    cache_path: Path,
) -> tuple[int, dict[int, list[list[float]]]]:
    with cache_path.open("r", encoding="utf-8") as cache_file:
        data = json.load(cache_file)
    if data.get("version") != DETECTION_CACHE_VERSION:
        raise ValueError(f"检测缓存版本不兼容: {cache_path}")
    fps = int(data["fps"])
    if fps <= 0:
        raise ValueError(f"检测缓存 fps 无效: {cache_path}")
    video_labels = {
        int(frame_pos): [
            [float(value) for value in detection] for detection in detections
        ]
        for frame_pos, detections in data["video_labels"].items()
    }
    return fps, dict(sorted(video_labels.items()))


def save_detection_cache(
    cache_path: Path,
    fps: int,
    video_labels: dict[int, list[list[float]]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {
        "version": DETECTION_CACHE_VERSION,
        "fps": int(fps),
        "video_labels": video_labels,
    }
    with temporary_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, ensure_ascii=False, separators=(",", ":"))
    temporary_path.replace(cache_path)


def load_algorithm() -> Any:
    """Load the packaged, offline identification implementation."""

    missing = [
        module_name
        for module_name in ("av", "cv2", "numpy", "ultralytics")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing inference dependencies: "
            + ", ".join(missing)
            + ". Install the project with `pip install -e .`."
        )

    from chicken_identify import matching

    return matching


def normalize_prediction(value: str | None) -> str:
    # 线上算法在低置信度预测后添加 *；计算识别是否正确时忽略该标记，
    # 同时另存 hen_exact 以便统计高置信度的严格命中。
    # 不使用 str.removesuffix，以兼容服务器上可能仍在使用的 Python 3.8。
    normalized = value or ""
    if normalized.endswith("*"):
        normalized = normalized[:-1]
    if normalized.endswith(","):
        normalized = normalized[:-1]
    return normalized


def predict_nearest_center_baseline(
    algorithm: Any,
    rooster_boxes: dict[int, list[float]],
    video_labels: dict[int, list[list[float]]],
    mate_start: int,
    mate_end_exclusive: int,
    roi_area_ratio: float,
) -> tuple[str | None, str | None]:
    """使用相同公鸡识别，仅将母鸡判定替换为最近中心点时间投票。"""

    rooster_cls = algorithm.find_rooster_label(
        rooster_boxes,
        video_labels,
        roi_area_ratio=roi_area_ratio,
    )
    if rooster_cls is None or rooster_cls not in algorithm.chicken_label.label_map:
        return None, None

    rooster_mating_boxes = {
        frame_pos: box
        for frame_pos, box in rooster_boxes.items()
        if mate_start <= frame_pos < mate_end_exclusive
    }
    if not rooster_mating_boxes:
        raise ValueError("交配区间内没有公鸡框")

    hen_cls = algorithm.find_hen_nearest_center_baseline(
        rooster_cls,
        rooster_mating_boxes,
        video_labels,
    )
    rooster_label = algorithm.chicken_label.label_map[rooster_cls]
    hen_label = (
        algorithm.chicken_label.label_map[hen_cls]
        if hen_cls in algorithm.chicken_label.label_map
        else None
    )
    return rooster_label, hen_label


class DetectionRecorder:
    """记录线上算法每次调用标签检测器时返回的框，供结果视频复用。"""

    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.detections: dict[int, list[list[float]]] = {}
        # detect_clip_labels 只在 1、3、5……这些奇数帧调用 detect_all。
        self.next_frame_pos = 1

    def __call__(self, image: Any) -> list[list[float]]:
        detections = self.detector(image)
        self.detections[self.next_frame_pos] = [
            [float(value) for value in detection] for detection in detections
        ]
        self.next_frame_pos += 2
        return detections


def class_color(class_id: int) -> tuple[int, int, int]:
    """为每个标签类别生成稳定、较明亮的 BGR 颜色。"""

    return (
        80 + (class_id * 67) % 176,
        80 + (class_id * 131) % 176,
        80 + (class_id * 193) % 176,
    )


def draw_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    import cv2

    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(
        image,
        (x, max(0, y - height - baseline - 4)),
        (x + width + 6, y + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_annotated_video(
    source_path: Path,
    output_path: Path,
    detections: dict[int, list[list[float]]],
    rooster_boxes: dict[int, list[float]],
    label_map: dict[int, str],
    mate_start: int,
    mate_end_exclusive: int,
    expected_rooster: str,
    expected_hen: str,
    predicted_rooster: str | None,
    predicted_hen: str | None,
    output_scale: float,
    roi_area_ratio: float = DEFAULT_ROI_AREA_RATIO,
) -> None:
    """绘制检测框、公鸡中心区域和预测结果，并保存为 MP4。"""

    import cv2

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开待标注视频: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"无法读取视频参数: {source_path}")

    output_width = max(2, int(width * output_scale))
    output_height = max(2, int(height * output_scale))
    # 常见视频编码器要求宽高为偶数。
    output_width -= output_width % 2
    output_height -= output_height % 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建结果视频: {output_path}")

    rooster_poses = sorted(rooster_boxes)
    line_width = max(2, round(min(width, height) / 540))
    text_scale = max(0.8, min(width, height) / 1800)
    frame_pos = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            in_mating_range = mate_start <= frame_pos < mate_end_exclusive

            # 日志可能缺少少量帧；显示时选最邻近的框，但最多容许相差 2 帧。
            insert_at = bisect.bisect_left(rooster_poses, frame_pos)
            nearby = rooster_poses[
                max(0, insert_at - 1) : min(len(rooster_poses), insert_at + 1)
            ]
            rooster_pos = (
                min(nearby, key=lambda pos: abs(pos - frame_pos)) if nearby else None
            )
            if rooster_pos is not None and abs(rooster_pos - frame_pos) <= 2:
                box = rooster_boxes[rooster_pos]
                x1, y1, x2, y2 = (int(value) for value in box[:4])
                rooster_color = (255, 0, 255)
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), rooster_color, line_width
                )
                draw_text(
                    frame,
                    f"rooster log frame={rooster_pos}",
                    (x1, max(30, y1)),
                    rooster_color,
                    text_scale,
                    line_width,
                )

                # 面积比例开平方得到宽、高方向各自保留的边长比例。
                box_width, box_height = x2 - x1, y2 - y1
                roi_side_ratio = math.sqrt(roi_area_ratio)
                x_margin = box_width * (1 - roi_side_ratio) / 2
                y_margin = box_height * (1 - roi_side_ratio) / 2
                center_box = (
                    int(x1 + x_margin),
                    int(y1 + y_margin),
                    int(x2 - x_margin),
                    int(y2 - y_margin),
                )
                cv2.rectangle(
                    frame,
                    center_box[:2],
                    center_box[2:],
                    (255, 255, 0),
                    line_width,
                )

            # 模型只检测奇数帧；把奇数帧结果保留到其后的一个偶数帧，避免闪烁。
            detection_pos = frame_pos if frame_pos % 2 == 1 else frame_pos - 1
            for detection in detections.get(detection_pos, []):
                if len(detection) < 6:
                    continue
                x1, y1, x2, y2 = (int(value) for value in detection[:4])
                confidence = float(detection[4])
                class_id = int(detection[5])
                color = class_color(class_id)
                label_name = label_map.get(class_id, f"class-{class_id}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
                cv2.circle(
                    frame,
                    ((x1 + x2) // 2, (y1 + y2) // 2),
                    line_width + 2,
                    color,
                    -1,
                )
                draw_text(
                    frame,
                    f"{label_name} {confidence:.2f}",
                    (x1, max(30, y1)),
                    color,
                    text_scale,
                    line_width,
                )

            state_color = (0, 255, 0) if in_mating_range else (180, 180, 180)
            header_lines = [
                f"frame={frame_pos} mating={'YES' if in_mating_range else 'NO'} "
                f"range=[{mate_start},{mate_end_exclusive - 1}]",
                f"expected rooster={expected_rooster or '-'} hen={expected_hen or '-'}",
                f"predicted rooster={predicted_rooster or '-'} "
                f"hen={predicted_hen or '-'}",
                "magenta=rooster log  cyan=rooster center  colored=label detection",
            ]
            header_y = max(38, int(42 * text_scale))
            for index, line in enumerate(header_lines):
                draw_text(
                    frame,
                    line,
                    (20, header_y * (index + 1)),
                    state_color if index == 0 else (255, 255, 255),
                    text_scale,
                    line_width,
                )

            if output_width != width or output_height != height:
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            writer.write(frame)
            frame_pos += 1
    finally:
        capture.release()
        writer.release()

    if frame_pos == 0 or not output_path.is_file():
        raise RuntimeError(f"结果视频没有成功写入帧: {output_path}")


def resolve_mate_range(
    case: ClipCase,
    rooster_boxes: dict[int, list[float]],
    override_start: int | None,
    override_end: int | None,
) -> tuple[int, int, str]:
    log_start = min(rooster_boxes)
    log_end = max(rooster_boxes)
    start = override_start
    end = override_end
    source = "command-line"

    if start is None:
        start = case.mate_start_pos
        source = "csv"
    if end is None:
        end = case.mate_end_pos
        source = "csv"

    if start is None or end is None:
        start, end = log_start, log_end
        source = "log-range"

    if start > end:
        raise ValueError(f"交配起始帧 {start} 大于结束帧 {end}")
    # 线上函数的 mate_end_pos 是开区间，API 的 mark_pos2 则是闭区间。
    return start, end + 1, source


def validate_dataset(
    cases: list[ClipCase], videos_dir: Path, logs_dir: Path
) -> tuple[list[str], list[str], list[str]]:
    missing_videos: list[str] = []
    missing_logs: list[str] = []
    bad_logs: list[str] = []
    for case in cases:
        if not (videos_dir / case.clip_name).is_file():
            missing_videos.append(case.clip_name)
        log_path = logs_dir / f"{case.clip_name}.csv"
        if not log_path.is_file():
            missing_logs.append(case.clip_name)
            continue
        try:
            load_rooster_boxes(log_path)
        except (OSError, ValueError) as exc:
            bad_logs.append(f"{case.clip_name}: {exc}")
    return missing_videos, missing_logs, bad_logs


def load_completed(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    with output_path.open("r", encoding="utf-8-sig", newline="") as output_file:
        reader = csv.DictReader(output_file)
        if set(reader.fieldnames or ()) != set(RESULT_FIELDS):
            raise ValueError(
                f"已有结果文件字段不兼容，请改用新 --output，或加 "
                f"--overwrite-results: {output_path}"
            )
        return {
            row["clip_name"]
            for row in reader
            if row.get("clip_name") and row.get("status") == "ok"
        }


def print_summary(rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row["status"] == "ok"]
    rooster_scored = [row for row in successful if row["rooster_correct"] != ""]
    hen_scored = [row for row in successful if row["hen_correct"] != ""]
    hen_exact_scored = [row for row in successful if row["hen_exact"] != ""]
    pair_scored = [
        row
        for row in successful
        if row["rooster_correct"] != "" and row["hen_correct"] != ""
    ]
    rooster_hits = sum(row["rooster_correct"] == "1" for row in rooster_scored)
    hen_hits = sum(row["hen_correct"] == "1" for row in hen_scored)
    hen_exact_hits = sum(row["hen_exact"] == "1" for row in hen_exact_scored)
    pair_hits = sum(
        row["rooster_correct"] == "1" and row["hen_correct"] == "1"
        for row in pair_scored
    )
    hen_predictions = sum(bool(row["predicted_hen"]) for row in hen_scored)

    def ratio(hits: int, total: int) -> str:
        return f"{hits / total:.2%}" if total else "N/A"

    print(
        "本次完成 "
        f"{len(successful)}/{len(rows)} 条；"
        f"公鸡准确率 {rooster_hits}/{len(rooster_scored)} "
        f"({ratio(rooster_hits, len(rooster_scored))})；"
        f"母鸡准确率 {hen_hits}/{len(hen_scored)} "
        f"({ratio(hen_hits, len(hen_scored))})；"
        f"成对准确率 {pair_hits}/{len(pair_scored)} "
        f"({ratio(pair_hits, len(pair_scored))})；"
        f"母鸡预测覆盖率 {hen_predictions}/{len(hen_scored)} "
        f"({ratio(hen_predictions, len(hen_scored))})；"
        f"母鸡严格准确率 {hen_exact_hits}/{len(hen_exact_scored)} "
        f"({ratio(hen_exact_hits, len(hen_exact_scored))})。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用本地视频、日志和真值批量复现线上公鸡/母鸡标签识别"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument(
        "--label-model",
        type=Path,
        default=DEFAULT_LABEL_MODEL,
        help=f"本地标签 YOLO 模型（默认: {DEFAULT_LABEL_MODEL}）",
    )
    parser.add_argument(
        "--label-conf",
        type=float,
        default=0.25,
        help="本地标签模型置信度阈值（默认: 0.25）",
    )
    parser.add_argument(
        "--label-iou",
        type=float,
        default=0.70,
        help="本地标签模型 NMS IoU 阈值（默认: 0.70）",
    )
    parser.add_argument(
        "--label-imgsz",
        type=int,
        default=640,
        help="本地标签模型输入尺寸（默认: 640）",
    )
    parser.add_argument(
        "--device",
        help="Ultralytics 推理设备，例如 0、0,1 或 cpu；默认自动选择",
    )
    parser.add_argument(
        "--no-label-slices",
        "--no-label-tiles",
        dest="no_label_slices",
        action="store_true",
        help="关闭原图切片，直接对整帧推理（仅用于对照实验）",
    )
    parser.add_argument(
        "--slice-width",
        type=int,
        default=1020,
        help="标签推理切片宽度（默认: 1020）",
    )
    parser.add_argument(
        "--slice-height",
        type=int,
        default=1120,
        help="标签推理切片高度（默认: 1120）",
    )
    parser.add_argument(
        "--slice-overlap",
        type=int,
        default=80,
        help="横纵相邻切片重叠像素数（默认: 80）",
    )
    parser.add_argument(
        "--slice-batch-size",
        type=int,
        default=16,
        help="一次送入标签模型的切片数量（默认: 16）",
    )
    parser.add_argument(
        "--slice-nms-iou",
        type=float,
        default=0.50,
        help="切片结果映射回原图后的去重 IoU（默认: 0.50）",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="保存绘有检测框、公鸡框和预测结果的可视化视频",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=DEFAULT_ANNOTATED_DIR,
        help=f"可视化视频目录（默认: {DEFAULT_ANNOTATED_DIR}）",
    )
    parser.add_argument(
        "--video-scale",
        type=float,
        default=0.5,
        help="结果视频相对原视频的缩放比例（默认: 0.5）",
    )
    parser.add_argument(
        "--roi-area-ratio",
        type=parse_roi_area_ratio,
        default=DEFAULT_ROI_AREA_RATIO,
        help=(
            "公鸡框中央 ROI 的面积比例，可写成 1/9 或 0.25；"
            "默认 1/9"
        ),
    )
    parser.add_argument(
        "--tau-gap-seconds",
        type=parse_positive_seconds,
        default=DEFAULT_TAU_GAP_SECONDS,
        help=(
            "完整遮挡方法判定标签消失区间的最小时长（秒）；"
            "默认 1.0。轨迹窗口保持固定，不随该值变化"
        ),
    )
    trajectory_window_group = parser.add_mutually_exclusive_group()
    trajectory_window_group.add_argument(
        "--trajectory-window-seconds",
        type=parse_positive_seconds,
        help=(
            "完整遮挡方法在标签消失前使用的轨迹时间窗口 w（秒）；"
            "与 --trajectory-window-fps-multiplier 互斥"
        ),
    )
    trajectory_window_group.add_argument(
        "--trajectory-window-fps-multiplier",
        type=parse_positive_number,
        metavar="K",
        help=(
            "按有效检测点数定义轨迹窗口：Nw=ceil(K*F)，其中 F 为视频 fps；"
            "例如 K=1.0 精确复现线上最后 fps 个检测点的配置"
        ),
    )
    parser.add_argument(
        "--disable-tau-gap",
        action="store_true",
        help=(
            "消融 tau_gap 持续时长阈值：只要缺失一次按当前检测步长应出现的"
            "标签检测就视为 gap；动态 ROI 判断仍保留"
        ),
    )
    parser.add_argument(
        "--disable-trajectory-window",
        action="store_true",
        help=(
            "消融多点轨迹窗口：主路径使用 gap 前最后一个有效点，回退路径使用"
            "交配区间中点后的第一个有效点，并绕过多点轨迹异常值过滤"
        ),
    )
    parser.add_argument(
        "--disable-outlier-filter",
        action="store_true",
        help=(
            "完全关闭主路径和交配区间中点回退路径的轨迹异常值过滤"
        ),
    )
    parser.add_argument(
        "--outlier-sigma-multiplier",
        type=parse_positive_number,
        default=DEFAULT_OUTLIER_SIGMA_MULTIPLIER,
        metavar="K",
        help="轨迹距离过滤 μ_d±Kσ_d 的标准差倍率（默认: 2）",
    )
    parser.add_argument(
        "--detections-cache-dir",
        type=Path,
        help=(
            "可选：缓存逐帧标签检测结果；ROI 扫描或基线对比时复用缓存，"
            "避免重复运行模型"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("proposed", "nearest-baseline"),
        default="proposed",
        help=(
            "母鸡身份判定方法：proposed 为完整遮挡推理；"
            "nearest-baseline 为逐帧最近中心点时间投票"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--clip",
        action="append",
        help="只测试指定 clip_name；可重复传入",
    )
    parser.add_argument("--limit", type=int, help="最多测试多少条")
    parser.add_argument(
        "--mate-start",
        type=int,
        help="单视频调试时覆盖交配起始帧（闭区间）",
    )
    parser.add_argument(
        "--mate-end",
        type=int,
        help="单视频调试时覆盖交配结束帧（闭区间）",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="覆盖已有结果；默认跳过已有结果中的成功记录",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验 CSV、视频和日志是否对应，不加载识别模型",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    try:
        cases = load_cases(args.csv.resolve())
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.clip:
        requested = set(args.clip)
        cases = [case for case in cases if case.clip_name in requested]
        found = {case.clip_name for case in cases}
        missing = requested - found
        if missing:
            print(
                "错误: 数据集 CSV 中不存在: " + ", ".join(sorted(missing)),
                file=sys.stderr,
            )
            return 2
    if args.limit is not None:
        if args.limit < 1:
            print("错误: --limit 必须大于 0", file=sys.stderr)
            return 2
        cases = cases[: args.limit]
    if (args.mate_start is not None or args.mate_end is not None) and len(cases) != 1:
        print("错误: --mate-start/--mate-end 只能用于单个 --clip", file=sys.stderr)
        return 2
    if not 0 < args.video_scale <= 1:
        print("错误: --video-scale 必须大于 0 且不超过 1", file=sys.stderr)
        return 2
    tau_gap_text = (
        "disabled (any missed scheduled detection)"
        if args.disable_tau_gap
        else f"{args.tau_gap_seconds:.6g}s"
    )
    if args.disable_trajectory_window:
        trajectory_window_text = "disabled (single immediate point)"
    elif args.trajectory_window_seconds is not None:
        trajectory_window_text = f"{args.trajectory_window_seconds:.6g}s"
    elif args.trajectory_window_fps_multiplier is not None:
        trajectory_window_text = (
            f"Nw={args.trajectory_window_fps_multiplier:.6g}F "
            "(valid detection points)"
        )
    else:
        trajectory_window_text = "deployed-Nw=1F-valid-detection-points"
    if args.disable_trajectory_window:
        outlier_filter_text = "bypassed (trajectory window disabled)"
    elif args.disable_outlier_filter:
        outlier_filter_text = "disabled"
    else:
        outlier_filter_text = (
            "distance-mean-std:"
            f"mu+/-{args.outlier_sigma_multiplier:.6g}sigma"
        )
    print(
        f"评估方法={args.method}；"
        f"ROI 面积比例={args.roi_area_ratio:.6g}，"
        f"边长比例={math.sqrt(args.roi_area_ratio):.6g}；"
        f"tau_gap={tau_gap_text}；"
        f"w={trajectory_window_text}；"
        f"异常值过滤={outlier_filter_text}"
    )
    if (
        args.method == "nearest-baseline"
        and (
            args.tau_gap_seconds != DEFAULT_TAU_GAP_SECONDS
            or args.disable_tau_gap
            or args.disable_trajectory_window
            or args.trajectory_window_seconds is not None
            or args.trajectory_window_fps_multiplier is not None
            or args.disable_outlier_filter
            or (
                args.outlier_sigma_multiplier
                != DEFAULT_OUTLIER_SIGMA_MULTIPLIER
            )
        )
    ):
        print(
            "提示: nearest-baseline 不使用 tau_gap、w 或异常值过滤；"
            "这些参数仅影响 proposed 方法。"
        )

    videos_dir = args.videos_dir.resolve()
    logs_dir = args.logs_dir.resolve()
    missing_videos, missing_logs, bad_logs = validate_dataset(
        cases, videos_dir, logs_dir
    )
    print(
        f"已校验 {len(cases)} 条：缺少视频 {len(missing_videos)}，"
        f"缺少日志 {len(missing_logs)}，错误日志 {len(bad_logs)}。"
    )
    for title, values in (
        ("缺少视频", missing_videos),
        ("缺少日志", missing_logs),
        ("错误日志", bad_logs),
    ):
        if values:
            print(f"{title}示例:")
            for value in values[:10]:
                print(f"  {value}")

    if args.validate_only:
        return 1 if missing_videos or missing_logs or bad_logs else 0
    if missing_videos or missing_logs or bad_logs:
        print("错误: 数据集不完整；请补齐后运行，或用 --clip 选择完整样本。")
        return 2

    try:
        algorithm = load_algorithm()
        if hasattr(algorithm.label_detector, "configure"):
            algorithm.label_detector.configure(
                model_path=args.label_model,
                confidence=args.label_conf,
                iou=args.label_iou,
                image_size=args.label_imgsz,
                device=args.device,
                use_tiles=not args.no_label_slices,
                slice_width=args.slice_width,
                slice_height=args.slice_height,
                slice_overlap=args.slice_overlap,
                slice_batch_size=args.slice_batch_size,
                slice_nms_iou=args.slice_nms_iou,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detections_cache_dir = (
        args.detections_cache_dir.resolve() if args.detections_cache_dir else None
    )
    try:
        completed = (
            set() if args.overwrite_results else load_completed(output_path)
        )
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    pending = [case for case in cases if case.clip_name not in completed]
    if not pending:
        print("所选视频均已有成功结果，无需重复运行。")
        return 0

    mode = "w" if args.overwrite_results or not output_path.exists() else "a"
    rows: list[dict[str, Any]] = []
    with output_path.open(mode, encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        if mode == "w":
            writer.writeheader()

        for index, case in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {case.clip_name}", flush=True)
            row: dict[str, Any] = {
                "clip_name": case.clip_name,
                "expected_rooster": case.expected_rooster,
                "predicted_rooster": "",
                "rooster_correct": "",
                "expected_hen": case.expected_hen,
                "predicted_hen": "",
                "hen_correct": "",
                "hen_exact": "",
                "mate_start_pos": "",
                "mate_end_pos": "",
                "range_source": "",
                "status": "failed",
                "error": "",
            }
            try:
                rooster_boxes = load_rooster_boxes(
                    logs_dir / f"{case.clip_name}.csv"
                )
                mate_start, mate_end_exclusive, range_source = resolve_mate_range(
                    case, rooster_boxes, args.mate_start, args.mate_end
                )
                row["mate_start_pos"] = mate_start
                row["mate_end_pos"] = mate_end_exclusive - 1
                row["range_source"] = range_source

                # 使用文件对象是为了兼容线上函数在保存异常帧前调用 seek(0)。
                original_detector = algorithm.label_detector.detect_all
                recorder = DetectionRecorder(original_detector)
                precomputed_detection = None
                if detections_cache_dir is not None:
                    cache_path = detections_cache_dir / f"{case.clip_name}.json"
                    if cache_path.is_file():
                        precomputed_detection = load_detection_cache(cache_path)
                        print(f"  使用检测缓存: {cache_path}")
                    else:
                        with (videos_dir / case.clip_name).open("rb") as video_source:
                            precomputed_detection = algorithm.detect_clip_labels(
                                video_source
                            )
                        if not precomputed_detection:
                            raise RuntimeError("标签检测失败，无法写入检测缓存")
                        save_detection_cache(
                            cache_path,
                            precomputed_detection[0],
                            precomputed_detection[1],
                        )
                        print(f"  已写入检测缓存: {cache_path}")
                    recorder.detections = precomputed_detection[1]
                elif args.method == "nearest-baseline":
                    with (videos_dir / case.clip_name).open("rb") as video_source:
                        precomputed_detection = algorithm.detect_clip_labels(
                            video_source
                        )
                    if not precomputed_detection:
                        raise RuntimeError("标签检测失败，无法运行最近中心点基线")
                    recorder.detections = precomputed_detection[1]
                elif args.save_video:
                    algorithm.label_detector.detect_all = recorder
                try:
                    if args.method == "nearest-baseline":
                        rooster, hen = predict_nearest_center_baseline(
                            algorithm,
                            rooster_boxes,
                            precomputed_detection[1],
                            mate_start,
                            mate_end_exclusive,
                            args.roi_area_ratio,
                        )
                    else:
                        with (videos_dir / case.clip_name).open("rb") as video_source:
                            rooster, hen = algorithm.find_rooster_and_hen_from_clip(
                                case.clip_name,
                                video_source,
                                rooster_boxes,
                                mate_start,
                                mate_end_exclusive,
                                roi_area_ratio=args.roi_area_ratio,
                                tau_gap_seconds=args.tau_gap_seconds,
                                trajectory_window_seconds=(
                                    args.trajectory_window_seconds
                                ),
                                precomputed_detection=precomputed_detection,
                                trajectory_window_fps_multiplier=(
                                    args.trajectory_window_fps_multiplier
                                ),
                                outlier_filter_enabled=(not args.disable_outlier_filter),
                                outlier_sigma_multiplier=(
                                    args.outlier_sigma_multiplier
                                ),
                                disable_tau_gap=args.disable_tau_gap,
                                disable_trajectory_window=(
                                    args.disable_trajectory_window
                                ),
                            )
                finally:
                    algorithm.label_detector.detect_all = original_detector

                row["predicted_rooster"] = rooster or ""
                row["predicted_hen"] = hen or ""
                if case.expected_rooster:
                    row["rooster_correct"] = str(
                        int(
                            normalize_prediction(rooster)
                            == normalize_prediction(case.expected_rooster)
                        )
                    )
                if case.expected_hen:
                    row["hen_correct"] = str(
                        int(
                            normalize_prediction(hen)
                            == normalize_prediction(case.expected_hen)
                        )
                    )
                    row["hen_exact"] = str(int((hen or "") == case.expected_hen))

                if args.save_video:
                    annotated_suffix = (
                        "-nearest-baseline-annotated.mp4"
                        if args.method == "nearest-baseline"
                        else "-annotated.mp4"
                    )
                    annotated_path = (
                        args.annotated_dir.resolve()
                        / f"{Path(case.clip_name).stem}{annotated_suffix}"
                    )
                    save_annotated_video(
                        source_path=videos_dir / case.clip_name,
                        output_path=annotated_path,
                        detections=recorder.detections,
                        rooster_boxes=rooster_boxes,
                        label_map=algorithm.chicken_label.label_map,
                        mate_start=mate_start,
                        mate_end_exclusive=mate_end_exclusive,
                        expected_rooster=case.expected_rooster,
                        expected_hen=case.expected_hen,
                        predicted_rooster=rooster,
                        predicted_hen=hen,
                        output_scale=args.video_scale,
                        roi_area_ratio=args.roi_area_ratio,
                    )
                    print(f"  可视化视频: {annotated_path}")
                row["status"] = "ok"
            except Exception as exc:  # 单条失败不应中止数百条批处理
                logging.exception("处理 %s 失败", case.clip_name)
                row["error"] = f"{type(exc).__name__}: {exc}"

            writer.writerow(row)
            output_file.flush()
            rows.append(row)

    print_summary(rows)
    print(f"逐条结果已保存到: {output_path}")
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
