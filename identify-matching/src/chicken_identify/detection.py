"""使用本地 Ultralytics YOLO 权重实现线上 label_detector 接口。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "chicken-label-12-23.pt"

_model_path = DEFAULT_MODEL
_confidence = 0.5
_iou = 0.5
_image_size = 1120
_device: str | None = None
_use_tiles = True
_slice_width = 1020
_slice_height = 1120
_slice_overlap = 80
_slice_batch_size = 16
_slice_nms_iou = 0.50
_model: Any = None

_SLICE_DUPLICATE_IOS = 0.70
_SLICE_EDGE_CONFIDENCE_PENALTY = 0.50


def detect_clip_labels(source: str | Path, default_fps: int = 13):
    """Run label detection on every other video frame.

    PyAV is imported lazily so trajectory-only analysis and unit tests do not
    require the video inference dependencies.
    """

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Video inference requires PyAV: pip install av") from exc

    av_error_module = getattr(av, "error", None)
    ffmpeg_error = getattr(
        av,
        "FFmpegError",
        getattr(av_error_module, "FFmpegError", getattr(av, "AVError", OSError)),
    )
    try:
        src_video = av.open(source)
    except ffmpeg_error:
        logger.exception("error on open video source")
        return None

    try:
        video_stream = src_video.streams.video[0]
        video_stream.thread_type = "AUTO"
        video_stream.thread_count = 0
        rate = getattr(video_stream, "average_rate", None)
        if rate is None:
            rate = getattr(video_stream, "base_rate", None)
        fps = default_fps if rate is None else int(float(rate))

        video_labels = {}
        for index, frame in enumerate(src_video.decode(video=0)):
            if index % 2 == 0:
                continue
            video_labels[index] = detect_all(frame.to_ndarray(format="bgr24"))
            if (index + 1) % fps == 0:
                logger.info(
                    "processed frame count %s, finish seconds %s",
                    index + 1,
                    (index + 1) // fps,
                )
    except ffmpeg_error:
        logger.exception("error while decoding video")
        return None
    finally:
        src_video.close()

    return fps, video_labels


@dataclass(frozen=True)
class _TiledDetection:
    values: tuple[float, ...]
    tile_index: int
    tile_bounds: tuple[int, int, int, int]


def configure(
    model_path: str | Path = DEFAULT_MODEL,
    confidence: float = 0.25,
    iou: float = 0.70,
    image_size: int = 640,
    device: str | None = None,
    use_tiles: bool = True,
    slice_width: int = 1020,
    slice_height: int = 1120,
    slice_overlap: int = 80,
    slice_batch_size: int = 16,
    slice_nms_iou: float = 0.50,
) -> None:
    """在第一次推理前配置模型和预测参数。"""

    global _model_path, _confidence, _iou, _image_size, _device
    global _use_tiles, _slice_width, _slice_height, _slice_overlap
    global _slice_batch_size, _slice_nms_iou, _model
    resolved = Path(model_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"标签模型不存在: {resolved}")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence 必须在 0 到 1 之间")
    if not 0 <= iou <= 1:
        raise ValueError("iou 必须在 0 到 1 之间")
    if image_size < 32:
        raise ValueError("image_size 不能小于 32")
    if slice_width < 32 or slice_height < 32:
        raise ValueError("slice_width 和 slice_height 不能小于 32")
    if not 0 <= slice_overlap < min(slice_width, slice_height):
        raise ValueError(
            "slice_overlap 必须大于等于 0，且小于切片的宽度和高度"
        )
    if slice_batch_size < 1:
        raise ValueError("slice_batch_size 必须大于 0")
    if not 0 <= slice_nms_iou <= 1:
        raise ValueError("slice_nms_iou 必须在 0 到 1 之间")

    if resolved != _model_path:
        _model = None
    _model_path = resolved
    _confidence = confidence
    _iou = iou
    _image_size = image_size
    _device = device
    _use_tiles = use_tiles
    _slice_width = slice_width
    _slice_height = slice_height
    _slice_overlap = slice_overlap
    _slice_batch_size = slice_batch_size
    _slice_nms_iou = slice_nms_iou


def _get_model() -> Any:
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics，请先运行: pip install ultralytics"
            ) from exc
        if not _model_path.is_file():
            raise FileNotFoundError(f"标签模型不存在: {_model_path}")
        _model = YOLO(str(_model_path))
    return _model


def _predict(images: np.ndarray | list[np.ndarray]) -> list[Any]:
    predict_options: dict[str, Any] = {
        "source": images,
        "conf": _confidence,
        "iou": _iou,
        "imgsz": _image_size,
        "verbose": False,
    }
    if _device:
        predict_options["device"] = _device

    return _get_model().predict(**predict_options)


def _slice_starts(length: int, slice_size: int, overlap: int) -> list[int]:
    """生成覆盖完整边长的裁块起点，最后一个裁块贴齐图像边缘。"""

    if length <= slice_size:
        return [0]
    stride = slice_size - overlap
    starts = list(range(0, length - slice_size + 1, stride))
    final_start = length - slice_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _iou_with_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )


def _box_overlap_metrics(
    first: tuple[float, ...], second: tuple[float, ...]
) -> tuple[float, float]:
    """返回两个框的 IoU 和交集占较小框面积的比例（IoS）。"""

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    smaller_area = min(first_area, second_area)
    iou = intersection / union if union > 0 else 0.0
    ios = intersection / smaller_area if smaller_area > 0 else 0.0
    return iou, ios


def _tile_bounds_overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return min(first[2], second[2]) > max(first[0], second[0]) and min(
        first[3], second[3]
    ) > max(first[1], second[1])


def _touches_internal_tile_edge(
    detection: _TiledDetection,
    image_width: int,
    image_height: int,
    edge_margin: float,
) -> bool:
    x1, y1, x2, y2 = detection.values[:4]
    tile_x1, tile_y1, tile_x2, tile_y2 = detection.tile_bounds
    return (
        (tile_x1 > 0 and x1 <= tile_x1 + edge_margin)
        or (tile_y1 > 0 and y1 <= tile_y1 + edge_margin)
        or (tile_x2 < image_width and x2 >= tile_x2 - edge_margin)
        or (tile_y2 < image_height and y2 >= tile_y2 - edge_margin)
    )


def _deduplicate_slice_seams(
    detections: list[_TiledDetection],
    image_width: int,
    image_height: int,
    iou_threshold: float,
    overlap: int,
) -> list[_TiledDetection]:
    """合并相邻切片在接缝处对同一物理标签产生的重复框。

    只比较来源不同且切片范围相交的框，避免把同一切片中确实靠近的两个标签
    当成重复目标。IoS 用于识别“完整框 + 接缝截断框”这类 IoU 偏低的组合；
    类别不作为分组条件，以消除同一标签在相邻切片中的类别冲突。
    """

    if len(detections) < 2:
        return detections

    parents = list(range(len(detections)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(detections):
        for second_index in range(first_index + 1, len(detections)):
            second = detections[second_index]
            if first.tile_index == second.tile_index:
                continue
            if not _tile_bounds_overlap(first.tile_bounds, second.tile_bounds):
                continue
            iou, ios = _box_overlap_metrics(first.values, second.values)
            if iou > iou_threshold or ios >= _SLICE_DUPLICATE_IOS:
                union(first_index, second_index)

    components: dict[int, list[int]] = {}
    for index in range(len(detections)):
        components.setdefault(find(index), []).append(index)

    edge_margin = max(2.0, min(16.0, overlap * 0.10))

    def quality(index: int) -> tuple[float, float, float, int]:
        detection = detections[index]
        confidence = float(detection.values[4])
        touches_edge = _touches_internal_tile_edge(
            detection,
            image_width,
            image_height,
            edge_margin,
        )
        adjusted_confidence = confidence * (
            _SLICE_EDGE_CONFIDENCE_PENALTY if touches_edge else 1.0
        )
        width = max(0.0, detection.values[2] - detection.values[0])
        height = max(0.0, detection.values[3] - detection.values[1])
        return adjusted_confidence, confidence, width * height, -detection.tile_index

    kept = [detections[max(indices, key=quality)] for indices in components.values()]
    kept.sort(key=lambda detection: detection.values[4], reverse=True)
    return kept


def _class_aware_nms(
    detections: list[list[float]], iou_threshold: float
) -> list[list[float]]:
    """在映射回原图后，按类别消除重叠裁块产生的重复检测。"""

    if not detections:
        return []
    array = np.asarray(detections, dtype=float)
    kept_indices: list[int] = []
    for class_id in np.unique(array[:, 5].astype(int)):
        class_indices = np.flatnonzero(array[:, 5].astype(int) == class_id)
        order = class_indices[np.argsort(array[class_indices, 4])[::-1]]
        while order.size:
            best = int(order[0])
            kept_indices.append(best)
            if order.size == 1:
                break
            remaining = order[1:]
            overlaps = _iou_with_many(array[best, :4], array[remaining, :4])
            order = remaining[overlaps <= iou_threshold]

    kept_indices.sort(key=lambda index: array[index, 4], reverse=True)
    return array[kept_indices].tolist()


def _detect_full_image(image: np.ndarray) -> list[list[float]]:
    results = _predict(image)
    if not results or results[0].boxes is None:
        return []
    return results[0].boxes.data.detach().cpu().numpy().tolist()


def _detect_tiled(image: np.ndarray) -> list[list[float]]:
    image_height, image_width = image.shape[:2]
    x_starts = _slice_starts(image_width, _slice_width, _slice_overlap)
    y_starts = _slice_starts(image_height, _slice_height, _slice_overlap)

    tiles: list[np.ndarray] = []
    tile_specs: list[tuple[int, tuple[int, int, int, int]]] = []
    for y_start in y_starts:
        for x_start in x_starts:
            x_end = min(image_width, x_start + _slice_width)
            y_end = min(image_height, y_start + _slice_height)
            tiles.append(image[y_start:y_end, x_start:x_end])
            tile_specs.append(
                (
                    len(tile_specs),
                    (x_start, y_start, x_end, y_end),
                )
            )

    mapped_detections: list[_TiledDetection] = []
    for batch_start in range(0, len(tiles), _slice_batch_size):
        batch_tiles = tiles[batch_start : batch_start + _slice_batch_size]
        batch_specs = tile_specs[batch_start : batch_start + _slice_batch_size]
        results = _predict(batch_tiles)
        if len(results) != len(batch_tiles):
            raise RuntimeError(
                f"裁块数量与模型结果数量不一致: {len(batch_tiles)} != "
                f"{len(results)}"
            )

        for result, (tile_index, tile_bounds) in zip(results, batch_specs):
            if result.boxes is None:
                continue
            x_offset, y_offset = tile_bounds[:2]
            local_detections = result.boxes.data.detach().cpu().numpy()
            for detection in local_detections:
                mapped = detection.astype(float).tolist()
                # 将裁块局部坐标恢复成原始 4K 图像坐标。
                mapped[0] += x_offset
                mapped[2] += x_offset
                mapped[1] += y_offset
                mapped[3] += y_offset
                mapped_detections.append(
                    _TiledDetection(
                        values=tuple(mapped),
                        tile_index=tile_index,
                        tile_bounds=tile_bounds,
                    )
                )

    seam_deduplicated = _deduplicate_slice_seams(
        mapped_detections,
        image_width,
        image_height,
        _slice_nms_iou,
        _slice_overlap,
    )
    return _class_aware_nms(
        [list(detection.values) for detection in seam_deduplicated],
        _slice_nms_iou,
    )


def detect_all(image: np.ndarray) -> list[list[float]]:
    """返回原图坐标的 [x1, y1, x2, y2, confidence, class_id]。"""

    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("detect_all 需要 H×W×C 的 NumPy 图像")
    if _use_tiles:
        return _detect_tiled(image)
    return _detect_full_image(image)
