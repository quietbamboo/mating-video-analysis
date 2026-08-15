# -*- coding: utf-8 -*-
"""
尝试找一下交配的公鸡和母鸡的标签
"""

import bisect
import logging
import math
from collections import defaultdict

import numpy as np

from . import detection as label_detector
from . import labels as chicken_label

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

default_rooster_roi_area_ratio = 1.0 / 9.0
default_tau_gap_seconds = 1.0
default_outlier_sigma_multiplier = 2.0


# 已有的label是每分钟检测一次，现在每隔一帧检测一次，增加计算数据
# 视频是21s的4k视频，带交配的上下文，且无需再矫正畸变
detect_clip_labels = label_detector.detect_clip_labels


def get_center(box):
    """计算边界框中心点坐标"""
    x1, y1, x2, y2 = box[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def seconds_to_frame_threshold(seconds, fps):
    """把秒数转换为不短于该时长的整数帧阈值。"""
    if seconds <= 0:
        raise ValueError("seconds must be greater than 0")
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    return max(1, int(math.ceil(seconds * fps)))


def infer_analyzed_frame_step(frame_positions, default_step=2):
    """Infer the detector's frame-index stride from analyzed frame positions."""
    positions = sorted(set(frame_positions))
    positive_steps = [
        right - left
        for left, right in zip(positions, positions[1:])
        if right > left
    ]
    return min(positive_steps) if positive_steps else default_step


def fps_multiplier_to_point_count(multiplier, fps):
    """把 Nw/F 倍率转换为有效检测点数量，向上取整避免缩短窗口。"""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("trajectory window FPS multiplier must be finite and positive")
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    return max(1, int(math.ceil(multiplier * fps)))


def is_in_rooster_center_area(
    rooster_box,
    target_center,
    minus_margin_ratio=3,
    roi_area_ratio=None,
):
    """判断点是否位于公鸡框的中心 ROI。

    ``roi_area_ratio`` 是 ROI 面积占完整公鸡框的比例。默认保留旧接口：
    ``minus_margin_ratio=3`` 等价于中心 1/9 面积。新代码应优先传
    ``roi_area_ratio``，其含义更直观。
    """
    w, h = rooster_box[2] - rooster_box[0], rooster_box[3] - rooster_box[1]
    if roi_area_ratio is None:
        if minus_margin_ratio <= 2:
            raise ValueError("minus_margin_ratio must be greater than 2")
        roi_side_ratio = 1.0 - 2.0 / minus_margin_ratio
    else:
        if not 0 < roi_area_ratio <= 1:
            raise ValueError("roi_area_ratio must be in (0, 1]")
        roi_side_ratio = math.sqrt(roi_area_ratio)

    # 保持默认 1/9 与旧版 w/3、h/3 的浮点边界行为完全一致。
    if (
        roi_area_ratio == default_rooster_roi_area_ratio
        or (roi_area_ratio is None and minus_margin_ratio == 3)
    ):
        x_margin = w / 3.0
        y_margin = h / 3.0
    else:
        x_margin = w * (1.0 - roi_side_ratio) / 2.0
        y_margin = h * (1.0 - roi_side_ratio) / 2.0
    rooster_label_x_min = rooster_box[0] + x_margin
    rooster_label_y_min = rooster_box[1] + y_margin
    rooster_label_x_max = rooster_box[2] - x_margin
    rooster_label_y_max = rooster_box[3] - y_margin

    return (
        rooster_label_x_min <= target_center[0] <= rooster_label_x_max
        and rooster_label_y_min <= target_center[1] <= rooster_label_y_max
    )


# 给定一个数值，从列表中找到跟他最相近的值
def find_closest_value(target_value, sorted_values):
    # 使用 bisect 找到插入位置    all(sorted_values[:idx]) < target_value <= all(sorted_values[idx:])
    idx = bisect.bisect_left(sorted_values, target_value)
    # 处理边界情况
    if idx == 0:
        return sorted_values[0]
    if idx == len(sorted_values):
        return sorted_values[-1]

    # 比较左右两个值，返回更接近的
    left = sorted_values[idx - 1]
    right = sorted_values[idx]
    if abs(target_value - left) <= abs(target_value - right):
        return left
    else:
        return right


# 找到公鸡的框中心区域出现的次数最多的label
# video_labels: key=frame_pos, value=list[box1, box2,...] labels boxes of each frame
def find_rooster_label(
    rooster_boxes: dict,
    video_labels: dict,
    roi_area_ratio=default_rooster_roi_area_ratio,
):
    class_scores = defaultdict(float)
    class_frame_votes = defaultdict(int)
    mating_poses = list(rooster_boxes.keys())
    mate_start_pos = min(mating_poses)
    mate_end_pos = max(mating_poses)

    for frame_pos, boxes in video_labels.items():
        if frame_pos < mate_start_pos or frame_pos > mate_end_pos:
            continue

        fit_pos = find_closest_value(frame_pos, mating_poses)
        rooster_box = rooster_boxes[fit_pos]

        # 每个检测帧最多贡献一个候选，避免切片重复框在同一帧重复计票。
        frame_candidate = None
        for label_data in boxes:
            if len(label_data) < 6:
                continue
            chicken_center = get_center(label_data)  # 标签框中心点

            # 看label中心点是否在公鸡的box中央区域
            if is_in_rooster_center_area(
                rooster_box,
                chicken_center,
                roi_area_ratio=roi_area_ratio,
            ):
                candidate = int(label_data[5])
                confidence = float(label_data[4])
                candidate_rank = (confidence, -candidate)
                if frame_candidate is None or candidate_rank > frame_candidate[0]:
                    frame_candidate = (candidate_rank, candidate, confidence)

        if frame_candidate is not None:
            _, candidate, confidence = frame_candidate
            logger.debug(
                "find candidate rooster class: %s, confidence: %.4f",
                chicken_label.label_map.get(candidate, candidate),
                confidence,
            )
            class_scores[candidate] += confidence
            class_frame_votes[candidate] += 1

    logger.info(
        "final candidate rooster weighted scores: %s, frame votes: %s",
        dict(class_scores),
        dict(class_frame_votes),
    )
    return (
        min(
            class_scores,
            key=lambda candidate: (
                -class_scores[candidate],
                -class_frame_votes[candidate],
                candidate,
            ),
        )
        if class_scores
        else None
    )


def build_high_confidence_chicken_centers(
    video_labels: dict,
    excluded_class,
):
    """每个类别每帧只保留置信度最高的中心点。

    检测结果通常按置信度从高到低排列，旧逻辑却让后出现的低置信度重复框
    覆盖先出现的框。显式比较置信度可以消除这一顺序依赖。
    """

    chicken_centers = {}
    chicken_confidences = {}
    excluded_class = int(excluded_class)
    for frame_pos, labels in video_labels.items():
        for label in labels:
            if len(label) < 6:
                continue
            chicken_cls = int(label[5])
            if chicken_cls == excluded_class:
                continue
            confidence = float(label[4])
            confidence_key = (chicken_cls, frame_pos)
            previous_confidence = chicken_confidences.get(confidence_key)
            if previous_confidence is not None and confidence <= previous_confidence:
                continue

            chicken_confidences[confidence_key] = confidence
            chicken_centers.setdefault(chicken_cls, {})[frame_pos] = get_center(label)

    return chicken_centers


def find_hen_nearest_center_baseline(
    rooster_cls_id,
    rooster_boxes: dict,
    video_labels: dict,
):
    """用逐帧最近中心点投票确定母鸡标签，不使用任何遮挡推理。

    对交配区间内的每个检测帧，排除已识别的公鸡标签后，选取中心点到
    公鸡框中心最近的可见标签并投一票。距离除以公鸡框对角线以消除框尺度
    影响。事件级结果取票数最多的类别；票数相同时取平均归一化距离更小者，
    再相同时取类别 ID 更小者，保证结果可复现。
    """
    if rooster_cls_id is None or not rooster_boxes or not video_labels:
        return None

    mating_poses = sorted(rooster_boxes)
    mating_start_pos = mating_poses[0]
    mating_end_pos = mating_poses[-1]
    class_votes = defaultdict(int)
    class_distance_sums = defaultdict(float)

    for frame_pos in sorted(video_labels):
        if frame_pos < mating_start_pos or frame_pos > mating_end_pos:
            continue

        fit_pos = find_closest_value(frame_pos, mating_poses)
        rooster_box = rooster_boxes[fit_pos]
        rooster_center = get_center(rooster_box)
        box_width = rooster_box[2] - rooster_box[0]
        box_height = rooster_box[3] - rooster_box[1]
        box_diagonal = math.hypot(box_width, box_height)
        if box_diagonal <= 0:
            continue

        nearest_candidate = None
        for label_data in video_labels[frame_pos]:
            if len(label_data) < 6:
                continue
            label_cls = int(label_data[5])
            if label_cls == rooster_cls_id:
                continue

            label_center = get_center(label_data)
            normalized_distance = (
                math.hypot(
                    label_center[0] - rooster_center[0],
                    label_center[1] - rooster_center[1],
                )
                / box_diagonal
            )
            candidate = (normalized_distance, label_cls)
            if nearest_candidate is None or candidate < nearest_candidate:
                nearest_candidate = candidate

        if nearest_candidate is None:
            continue

        normalized_distance, label_cls = nearest_candidate
        class_votes[label_cls] += 1
        class_distance_sums[label_cls] += normalized_distance

    if not class_votes:
        logger.info("nearest-center baseline found no visible non-rooster label")
        return None

    mean_distances = {
        label_cls: class_distance_sums[label_cls] / vote_count
        for label_cls, vote_count in class_votes.items()
    }
    chosen_cls = min(
        class_votes,
        key=lambda label_cls: (
            -class_votes[label_cls],
            mean_distances[label_cls],
            label_cls,
        ),
    )
    logger.info(
        "nearest-center baseline votes: %s, mean normalized distances: %s, "
        "chosen class: %s",
        dict(class_votes),
        mean_distances,
        chosen_cls,
    )
    return chosen_cls


def find_outliers_via_mean_std(
    points,
    sigma_multiplier=default_outlier_sigma_multiplier,
):
    """按点到轨迹均值中心的距离应用 ``mu +/- k*sigma`` 规则。"""
    if not points:
        return []
    if not math.isfinite(sigma_multiplier) or sigma_multiplier <= 0:
        raise ValueError(
            "outlier sigma multiplier must be finite and positive"
        )

    points_array = np.asarray(points, dtype=float)
    trajectory_center = np.mean(points_array, axis=0)
    distances = np.linalg.norm(points_array - trajectory_center, axis=1)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances, ddof=0)
    lower_bound = mean_distance - sigma_multiplier * std_distance
    upper_bound = mean_distance + sigma_multiplier * std_distance
    outlier_indices = (distances < lower_bound) | (distances > upper_bound)
    return outlier_indices.nonzero()[0].tolist()


def find_trajectory_outliers(
    points,
    enabled=True,
    sigma_multiplier=default_outlier_sigma_multiplier,
):
    """Apply the manuscript's distance rule ``mu_d +/- k*sigma_d``."""
    if not enabled:
        return []
    return find_outliers_via_mean_std(
        points,
        sigma_multiplier=sigma_multiplier,
    )


def find_discontinuous_gaps(indices, min_gap, max_idx=None):
    """找出不连续区间"""
    if not indices:
        return []
    logger.debug(f"find discontinuous gaps from indices: {indices}")
    sorted_indices = sorted(indices)
    gaps = []
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] - sorted_indices[i - 1] >= min_gap:  # 发现不连续
            gaps.append((sorted_indices[i - 1], sorted_indices[i]))  # 记录前一个 index

    if max_idx and sorted_indices[-1] < max_idx:
        gaps.append((sorted_indices[-1], max_idx))

    if len(gaps) < 2:
        return gaps

    # 如果两个gap0和gap1  gap0.end = gap1.start，可能是误检的离群点，合并为一个大的gap
    merged = []  # 用于存放合并后的结果
    # 取第一个区间作为当前区间
    current_start, current_end = gaps[0]

    for next_start, next_end in gaps[1:]:
        if current_end == next_start:  # 满足合并条件：前一个end等于后一个start
            # 合并区间，更新当前区间的结束坐标为下一个区间的结束坐标
            current_end = next_end
        else:
            # 如果不满足合并条件，将当前区间加入结果列表
            merged.append((current_start, current_end))
            # 并将下一个区间设为新的当前区间
            current_start, current_end = next_start, next_end

    # 循环结束后，将最后一个当前区间加入结果列表
    merged.append((current_start, current_end))
    logger.debug(f"discontinuous gaps: {merged}")
    return merged


# 找到下一个分割区间的end pos
# current (4, 5) gaps [(1, 2), (4, 5), (8, 10)] => 10
def find_gap_next_end(current_gap, gaps):
    current_idx = 0
    current_start, current_end = current_gap
    for i in range(len(gaps)):
        start, end = gaps[i]
        if current_start == start and current_end == end:
            current_idx = i

    if current_idx + 1 < len(gaps):
        return gaps[current_idx + 1][1]
    else:
        return current_end


def select_trajectory_prefix(
    index_dict,
    end_frame,
    trajectory_window_points=None,
    trajectory_window_frames=None,
):
    """选择截止到 ``end_frame`` 的轨迹前缀。

    ``trajectory_window_frames`` 按原始视频帧号定义真实时间窗口；
    ``trajectory_window_points`` 仅用于保留旧版“最后 N 个采样点”的行为。
    """
    if (
        trajectory_window_points is not None
        and trajectory_window_frames is not None
    ):
        raise ValueError(
            "trajectory_window_points and trajectory_window_frames "
            "cannot both be set"
        )
    if trajectory_window_points is not None and trajectory_window_points < 1:
        raise ValueError("trajectory_window_points must be at least 1")
    if trajectory_window_frames is not None and trajectory_window_frames < 1:
        raise ValueError("trajectory_window_frames must be at least 1")

    prefix_items = [
        (frame_pos, center)
        for frame_pos, center in index_dict.items()
        if frame_pos <= end_frame
    ]
    if trajectory_window_frames is not None:
        first_frame = end_frame - trajectory_window_frames
        prefix_items = [
            item for item in prefix_items if item[0] >= first_frame
        ]
    elif trajectory_window_points is not None:
        prefix_items = prefix_items[-trajectory_window_points:]
    return prefix_items


# 返回[tuple(label-class-id, count)], dict{key=label-class-id, value=set(frame pos)}
def count_discontinuous_centers(
    chicken_centers: dict,
    rooster_boxes: dict,
    min_gap: int,
    roi_area_ratio=default_rooster_roi_area_ratio,
    trajectory_window_points=None,
    trajectory_window_frames=None,
    outlier_filter_enabled=True,
    outlier_sigma_multiplier=default_outlier_sigma_multiplier,
):
    """统计每个 标签 中不连续区间的第一个中心点一直位于公鸡 box 内的次数"""
    if trajectory_window_points is None and trajectory_window_frames is None:
        # 保留旧接口行为；参数敏感性实验会显式传入固定窗口以避免与 τ_gap 联动。
        trajectory_window_points = min_gap

    result = []
    # 可能检测有误的标签 所在的frame索引列表
    false_label_frame_poses = defaultdict(set)
    rooster_start_pos = min(rooster_boxes.keys())
    rooster_max_pos = max(rooster_boxes.keys())
    for label_cls, index_dict in chicken_centers.items():
        logging.info(f"check label: {label_cls}, {chicken_label.label_map[label_cls]}")

        frame_indices = list(index_dict.keys())
        gap_indices = find_discontinuous_gaps(
            frame_indices, min_gap, max_idx=rooster_max_pos
        )
        if not gap_indices:
            logging.info(f"no discontinuous gaps found, skip")
            continue

        candidate_gaps = sorted(gap_indices, key=lambda x: x[1] - x[0], reverse=True)
        candidate_counts = [0] * len(candidate_gaps)
        for candidate_idx, candidate_gap in enumerate(candidate_gaps):
            logger.info(f"check candidate gap at index : {candidate_idx}")
            count = 0
            start, end = candidate_gap
            if start > rooster_max_pos:
                logger.info(f"gap start pos is larger than rooster max pos, skip")
                continue
            if end < rooster_start_pos:
                logger.info(f"gap end pos is smaller than rooster start pos, skip")
                continue

            # 只取消失前的部分点，太多的点可能因为不断移动，导致错误删除了离群点
            prefix_items = select_trajectory_prefix(
                index_dict,
                start,
                trajectory_window_points=trajectory_window_points,
                trajectory_window_frames=trajectory_window_frames,
            )
            prefix_poses = [frame_pos for frame_pos, _ in prefix_items]
            prefix_points = [center for _, center in prefix_items]

            # 删除离群点；敏感性实验可显式关闭或改变标准差倍率。
            outlier_indices = find_trajectory_outliers(
                prefix_points,
                enabled=outlier_filter_enabled,
                sigma_multiplier=outlier_sigma_multiplier,
            )
            if outlier_indices:
                for outlier_idx in outlier_indices:
                    if outlier_idx < len(prefix_poses):
                        frame_pos = prefix_poses[outlier_idx]
                        logging.warning(f"find outlier at frame pos: {frame_pos}")
                        false_label_frame_poses[label_cls].add(frame_pos)
                prefix_points = [
                    p
                    for idx, p in enumerate(prefix_points)
                    if idx not in outlier_indices
                ]

            if not prefix_points:
                logging.info("no prefix points available, skip")
                continue

            center_point = prefix_points[-1]  # 最后一次出现的中心点
            logging.info(
                f"gap start: {start}, gap end: {end}, last seen label center: {center_point}"
            )
            if end < rooster_max_pos:
                # 延长的目的，防止中间因为标签误识别，断开了当前的标签gap，即使公鸡交配提前完成，因为走出了实际交配的区域，对计数的影响不大
                end = find_gap_next_end(candidate_gap, gap_indices)
                logger.info(
                    f"gap end is smaller than rooster max pos, extend it to next gap end {end}"
                )

            for idx in range(start + 1, end - 1):
                if idx in rooster_boxes:  # 确保 box_dict 中有该 index
                    box = rooster_boxes[idx]

                    if is_in_rooster_center_area(
                        box,
                        center_point,
                        roi_area_ratio=roi_area_ratio,
                    ):
                        count += 1

            logger.info(f"valid count: {count}")
            candidate_counts[candidate_idx] = count

        if not candidate_counts:
            continue

        max_cnt, idx_of_max_val = max(
            (value, idx) for idx, value in enumerate(candidate_counts)
        )
        result.append((label_cls, max_cnt))
        for idx in range(len(candidate_counts)):
            if idx < idx_of_max_val:
                logger.warning(
                    f"candidate gap inside count less than max count, "
                    f"may have outliers that cuts the entier gap,"
                    f"add frame pos {candidate_gaps[idx][0]} to false label detect"
                )
                false_label_frame_poses[label_cls].add(candidate_gaps[idx][0])
            else:
                break

    return result, false_label_frame_poses


def calculate_midpoint_reappear_distance_to_rooster(
    chicken_centers: dict,
    rooster_mating_boxes: dict,
    fps,
    outlier_filter_enabled=True,
    outlier_sigma_multiplier=default_outlier_sigma_multiplier,
    trajectory_window_enabled=True,
):
    label_distance = []
    # 可能检测有误的标签 所在的frame索引列表
    false_label_frame_poses = defaultdict(set)
    mating_poses = list(rooster_mating_boxes.keys())
    mating_start_pos = min(mating_poses)
    mating_end_pos = max(mating_poses)
    # Fallback candidates start after the temporal midpoint of the mating interval.
    post_thresh = (mating_start_pos + mating_end_pos) // 2

    for label_cls, index_dict in chicken_centers.items():
        logging.info(f"check label: {label_cls}, {chicken_label.label_map[label_cls]}")
        frame_indices = list(index_dict.keys())
        # 查找交配区间中点后的第一个出现点。
        appear_pos = 0
        for frame_pos in frame_indices:
            if frame_pos > post_thresh:
                appear_pos = frame_pos
                break

        if appear_pos == 0:
            logger.info(f"no valid reappear frame pos, skip")
            continue

        prefix_count = len([pos for pos in frame_indices if pos < appear_pos])
        if prefix_count > fps // 2:
            logger.info(f"too many prefix appearances, count: {prefix_count}, skip")
            continue

        post_dict = {k: v for k, v in index_dict.items() if k >= appear_pos}
        if trajectory_window_enabled:
            # At least three points are needed to identify a trajectory outlier.
            if len(post_dict) < 3:
                continue
            post_keep = fps // 2
            post_centers = list(post_dict.values())[:post_keep]
            post_poses = list(post_dict.keys())[:post_keep]
            outlier_indices = find_trajectory_outliers(
                post_centers,
                enabled=outlier_filter_enabled,
                sigma_multiplier=outlier_sigma_multiplier,
            )
        else:
            # Ablation: use the first valid point after the mating midpoint.
            post_poses = [appear_pos]
            outlier_indices = []
        if outlier_indices:
            for outlier_idx in outlier_indices:
                if outlier_idx < len(post_poses):
                    frame_pos = post_poses[outlier_idx]
                    logging.warning(f"find outlier at frame pos: {frame_pos}")
                    false_label_frame_poses[label_cls].add(frame_pos)

            post_poses = [
                p for idx, p in enumerate(post_poses) if idx not in outlier_indices
            ]

        if not post_poses:
            continue

        appear_pos = post_poses[0]

        rooster_pos = mating_end_pos
        for mate_pos in mating_poses:
            if mate_pos >= appear_pos:
                rooster_pos = mate_pos
                break

        rooster_box = rooster_mating_boxes[rooster_pos]
        rooster_center = (
            (rooster_box[0] + rooster_box[2]) / 2,
            (rooster_box[1] + rooster_box[3]) / 2,
        )
        hen_center = index_dict[appear_pos]
        logger.info(
            f"calculate distance with rooster pos {rooster_pos}, "
            f"at appear pos {appear_pos}, label center: {hen_center}"
        )

        distance = math.sqrt(
            (rooster_center[0] - hen_center[0]) ** 2
            + (rooster_center[1] - hen_center[1]) ** 2
        )
        label_distance.append((label_cls, distance))

    return label_distance, false_label_frame_poses


# rooster_boxes: key=frame_pos, value=box
def find_rooster_and_hen_from_clip(
    clip_name,
    clip_src,
    rooster_boxes: dict,
    mate_start_pos: int,
    mate_end_pos: int,
    roi_area_ratio=default_rooster_roi_area_ratio,
    tau_gap_seconds=default_tau_gap_seconds,
    trajectory_window_seconds=None,
    precomputed_detection=None,
    trajectory_window_fps_multiplier=None,
    outlier_filter_enabled=True,
    outlier_sigma_multiplier=default_outlier_sigma_multiplier,
    disable_tau_gap=False,
    disable_trajectory_window=False,
    return_metadata=False,
):
    if not 0 < roi_area_ratio <= 1:
        raise ValueError("roi_area_ratio must be in (0, 1]")
    if tau_gap_seconds <= 0:
        raise ValueError("tau_gap_seconds must be greater than 0")
    if (
        trajectory_window_seconds is not None
        and trajectory_window_seconds <= 0
    ):
        raise ValueError("trajectory_window_seconds must be greater than 0")
    if (
        trajectory_window_seconds is not None
        and trajectory_window_fps_multiplier is not None
    ):
        raise ValueError(
            "trajectory_window_seconds and trajectory_window_fps_multiplier "
            "are mutually exclusive"
        )
    if (
        trajectory_window_fps_multiplier is not None
        and (
            not math.isfinite(trajectory_window_fps_multiplier)
            or trajectory_window_fps_multiplier <= 0
        )
    ):
        raise ValueError(
            "trajectory_window_fps_multiplier must be finite and positive"
        )
    if (
        not math.isfinite(outlier_sigma_multiplier)
        or outlier_sigma_multiplier <= 0
    ):
        raise ValueError(
            "outlier_sigma_multiplier must be finite and positive"
        )

    if precomputed_detection is None:
        fps, video_labels = detect_clip_labels(clip_src)
    else:
        fps, video_labels = precomputed_detection
    if not video_labels:
        logger.error("no video labels detected from clip source")
        result = (None, None, "unresolved") if return_metadata else (None, None)
        return result

    logger.info("try to find rooster label")
    class_idx = find_rooster_label(
        rooster_boxes,
        video_labels,
        roi_area_ratio=roi_area_ratio,
    )
    if class_idx is None:
        logger.error("cannot determine rooster label")
        result = (None, None, "unresolved") if return_metadata else (None, None)
        return result

    if class_idx not in chicken_label.label_map:
        logger.error("rooster label id not in labels map")
        result = (None, None, "unresolved") if return_metadata else (None, None)
        return result

    rooster_cls_id = class_idx
    rooster_label = chicken_label.label_map[class_idx]
    logger.info(
        f"rooster label found, class index is {class_idx}, label name is {rooster_label}"
    )

    logger.info(f"now try to find hen label")

    logger.info(f"rooster boxes count including margin: {len(rooster_boxes)}")
    if mate_end_pos < 0:
        mate_end_pos = 10**8
    # 过滤公鸡的box，只保留交配区间的box，这对找母鸡的标签很重要
    rooster_mating_boxes = {
        pos: box
        for pos, box in rooster_boxes.items()
        if mate_start_pos <= pos < mate_end_pos
    }
    logger.info(f"rooster mating boxes count: {len(rooster_mating_boxes)}")

    # 把每个 label 单独拎出来；同一类别同一帧只保留最高置信度检测框。
    # key=label_id, value={frame_pos: label_center}
    chicken_centers = build_high_confidence_chicken_centers(
        video_labels,
        excluded_class=rooster_cls_id,
    )

    # Main path: find labels that disappear while their last valid point is in
    # the dynamic rooster ROI.
    if disable_tau_gap:
        analyzed_frame_step = infer_analyzed_frame_step(video_labels.keys())
        # Start a gap as soon as one scheduled label-detection frame is absent.
        min_gap_frames = analyzed_frame_step + 1
        tau_gap_description = (
            "disabled; any missed scheduled detection "
            f"(inferred frame step {analyzed_frame_step})"
        )
    else:
        min_gap_frames = seconds_to_frame_threshold(tau_gap_seconds, fps)
        tau_gap_description = f"{tau_gap_seconds:.3f}s = {min_gap_frames} frames"

    if disable_trajectory_window:
        trajectory_window_points = 1
        trajectory_window_frames = None
        trajectory_window_description = "disabled; immediate last valid point"
    elif trajectory_window_fps_multiplier is not None:
        trajectory_window_points = fps_multiplier_to_point_count(
            trajectory_window_fps_multiplier, fps
        )
        trajectory_window_frames = None
        trajectory_window_description = (
            f"Nw={trajectory_window_fps_multiplier:.3f}F = "
            f"{trajectory_window_points} valid detection points"
        )
    elif trajectory_window_seconds is None:
        trajectory_window_points = fps
        trajectory_window_frames = None
        trajectory_window_description = (
            f"deployed Nw=1.000F = {trajectory_window_points} "
            "valid detection points"
        )
    else:
        trajectory_window_points = None
        trajectory_window_frames = seconds_to_frame_threshold(
            trajectory_window_seconds, fps
        )
        trajectory_window_description = (
            f"{trajectory_window_seconds:.3f}s = "
            f"{trajectory_window_frames} frames"
        )
    logger.info(
        "tau_gap %s; trajectory window %s",
        tau_gap_description,
        trajectory_window_description,
    )
    trajectory_outlier_filter_enabled = (
        outlier_filter_enabled and not disable_trajectory_window
    )
    if trajectory_outlier_filter_enabled:
        logger.info(
            "use distance mean-std outlier filter: mu +/- %.3g sigma",
            outlier_sigma_multiplier,
        )
    else:
        logger.info("disable trajectory outlier filtering")
    possible, false_label_frame_poses = count_discontinuous_centers(
        chicken_centers,
        rooster_mating_boxes,
        min_gap_frames,
        roi_area_ratio=roi_area_ratio,
        trajectory_window_points=trajectory_window_points,
        trajectory_window_frames=trajectory_window_frames,
        outlier_filter_enabled=trajectory_outlier_filter_enabled,
        outlier_sigma_multiplier=outlier_sigma_multiplier,
    )

    possible.sort(key=lambda x: x[1], reverse=True)
    for p in possible:
        if p[1] > 0 and p[0] in chicken_label.label_map:
            logger.info(
                f"candidate hen: {chicken_label.label_map[p[0]]}, under rooster mating center count {p[1]}"
            )
    chosen_hen = possible[0] if possible else (None, 0)

    hen_label = None
    hen_path = "unresolved"
    # 主路径至少需要累计约半秒的动态 ROI 内点数。
    if chosen_hen[1] >= fps // 2:
        if chosen_hen[0] not in chicken_label.label_map:
            logger.error(f"hen label id not in chicken labels map")
        else:
            hen_label = chicken_label.label_map[chosen_hen[0]]
            hen_path = "main"
            logger.info(f"chosen hen label is {hen_label}")
    else:
        if chosen_hen[1] > 0:
            logger.warning(
                f"most possible hen center under rooster mating center count {chosen_hen[1]}, "
                f"too low, set it suspicious"
            )
            hen_label = chicken_label.label_map[chosen_hen[0]] + "*"
            hen_path = "main"
        else:
            logger.error(f"cannot find hen label")

    # 相对不太可靠，前置的标签没有出现，看后置标签出现的位置，离公鸡框最近的作为母鸡的标签
    if not hen_label:
        hen_path = "fallback"
        logger.warning(
            "cannot determine hen label with normal method; "
            "checking eligible detections after the mating-interval midpoint"
        )
        label_distances, false_label_frame_poses = (
            calculate_midpoint_reappear_distance_to_rooster(
                chicken_centers,
                rooster_mating_boxes,
                fps,
                outlier_filter_enabled=trajectory_outlier_filter_enabled,
                outlier_sigma_multiplier=outlier_sigma_multiplier,
                trajectory_window_enabled=not disable_trajectory_window,
            )
        )
        if label_distances:
            label_distances.sort(key=lambda x: x[1])
            label_distances = label_distances[:3]
            for p in label_distances:
                if p[0] in chicken_label.label_map:
                    logger.info(
                        f"candidate hen: {chicken_label.label_map[p[0]]}, reappear distance to rooster: {p[1]}"
                    )

            chosen_hen = label_distances[0]
            if chosen_hen[0] in chicken_label.label_map:
                hen_label = chicken_label.label_map[chosen_hen[0]] + "*"
                logger.info(
                    f"chosen hen label via midpoint-fallback distance is {hen_label}"
                )

    if return_metadata:
        return rooster_label, hen_label, hen_path
    return rooster_label, hen_label


