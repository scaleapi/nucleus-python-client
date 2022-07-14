from functools import wraps
from typing import Dict, List, Tuple

import numpy as np

try:
    from shapely.geometry import Polygon
except (ModuleNotFoundError, OSError):
    from ..package_not_installed import PackageNotInstalled

    Polygon = PackageNotInstalled


from nucleus.annotation import CuboidAnnotation
from nucleus.prediction import CuboidPrediction

from .base import ScalarResult


def group_cuboids_by_label(
    annotations: List[CuboidAnnotation],
    predictions: List[CuboidPrediction],
) -> Dict[str, Tuple[List[CuboidAnnotation], List[CuboidPrediction]]]:
    """Groups input annotations and predictions by label.

    Args:
        annotations: list of input cuboid annotations
        predictions: list of input cuboid predictions

    Returns:
        Mapping from each label to (annotations, predictions) tuple
    """
    labels = set(annotation.label for annotation in annotations)
    labels |= set(prediction.label for prediction in predictions)
    grouped: Dict[
        str, Tuple[List[CuboidAnnotation], List[CuboidPrediction]]
    ] = {label: ([], []) for label in labels}
    for annotation in annotations:
        grouped[annotation.label][0].append(annotation)
    for prediction in predictions:
        grouped[prediction.label][1].append(prediction)
    return grouped


def label_match_wrapper(metric_fn):
    """Decorator to add the ability to only apply metric to annotations and
    predictions with matching labels.

    Args:
        metric_fn: Metric function that takes a list of annotations, a list
            of predictions, and optional args and kwargs.

    Returns:
        Metric function which can optionally enforce matching labels.
    """

    @wraps(metric_fn)
    def wrapper(
        annotations: List[CuboidAnnotation],
        predictions: List[CuboidPrediction],
        *args,
        enforce_label_match: bool = False,
        **kwargs,
    ) -> ScalarResult:
        # Simply return the metric if we are not enforcing label matches.
        if not enforce_label_match:
            return metric_fn(annotations, predictions, *args, **kwargs)

        # For each bin of annotations/predictions, compute the metric applied
        # only to that bin. Then aggregate results across all bins.
        grouped_inputs = group_cuboids_by_label(annotations, predictions)
        metric_results = []
        for binned_annotations, binned_predictions in grouped_inputs.values():
            metric_result = metric_fn(
                binned_annotations, binned_predictions, *args, **kwargs
            )
            metric_results.append(metric_result)
        assert all(
            isinstance(r, ScalarResult) for r in metric_results
        ), "Expected every result to be a ScalarResult"
        return ScalarResult.aggregate(metric_results)

    return wrapper


def process_dataitem(dataitem):
    processed_item = {}
    processed_item["xyz"] = np.array(
        [[ann.position.x, ann.position.y, ann.position.z] for ann in dataitem]
    )
    processed_item["wlh"] = np.array(
        [
            [ann.dimensions.x, ann.dimensions.y, ann.dimensions.z]
            for ann in dataitem
        ]
    )
    processed_item["yaw"] = np.array([ann.yaw for ann in dataitem])
    return processed_item


def compute_outer_iou(
    xyz_0: np.ndarray,
    wlh_0: np.ndarray,
    yaw_0: np.ndarray,
    xyz_1: np.ndarray,
    wlh_1: np.ndarray,
    yaw_1: np.ndarray,
    scale_convention: bool = True,
    distance_threshold=25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes outer 3D and 2D IoU
    :param xyz_0: (n, 3)
    :param wlh_0: (n, 3)
    :param yaw_0: (n,)
    :param xyz_1: (m, 3)
    :param wlh_1: (m, 3)
    :param yaw_1: (m,)
    :param scale_convention: flag whether the internal Scale convention is used (have to be adjusted by pi/2)
    :param distance_threshold: computes iou only within this distance (~3x speedup)
    :return: (n, m) 3D IoU, (n, m) 2D IoU
    """

    bottom_z = np.maximum.outer(
        xyz_0[:, 2] - (wlh_0[:, 2] / 2), xyz_1[:, 2] - (wlh_1[:, 2] / 2)
    )
    top_z = np.minimum.outer(
        xyz_0[:, 2] + (wlh_0[:, 2] / 2), xyz_1[:, 2] + (wlh_1[:, 2] / 2)
    )
    height_intersection = np.maximum(0, top_z - bottom_z)

    cuboid_corners_0 = get_batch_cuboid_corners(
        xyz_0, wlh_0, yaw_0, scale_convention=scale_convention
    )
    cuboid_corners_1 = get_batch_cuboid_corners(
        xyz_1, wlh_1, yaw_1, scale_convention=scale_convention
    )
    polygons_1 = [
        Polygon(corners_1[[1, 0, 4, 5, 1], :2])
        for corners_1 in cuboid_corners_1
    ]
    area_intersection = np.zeros(
        (cuboid_corners_0.shape[0], cuboid_corners_1.shape[0]),
        dtype=np.float32,
    )

    if cuboid_corners_0.shape[0] != 0 and cuboid_corners_1.shape[0] != 0:
        distance_mask = (
            np.linalg.norm(
                xyz_0[:, np.newaxis, :] - xyz_1[np.newaxis, :, :], axis=2
            )
            < distance_threshold
        )

    for i, corners_0 in enumerate(cuboid_corners_0):
        for j, polygon_1 in enumerate(polygons_1):
            if distance_mask[i, j]:
                area_intersection[i, j] = (
                    Polygon(corners_0[[1, 0, 4, 5, 1], :2])
                    .intersection(polygon_1)
                    .area
                )

    intersection = height_intersection * area_intersection
    area_0 = wlh_0[:, 0] * wlh_0[:, 1]
    area_1 = wlh_1[:, 0] * wlh_1[:, 1]
    union_2d = np.add.outer(area_0, area_1) - area_intersection

    volume_0 = area_0 * wlh_0[:, 2]
    volume_1 = area_1 * wlh_1[:, 2]
    union = np.add.outer(volume_0, volume_1) - intersection
    return intersection / union, area_intersection / union_2d


def get_batch_cuboid_corners(
    xyz: np.ndarray,
    wlh: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray = None,
    roll: np.ndarray = None,
    scale_convention: bool = True,
) -> np.ndarray:
    """
    Vectorized batch version of get_cuboid_corners
    :param xyz: (n, 3)
    :param wlh: (n, 3)
    :param yaw: (n,)
    :param pitch: (n,)
    :param roll: (n,)
    :param scale_convention: flag whether the internal Scale convention is used (have to be adjusted by pi/2)
    :return: (n, 8, 3)
    """
    if scale_convention:
        yaw = yaw.copy() + np.pi / 2

    w, l, h = wlh[:, 0, None], wlh[:, 1, None], wlh[:, 2, None]

    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.stack((x_corners, y_corners, z_corners), axis=1)

    rot_mats = get_batch_rotation_matrices(yaw, pitch, roll)
    corners = np.matmul(rot_mats, corners)

    x, y, z = xyz[:, 0, None], xyz[:, 1, None], xyz[:, 2, None]
    corners[:, 0, :] = corners[:, 0, :] + x
    corners[:, 1, :] = corners[:, 1, :] + y
    corners[:, 2, :] = corners[:, 2, :] + z
    return corners.swapaxes(1, 2)


def get_batch_rotation_matrices(
    yaw: np.ndarray, pitch: np.ndarray = None, roll: np.ndarray = None
) -> np.ndarray:
    if pitch is None:
        pitch = np.zeros_like(yaw)
    if roll is None:
        roll = np.zeros_like(yaw)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    return np.stack(
        (
            np.stack(
                (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), 1
            ),
            np.stack(
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), 1
            ),
            np.stack((-sp, cp * sr, cp * cr), 1),
        ),
        1,
    )


def associate_cuboids_on_iou(
    xyz_0: np.ndarray,
    wlh_0: np.ndarray,
    yaw_0: np.ndarray,
    xyz_1: np.ndarray,
    wlh_1: np.ndarray,
    yaw_1: np.ndarray,
    threshold_in_overlap_ratio: float = 0.1,
) -> List[Tuple[int, int]]:
    if xyz_0.shape[0] < 1 or xyz_1.shape[0] < 1:
        return []
    iou_matrix, _ = compute_outer_iou(xyz_0, wlh_0, yaw_0, xyz_1, wlh_1, yaw_1)
    mapping = []
    for i, m in enumerate(iou_matrix.max(axis=1)):
        if m >= threshold_in_overlap_ratio:
            mapping.append((i, iou_matrix[i].argmax()))
    return mapping


def recall_precision(
    prediction: List[CuboidPrediction],
    groundtruth: List[CuboidAnnotation],
    threshold_in_overlap_ratio: float,
) -> Dict[str, float]:
    """
    Calculates the precision and recall of each lidar frame.

    Args:
        :param predictions: list of cuboid annotation predictions.
        :param ground_truth: list of cuboid annotation groundtruths.
        :param threshold: IOU threshold to consider detection as valid. Must be in [0, 1].
    """

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    num_predicted = 0
    num_instances = 0

    gt_items = process_dataitem(groundtruth)
    pred_items = process_dataitem(prediction)

    num_predicted += pred_items["xyz"].shape[0]
    num_instances += gt_items["xyz"].shape[0]

    tp = np.zeros(pred_items["xyz"].shape[0])
    fp = np.ones(pred_items["xyz"].shape[0])
    fn = np.ones(gt_items["xyz"].shape[0])

    mapping = associate_cuboids_on_iou(
        pred_items["xyz"],
        pred_items["wlh"],
        pred_items["yaw"] + np.pi / 2,
        gt_items["xyz"],
        gt_items["wlh"],
        gt_items["yaw"] + np.pi / 2,
        threshold_in_overlap_ratio=threshold_in_overlap_ratio,
    )

    for pred_id, gt_id in mapping:
        if fn[gt_id] == 0:
            continue
        tp[pred_id] = 1
        fp[pred_id] = 0
        fn[gt_id] = 0

    tp_sum += tp.sum()
    fp_sum += fp.sum()
    fn_sum += fn.sum()

    return {
        "tp_sum": tp_sum,
        "fp_sum": fp_sum,
        "fn_sum": fn_sum,
        "precision": tp_sum / (tp_sum + fp_sum),
        "recall": tp_sum / (tp_sum + fn_sum),
        "num_predicted": num_predicted,
        "num_instances": num_instances,
    }


def detection_iou(
    prediction: List[CuboidPrediction],
    groundtruth: List[CuboidAnnotation],
    threshold_in_overlap_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the 2D IOU and 3D IOU overlap between predictions and groundtruth.
    Uses linear sum assignment to associate cuboids.

    Args:
        :param predictions: list of cuboid annotation predictions.
        :param ground_truth: list of cuboid annotation groundtruths.
        :param threshold: IOU threshold to consider detection as valid. Must be in [0, 1].
    """

    gt_items = process_dataitem(groundtruth)
    pred_items = process_dataitem(prediction)

    meter_2d = []
    meter_3d = []

    if gt_items["xyz"].shape[0] == 0 or pred_items["xyz"].shape[0] == 0:
        return np.array([0.0]), np.array([0.0])

    iou_3d, iou_2d = compute_outer_iou(
        gt_items["xyz"],
        gt_items["wlh"],
        gt_items["yaw"],
        pred_items["xyz"],
        pred_items["wlh"],
        pred_items["yaw"],
    )

    for i, m in enumerate(iou_3d.max(axis=1)):
        if m >= threshold_in_overlap_ratio:
            j = iou_3d[i].argmax()
            meter_3d.append(iou_3d[i, j])
            meter_2d.append(iou_2d[i, j])

    return np.array(meter_3d), np.array(meter_2d)
