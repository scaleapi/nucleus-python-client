import logging
from collections import defaultdict
from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from nucleus import Point, PolygonPrediction, Segment
from nucleus.metrics.custom_types import BoxOrPolygonAnnotation
from nucleus.metrics.polygon_utils import polygon_annotation_to_shape
from nucleus.package_not_installed import (  # pylint: disable=ungrouped-imports
    PackageNotInstalled,
)

FALSE_POSITIVES = "__non_max_false_positive"

try:
    from shapely import geometry
except (ModuleNotFoundError, OSError):
    geometry = PackageNotInstalled


try:
    from rasterio import features
except (ModuleNotFoundError, OSError):
    rasterio = PackageNotInstalled


def instance_mask_to_polys(instance_mask: np.ndarray, background_code=None):
    mask_values = []
    all_polygons = []
    not_background_mask = (
        (instance_mask != background_code) if background_code else None
    )
    for shape, value in features.shapes(
        instance_mask.astype(np.int32),
        mask=not_background_mask,
    ):
        poly = geometry.shape(shape)
        all_polygons.append(poly)
        mask_values.append(int(value))

    return mask_values, all_polygons


def transform_poly_codes_to_poly_preds(
    dataset_item_id: str, pred_value, pred_polys, code_to_label
):
    polygon_predictions = []
    for code, poly in zip(pred_value, pred_polys):
        if poly.type != "Polygon":
            continue
        label = code_to_label[code]
        x_stack, y_stack = poly.exterior.coords.xy
        pred = PolygonPrediction(
            label,
            vertices=[Point(x, y) for x, y in zip(x_stack, y_stack)],
            reference_id=dataset_item_id,
        )
        polygon_predictions.append(pred)
    return polygon_predictions


def max_iou_match_from_confusion(confusion):
    """Calculate iou from confusion matrix and do linear sum assignment to get strongest candiaate for each GT

    Returns:
        iou_matrix with same dims as confusion and 1-d best match rows, 1-d best match cols
    """
    iou = np.zeros(confusion.shape, dtype=np.float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                intersection = confusion[i, j]
                union = (
                    confusion[i, :].sum()
                    + confusion[:, j].sum()
                    - intersection
                )
                if union > 0:
                    iou[i, j] = intersection / union
    iou = np.nan_to_num(iou)
    iou_match_row, iou_match_col = linear_sum_assignment(-iou)
    return iou, iou_match_row, iou_match_col


def fast_confusion_matrix(
    label_true: np.ndarray, label_pred: np.ndarray, n_class: int
) -> np.ndarray:
    """Calculates confusion matrix - fast!

    Outputs a confusion matrix where each row is GT confusion and column is prediction confusion
    Example:
        _fast_hist(np.array([0, 1, 2, 3], dtype=np.int32), np.array([0, 1, 1, 1], dtype=np.int32), n_class=4)
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0]])
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def non_max_suppress_confusion(confusion: np.ndarray, iou_threshold):
    """Uses linear sum assignment to find biggest pixel-wise IOU match. Secondary matches are moved to last column
    as false positives (since they are outside of instance boundaries).

    TODO(gunnar): Change logic to only move suppressed TP to FP so we can maintain the original confusion

    Arguments:
        confusion: Confusion matrix
        iou_threshold: Detections under iou threshold are considered false positives

    Returns:
        Non max suppressed confusion matrix (NxN) with dimension ((N+1)x(N+1)) where the last column are suppressed
        positives

    """
    original_count = confusion.sum()
    iou, max_iou_row, max_iou_col = max_iou_match_from_confusion(confusion)
    # Prepare the new confusion with +1 added to the shape
    non_max_suppressed = np.zeros(np.add(confusion.shape, 1), dtype=np.int64)

    # ----  IOU filtering from diagonal
    keep_diagonal = iou.diagonal() >= iou_threshold
    # Move false positives from diag to new false_positive class
    move_indexes = np.where(~keep_diagonal)

    # log iou suppressed as FPs
    non_max_suppressed[:, -1].put(
        move_indexes, confusion.diagonal().take(move_indexes)
    )
    # Zero false positives on diagonal
    keep_indexes = np.where(keep_diagonal)
    new_diagonal = np.zeros(len(confusion.diagonal()))
    new_diagonal.put(keep_indexes, confusion.diagonal()[keep_indexes])
    np.fill_diagonal(confusion, new_diagonal)
    # ----

    # -- move max over
    non_max_suppressed[max_iou_row, max_iou_col] = confusion[
        max_iou_row, max_iou_col
    ]
    confusion[max_iou_row, max_iou_col] = np.zeros(len(max_iou_col))
    # --

    # -- move left on diagonal to FPs
    non_max_suppressed[:, -1] = (
        np.r_[confusion.diagonal(), np.zeros(1)] + non_max_suppressed[:, -1]
    )
    np.fill_diagonal(confusion, np.zeros(len(confusion.diagonal())))
    # --

    # -- move valid confusions over
    valid_confusion = confusion > 0
    valid_row, valid_col = np.where(confusion > 0)
    flat_idxs = valid_col + valid_row * non_max_suppressed.shape[1]
    non_max_suppressed.put(flat_idxs, confusion[valid_confusion])
    # --
    assert original_count == non_max_suppressed.sum()
    return non_max_suppressed


def rasterize_polygons_to_segmentation_mask(
    annotations: Sequence[BoxOrPolygonAnnotation], shape: Tuple[int, int]
) -> Tuple[np.ndarray, List[Segment]]:
    polys = [polygon_annotation_to_shape(a) for a in annotations]
    segments = [
        Segment(ann.label, index=idx + 1, metadata=ann.metadata)
        for idx, ann in enumerate(annotations)
    ]
    poly_vals = [
        (poly, segment.index) for poly, segment in zip(polys, segments)
    ]
    rasterized = features.rasterize(
        poly_vals,
        out_shape=shape,
        fill=0,
        out=None,
        all_touched=False,
        dtype=None,
    )
    return rasterized, segments


def convert_to_instance_seg_confusion(confusion, annotation, prediction):
    pred_index_to_label = {s.index: s.label for s in prediction.annotations}

    gt_label_to_old_indexes = defaultdict(set)
    for segment in annotation.annotations:
        gt_label_to_old_indexes[segment.label].add(segment.index)

    pr_label_to_old_indexes = defaultdict(set)
    for segment in prediction.annotations:
        pr_label_to_old_indexes[segment.label].add(segment.index)

    new_labels = list(
        dict.fromkeys(
            list(pr_label_to_old_indexes)[:-1]
            + list(gt_label_to_old_indexes)[:-1]
        )
    )
    # NOTE: We make sure that FALSE_POSITIVES are at the back
    false_positive_label = list(pr_label_to_old_indexes.keys())[-1]
    new_labels.append(false_positive_label)
    non_taxonomy_classes = {len(new_labels) - 1}

    num_classes = len(new_labels)
    new_confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int32,
    )

    for gt_class_idx, from_label in enumerate(new_labels):
        from_indexes = gt_label_to_old_indexes[from_label]
        tp, fp = 0, 0
        if len(from_indexes) == 0:
            logging.warning(
                "No annotations with label '%s', interpreted as false positives.",
                from_label,
            )
            non_taxonomy_classes.add(gt_class_idx)
            # NOTE: If the index is not in the gt segments it comes from the predictions, we get the "old_indexes"
            # from there even though they are all FPs
            from_indexes = pr_label_to_old_indexes[from_label]
        for gt_instance_idx in from_indexes:
            max_col = np.argmax(
                confusion[gt_instance_idx, :]
            )  # TODO: Get from IOU
            if confusion[gt_instance_idx, max_col] == 0:
                continue

            for pred_class_idx, to_label in enumerate(new_labels):
                to_indexes = pr_label_to_old_indexes[to_label]
                if from_label == to_label:
                    if pred_index_to_label.get(max_col, None) == from_label:
                        tp += confusion[gt_instance_idx, max_col]
                        fp_indexes = to_indexes - {max_col}
                    else:
                        fp_indexes = to_indexes
                    fp += (
                        confusion[gt_instance_idx, :]
                        .take(list(fp_indexes))
                        .sum()
                        + confusion[gt_instance_idx, -1]
                    )
                else:
                    new_confusion[gt_class_idx, pred_class_idx] += (
                        confusion[gt_instance_idx, :]
                        .take(list(to_indexes))
                        .sum()
                    )

        new_confusion[gt_class_idx, gt_class_idx] = tp
        new_confusion[gt_class_idx, -1] = fp

    assert confusion.sum() == new_confusion.sum()
    return new_confusion, new_labels, non_taxonomy_classes


def setup_iou_thresholds(iou_thresholds: Union[Sequence[float], str] = "coco"):
    supported_iou_setups = {"coco"}
    if isinstance(iou_thresholds, (list, np.ndarray)):
        return np.array(iou_thresholds, np.float_)
    elif isinstance(iou_thresholds, str):
        if iou_thresholds in supported_iou_setups:
            return np.arange(0.5, 1.0, 0.05)
        else:
            raise RuntimeError(
                f"Got invalid configuration value: {iou_thresholds}, expected one of: {supported_iou_setups}"
            )
    else:
        raise RuntimeError(
            f"Got invalid configuration: {iou_thresholds}. Expected list of floats or one of: {supported_iou_setups}"
        )
