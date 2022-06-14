import numpy as np
from scipy.optimize import linear_sum_assignment

from nucleus import Point, PolygonPrediction
from nucleus.package_not_installed import (  # pylint: disable=ungrouped-imports
    PackageNotInstalled,
)

try:
    from shapely import geometry
except ModuleNotFoundError:
    geometry = PackageNotInstalled


try:
    from rasterio import features
except ModuleNotFoundError:
    rasterio = PackageNotInstalled


def instance_mask_to_polys(instance_mask: np.ndarray, background_code=None):
    mask_values = []
    all_polygons = []
    not_background_mask = (
        (instance_mask != background_code) if background_code else None
    )
    for shape, value in features.shapes(
        instance_mask.astype(np.int16),
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
        _fast_hist(np.array([0, 1, 2, 3], dtype=np.int16), np.array([0, 1, 1, 1], dtype=np.int16), n_class=4)
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
    iou, max_iou_row, max_iou_col = max_iou_match_from_confusion(confusion)
    # Prepare the new confusion with +1 added to the shape
    non_max_suppressed = np.zeros(np.add(confusion.shape, 1), dtype=np.int16)
    original_confusion = np.copy(confusion)

    keep_diagonal = iou.diagonal() >= iou_threshold
    # Move false positives from diag to new false_positive class
    move_indexes = np.where(~keep_diagonal)
    non_max_suppressed[:, -1].put(
        move_indexes, confusion.diagonal().take(move_indexes)
    )
    # Zero false positives on diagonal
    keep_indexes = np.where(keep_diagonal)
    new_diagonal = np.zeros(len(confusion.diagonal()))
    new_diagonal.put(keep_indexes, confusion.diagonal()[keep_indexes])
    np.fill_diagonal(confusion, new_diagonal)

    matches_flat_indexes = max_iou_col + max_iou_row * confusion.shape[1]
    dest_flat_indexes = max_iou_col + max_iou_row * non_max_suppressed.shape[1]

    non_max_suppressed.put(
        dest_flat_indexes, confusion.take(matches_flat_indexes)
    )
    confusion.put(matches_flat_indexes, np.zeros(len(matches_flat_indexes)))
    valid_confusion = confusion > 0
    valid_row, valid_col = np.where(confusion > 0)
    flat_idxs = valid_col + valid_row * non_max_suppressed.shape[1]
    non_max_suppressed.put(flat_idxs, confusion[valid_confusion])
    return non_max_suppressed
