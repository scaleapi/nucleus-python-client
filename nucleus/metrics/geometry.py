from typing import List, Tuple, Union

import numpy as np

TOLERANCE = 1e-8


class GeometryPolygon:
    def __init__(
        self,
        points: Union[np.ndarray, List[Tuple[float, float]]],
        is_rectangle: bool = False,
    ):
        self.points = (
            points if isinstance(points, np.ndarray) else np.array(points)
        )
        self.is_rectangle = is_rectangle
        points_x = self.points[:, 0]
        points_y = self.points[:, 1]
        if is_rectangle:
            self.signed_area = np.abs(self.points[2] - self.points[0]).prod()
            self.area = self.signed_area
        else:
            self.signed_area = (
                points_x @ np.roll(points_y, -1)
                - points_x @ np.roll(points_y, 1)
            ) / 2
            self.area = np.abs(self.signed_area)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def __repr__(self) -> str:
        return f"GeometryPolygon({self.points})"


# alpha * a1 + (1 - alpha) * a2 = beta * b1 + (1 - beta) * b2
def segment_intersection(
    segment1: Tuple[np.ndarray, np.ndarray],
    segment2: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, float, np.ndarray]:
    a1, a2 = segment1
    b1, b2 = segment2
    x2_x2 = b2[0] - a2[0]
    y2_y2 = b2[1] - a2[1]
    x1x2 = a1[0] - a2[0]
    y1y2 = a1[1] - a2[1]
    y1_y2_ = b1[1] - b2[1]
    x1_x2_ = b1[0] - b2[0]

    if np.abs(y1_y2_ * x1x2 - x1_x2_ * y1y2) < TOLERANCE:
        beta = 1.0
    else:
        beta = (x2_x2 * y1y2 - y2_y2 * x1x2) / (y1_y2_ * x1x2 - x1_x2_ * y1y2)

    if x1x2 == 0:
        alpha = (y2_y2 + y1_y2_ * beta) / (y1y2 + TOLERANCE)
    else:
        alpha = (x2_x2 + x1_x2_ * beta) / (x1x2 + TOLERANCE)

    return alpha, beta, alpha * a1 + (1 - alpha) * a2


def convex_polygon_intersection_area(
    polygon_a: GeometryPolygon, polygon_b: GeometryPolygon
) -> float:
    # pylint: disable=R0912
    sa = polygon_a.signed_area
    sb = polygon_b.signed_area
    if sa * sb < 0:
        sign = -1
    else:
        sign = 1
    na = len(polygon_a)
    nb = len(polygon_b)
    ps = []  # point set
    for i in range(na):
        a1 = polygon_a[i - 1]
        a2 = polygon_a[i]
        flag = False
        sum_s = 0
        for j in range(nb):
            b1 = polygon_b[j - 1]
            b2 = polygon_b[j]
            sum_s += np.abs(GeometryPolygon([a1, b1, b2]).signed_area)

        if np.abs(np.abs(sum_s) - np.abs(sb)) < TOLERANCE:
            flag = True

        if flag:
            ps.append(a1)
        for j in range(nb):
            b1 = polygon_b[j - 1]
            b2 = polygon_b[j]
            a, b, p = segment_intersection((a1, a2), (b1, b2))
            if 0 < a < 1 and 0 < b < 1:
                ps.append(p)

    for i in range(nb):
        a1 = polygon_b[i - 1]
        a2 = polygon_b[i]
        flag = False
        sum_s = 0
        for j in range(na):
            b1 = polygon_a[j - 1]
            b2 = polygon_a[j]
            sum_s += np.abs(GeometryPolygon([a1, b1, b2]).signed_area)
        if np.abs(np.abs(sum_s) - np.abs(sa)) < TOLERANCE:
            flag = True
        if flag:
            ps.append(a1)

    def unique(ar):
        res = []
        for i, a in enumerate(ar):
            if np.abs(a - ar[i - 1]).sum() > TOLERANCE:
                res.append(a)

        return res

    ps = sorted(ps, key=lambda x: (x[0] + TOLERANCE * x[1]))
    ps = unique(ps)

    if len(ps) == 0:
        return 0

    tmp = ps[0]

    res = []
    res.append(tmp)
    ps = sorted(
        ps[1:],
        key=lambda x: -((x - tmp) @ np.array((0, 1)) / len(x - tmp)),
    )
    res.extend(ps)

    return GeometryPolygon(res).signed_area * sign


def area(box):
    if box[2] <= box[0] or box[3] <= box[1]:
        return 0
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(box_a, box_b):
    box_c = intersection(box_a, box_b)
    return area(box_c) / (area(box_a) + area(box_b) - area(box_c))


def intersection(box_a, box_b):
    """boxes are left, top, right, bottom where left < right and top < bottom"""
    box_c = [
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    ]
    return box_c


def rectangle_intersection_area(
    polygon_a: GeometryPolygon, polygon_b: GeometryPolygon
) -> float:
    minx_a, miny_a = np.min(polygon_a.points, axis=0)
    maxx_a, maxy_a = np.max(polygon_a.points, axis=0)
    minx_b, miny_b = np.min(polygon_b.points, axis=0)
    maxx_b, maxy_b = np.max(polygon_b.points, axis=0)

    minx_c = max(minx_a, minx_b)
    miny_c = max(miny_a, miny_b)
    maxx_c = min(maxx_a, maxx_b)
    maxy_c = min(maxy_a, maxy_b)
    return max(maxx_c - minx_c, 0) * max(maxy_c - miny_c, 0)


def polygon_intersection_area(
    polygon_a: GeometryPolygon, polygon_b: GeometryPolygon
) -> float:
    if polygon_a.is_rectangle and polygon_b.is_rectangle:
        return rectangle_intersection_area(polygon_a, polygon_b)

    na = len(polygon_a)
    nb = len(polygon_b)
    res = 0.0
    for i in range(1, na - 1):
        sa = polygon_a[[0, i, i + 1]]
        for j in range(1, nb - 1):
            sb = polygon_b[[0, j, j + 1]]
            tmp = convex_polygon_intersection_area(
                GeometryPolygon(sa), GeometryPolygon(sb)
            )
            res += tmp

    return np.abs(res)
