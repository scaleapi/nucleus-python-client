from typing import List, Tuple, Union

import numpy as np

TOLERANCE = 1e-8


class GeometryPoint:
    def __init__(self, xy: Union[Tuple[float, float], np.ndarray]):
        self.xy = np.array(xy)
        self.x = xy[0]
        self.y = xy[1]

    def __repr__(self) -> str:
        return f"GeometryPoint(xy=({self.xy[0]}, {self.xy[1]})"

    def __add__(self, p: "GeometryPoint") -> "GeometryPoint":
        return GeometryPoint(self.xy + p.xy)

    def __sub__(self, p: "GeometryPoint") -> "GeometryPoint":
        return GeometryPoint(self.xy - p.xy)

    def __rmul__(self, scalar: float) -> "GeometryPoint":
        return GeometryPoint(self.xy * scalar)

    def __mul__(self, scalar: float) -> "GeometryPoint":
        return GeometryPoint(self.xy * scalar)

    # Operator @
    def __matmul__(self, p: "GeometryPoint") -> float:
        return self.xy @ p.xy

    def length(self) -> float:
        return np.linalg.norm(self.xy)

    def cmp(self, p: "GeometryPoint") -> float:
        return np.abs(self.xy - p.xy).sum()


class GeometryPolygon:
    def __init__(self, points: List[GeometryPoint]):
        self.points = points

        points_x = np.array([point.x for point in points])
        points_y = np.array([point.y for point in points])
        self.signed_area = (
            points_x @ np.roll(points_y, 1) - points_x @ np.roll(points_y, -1)
        ) / 2
        self.area = np.abs(self.signed_area)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def __repr__(self) -> str:
        points = ", ".join([str(point) for point in self.points])
        return f"GeometryPolygon({points})"


Segment = Tuple[GeometryPoint, GeometryPoint]


# alpha * a1 + (1 - alpha) * a2 = beta * b1 + (1 - beta) * b2
def segment_intersection(
    segment1: Segment, segment2: Segment
) -> Tuple[float, float, GeometryPoint]:
    a1, a2 = segment1
    b1, b2 = segment2
    x2_x2 = b2.x - a2.x
    y2_y2 = b2.y - a2.y
    x1x2 = a1.x - a2.x
    y1y2 = a1.y - a2.y
    y1_y2_ = b1.y - b2.y
    x1_x2_ = b1.x - b2.x

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
        for i, _ in enumerate(ar):
            if _.cmp(ar[i - 1]) > TOLERANCE:
                res.append(_)

        return res

    ps = sorted(ps, key=lambda x: (x.x + TOLERANCE * x.y))
    ps = unique(ps)

    if len(ps) == 0:
        return 0

    tmp = ps[0]

    res = []
    res.append(tmp)
    ps = sorted(
        ps[1:],
        key=lambda x: -(
            (x - tmp) @ GeometryPoint((0, 1)) / (x - tmp).length()
        ),
    )
    res.extend(ps)

    return GeometryPolygon(res).signed_area * sign


def polygon_intersection_area(
    polygon_a: GeometryPolygon, polygon_b: GeometryPolygon
) -> float:
    na = len(polygon_a)
    nb = len(polygon_b)
    res = 0.0
    for i in range(1, na - 1):
        sa = [polygon_a[0], polygon_a[i], polygon_a[i + 1]]
        for j in range(1, nb - 1):
            sb = [polygon_b[0], polygon_b[j], polygon_b[j + 1]]
            tmp = convex_polygon_intersection_area(
                GeometryPolygon(sa), GeometryPolygon(sb)
            )
            res += tmp

    return np.abs(res)
