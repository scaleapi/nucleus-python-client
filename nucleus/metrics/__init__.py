import sys

from .base import Metric, ScalarResult
from .categorization_metrics import CategorizationF1
from .cuboid_metrics import CuboidIOU, CuboidPrecision, CuboidRecall
from .filtering import (
    FieldFilter,
    ListOfOrAndFilters,
    MetadataFilter,
    apply_filters,
)
from .polygon_metrics import (
    PolygonAveragePrecision,
    PolygonIOU,
    PolygonMAP,
    PolygonMetric,
    PolygonPrecision,
    PolygonRecall,
)

try:
    import shapely
except ModuleNotFoundError as e:
    if sys.platform.startswith("darwin"):
        platform_specific_msg = (
            "Depending on Python environment used GEOS might need to be installed via "
            "`brew install geos`."
        )
    elif sys.platform.startswith("linux"):
        platform_specific_msg = (
            "Depending on Python environment used GEOS might need to be installed via "
            "system package `libgeos-dev`."
        )
    else:
        platform_specific_msg = "GEOS package will need to be installed see (https://trac.osgeo.org/geos/)"
    raise ModuleNotFoundError(
        f"Module 'shapely' not found. Install optionally with `scale-nucleus[shapely]` or when developing "
        f"`poetry install -E shapely`. {platform_specific_msg}"
    ) from e
