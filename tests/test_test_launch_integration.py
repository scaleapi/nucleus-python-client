from typing import Any, Callable, List

import pytest

from nucleus.test_launch_integration import (
    verify_box_output,
    verify_category_output,
    verify_line_output,
    verify_polygon_output,
)

_GOOD_POINT_ARRAY = [
    {"x": 1, "y": 2},
    {"x": 3, "y": 4},
    {"x": 5, "y": 6},
]

_BAD_POINT_ARRAY_KEYS = [
    {"x": 1, "z": 2},
    {"x": 3, "y": 4},
    {"x": 5, "y": 6},
]

_BAD_POINT_ARRAY_TYPES = [
    {"x": 1, "y": "hi"},
    {"x": 3, "y": "here"},
    {"x": 5, "y": "there"},
]


def _assert_all_valueerror(fn: Callable[[Any], Any], items: List[Any]):
    for item in items:
        try:
            fn(item)
        except ValueError:
            pass
        else:
            print(item)
            pytest.fail("Expected function to throw ValueError")


def test_verify_box_output_good():
    good_bboxes = [
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "car",
            "confidence": 0.99,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "truck",
            "confidence": 0.98,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "person",
            "confidence": 0.97,
        },
    ]
    verify_box_output(good_bboxes)


def test_verify_box_output_bad():
    # missing type
    bad_bboxes_1 = [
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "label": "car",
            "confidence": 0.99,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "truck",
            "confidence": 0.98,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "person",
            "confidence": 0.97,
        },
    ]
    # extra top level key
    bad_bboxes_2 = [
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "truck",
            "confidence": 0.98,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "person",
            "confidence": 0.97,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "car",
            "confidence": 0.99,
            "extra": "extra",
        },
    ]
    # missing item in geometry
    bad_bboxes_3 = [
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "truck",
            "confidence": 0.98,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1},
            "type": "box",
            "label": "car",
            "confidence": 0.99,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "person",
            "confidence": 0.97,
        },
    ]
    # extra item in geometry
    bad_bboxes_4 = [
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "truck",
            "confidence": 0.98,
        },
        {
            "geometry": {
                "x": 1,
                "y": 1,
                "width": 1,
                "height": 1,
                "extra": "extra",
            },
            "type": "box",
            "label": "car",
            "confidence": 0.99,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "person",
            "confidence": 0.97,
        },
    ]
    # incorrect type
    bad_bboxes_5 = [
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "truck",
            "confidence": 0.98,
        },
        {
            "geometry": {
                "x": 1,
                "y": 1,
                "width": 1,
                "height": 1,
            },
            "type": "polygon",
            "label": "car",
            "confidence": 0.99,
        },
        {
            "geometry": {"x": 1, "y": 1, "width": 1, "height": 1},
            "type": "box",
            "label": "person",
            "confidence": 0.97,
        },
    ]

    bad_bboxes = [
        bad_bboxes_1,
        bad_bboxes_2,
        bad_bboxes_3,
        bad_bboxes_4,
        bad_bboxes_5,
    ]

    _assert_all_valueerror(verify_box_output, bad_bboxes)


def test_verify_category_output_good():
    good_category = [
        {
            "geometry": {},
            "type": "category",
            "label": "person",
            "confidence": 0.42,
        }
    ]
    verify_category_output(good_category)


def test_verify_category_output_bad():
    # missing category
    bad_category_1 = [
        {
            "type": "category",
            "label": "person",
            "confidence": 0.42,
        }
    ]
    # extra category
    bad_category_2 = [
        {
            "geometry": {},
            "type": "category",
            "label": "person",
            "confidence": 0.42,
            "extra": "extra",
        }
    ]
    # wrong type of a field
    bad_category_3 = [
        {
            "geometry": {},
            "type": "category",
            "label": "person",
            "confidence": "extremely",  # Note: passing in "0.42" will not cause pydantic to throw an error
        }
    ]
    # incorrect geometry
    bad_category_4 = [
        {
            "geometry": {"x": 3.1, "y": 4.2, "width": 5.3, "height": 6.4},
            "type": "category",
            "label": "person",
            "confidence": 0.42,
            "extra": "extra",
        }
    ]
    # wrong type
    bad_category_5 = [
        {
            "geometry": {},
            "type": "box",
            "label": "person",
            "confidence": 0.42,
            "extra": "extra",
        }
    ]
    bad_categories = [
        bad_category_1,
        bad_category_2,
        bad_category_3,
        bad_category_4,
        bad_category_5,
    ]

    _assert_all_valueerror(verify_category_output, bad_categories)


def test_verify_line_output_good():
    good_line = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "line",
            "label": "label1",
            "confidence": 0.99,
        }
    ]

    verify_line_output(good_line)


def test_verify_line_output_bad():
    bad_line_1 = [
        {
            "geometry": {"vertices": _BAD_POINT_ARRAY_TYPES},
            "type": "line",
            "label": "label1",
            "confidence": 0.99,
        }
    ]
    bad_line_2 = [
        {
            "geometry": {"vertices": _BAD_POINT_ARRAY_KEYS},
            "type": "line",
            "label": "label1",
            "confidence": 0.99,
        }
    ]
    bad_line_3 = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "notline",
            "label": "label1",
            "confidence": 0.99,
        }
    ]
    bad_line_4 = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "line",
            "label": "label1",
            "confidence": "extremely",
        }
    ]
    bad_line_5 = [
        {"geometry": {}, "type": "line", "label": "label1", "confidence": 0.99}
    ]
    bad_line_6 = [{"type": "line", "label": "label1", "confidence": 0.99}]
    bad_line_7 = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "line",
            "label": "label1",
            "confidence": 0.99,
            "extra": "extra",
        }
    ]

    bad_lines = [
        bad_line_1,
        bad_line_2,
        bad_line_3,
        bad_line_4,
        bad_line_5,
        bad_line_6,
        bad_line_7,
    ]
    _assert_all_valueerror(verify_line_output, bad_lines)


def test_verify_polygon_output_good():
    good_polygon = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "polygon",
            "label": "label1",
            "confidence": 0.99,
        }
    ]

    verify_polygon_output(good_polygon)


def test_verify_polygon_output_bad():
    bad_polygon_1 = [
        {
            "geometry": {"vertices": _BAD_POINT_ARRAY_TYPES},
            "type": "polygon",
            "label": "label1",
            "confidence": 0.99,
        }
    ]
    bad_polygon_2 = [
        {
            "geometry": {"vertices": _BAD_POINT_ARRAY_KEYS},
            "type": "polygon",
            "label": "label1",
            "confidence": 0.99,
        }
    ]
    bad_polygon_3 = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "notpolygon",
            "label": "label1",
            "confidence": 0.99,
        }
    ]
    bad_polygon_4 = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "polygon",
            "label": "label1",
            "confidence": "extremely",
        }
    ]
    bad_polygon_5 = [
        {
            "geometry": {},
            "type": "polygon",
            "label": "label1",
            "confidence": 0.99,
        }
    ]
    bad_polygon_6 = [
        {"type": "polygon", "label": "label1", "confidence": 0.99}
    ]
    bad_polygon_7 = [
        {
            "geometry": {"vertices": _GOOD_POINT_ARRAY},
            "type": "polygon",
            "label": "label1",
            "confidence": 0.99,
            "extra": "extra",
        }
    ]

    bad_polygons = [
        bad_polygon_1,
        bad_polygon_2,
        bad_polygon_3,
        bad_polygon_4,
        bad_polygon_5,
        bad_polygon_6,
        bad_polygon_7,
    ]
    _assert_all_valueerror(verify_polygon_output, bad_polygons)
