import pytest

from nucleus.test_launch_integration import verify_box_output


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

    for bad_bbox in bad_bboxes:
        try:
            verify_box_output(bad_bbox)
        except ValueError:
            pass
        else:
            pytest.fail("Expected verify_box_output to throw ValueError")
