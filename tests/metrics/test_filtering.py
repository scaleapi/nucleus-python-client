import json

import pytest

from nucleus.metrics import FieldFilter, MetadataFilter, apply_filters
from tests.metrics.helpers import (
    TEST_BOX_ANNOTATION_LIST,
    TEST_BOX_PREDICTION_LIST,
)


@pytest.fixture(
    params=[
        TEST_BOX_ANNOTATION_LIST.box_annotations,
        TEST_BOX_PREDICTION_LIST.box_predictions,
    ]
)
def annotations_or_predictions(request):
    yield request.param


def test_filter_field(annotations_or_predictions):
    dnf_filters = [
        FieldFilter("label", "==", annotations_or_predictions[0].label)
    ]
    filtered = apply_filters(annotations_or_predictions, dnf_filters)
    assert filtered == [annotations_or_predictions[0]]


def test_filter_metadata_field(annotations_or_predictions):
    dnf_filters = [
        MetadataFilter(
            "index", "==", annotations_or_predictions[0].metadata["index"]
        )
    ]
    filtered = apply_filters(annotations_or_predictions, dnf_filters)
    assert filtered == [annotations_or_predictions[0]]


def test_only_and(annotations_or_predictions):
    and_filters = [
        MetadataFilter(
            "index", "==", annotations_or_predictions[0].metadata["index"]
        )
    ]
    or_filters = [and_filters]
    filtered_and = apply_filters(annotations_or_predictions, and_filters)
    filtered_or = apply_filters(annotations_or_predictions, or_filters)
    assert filtered_and == filtered_or


def test_json_encoded_filters(annotations_or_predictions):
    filters = [
        [MetadataFilter("index", ">", 0), MetadataFilter("index", "<", 4)]
    ]
    expected = apply_filters(annotations_or_predictions, filters)
    json_string = json.dumps(filters)
    filters_from_json = json.loads(json_string)
    json_filtered = apply_filters(
        annotations_or_predictions, filters_from_json
    )
    assert json_filtered == expected


def test_json_encoded_and_filters(annotations_or_predictions):
    filters = [
        MetadataFilter("index", ">", 0),
        MetadataFilter("index", "<", 4),
    ]
    expected = apply_filters(annotations_or_predictions, filters)
    json_string = json.dumps(filters)
    filters_from_json = json.loads(json_string)
    json_filtered = apply_filters(
        annotations_or_predictions, filters_from_json
    )
    assert json_filtered == expected


def test_or_branches(annotations_or_predictions):
    index_0_or_2 = [
        [MetadataFilter("index", "==", 0)],
        [MetadataFilter("index", "==", 2)],
    ]
    filtered = apply_filters(annotations_or_predictions, index_0_or_2)
    assert filtered == [
        annotations_or_predictions[0],
        annotations_or_predictions[2],
    ]


def test_only_one_or(annotations_or_predictions):
    later_matches = [
        [MetadataFilter("index", "==", -1)],
        [MetadataFilter("index", "==", 2)],
    ]
    filtered = apply_filters(annotations_or_predictions, later_matches)
    assert filtered == [
        annotations_or_predictions[2],
    ]


def test_and_branches(annotations_or_predictions):
    index_0_or_2 = [
        [
            MetadataFilter("index", "==", 0),
            FieldFilter("label", "==", annotations_or_predictions[0].label),
        ],
        [
            MetadataFilter("index", "==", 2),
            FieldFilter("label", "==", annotations_or_predictions[2].label),
        ],
    ]
    filtered = apply_filters(annotations_or_predictions, index_0_or_2)
    assert filtered == [
        annotations_or_predictions[0],
        annotations_or_predictions[2],
    ]


def test_multi_or(annotations_or_predictions):
    all_match = [
        [MetadataFilter("index", "==", i)]
        for i in range(len(annotations_or_predictions))
    ]
    filtered = apply_filters(annotations_or_predictions, all_match)
    assert filtered == annotations_or_predictions


def test_missing_field_raises(annotations_or_predictions):
    missing_field = [[FieldFilter("i_dont_exist", "==", 1)]]
    with pytest.raises(AttributeError):
        apply_filters(annotations_or_predictions, missing_field)


def test_allow_missing_field(annotations_or_predictions):
    missing_field = [
        [FieldFilter("i_dont_exist", "==", 1, allow_missing=True)]
    ]
    filtered = apply_filters(annotations_or_predictions, missing_field)
    assert filtered == []


def test_missing_metadata_raises(annotations_or_predictions):
    missing_field = [[MetadataFilter("i_dont_exist", "==", 1)]]
    with pytest.raises(KeyError):
        apply_filters(annotations_or_predictions, missing_field)


def test_allow_missing_metadata_field(annotations_or_predictions):
    missing_field = [
        [FieldFilter("i_dont_exist", "==", 1, allow_missing=True)]
    ]
    filtered = apply_filters(annotations_or_predictions, missing_field)
    assert filtered == []


def test_gt_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", ">", 0)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[1:]


def test_gt_field(annotations_or_predictions):
    valid_gt = [FieldFilter("x", ">", annotations_or_predictions[0].x)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[1:]


def test_gte_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", ">=", 1)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[1:]


def test_gte_field(annotations_or_predictions):
    valid_gt = [FieldFilter("x", ">=", annotations_or_predictions[1].x)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[1:]


def test_lt_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", "<", 1)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[:1]


def test_lt_field(annotations_or_predictions):
    valid_gt = [FieldFilter("x", "<", annotations_or_predictions[1].x)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[:1]


def test_lte_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", "<=", 1)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[:2]


def test_lte_field(annotations_or_predictions):
    valid_gt = [FieldFilter("x", "<=", annotations_or_predictions[1].x)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[:2]


def test_eqeq_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", "==", 0)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == [annotations_or_predictions[0]]


def test_eqeq_field(annotations_or_predictions):
    valid_gt = [FieldFilter("x", "==", annotations_or_predictions[0].x)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == [annotations_or_predictions[0]]


def test_eq_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", "=", 0)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == [annotations_or_predictions[0]]


def test_eq_field(annotations_or_predictions):
    valid_gt = [FieldFilter("x", "=", annotations_or_predictions[0].x)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == [annotations_or_predictions[0]]


def test_neq_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", "!=", 0)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[1:]


def test_neq_field(annotations_or_predictions):
    valid_gt = [FieldFilter("x", "!=", annotations_or_predictions[0].x)]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[1:]


def test_in_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", "in", [0, 2])]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == [
        annotations_or_predictions[0],
        annotations_or_predictions[2],
    ]


def test_in_field(annotations_or_predictions):
    valid_gt = [
        FieldFilter(
            "x",
            "in",
            [annotations_or_predictions[0].x, annotations_or_predictions[2].x],
        )
    ]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == [
        annotations_or_predictions[0],
        annotations_or_predictions[2],
    ]


def test_not_in_metadata(annotations_or_predictions):
    valid_gt = [MetadataFilter("index", "not in", [0, 1])]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[2:]


def test_not_in_field(annotations_or_predictions):
    valid_gt = [
        FieldFilter(
            "x",
            "not in",
            [annotations_or_predictions[0].x, annotations_or_predictions[1].x],
        )
    ]
    filtered = apply_filters(annotations_or_predictions, valid_gt)
    assert filtered == annotations_or_predictions[2:]
