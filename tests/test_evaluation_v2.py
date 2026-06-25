"""Unit tests for Evaluations V2 client (no live API)."""

from unittest.mock import MagicMock

import pytest
import requests

from nucleus import (
    AllowedLabelMatch,
    BoxAreaExclusionRule,
    EvaluationV2,
    LabelExclusionRule,
    MetadataExclusionRule,
    NucleusClient,
)
from nucleus.data_transfer_object.evaluation_v2 import (
    EvaluationV2Charts,
    EvaluationV2FilterArgs,
    MetadataPredicate,
    RangeNum,
    _camelize_filter_value,
)


def _charts_response():
    return {
        "mapSummary": {"mapAt50": 0.1, "mapAt75": 0.2, "mapAt5095": 0.15},
        "perClassAp": [],
        "confusionMatrix": [],
        "scoreHistogram": [],
        "computedIouRanges": [],
        "totalCounts": {"tp": 0, "fp": 0, "fn": 0, "predsWithConfidence": 0},
        "apBySize": {"small": None, "medium": None, "large": None},
        "prCurve": [],
        "tideAttribution": {
            "truePositive": 0,
            "localization": 0,
            "classification": 0,
            "both": 0,
            "duplicate": 0,
            "background": 0,
            "missed": 0,
        },
    }


def test_evaluation_v2_filter_args_to_api_filters():
    filters = EvaluationV2FilterArgs(
        confidence_range=RangeNum(min=0.1, max=0.9),
        pred_labels=["cat"],
        item_metadata=[MetadataPredicate(key="tier", op="EQ", value="gold")],
        has_ground_truth=True,
    )
    assert filters.to_api_filters() == {
        "confidenceRange": {"min": 0.1, "max": 0.9},
        "predLabels": ["cat"],
        "itemMetadata": [{"key": "tier", "op": "EQ", "value": "gold"}],
        "hasGroundTruth": True,
    }


def test_camelize_filter_value_nested_keys():
    assert _camelize_filter_value({"bucket_min": 1.0, "bucket_max": 2.0}) == {
        "bucketMin": 1.0,
        "bucketMax": 2.0,
    }


def test_camelize_filter_value_preserves_predicate_value():
    assert _camelize_filter_value(
        {"key": "k", "op": "EQ", "value": {"keep_snake": 1}}
    ) == {"key": "k", "op": "EQ", "value": {"keep_snake": 1}}


def test_allowed_label_match_to_api_dict():
    m = AllowedLabelMatch(ground_truth_label="a", model_prediction_label="b")
    assert m.to_api_dict() == {
        "ground_truth_label": "a",
        "model_prediction_label": "b",
    }


def test_evaluation_v2_from_json_with_matches():
    client = NucleusClient(api_key="k")
    payload = {
        "id": "evalv2_1",
        "model_run_id": "run_1",
        "dataset_id": "ds_1",
        "status": "pending",
        "allowed_label_matches": [
            {"groundTruthLabel": "x", "modelPredictionLabel": "y"},
        ],
    }
    ev = EvaluationV2.from_json(payload, client)
    assert ev.id == "evalv2_1"
    assert ev.allowed_label_matches is not None
    assert len(ev.allowed_label_matches) == 1
    assert ev.allowed_label_matches[0].ground_truth_label == "x"


def test_list_evaluations_v2_empty():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=[])
    result = client.list_evaluations_v2("run_1")
    assert result == []
    client.connection.get.assert_called_once_with(
        "modelRun/run_1/evaluationsV2"
    )


def test_list_evaluations_v2_returns_rows():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value=[
            {
                "id": "evalv2_1",
                "model_run_id": "run_1",
                "dataset_id": "ds_1",
                "status": "succeeded",
            },
        ]
    )
    result = client.list_evaluations_v2("run_1")
    assert len(result) == 1
    assert result[0].id == "evalv2_1"
    assert result[0]._client is client


def test_list_evaluations_v2_invalid_response():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value={"evaluations": []})
    with pytest.raises(RuntimeError, match="Unexpected list evaluations V2"):
        client.list_evaluations_v2("run_1")


def test_create_evaluation_v2_then_get():
    client = NucleusClient(api_key="test")
    client.connection.make_request = MagicMock(
        return_value={
            "evaluation_id": "evalv2_new",
            "status": "pending",
            "workflow_id": "w",
        }
    )
    client.connection.get = MagicMock(
        return_value={
            "id": "evalv2_new",
            "model_run_id": "run_1",
            "dataset_id": "ds_1",
            "status": "pending",
        }
    )

    ev = client.create_evaluation_v2(
        "run_1",
        name="n1",
        allowed_label_matches=[
            AllowedLabelMatch("gt", "pred"),
        ],
    )
    assert ev.id == "evalv2_new"
    client.connection.make_request.assert_called_once()
    client.connection.get.assert_called_once_with("evaluationsV2/evalv2_new")


def test_create_evaluation_v2_with_slice_and_exclusion_rules():
    client = NucleusClient(api_key="test")
    client.connection.make_request = MagicMock(
        return_value={"evaluation_id": "evalv2_new", "status": "pending"}
    )
    client.connection.get = MagicMock(
        return_value={
            "id": "evalv2_new",
            "model_run_id": "run_1",
            "dataset_id": "ds_1",
            "status": "pending",
        }
    )
    client.create_evaluation_v2(
        "run_1",
        slice_id="slc_x",
        exclusion_rules=[
            BoxAreaExclusionRule(
                scope="annotation", target="groundTruth", min=1024
            ),
            LabelExclusionRule(
                scope="item", target="prediction", labels=["ignore"]
            ),
            MetadataExclusionRule(key="is_dark", op="EQ", value=True),
            {
                "type": "labels",
                "scope": "item",
                "target": "groundTruth",
                "labels": ["x"],
            },
        ],
    )
    payload = client.connection.make_request.call_args[0][0]
    assert payload["sliceId"] == "slc_x"
    assert payload["exclusionRules"] == [
        {
            "type": "boxArea",
            "scope": "annotation",
            "target": "groundTruth",
            "min": 1024,
        },
        {
            "type": "labels",
            "scope": "item",
            "target": "prediction",
            "labels": ["ignore"],
        },
        {
            "type": "metadata",
            "scope": "item",
            "key": "is_dark",
            "op": "EQ",
            "value": True,
        },
        {
            "type": "labels",
            "scope": "item",
            "target": "groundTruth",
            "labels": ["x"],
        },
    ]


def test_evaluation_v2_filter_args_gt_area_and_slices():
    filters = EvaluationV2FilterArgs(
        gt_area_range=RangeNum(min=1024, max=9216),
        slice_ids=["slc_a"],
    )
    assert filters.to_api_filters() == {
        "gtAreaRange": {"min": 1024.0, "max": 9216.0},
        "sliceIds": ["slc_a"],
    }


def test_evaluation_v2_from_json_slice_and_exclusions():
    # exclusion_rules as a JSON string (raw jsonb), exclusion_stats as a dict.
    ev = EvaluationV2.from_json(
        {
            "id": "evalv2_1",
            "model_run_id": "run_1",
            "dataset_id": "ds_1",
            "status": "succeeded",
            "slice_id": "slc_x",
            "exclusion_rules": '[{"type":"labels","scope":"item","target":"prediction","labels":["ignore"]}]',
            "exclusion_stats": {"totals": {"itemsDropped": 3}},
        }
    )
    assert ev.slice_id == "slc_x"
    assert ev.exclusion_rules == [
        {
            "type": "labels",
            "scope": "item",
            "target": "prediction",
            "labels": ["ignore"],
        }
    ]
    assert ev.exclusion_stats == {"totals": {"itemsDropped": 3}}


def test_evaluation_v2_from_json_exclusions_absent():
    ev = EvaluationV2.from_json(
        {
            "id": "evalv2_1",
            "model_run_id": "run_1",
            "dataset_id": "ds_1",
            "status": "succeeded",
        }
    )
    assert ev.slice_id is None
    assert ev.exclusion_rules is None
    assert ev.exclusion_stats is None


def test_charts_post_body():
    client = MagicMock(spec=NucleusClient)
    client.post.return_value = _charts_response()
    ev = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="succeeded",
        _client=client,
    )
    charts = ev.charts(iou_threshold=0.5)
    assert isinstance(charts, EvaluationV2Charts)
    client.post.assert_called_once()
    payload, route = client.post.call_args[0]
    assert route == "evaluationsV2/evalv2_1/charts"
    assert payload == {"iouThreshold": 0.5}


def test_charts_with_filter_args():
    client = MagicMock(spec=NucleusClient)
    client.post.return_value = _charts_response()
    ev = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="succeeded",
        _client=client,
    )
    filters = EvaluationV2FilterArgs(
        gt_area_range=RangeNum(min=1024),
        slice_ids=["slc_a", "slc_b"],
    )
    ev.charts(iou_threshold=0.75, filters=filters, query="dog")
    payload, route = client.post.call_args[0]
    assert route == "evaluationsV2/evalv2_1/charts"
    assert payload["iouThreshold"] == 0.75
    assert payload["query"] == "dog"
    assert payload["filters"] == {
        "gtAreaRange": {"min": 1024.0},
        "sliceIds": ["slc_a", "slc_b"],
    }


def test_examples_post_body():
    client = MagicMock(spec=NucleusClient)
    client.post.return_value = {"rows": [], "total": 0}
    ev = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="succeeded",
        _client=client,
    )
    page = ev.examples("TP", limit=20, offset=5)
    assert page.total == 0
    client.post.assert_called_once()
    args, kwargs = client.post.call_args
    payload, route = args
    assert route == "evaluationsV2/evalv2_1/examples"
    assert payload["match_type"] == "TP"
    assert payload["limit"] == 20
    assert payload["offset"] == 5


def test_examples_with_filter_args():
    client = MagicMock(spec=NucleusClient)
    client.post.return_value = {"rows": [], "total": 0}
    ev = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="succeeded",
        _client=client,
    )
    filters = EvaluationV2FilterArgs(
        confidence_range=RangeNum(min=0.1, max=0.9),
        pred_labels=["cat"],
        has_ground_truth=True,
    )
    ev.examples("FP", limit=10, filters=filters)
    payload = client.post.call_args[0][0]
    assert payload["filters"] == {
        "confidenceRange": {"min": 0.1, "max": 0.9},
        "predLabels": ["cat"],
        "hasGroundTruth": True,
    }


def test_wait_for_completion():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        side_effect=[
            {
                "id": "evalv2_1",
                "model_run_id": "run_1",
                "dataset_id": "ds_1",
                "status": "pending",
            },
            {
                "id": "evalv2_1",
                "model_run_id": "run_1",
                "dataset_id": "ds_1",
                "status": "succeeded",
            },
        ]
    )
    ev = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="pending",
        _client=client,
    )
    ev.wait_for_completion(timeout_sec=5, poll_interval=0.01)
    assert ev.status == "succeeded"


@pytest.mark.parametrize("status_code", [200, 204])
def test_delete_success(status_code):
    client = NucleusClient(api_key="test")
    resp = MagicMock()
    resp.status_code = status_code
    client.connection.make_request = MagicMock(return_value=resp)
    ev = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="succeeded",
        _client=client,
    )
    ev.delete()
    assert client.connection.make_request.call_count == 1
    cargs = client.connection.make_request.call_args
    assert cargs[0][0] == {}
    assert cargs[0][1] == "evaluationsV2/evalv2_1"
    assert cargs[0][2] is requests.delete
    assert cargs[0][3] is True
