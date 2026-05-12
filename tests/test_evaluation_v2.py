"""Unit tests for Evaluations V2 client (no live API)."""

from unittest.mock import MagicMock

import requests

from nucleus import AllowedLabelMatch, EvaluationV2, NucleusClient
from nucleus.data_transfer_object.evaluation_v2 import EvaluationV2Charts


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


def test_charts_get_query_string():
    client = MagicMock(spec=NucleusClient)
    client.get.return_value = {
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
    ev = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="succeeded",
        _client=client,
    )
    charts = ev.charts(iou_threshold=0.5)
    assert isinstance(charts, EvaluationV2Charts)
    call_route = client.get.call_args[0][0]
    assert "evaluationsV2/evalv2_1/charts" in call_route
    assert "iouThreshold=0.5" in call_route


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


def test_delete_204():
    client = NucleusClient(api_key="test")
    resp = MagicMock()
    resp.status_code = 204
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
