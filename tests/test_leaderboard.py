"""Unit tests for benchmark leaderboard and filter-schema client methods
(no live API).

The backing REST endpoints (``POST /nucleus/leaderboard/ranking``,
``POST /nucleus/leaderboard/f1Curve``,
``GET /nucleus/evaluationsV2/:id/filterSchema``) ship with the scaleapi
server; these tests only cover request/response wiring.
"""

from unittest.mock import MagicMock

import pytest

from nucleus import EvaluationV2, NucleusClient
from nucleus.data_transfer_object.evaluation_v2 import (
    EvaluationV2FilterSchema,
    LeaderboardF1CurveEntry,
    LeaderboardRankingEntry,
)

_RANKING_ROW = {
    "evaluation_id": "evalv2_1",
    "evaluation_name": "eval",
    "model_run_id": "run_1",
    "model_run_name": "run",
    "model_id": "prj_1",
    "model_name": "model",
    "model_version_major": 1,
    "model_version_minor": 2,
    "model_version_label": "1.2",
    "parent_model_project_id": None,
    "score": 0.42,
    "rank": 1,
}

_F1_ROW = {
    "evaluation_id": "evalv2_1",
    "model_run_id": "run_1",
    "model_name": "model",
    "best_f1": 0.8,
    "points": [
        {"confidence_threshold": 0.25, "score": 0.7},
        {"confidence_threshold": 0.5, "score": 0.8},
    ],
    "rank": 1,
}

_FILTER_SCHEMA = {
    "gt_labels": ["car", "truck"],
    "pred_labels": ["car"],
    "metadata_fields": [
        {"key": "weather", "value_type": "string", "values": ["rain", "sun"]}
    ],
}


def test_leaderboard_ranking_payload_and_parsing():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(return_value=[dict(_RANKING_ROW)])
    rows = client.leaderboard_ranking(
        "MAP_50",
        ["bm_1", "bm_2"],
        confidence_threshold=0.5,
        model_ids=["prj_1"],
        scope="mine",
        collapse="allRuns",
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "leaderboard/ranking"
    assert payload == {
        "metric_type": "MAP_50",
        "benchmark_ids": ["bm_1", "bm_2"],
        "confidence_threshold": 0.5,
        "model_ids": ["prj_1"],
        "scope": "mine",
        "collapse": "allRuns",
    }
    assert rows == [LeaderboardRankingEntry.parse_obj(_RANKING_ROW)]
    assert rows[0].score == 0.42
    assert rows[0].rank == 1


def test_leaderboard_ranking_parses_run_free_row():
    # Run-free evaluations have no model run: model_run_id / model_run_name
    # arrive absent (or null) and must parse to None, not raise.
    client = NucleusClient(api_key="test")
    run_free_row = {
        "evaluation_id": "evalv2_1",
        "model_id": "prj_1",
        "model_name": "model",
        "score": 0.5,
        "rank": 1,
    }
    client.connection.post = MagicMock(return_value=[run_free_row])
    rows = client.leaderboard_ranking("MAP_50", ["bm_1"])
    assert rows[0].model_run_id is None
    assert rows[0].model_run_name is None
    assert rows[0].model_id == "prj_1"


def test_leaderboard_ranking_minimal_payload():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(return_value=[])
    client.leaderboard_ranking("F1", ["bm_1"])
    payload = client.connection.post.call_args[0][0]
    assert payload == {"metric_type": "F1", "benchmark_ids": ["bm_1"]}


def test_leaderboard_ranking_invalid_response():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(return_value={"error": "nope"})
    with pytest.raises(RuntimeError, match="Unexpected leaderboard ranking"):
        client.leaderboard_ranking("MAP_50", ["bm_1"])


def test_leaderboard_f1_curve_payload_and_parsing():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(return_value=[dict(_F1_ROW)])
    rows = client.leaderboard_f1_curve(["bm_1"], top_n=3)
    payload, route = client.connection.post.call_args[0]
    assert route == "leaderboard/f1Curve"
    assert payload == {"benchmark_ids": ["bm_1"], "top_n": 3}
    assert rows == [LeaderboardF1CurveEntry.parse_obj(_F1_ROW)]
    assert rows[0].best_f1 == 0.8
    assert rows[0].points[1].confidence_threshold == 0.5


def test_get_evaluation_v2_filter_schema():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=dict(_FILTER_SCHEMA))
    schema = client.get_evaluation_v2_filter_schema("evalv2_1")
    client.connection.get.assert_called_once_with(
        "evaluationsV2/evalv2_1/filterSchema"
    )
    assert schema == EvaluationV2FilterSchema.parse_obj(_FILTER_SCHEMA)
    assert schema.gt_labels == ["car", "truck"]
    assert schema.metadata_fields[0].key == "weather"


def test_evaluation_filter_schema_instance_method():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=dict(_FILTER_SCHEMA))
    evaluation = EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        status="succeeded",
        _client=client,
    )
    schema = evaluation.filter_schema()
    client.connection.get.assert_called_once_with(
        "evaluationsV2/evalv2_1/filterSchema"
    )
    assert schema.pred_labels == ["car"]
