"""Unit tests for Evaluation V2 presets, cancel/retry, and label-schema
discovery (no live API)."""

from unittest.mock import MagicMock

import requests

from nucleus import (
    AllowedLabelMatch,
    EvaluationV2,
    EvaluationV2Preset,
    LabelExclusionRule,
    NucleusClient,
)
from nucleus.dataset import Dataset


# --------------------------------------------------------------------------- #
# Preset CRUD
# --------------------------------------------------------------------------- #
def test_list_evaluation_v2_presets():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value=[
            {
                "id": "prev_1",
                "name": "vehicles",
                "allowed_label_matches": [
                    {
                        "groundTruthLabel": "car",
                        "modelPredictionLabel": "vehicle",
                    }
                ],
                "exclusion_rules": None,
                "created_by_user_id": "u_1",
            }
        ]
    )
    presets = client.list_evaluation_v2_presets()
    client.connection.get.assert_called_once_with("evaluationV2Presets")
    assert len(presets) == 1
    assert presets[0].id == "prev_1"
    assert presets[0].name == "vehicles"
    assert presets[0].allowed_label_matches[0] == AllowedLabelMatch(
        ground_truth_label="car", model_prediction_label="vehicle"
    )


def test_create_evaluation_v2_preset_payload():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(
        return_value={
            "id": "prev_1",
            "name": "vehicles",
            "allowed_label_matches": [],
            "exclusion_rules": None,
        }
    )
    preset = client.create_evaluation_v2_preset(
        "vehicles",
        allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
        exclusion_rules=[
            LabelExclusionRule(
                scope="item", target="prediction", labels=["ignore"]
            )
        ],
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "evaluationV2Presets"
    assert payload["name"] == "vehicles"
    assert payload["allowedLabelMatches"] == [
        {"ground_truth_label": "car", "model_prediction_label": "vehicle"}
    ]
    assert payload["exclusionRules"] == [
        {
            "type": "labels",
            "scope": "item",
            "target": "prediction",
            "labels": ["ignore"],
        }
    ]
    assert preset.id == "prev_1"


def test_update_evaluation_v2_preset_name_only_omits_other_fields():
    client = NucleusClient(api_key="test")
    client.connection.patch = MagicMock(
        return_value={"id": "prev_1", "name": "renamed"}
    )
    client.update_evaluation_v2_preset("prev_1", name="renamed")
    payload, route = client.connection.patch.call_args[0]
    assert route == "evaluationV2Presets/prev_1"
    # Only the provided field is sent; matches/rules untouched.
    assert payload == {"name": "renamed"}


def test_update_evaluation_v2_preset_clear_rules_sends_null():
    client = NucleusClient(api_key="test")
    client.connection.patch = MagicMock(
        return_value={"id": "prev_1", "name": "p"}
    )
    client.update_evaluation_v2_preset("prev_1", exclusion_rules=None)
    payload = client.connection.patch.call_args[0][0]
    # Explicit None clears the rules (distinct from "leave unchanged").
    assert payload == {"exclusionRules": None}


def test_delete_evaluation_v2_preset():
    client = NucleusClient(api_key="test")
    client.connection.make_request = MagicMock(return_value=MagicMock())
    client.delete_evaluation_v2_preset("prev_1")
    # NucleusClient.make_request forwards args positionally to the connection:
    # (payload, route, requests_command, return_raw_response).
    args = client.connection.make_request.call_args[0]
    assert args[1] == "evaluationV2Presets/prev_1"
    assert args[2] is requests.delete


def test_preset_instance_update_and_delete_delegate_to_client():
    client = MagicMock(spec=NucleusClient)
    preset = EvaluationV2Preset(id="prev_1", name="p", _client=client)
    client.update_evaluation_v2_preset.return_value = EvaluationV2Preset(
        id="prev_1", name="renamed", _client=client
    )
    preset.update(name="renamed")
    assert preset.name == "renamed"
    preset.delete()
    client.delete_evaluation_v2_preset.assert_called_once_with("prev_1")


# --------------------------------------------------------------------------- #
# Cancel / retry
# --------------------------------------------------------------------------- #
def _eval(client, status="computing"):
    return EvaluationV2(
        id="evalv2_1",
        model_run_id="run_1",
        dataset_id="ds_1",
        status=status,
        _client=client,
    )


def test_evaluation_cancel_posts_and_refreshes():
    client = MagicMock(spec=NucleusClient)
    client.get.return_value = {
        "id": "evalv2_1",
        "model_run_id": "run_1",
        "dataset_id": "ds_1",
        "status": "cancelled",
    }
    ev = _eval(client)
    ev.cancel()
    args, kwargs = client.make_request.call_args
    assert args[1] == "evaluationsV2/evalv2_1/cancel"
    assert kwargs["requests_command"] is requests.post
    assert ev.status == "cancelled"


def test_evaluation_retry_resolves_new_evaluation():
    client = MagicMock(spec=NucleusClient)
    client.post.return_value = {"evaluation_id": "evalv2_retry"}
    client.get_evaluation_v2.return_value = EvaluationV2(
        id="evalv2_retry",
        model_run_id="run_1",
        dataset_id="ds_1",
        status="pending",
        _client=client,
    )
    ev = _eval(client, status="failed")
    new_eval = ev.retry()
    _, route = client.post.call_args[0]
    assert route == "evaluationsV2/evalv2_1/retry"
    assert new_eval.id == "evalv2_retry"
    client.get_evaluation_v2.assert_called_once_with("evalv2_retry")


# --------------------------------------------------------------------------- #
# Examples optional match_type
# --------------------------------------------------------------------------- #
def test_examples_match_type_optional():
    client = MagicMock(spec=NucleusClient)
    client.post.return_value = {"rows": [], "total": 0}
    ev = _eval(client, status="succeeded")

    ev.examples()
    payload = client.post.call_args[0][0]
    assert "match_type" not in payload

    ev.examples(match_type="FP")
    payload2 = client.post.call_args[0][0]
    assert payload2["match_type"] == "FP"


# --------------------------------------------------------------------------- #
# Label schema discovery
# --------------------------------------------------------------------------- #
def test_dataset_evaluation_label_schema():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={"gt_labels": ["car"], "prediction_labels": ["vehicle"]}
    )
    dataset = Dataset("ds_1", client)
    out = dataset.evaluation_label_schema()
    assert out == {"gt_labels": ["car"], "prediction_labels": ["vehicle"]}
    client.connection.get.assert_called_once_with("dataset/ds_1/labelSchema")


# --------------------------------------------------------------------------- #
# Preset rollup groups
# --------------------------------------------------------------------------- #
def test_create_evaluation_v2_preset_with_rollup_groups():
    from nucleus import RollupGroup

    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(
        return_value={
            "id": "prev_1",
            "name": "vehicles",
            "rollup_groups": [{"className": "vehicle", "labels": ["car"]}],
        }
    )
    preset = client.create_evaluation_v2_preset(
        "vehicles",
        rollup_groups=[RollupGroup("vehicle", ["car"])],
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "evaluationV2Presets"
    assert payload["rollupGroups"] == [
        {"class_name": "vehicle", "labels": ["car"]}
    ]
    assert preset.rollup_groups is not None
    assert preset.rollup_groups[0].class_name == "vehicle"


def test_create_evaluation_v2_preset_rollup_and_matches_mutually_exclusive():
    from nucleus import RollupGroup

    client = NucleusClient(api_key="test")
    try:
        client.create_evaluation_v2_preset(
            "p",
            rollup_groups=[RollupGroup("vehicle", ["car"])],
            allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "cannot both be set" in str(e)


def test_update_evaluation_v2_preset_rollup_groups_and_clear():
    from nucleus import RollupGroup

    client = NucleusClient(api_key="test")
    client.connection.patch = MagicMock(
        return_value={"id": "prev_1", "name": "p"}
    )
    client.update_evaluation_v2_preset(
        "prev_1", rollup_groups=[RollupGroup("vehicle", ["car"])]
    )
    payload = client.connection.patch.call_args[0][0]
    assert payload == {
        "rollupGroups": [{"class_name": "vehicle", "labels": ["car"]}]
    }

    client.update_evaluation_v2_preset("prev_1", rollup_groups=None)
    payload2 = client.connection.patch.call_args[0][0]
    # Explicit None clears the rollup groups (distinct from "leave unchanged").
    assert payload2 == {"rollupGroups": None}


def test_preset_from_json_parses_rollup_groups_both_casings():
    for key in ("rollup_groups", "rollupGroups"):
        preset = EvaluationV2Preset.from_json(
            {
                "id": "prev_1",
                "name": "p",
                key: [{"className": "vehicle", "labels": ["car", "truck"]}],
            }
        )
        assert preset.rollup_groups is not None
        assert preset.rollup_groups[0].labels == ["car", "truck"]
