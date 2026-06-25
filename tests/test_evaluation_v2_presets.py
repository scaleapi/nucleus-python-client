"""Unit tests for Evaluation V2 presets, batch create, cancel/retry, and
label-schema discovery (no live API)."""

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
                    {"groundTruthLabel": "car", "modelPredictionLabel": "vehicle"}
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
# Apply preset + only_items_with_predictions on create
# --------------------------------------------------------------------------- #
def _stub_create(client):
    client.connection.make_request = MagicMock(
        return_value={"evaluation_id": "evalv2_new"}
    )
    client.connection.get = MagicMock(
        return_value={
            "id": "evalv2_new",
            "model_run_id": "run_1",
            "dataset_id": "ds_1",
            "status": "pending",
        }
    )


def test_create_evaluation_v2_with_preset_seeds_config():
    client = NucleusClient(api_key="test")
    _stub_create(client)
    preset = EvaluationV2Preset(
        id="prev_1",
        name="p",
        allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
        exclusion_rules=[
            {
                "type": "labels",
                "scope": "item",
                "target": "groundTruth",
                "labels": ["x"],
            }
        ],
    )
    client.create_evaluation_v2("run_1", preset=preset)
    payload = client.connection.make_request.call_args[0][0]
    assert payload["allowed_label_matches"] == [
        {"ground_truth_label": "car", "model_prediction_label": "vehicle"}
    ]
    assert payload["exclusionRules"] == [
        {
            "type": "labels",
            "scope": "item",
            "target": "groundTruth",
            "labels": ["x"],
        }
    ]


def test_create_evaluation_v2_explicit_args_override_preset():
    client = NucleusClient(api_key="test")
    _stub_create(client)
    preset = EvaluationV2Preset(
        id="prev_1",
        name="p",
        allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
    )
    client.create_evaluation_v2(
        "run_1",
        preset=preset,
        allowed_label_matches=[AllowedLabelMatch("dog", "animal")],
    )
    payload = client.connection.make_request.call_args[0][0]
    assert payload["allowed_label_matches"] == [
        {"ground_truth_label": "dog", "model_prediction_label": "animal"}
    ]


def test_create_evaluation_v2_only_items_with_predictions():
    client = NucleusClient(api_key="test")
    _stub_create(client)
    client.create_evaluation_v2("run_1", only_items_with_predictions=True)
    payload = client.connection.make_request.call_args[0][0]
    assert payload["onlyItemsWithPredictions"] is True


# --------------------------------------------------------------------------- #
# Batch create
# --------------------------------------------------------------------------- #
def test_create_evaluations_v2_batch_cross_product_and_error_capture():
    client = NucleusClient(api_key="test")
    seen = []

    def fake_create(run, **kwargs):
        seen.append((run, kwargs.get("slice_id"), kwargs.get("name")))
        if run == "run_bad":
            raise RuntimeError("boom")
        ev = MagicMock(spec=EvaluationV2)
        ev.id = f"eval_{run}_{kwargs.get('slice_id')}"
        return ev

    client.create_evaluation_v2 = fake_create
    results = client.create_evaluations_v2_batch(
        ["run_ok", "run_bad"],
        slice_ids=["slc_1", None],
        name_prefix="nightly",
    )

    # 2 runs x 2 targets = 4 jobs, returned in input order.
    assert len(results) == 4
    assert results[0].model_run_id == "run_ok"
    assert results[0].slice_id == "slc_1"
    assert results[0].name == "nightly — run_ok — slc_1"
    assert results[0].succeeded
    assert results[1].name == "nightly — run_ok"  # whole-dataset job
    # Failures are captured per-job, not raised.
    assert results[2].model_run_id == "run_bad"
    assert not results[2].succeeded
    assert results[2].error == "boom"


def test_create_evaluations_v2_batch_defaults_to_whole_dataset():
    client = NucleusClient(api_key="test")
    client.create_evaluation_v2 = MagicMock(
        return_value=MagicMock(spec=EvaluationV2)
    )
    results = client.create_evaluations_v2_batch(["run_1", "run_2"])
    assert len(results) == 2
    # No slice_ids -> one whole-dataset job per run.
    for call in client.create_evaluation_v2.call_args_list:
        assert call.kwargs["slice_id"] is None


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
    client.connection.make_request = MagicMock(
        return_value={"gt_labels": ["car"], "prediction_labels": ["vehicle"]}
    )
    dataset = Dataset("ds_1", client)
    out = dataset.evaluation_label_schema()
    assert out == {"gt_labels": ["car"], "prediction_labels": ["vehicle"]}
    args = client.connection.make_request.call_args[0]
    assert args[1] == "dataset/ds_1/labelSchema"
    assert args[2] is requests.get
