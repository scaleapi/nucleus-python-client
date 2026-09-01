from unittest.mock import MagicMock

import requests

from nucleus import Model


def _model_with_client():
    client = MagicMock()
    model = Model(
        model_id="prj_123",
        name="my-model",
        reference_id="my-ref",
        metadata=None,
        client=client,
    )
    return model, client


def test_model_runs_returns_ids_and_hits_the_route():
    model, client = _model_with_client()
    client.make_request.return_value = ["run_a", "run_b"]

    result = model.model_runs()

    assert result == ["run_a", "run_b"]
    payload, route, requests_command = client.make_request.call_args.args
    assert payload == {}
    assert route == "model/prj_123/modelRun"
    assert requests_command is requests.get


def test_model_runs_include_versions_appends_family_query():
    model, client = _model_with_client()
    client.make_request.return_value = []

    model.model_runs(include_versions=True)

    _, route, _ = client.make_request.call_args.args
    assert route == "model/prj_123/modelRun?family=true"
