"""Unit tests for dataset-less model runs (no live API).

A model run can now be created without naming a dataset up front. Its dataset
set starts empty and grows as predictions arrive, because each prediction
carries its own target item (``dataset_item_id``, or ``dataset_id`` +
``reference_id``). Uploads go to ``modelRun/{model_run_id}/uploadPredictions``,
where the server resolves each item, groups by dataset, and widens the run.

These tests pin the routing, the create call, the ``add_predictions`` upload,
and the per-prediction target ids emitted in ``to_payload`` — all with mocks,
so they run offline.
"""

from unittest.mock import MagicMock

import pytest

from nucleus import NucleusClient
from nucleus.annotation_uploader import PredictionUploader
from nucleus.errors import DuplicateIDError
from nucleus.model import Model
from nucleus.model_run import ModelRun
from nucleus.prediction import BoxPrediction, CategoryPrediction


def _client():
    return NucleusClient(api_key="test")


def _model(client=None):
    return Model(
        model_id="prj_1",
        name="My Model",
        reference_id="My-CNN",
        metadata={},
        client=client or _client(),
    )


def _predictions():
    return [
        BoxPrediction(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="item_1",
            confidence=0.9,
        )
    ]


# --------------------------------------------------------------------------- #
# PredictionUploader routing
# --------------------------------------------------------------------------- #
def test_model_run_id_alone_routes_to_the_dataset_less_endpoint():
    uploader = PredictionUploader(client=_client(), model_run_id="run_1")
    assert (
        uploader._route == "modelRun/run_1/uploadPredictions"
    )  # noqa: SLF001


def test_dataset_and_model_run_ids_still_route_to_the_widening_endpoint():
    uploader = PredictionUploader(
        client=_client(), dataset_id="ds_1", model_run_id="run_1"
    )
    assert (
        uploader._route
        == "dataset/ds_1/modelRun/run_1/uploadPredictions"  # noqa: SLF001
    )


def test_dataset_and_model_ids_still_route_to_the_model_endpoint():
    uploader = PredictionUploader(
        client=_client(), dataset_id="ds_1", model_id="prj_1"
    )
    assert (
        uploader._route
        == "dataset/ds_1/model/prj_1/uploadPredictions"  # noqa: SLF001
    )


def test_model_id_and_model_run_id_together_are_rejected():
    with pytest.raises(ValueError, match="not both"):
        PredictionUploader(
            client=_client(),
            dataset_id="ds_1",
            model_id="prj_1",
            model_run_id="run_1",
        )


def test_neither_model_nor_model_run_is_rejected():
    with pytest.raises(ValueError, match="required"):
        PredictionUploader(client=_client(), dataset_id="ds_1")


def test_model_id_without_dataset_id_is_rejected():
    with pytest.raises(ValueError, match="dataset_id is required"):
        PredictionUploader(client=_client(), model_id="prj_1")


# --------------------------------------------------------------------------- #
# Model.create_run (dataset-less)
# --------------------------------------------------------------------------- #
def test_create_run_posts_to_the_create_route():
    client = _client()
    client.make_request = MagicMock(return_value={"model_run_id": "run_1"})
    model = _model(client)

    run = model.create_run(
        name="my run", metadata={"k": "v"}, reference_id="ref_1"
    )

    payload = client.make_request.call_args[0][0]
    route = client.make_request.call_args[1]["route"]
    assert route == "model/prj_1/modelRun/create"
    assert payload == {
        "name": "my run",
        "reference_id": "ref_1",
        "metadata": {"k": "v"},
    }
    assert isinstance(run, ModelRun)
    assert run.model_run_id == "run_1"
    assert run.dataset_id is None


def test_create_run_defaults_metadata_to_empty_dict():
    client = _client()
    client.make_request = MagicMock(return_value={"model_run_id": "run_1"})
    model = _model(client)

    model.create_run(name="my run")

    payload = client.make_request.call_args[0][0]
    assert payload == {
        "name": "my run",
        "reference_id": None,
        "metadata": {},
    }


def test_create_run_with_predictions_and_no_dataset_uploads_them():
    client = _client()
    client.make_request = MagicMock(return_value={"model_run_id": "run_1"})
    model = _model(client)
    predictions = _predictions()

    with pytest.MonkeyPatch.context() as mp:
        captured = {}
        mp.setattr(
            ModelRun,
            "add_predictions",
            lambda self, preds, **kw: captured.update(
                run_id=self.model_run_id, preds=preds
            ),
        )
        run = model.create_run(name="my run", predictions=predictions)

    # The run is created dataset-less, then the predictions are attached to it.
    assert client.make_request.call_args[1]["route"] == (
        "model/prj_1/modelRun/create"
    )
    assert run.model_run_id == "run_1"
    assert run.dataset_id is None
    assert captured == {"run_id": "run_1", "preds": predictions}


# --------------------------------------------------------------------------- #
# ModelRun.add_predictions
# --------------------------------------------------------------------------- #
def test_add_predictions_uses_the_dataset_less_route_and_forwards_update():
    run = ModelRun(model_run_id="run_1", client=_client())
    captured = {}

    with pytest.MonkeyPatch.context() as mp:
        routes = []
        original_init = PredictionUploader.__init__

        def _spy_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            routes.append(self._route)  # noqa: SLF001

        mp.setattr(PredictionUploader, "__init__", _spy_init)
        mp.setattr(
            PredictionUploader,
            "upload",
            lambda self, **kw: captured.update(kw) or {},
        )
        run.add_predictions(_predictions(), update=True)

    assert routes == ["modelRun/run_1/uploadPredictions"]
    assert captured["update"] is True


def test_add_predictions_runs_the_duplicate_id_check():
    run = ModelRun(model_run_id="run_1", client=_client())
    duplicate = _predictions() * 2
    for pred in duplicate:
        pred.annotation_id = "ann_1"

    with pytest.raises(DuplicateIDError):
        run.add_predictions(duplicate)


def test_add_predictions_async_is_not_supported():
    run = ModelRun(model_run_id="run_1", client=_client())
    with pytest.raises(NotImplementedError):
        run.add_predictions(_predictions(), asynchronous=True)


# --------------------------------------------------------------------------- #
# Per-prediction target ids in to_payload
# --------------------------------------------------------------------------- #
def test_box_prediction_emits_item_id_from_dataset_item_id():
    pred = BoxPrediction(
        label="car",
        x=0,
        y=0,
        width=10,
        height=10,
        reference_id="item_1",
        dataset_item_id="di_1",
    )
    payload = pred.to_payload()
    assert payload["item_id"] == "di_1"
    assert "dataset_id" not in payload


def test_box_prediction_emits_dataset_id_and_reference_id():
    pred = BoxPrediction(
        label="car",
        x=0,
        y=0,
        width=10,
        height=10,
        reference_id="r1",
        dataset_id="ds_1",
    )
    payload = pred.to_payload()
    assert payload["dataset_id"] == "ds_1"
    assert payload["reference_id"] == "r1"
    assert "item_id" not in payload


def test_box_prediction_omits_target_ids_when_unset():
    payload = _predictions()[0].to_payload()
    assert "item_id" not in payload
    assert "dataset_id" not in payload


# --------------------------------------------------------------------------- #
# Predictions target an item by dataset_item_id alone (no reference_id)
# --------------------------------------------------------------------------- #
def test_box_prediction_builds_from_dataset_item_id_alone():
    pred = BoxPrediction(
        label="car",
        x=0,
        y=0,
        width=10,
        height=10,
        dataset_item_id="di_1",
        confidence=0.9,
    )
    payload = pred.to_payload()
    assert payload["item_id"] == "di_1"
    assert payload.get("reference_id") is None


def test_prediction_with_no_target_is_rejected():
    with pytest.raises(ValueError, match="reference_id or dataset_item_id"):
        BoxPrediction(label="car", x=0, y=0, width=10, height=10)


def test_category_prediction_builds_from_dataset_item_id_alone():
    pred = CategoryPrediction(label="car", dataset_item_id="di_1")
    payload = pred.to_payload()
    assert payload["item_id"] == "di_1"
    assert payload.get("reference_id") is None
