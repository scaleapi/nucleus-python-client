"""Unit tests for multi-dataset model runs (no live API).

A model run used to declare exactly one dataset, and prediction uploads had to
stay inside it. It now carries the *set* of datasets its predictions actually
land in, which is what lets one run be scored against a benchmark whose items
span several datasets.

These tests pin the routing, because the route is the whole difference: only
``dataset/{dataset_id}/modelRun/{model_run_id}/uploadPredictions`` can add a
dataset to a run. The other two prediction routes deliberately cannot, and
silently sending a widening upload to one of them would either create a second
run or be rejected server-side.
"""

from unittest.mock import MagicMock

import pytest

from nucleus import NucleusClient
from nucleus.annotation_uploader import PredictionUploader
from nucleus.dataset import Dataset
from nucleus.prediction import BoxPrediction


def _client():
    return NucleusClient(api_key="test")


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
# PredictionUploader routing — the three forms
# --------------------------------------------------------------------------- #
def test_dataset_and_model_run_ids_route_to_the_widening_endpoint():
    uploader = PredictionUploader(
        client=_client(), dataset_id="ds_1", model_run_id="run_1"
    )
    assert (
        uploader._route
        == "dataset/ds_1/modelRun/run_1/uploadPredictions"  # noqa: SLF001
    )


def test_dataset_and_model_ids_route_to_the_model_endpoint():
    """The (dataset, model) form is unchanged — it cannot widen a run."""
    uploader = PredictionUploader(
        client=_client(), dataset_id="ds_1", model_id="prj_1"
    )
    assert (
        uploader._route
        == "dataset/ds_1/model/prj_1/uploadPredictions"  # noqa: SLF001
    )


def test_model_run_id_alone_routes_to_the_deprecated_endpoint():
    """Kept working for single-dataset runs; the server infers the dataset."""
    uploader = PredictionUploader(client=_client(), model_run_id="run_1")
    assert uploader._route == "modelRun/run_1/predict"  # noqa: SLF001


def test_model_id_and_model_run_id_together_are_rejected():
    with pytest.raises(AssertionError):
        PredictionUploader(
            client=_client(),
            dataset_id="ds_1",
            model_id="prj_1",
            model_run_id="run_1",
        )


def test_neither_model_nor_model_run_is_rejected():
    with pytest.raises(AssertionError):
        PredictionUploader(client=_client(), dataset_id="ds_1")


# --------------------------------------------------------------------------- #
# Dataset.upload_predictions_for_model_run
# --------------------------------------------------------------------------- #
def test_upload_predictions_for_model_run_uses_the_widening_route():
    client = _client()
    dataset = Dataset("ds_1", client)
    uploaded = {}

    def _capture(**kwargs):
        uploaded.update(kwargs)
        return {"predictions_processed": 1, "predictions_ignored": 0}

    with pytest.MonkeyPatch.context() as mp:
        routes = []
        original_init = PredictionUploader.__init__

        def _spy_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            routes.append(self._route)  # noqa: SLF001

        mp.setattr(PredictionUploader, "__init__", _spy_init)
        mp.setattr(
            PredictionUploader, "upload", lambda self, **kw: _capture(**kw)
        )
        dataset.upload_predictions_for_model_run("run_1", _predictions())

    assert routes == ["dataset/ds_1/modelRun/run_1/uploadPredictions"]
    assert uploaded["update"] is False


def test_upload_predictions_for_model_run_async_hits_the_async_route():
    client = _client()
    dataset = Dataset("ds_1", client)
    client.make_request = MagicMock(
        return_value={
            "job_id": "job_1",
            "job_last_known_status": "Running",
            "job_type": "uploadPredictions",
            "job_creation_time": "2026-08-03T00:00:00.000Z",
        }
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "nucleus.dataset.serialize_and_write_to_presigned_url",
            lambda *args, **kwargs: "req_1",
        )
        dataset.upload_predictions_for_model_run(
            "run_1", _predictions(), asynchronous=True
        )

    route = client.make_request.call_args[1]["route"]
    assert route == "dataset/ds_1/modelRun/run_1/uploadPredictions?async=1"


def test_upload_predictions_for_model_run_forwards_trained_slice_id():
    client = _client()
    dataset = Dataset("ds_1", client)
    captured = {}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            PredictionUploader,
            "upload",
            lambda self, **kw: captured.update(kw) or {},
        )
        dataset.upload_predictions_for_model_run(
            "run_1", _predictions(), trained_slice_id="slc_1", update=True
        )

    assert captured["trained_slice_id"] == "slc_1"
    assert captured["update"] is True


def test_upload_predictions_for_model_run_rejects_duplicate_ids():
    """Inherited from PredictionUploader; asserted here so the new entry point
    is known to run the check rather than bypass it."""
    from nucleus.errors import DuplicateIDError

    dataset = Dataset("ds_1", _client())
    duplicate = _predictions() * 2
    for pred in duplicate:
        pred.annotation_id = "ann_1"

    with pytest.raises(DuplicateIDError):
        dataset.upload_predictions_for_model_run("run_1", duplicate)
