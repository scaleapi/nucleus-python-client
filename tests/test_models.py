import time
from pathlib import Path

import pytest

from nucleus import (
    BoxPrediction,
    Dataset,
    DatasetItem,
    Model,
    ModelRun,
    NucleusClient,
    UploadResponse,
)
from nucleus.constants import (
    DATASET_ID_KEY,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    IGNORED_ITEMS,
    NEW_ITEMS,
    UPDATED_ITEMS,
)

from .helpers import (
    TEST_BOX_PREDICTIONS,
    TEST_MODEL_NAME,
    TEST_MODEL_RUN,
    TEST_PREDS,
    assert_box_prediction_matches_dict,
    get_uuid,
)


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    client = NucleusClient(api_key="fake_key")
    test_repr(
        Model(
            client=client,
            model_id="fake_model_id",
            name="fake_name",
            reference_id="fake_reference_id",
            metadata={"fake": "metadata"},
        )
    )
    test_repr(
        ModelRun(
            client=client,
            dataset_id="fake_dataset_id",
            model_run_id="fake_model_run_id",
        )
    )


def test_model_creation_and_listing(CLIENT, dataset):
    model_reference = "model_" + str(time.time())
    # Creation
    model_name = TEST_MODEL_NAME + get_uuid()
    model = CLIENT.create_model(model_name, model_reference)
    model_run = TEST_MODEL_RUN + get_uuid()
    m_run = model.create_run(model_run, dataset, TEST_PREDS)

    assert isinstance(model, Model)
    assert isinstance(m_run, ModelRun)

    # List the models
    ms = CLIENT.models

    # Get a model
    m = CLIENT.get_model(model.id)
    m = CLIENT.get_model(m_run.model_run_id)
    assert m == model

    assert model in ms

    # Delete the model
    CLIENT.delete_model(model.id)
    ms = CLIENT.models

    assert model not in ms


# Until we fully remove the other endpoints (and then migrate those tests) just quickly test the basics of the new ones since they are basically just simple wrappers around the old ones.
def test_new_model_endpoints(CLIENT, dataset: Dataset):
    model_reference = "model_" + str(time.time())
    model = CLIENT.create_model(TEST_MODEL_NAME, model_reference)
    predictions = [BoxPrediction(**TEST_BOX_PREDICTIONS[0])]

    dataset.upload_predictions(model, predictions=predictions)

    dataset.calculate_evaluation_metrics(model)

    predictions_export = dataset.export_predictions(model)

    assert_box_prediction_matches_dict(
        predictions_export["box"][0], TEST_BOX_PREDICTIONS[0]
    )

    predictions_iloc = dataset.predictions_iloc(model, 0)
    assert_box_prediction_matches_dict(
        predictions_iloc["box"][0], TEST_BOX_PREDICTIONS[0]
    )

    predictions_refloc = dataset.predictions_refloc(
        model, predictions[0].reference_id
    )

    assert_box_prediction_matches_dict(
        predictions_refloc["box"][0], TEST_BOX_PREDICTIONS[0]
    )
    prediction_loc = dataset.prediction_loc(
        model, predictions[0].reference_id, predictions[0].annotation_id
    )
    assert_box_prediction_matches_dict(prediction_loc, TEST_BOX_PREDICTIONS[0])


def test_tag_model(CLIENT, dataset: Dataset):

    def testing_model(ref_id):
        models_from_backend = list(
            filter(lambda m: m.reference_id == ref_id, CLIENT.models)
        )
        assert len(models_from_backend) == 1
        return models_from_backend[0]

    model_reference = "model_" + str(time.time())
    model = CLIENT.create_model(
        TEST_MODEL_NAME, model_reference, tags=["first_tag"]
    )

    model.add_tag("single tag")
    model.add_tag(["tag_a", "tag_b"])

    backend_model = testing_model(model_reference)
    assert sorted(backend_model.tags) == sorted(
        ["first_tag", "single tag", "tag_a", "tag_b"]
    )

    model.remove_tag("tag_a")
    model.remove_tag(["first_tag", "tag_b"])

    backend_model = testing_model(model_reference)
    assert backend_model.tags == ["single tag"]
