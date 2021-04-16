from pathlib import Path
import time
import pytest
from nucleus import (
    Dataset,
    DatasetItem,
    UploadResponse,
    Model,
    ModelRun,
    BoxPrediction,
    NucleusClient,
)
from nucleus.constants import (
    NEW_ITEMS,
    UPDATED_ITEMS,
    IGNORED_ITEMS,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    DATASET_ID_KEY,
)
from helpers import (
    TEST_MODEL_NAME,
    TEST_MODEL_RUN,
    TEST_PREDS,
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
    test_repr(ModelRun(client=client, model_run_id="fake_model_run_id"))


def test_model_creation_and_listing(CLIENT, dataset):
    models_before = CLIENT.list_models()

    model_reference = "model_" + time.time()
    # Creation
    model = CLIENT.add_model(TEST_MODEL_NAME, model_reference)
    m_run = model.create_run(TEST_MODEL_RUN, dataset, TEST_PREDS)
    m_run.commit()

    assert isinstance(model, Model)
    assert isinstance(m_run, ModelRun)

    # List the models
    ms = CLIENT.list_models()

    assert model in ms
    assert list(set(ms) - set(models_before))[0] == model

    # Delete the model
    CLIENT.delete_model(model.id)
    ms = CLIENT.list_models()

    assert model not in ms
    assert ms == models_before
