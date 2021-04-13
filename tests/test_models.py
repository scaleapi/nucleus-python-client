from pathlib import Path
import pytest
from nucleus import (
    Dataset,
    DatasetItem,
    UploadResponse,
    Model,
    ModelRun,
    BoxPrediction,
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
    TEST_MODEL_REFERENCE,
    TEST_MODEL_RUN,
    TEST_PREDS,
)


def test_model_creation_and_listing(CLIENT, dataset):
    models_before = CLIENT.list_models()

    # Creation
    model = CLIENT.add_model(TEST_MODEL_NAME, TEST_MODEL_REFERENCE)
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
