from pathlib import Path
import pytest
from nucleus import (
    Dataset,
    DatasetItem,
    UploadResponse,
    Model,
    ModelRun,
    BoxPrediction
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
    dataset,
    TEST_MODEL_NAME,
    TEST_MODEL_REFERENCE,
    TEST_MODEL_RUN,
    TEST_PREDS
)

def test_model_creation_and_listing(CLIENT, dataset):
    # Creation
    m = CLIENT.add_model(TEST_MODEL_NAME, TEST_MODEL_REFERENCE)
    m_run =  m.create_run(TEST_MODEL_RUN, dataset, TEST_PREDS)
    m_run.commit()

    assert isinstance(m, Model)
    assert isinstance(m_run, ModelRun)

    # List the models
    ms = CLIENT.list_models()

    assert m in ms

    # Delete the model
    CLIENT.delete_model(m.id)
    ms = CLIENT.list_models()

    assert m not in ms