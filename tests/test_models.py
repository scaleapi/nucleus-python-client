from pathlib import Path
import pytest
from nucleus import Dataset, DatasetItem, UploadResponse, Model, ModelRun, BoxPrediction
from nucleus.constants import (
    NEW_ITEMS,
    UPDATED_ITEMS,
    IGNORED_ITEMS,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    DATASET_ID_KEY,
)

TEST_MODEL_NAME = '[PyTest] Test Model 3'
TEST_REFERENCE_ID = '[PyTest] Test Model 3'
TEST_METADATA = {
    'key': 'value'
}
TEST_MODEL_RUN_NAME = '[PyTest] Test ModelRun 3'
TEST_DATASET_NAME = '[PyTest] Test Dataset'
TEST_IMG_URLS = [
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/6dd63871-831611a6.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/82c1005c-e2d1d94f.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/7f2e1814-6591087d.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/06924f46-1708b96f.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/89b42832-10d662f4.jpg",
]
TEST_DATASET_ITEMS = [
    DatasetItem(TEST_IMG_URLS[0], '1'),
    DatasetItem(TEST_IMG_URLS[1], '2'),
    DatasetItem(TEST_IMG_URLS[2], '3'),
    DatasetItem(TEST_IMG_URLS[3], '4')
]
TEST_PREDS = [
    BoxPrediction('car', 0, 0, 100, 100, '1'),
    BoxPrediction('car', 0, 0, 100, 100, '2'),
    BoxPrediction('car', 0, 0, 100, 100, '3'),
    BoxPrediction('car', 0, 0, 100, 100, '4')
]

@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds.append(TEST_DATASET_ITEMS)
    yield ds

    CLIENT.delete_dataset(ds.id)

def test_model_creation_and_listing(CLIENT, dataset):
    # Creation
    m = CLIENT.add_model(TEST_MODEL_NAME, TEST_REFERENCE_ID, TEST_METADATA)
    m_run =  m.create_run(TEST_MODEL_RUN_NAME, dataset, TEST_PREDS, TEST_METADATA)
    m_run.commit()

    assert isinstance(m, Model)
    assert isinstance(m_run, ModelRun)

    # List
    ms = CLIENT.list_models()

    assert m in ms

    CLIENT.delete_model(m.id)

    ms = CLIENT.list_models()

    assert m not in ms