import time

import pytest

from nucleus import BoxAnnotation
from tests.helpers import (
    TEST_BOX_ANNOTATIONS,
    TEST_MODEL_NAME,
    TEST_SLICE_NAME,
    get_uuid,
)
from tests.modelci.helpers import create_box_annotations, create_predictions
from tests.test_dataset import make_dataset_items


@pytest.fixture(scope="module")
def modelci_dataset(CLIENT):
    """SHOULD NOT BE MUTATED IN TESTS. This dataset lives for the whole test module scope."""
    ds = CLIENT.create_dataset("[Test Model CI] Dataset", is_scene=False)
    yield ds

    CLIENT.delete_dataset(ds.id)


@pytest.fixture(scope="module")
def dataset_items(modelci_dataset):
    items = make_dataset_items()
    modelci_dataset.append(items)
    yield items


@pytest.fixture(scope="module")
def slice_items(dataset_items):
    yield dataset_items[:2]


@pytest.fixture(scope="module")
def test_slice(modelci_dataset, slice_items):
    slc = modelci_dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[item.reference_id for item in slice_items],
    )
    yield slc


@pytest.fixture(scope="module")
def model(CLIENT):
    model_reference = "model_" + str(time.time())
    model = CLIENT.create_model(TEST_MODEL_NAME, model_reference)
    yield model

    CLIENT.delete_model(model.id)


@pytest.fixture(scope="module")
def annotations(modelci_dataset, slice_items):
    annotations = create_box_annotations(modelci_dataset, slice_items)
    yield annotations


@pytest.fixture(scope="module")
def predictions(model, modelci_dataset, annotations):
    predictions = create_predictions(modelci_dataset, model, annotations)
    yield predictions


@pytest.fixture(scope="module")
@pytest.mark.usefixtures(
    "annotations"
)  # Unit test needs to have annotations in the slice
def unit_test(CLIENT, test_slice):
    test_name = "unit_test_" + get_uuid()  # use uuid to make unique
    unit_test = CLIENT.modelci.create_unit_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_criteria=[CLIENT.modelci.eval_functions.bbox_recall > 0.5],
    )
    yield unit_test

    CLIENT.modelci.delete_unit_test(unit_test.id)
