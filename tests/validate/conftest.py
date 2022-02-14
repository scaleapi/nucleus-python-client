import time

import pytest

from tests.helpers import (
    TEST_MODEL_NAME,
    TEST_SLICE_NAME,
    create_box_annotations,
    create_predictions,
    get_uuid,
)
from tests.test_dataset import make_dataset_items


@pytest.fixture(scope="module")
def validate_dataset(CLIENT):
    """SHOULD NOT BE MUTATED IN TESTS. This dataset lives for the whole test module scope."""
    ds = CLIENT.create_dataset("[Test Model CI] Dataset", is_scene=False)
    yield ds

    CLIENT.delete_dataset(ds.id)


@pytest.fixture(scope="module")
def dataset_items(validate_dataset):
    items = make_dataset_items()
    validate_dataset.append(items)
    yield items


@pytest.fixture(scope="module")
def slice_items(dataset_items):
    yield dataset_items[:2]


@pytest.fixture(scope="module")
def test_slice(validate_dataset, slice_items):
    slc = validate_dataset.create_slice(
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
def annotations(validate_dataset, slice_items):
    annotations = create_box_annotations(validate_dataset, slice_items)
    yield annotations


@pytest.fixture(scope="module")
def predictions(model, validate_dataset, annotations):
    predictions = create_predictions(validate_dataset, model, annotations)
    yield predictions


@pytest.fixture(scope="module")
@pytest.mark.usefixtures(
    "annotations"
)  # Scenario test needs to have annotations in the slice
def scenario_test(CLIENT, test_slice):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_criteria=[CLIENT.validate.eval_functions.bbox_recall > 0.5],
    )
    yield scenario_test

    CLIENT.validate.delete_scenario_test(scenario_test.id)
