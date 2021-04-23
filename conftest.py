# grequests must be imported before any module not designed for reentrancy,
# because it relies on aggressive monkey patching that breaks if done after many
# other common module imports, e.g. ssl.
#
# So we import it before everything else. For details see:
# https://github.com/gevent/gevent/issues/1016#issuecomment-328530533
# https://github.com/spyoungtech/grequests/issues/8
import grequests

################

import logging
import os

import requests
import pytest

import nucleus
from nucleus.constants import SUCCESS_STATUS_CODES

from tests.helpers import TEST_DATASET_NAME, TEST_DATASET_ITEMS

assert "NUCLEUS_PYTEST_API_KEY" in os.environ, (
    "You must set the 'NUCLEUS_PYTEST_API_KEY' environment variable to a valid "
    "Nucleus API key to run the test suite"
)

API_KEY = os.environ["NUCLEUS_PYTEST_API_KEY"]


@pytest.fixture(scope="session")
def CLIENT():
    client = nucleus.NucleusClient(API_KEY)
    return client


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds.append(TEST_DATASET_ITEMS)
    yield ds

    CLIENT.delete_dataset(ds.id)
