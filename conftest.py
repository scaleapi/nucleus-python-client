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
def monkeypatch_session(request):
    """This workaround is needed to allow monkeypatching in session-scoped fixtures.

    See https://github.com/pytest-dev/pytest/issues/363
    """
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="session")
def CLIENT(monkeypatch_session):
    client = nucleus.NucleusClient(API_KEY)

    # Change _make_request to raise AsssertionErrors when the
    # HTTP status code is not successful, so that tests fail if
    # the request was unsuccessful.
    def _make_request_patch(
        payload: dict, route: str, requests_command=requests.post
    ) -> dict:
        response = client._make_request_raw(payload, route, requests_command)
        assert response.status_code in SUCCESS_STATUS_CODES, (
            f"HTTP response had status code: {response.status_code}. "
            f"Full JSON: {response.json()}"
        )
        return response.json()

    monkeypatch_session.setattr(client, "_make_request", _make_request_patch)
    return client


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds.append(TEST_DATASET_ITEMS)
    yield ds

    CLIENT.delete_dataset(ds.id)
