import os

import pytest

from nucleus.deploy import DeployClient

API_KEY = os.environ["NUCLEUS_PYTEST_API_KEY"]


@pytest.fixture(scope="module")
def DEPLOY_CLIENT():
    return DeployClient(api_key=API_KEY)
