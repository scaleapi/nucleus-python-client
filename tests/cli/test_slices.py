from unittest import mock

import pytest
from click.testing import CliRunner

from cli.slices import list_slices, slices


@pytest.fixture(scope="module")
def cli_slices(test_slice):
    yield [test_slice]


# TODO(gunnar): Add actual slice data through fixture
def test_invoke_slices(runner):
    # NOTE: The list_slices method is tested elsewhere, just testing control flow
    with mock.patch("cli.slices.list_slices"):
        result = runner.invoke(slices)  # type: ignore
    assert result.exception is None
    assert result.exit_code == 0


@pytest.mark.integration
@pytest.mark.skip("Repeatedly hanging in tests")
def test_invoke_slices_list(runner, cli_slices):
    runner = CliRunner()
    result = runner.invoke(list_slices)  # type: ignore
    assert result.exception is None
    assert result.exit_code == 0
    assert cli_slices[0].id in result.output
