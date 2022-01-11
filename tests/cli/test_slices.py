import pytest
from click.testing import CliRunner

from cli.slices import list_slices, slices


@pytest.fixture(scope="module")
def cli_slices(test_slice):
    yield [test_slice]


# TODO(gunnar): Add actual slice data through fixture
def test_invoke_slices(cli_slices):
    runner = CliRunner()
    result = runner.invoke(slices)  # type: ignore
    assert result.exit_code == 0
    for slc in cli_slices:
        assert slc.id in result.output


def test_invoke_slices_list(cli_slices):
    runner = CliRunner()
    result = runner.invoke(list_slices)  # type: ignore
    assert result.exit_code == 0
    for slc in cli_slices:
        assert slc.id in result.output
