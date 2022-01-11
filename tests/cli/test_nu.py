from unittest import mock

import pytest
from click.testing import CliRunner

from cli.nu import nu


@pytest.mark.usefixtures("cli_datasets", "cli_models")
def test_invoke_nu_prints_help():
    runner = CliRunner()
    result = runner.invoke(nu)  # type: ignore
    assert "Nucleus CLI" in result.output


@pytest.mark.usefixtures("cli_datasets", "cli_models")
def test_invoke_nu_web():
    runner = CliRunner()
    with mock.patch("click.launch"):
        result = runner.invoke(nu, ["--web"])  # type: ignore
        assert result.exit_code == 0


@pytest.mark.usefixtures("cli_datasets", "cli_models")
def test_invoke_nu_web_subcommand_aborts():
    runner = CliRunner()
    with mock.patch("click.launch"):
        result = runner.invoke(nu, ["--web", "datasets"])  # type: ignore
        assert result.exit_code != 0
