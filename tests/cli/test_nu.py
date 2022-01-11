from unittest import mock

from click.testing import CliRunner

from cli.nu import nu


def test_invoke_nu_prints_help():
    runner = CliRunner()
    result = runner.invoke(nu)  # type: ignore
    assert result.exception is None
    assert "Nucleus CLI" in result.output


def test_invoke_nu_web():
    runner = CliRunner()
    with mock.patch("click.launch"):
        result = runner.invoke(nu, ["--web"])  # type: ignore
        assert result.exception is None
        assert result.exit_code == 0


def test_invoke_nu_web_subcommand_aborts():
    runner = CliRunner()
    with mock.patch("click.launch"):
        result = runner.invoke(nu, ["--web", "datasets"])  # type: ignore
        assert result.exit_code != 0
