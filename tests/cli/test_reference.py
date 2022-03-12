from unittest import mock

from cli.reference import reference


def test_invoke_reference(runner):
    with mock.patch("click.launch"):
        result = runner.invoke(reference)
        assert result.exception is None
        assert result.exit_code == 0
