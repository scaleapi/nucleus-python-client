from click.testing import CliRunner

from cli.models import list_models, models


def test_invoke_models(runner, cli_models):
    result = runner.invoke(models)  # type: ignore
    assert result.exception is None
    assert result.exit_code == 0
    for model in cli_models:
        assert model.id in result.output


def test_invoke_models_list(runner, cli_models):
    result = runner.invoke(list_models)  # type: ignore
    assert result.exception is None
    assert result.exit_code == 0
    for model in cli_models:
        assert model.id in result.output
