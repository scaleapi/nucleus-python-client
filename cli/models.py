import click
import questionary
from rich.console import Console
from rich.table import Column, Table

from cli.client import init_client
from cli.helpers.nucleus_url import nucleus_url
from cli.helpers.web_helper import launch_web_or_invoke


@click.group("models", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def models(ctx, web):
    """Models help you store and access your ML model data

    https://dashboard.scale.com/nucleus/models
    """
    launch_web_or_invoke("models", ctx, web, list_models)


@models.command("calculate-metrics")
def metrics():
    client = init_client()
    models = client.models
    prompt_to_id = {f"{m.id}: {m.name}": m.id for m in models}
    ans = questionary.select(
        "What model do you want to run metrics for?",
        choices=list(prompt_to_id.keys()),
    ).ask()
    model_id = prompt_to_id[ans]
    jobs = client.validate.metrics(model_id)
    console = Console()
    with console.status("Calculating metrics"):
        for job in jobs:
            job.sleep_until_complete()
    click.echo(click.style("Done", fg="green"))


@models.command("list")
def list_models():
    """List your Models"""
    console = Console()
    with console.status("Finding your Models!", spinner="dots4"):
        client = init_client()
        table = Table(
            Column("id", overflow="fold", min_width=24),
            "name",
            Column("url", overflow="fold"),
        )
        models = client.models
        for m in models:
            table.add_row(m.id, m.name, nucleus_url(m.id))
    console.print(table)
