import click
from rich.console import Console
from rich.table import Column, Table

from cli.client import init_client
from cli.helpers.nucleus_url import nucleus_url
from cli.helpers.web_helper import launch_web_or_invoke


@click.group("slices", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def slices(ctx, web):
    """Slices are named subsets of Datasets"""
    # TODO(gunnar): We don't have a natural landing for slices overview, until then we land on "/"
    launch_web_or_invoke("", ctx, web, list_slices)


@slices.command("list")
def list_slices():
    """List all available Slices"""
    client = init_client()
    console = Console()
    with console.status("Finding your Slices!", spinner="dots4"):
        table = Table(
            Column("id", overflow="fold", min_width=24),
            "name",
            "dataset_name",
            Column("url", overflow="fold"),
            title=":cake: Slices",
            title_justify="left",
        )
        datasets = client.datasets
        id_to_datasets = {d.id: d for d in datasets}
        all_slices = client.slices
        for s in all_slices:
            table.add_row(
                s.id,
                s.name,
                id_to_datasets[s.dataset_id].name,
                nucleus_url(f"{s.dataset_id}/{s.id}"),
            )

    console.print(table)
