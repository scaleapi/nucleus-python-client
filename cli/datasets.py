import click
from rich.console import Console
from rich.table import Column, Table

from cli.client import compose_client


@click.command("datasets")
def datasets():
    console = Console()
    with console.status("Finding your Datasets!", spinner="dots4"):
        client = compose_client()
        all_datasets = client.datasets
        table = Table(
            "Name",
            "id",
            Column("url", overflow="fold"),
            title=":fire: Datasets",
            title_justify="left",
        )
        for ds in all_datasets:
            table.add_row(
                ds.name, ds.id, f"https://dashboard.scale.com/nucleus/{ds.id}"
            )
    console.print(table)
