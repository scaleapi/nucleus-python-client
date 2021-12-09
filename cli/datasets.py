import click
from rich.console import Console
from rich.table import Column, Table

from cli.client import compose_client
from cli.helpers.nucleus_url import nucleus_url


@click.group("datasets")
def datasets():
    """Common operations for datasets"""
    pass


@datasets.command("list")
def list_datasets():
    """List all available datasets"""
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
            table.add_row(ds.name, ds.id, nucleus_url(ds.id))
    console.print(table)


@datasets.command("delete")
@click.option("--id", prompt=True)
def delete_dataset(id):
    """Delete a dataset"""
    console = Console()
    client = compose_client()
    dataset = [ds for ds in client.datasets if ds.id == id][0]
    delete_string = click.prompt(
        click.style(f"Type 'DELETE' to delete dataset: {dataset}", fg="red")
    )
    if delete_string == "DELETE":
        client.delete_dataset(dataset.id)
        console.print(f":fire: :anguished: Deleted {id}")
