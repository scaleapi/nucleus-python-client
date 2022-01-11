import click
from rich.console import Console
from rich.table import Column, Table

from cli.client import init_client
from cli.helpers.nucleus_url import nucleus_url
from cli.helpers.web_helper import launch_web_or_invoke


@click.group("datasets", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def datasets(ctx, web):
    """Datasets are the base collections of items in Nucleus

    https://dashboard.scale.com/nucleus/datasets
    """
    launch_web_or_invoke(
        sub_url="datasets", ctx=ctx, launch_browser=web, command=list_datasets
    )


@datasets.command("list")
def list_datasets():
    """List all available Datasets"""
    console = Console()
    with console.status("Finding your Datasets!", spinner="dots4"):
        client = init_client()
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
@click.pass_context
def delete_dataset(ctx, id):
    """Delete a Dataset"""
    console = Console()
    client = init_client()
    dataset = [ds for ds in client.datasets if ds.id == id][0]
    delete_string = click.prompt(
        click.style(f"Type 'DELETE' to delete dataset: {dataset}", fg="red")
    )
    if delete_string == "DELETE":
        client.delete_dataset(dataset.id)
        console.print(f":fire: :anguished: Deleted {id}")
    else:
        console.print(
            f":rotating_light: Refusing to delete {id}. Received '{delete_string}' instead of 'DELETE'"
        )
        ctx.abort()
