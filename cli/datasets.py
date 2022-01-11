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
@click.option(
    "-m", "--machine-readable", is_flag=True, help="Removes pretty printing"
)
def list_datasets(machine_readable):
    """List all available Datasets"""
    console = Console()
    with console.status("Finding your Datasets!", spinner="dots4"):
        client = init_client()
        all_datasets = client.datasets
        if machine_readable:
            table_params = {"box": None, "pad_edge": False}
        else:
            table_params = {
                "title": ":fire: Datasets",
                "title_justify": "left",
            }

        table = Table(
            "id", "Name", Column("url", overflow="fold"), **table_params
        )
        for ds in all_datasets:
            table.add_row(ds.id, ds.name, nucleus_url(ds.id))
    console.print(table)


@datasets.command("delete")
@click.option("--id", prompt=True)
@click.option(
    "--no-confirm-deletion",
    is_flag=True,
    help="WARNING: No confirmation for deletion",
)
@click.pass_context
def delete_dataset(ctx, id, no_confirm_deletion):
    """Delete a Dataset"""
    console = Console()
    client = init_client()
    id = id.strip()
    dataset = client.get_dataset(id)
    delete_string = ""
    if not no_confirm_deletion:
        delete_string = click.prompt(
            click.style(
                f"Type 'DELETE' to delete dataset: {dataset}", fg="red"
            )
        )
    if no_confirm_deletion or delete_string == "DELETE":
        client.delete_dataset(dataset.id)
        console.print(f":fire: :anguished: Deleted {id}")
    else:
        console.print(
            f":rotating_light: Refusing to delete {id}. Received '{delete_string}' instead of 'DELETE'"
        )
        ctx.abort()
