import click
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Column, Table
from rich.tree import Tree

from cli.client import init_client
from cli.helpers.nucleus_url import nucleus_url
from cli.helpers.web_helper import launch_web_or_invoke
from nucleus import NucleusAPIError


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
    with Live(
        Spinner("dots4", text="Finding your Slices!"),
        vertical_overflow="visible",
    ) as live:
        client = init_client()
        datasets = client.datasets
        table = Table(
            Column("id", overflow="fold", min_width=24),
            "name",
            "dataset_name",
            Column("url", overflow="fold"),
            title=":cake: Slices",
            title_justify="left",
        )
        errors = {}
        for ds in datasets:
            try:
                ds_slices = ds.slices
                if ds_slices:
                    for slc_id in ds_slices:
                        slice_url = nucleus_url(f"{ds.id}/{slc_id}")
                        slice_info = client.get_slice(slc_id).info()
                        table.add_row(
                            slc_id, slice_info["name"], ds.name, slice_url
                        )
                        live.update(table)
            except NucleusAPIError as e:
                errors[ds.id] = e

        error_tree = Tree(
            ":x: Encountered the following errors while fetching information"
        )
        for ds_id, error in errors.items():
            dataset_branch = error_tree.add(f"Dataset: {ds_id}")
            dataset_branch.add(f"Error: {error}")

        Console().print(error_tree)
