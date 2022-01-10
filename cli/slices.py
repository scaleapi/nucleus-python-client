import click
from rich.live import Live
from rich.spinner import Spinner
from rich.tree import Tree

from cli.client import init_client
from cli.helpers.nucleus_url import nucleus_url


@click.group("slices")
def slices():
    """Slices are named subsets of Datasets"""
    pass


@slices.command("list")
def list_slices():
    """List all available Slices"""
    with Live(
        Spinner("dots4", text="Finding your Slices!"),
        vertical_overflow="visible",
    ) as live:
        client = init_client()
        datasets = client.datasets
        tree = Tree(":cake: Slices")
        for ds in datasets:
            branch = tree.add(f"{ds.id}: {ds.name}")
            ds_slices = ds.slices
            if ds_slices:
                for slc_id in ds_slices:
                    slice_url = nucleus_url(f"{ds.id}/{slc_id}")
                    slice_info = client.get_slice(slc_id).info()
                    branch.add(f"{slc_id}: {slice_info['name']} ({slice_url})")
                    live.update(tree)
            else:
                branch.add("No slices in this dataset ... :sleeping:")
                live.update(tree)
