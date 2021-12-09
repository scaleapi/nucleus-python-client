import click
from rich.console import Console
from rich.table import Column, Table

from cli.client import compose_client
from cli.helpers.nucleus_url import nucleus_url


@click.group("modelci")
def modelci():
    """Interactions with Model CI"""
    pass


@modelci.group("unit-tests")
def unit_tests():
    """ Model CI Unit Tests """
    pass


@unit_tests.command("list")
def list_unit_tests():
    """View all your Unit Tests"""
    console = Console()
    with console.status("Finding your unit tests", spinner="dots4"):
        client = compose_client()
        unit_tests = client.modelci.list_unit_tests()
        table = Table(
            "Name",
            "id",
            "slice_id",
            Column("url", overflow="fold"),
            title=":triangular_flag_on_post: Unit tests",
            title_justify="left",
        )
        for ut in unit_tests:
            table.add_row(ut.name, ut.id, ut.slice_id, nucleus_url(ut.id))
    console.print(table)
