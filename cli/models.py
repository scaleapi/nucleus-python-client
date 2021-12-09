import click
from rich.console import Console
from rich.table import Column, Table

from cli.client import compose_client
from cli.helpers.nucleus_url import nucleus_url


@click.group("models")
def models():
    pass


@models.command("list")
def list_models():
    console = Console()
    with console.status("Finding your Models!", spinner="dots4"):
        client = compose_client()
        table = Table("name", "id", Column("url", overflow="fold"))
        models = client.models
        for m in models:
            table.add_row(m.name, m.id, nucleus_url(m.id))
    console.print(table)
