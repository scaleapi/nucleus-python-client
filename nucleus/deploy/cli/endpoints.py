import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Column, Table

from nucleus.deploy.cli.client import init_client
from nucleus.deploy.model_endpoint import AsyncModelEndpoint, Endpoint


@click.group("endpoints")
def endpoints():
    """Endpoints is a wrapper around model bundles in Scale Launch"""
    pass


@endpoints.command("list")
def list_endpoints():
    """List all of your Bundles"""
    client = init_client()

    table = Table(
        "Endpoint name",
        "Metadata",
        "Endpoint type",
        title="Endpoints",
        title_justify="left",
    )

    for endpoint_sync_async in client.list_model_endpoints():
        endpoint = endpoint_sync_async.endpoint
        table.add_row(
            endpoint.name,
            endpoint.metadata,
            endpoint.endpoint_type,
        )
    console = Console()
    console.print(table)


@endpoints.command("delete")
@click.argument("endpoint_name")
def delete_bundle(endpoint_name):
    """Delete a model bundle"""
    client = init_client()

    console = Console()
    endpoint = Endpoint(name=endpoint_name)
    dummy_endpoint = AsyncModelEndpoint(endpoint=endpoint, client=client)
    res = client.delete_model_endpoint(dummy_endpoint)
    console.print(res)
