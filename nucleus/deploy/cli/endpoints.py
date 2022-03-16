import click
from rich.console import Console
from rich.table import Column, Table
from rich.syntax import Syntax

from nucleus.deploy.model_endpoint import Endpoint, AsyncModelEndpoint
from nucleus.deploy.cli.client import init_client


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


@endpoints.command("get")
def list_bundles(bundle_name):
    """Priint bundle info"""
    client = init_client()

    model_bundle = client.get_model_bundle(bundle_name)

    console = Console()
    console.print(f"{model_bundle.bundle_id=}")
    console.print(f"{model_bundle.bundle_name=}")
    console.print(f"{model_bundle.location=}")
    console.print(f"{model_bundle.packaging_type=}")
    console.print(f"{model_bundle.env_params=}")
    console.print(f"{model_bundle.requirements=}")

    console.print("model_bundle.metadata:")
    for meta_name, meta_value in model_bundle.metadata.items():
        # TODO print non-code metadata differently
        console.print(f"{meta_name}:", style="yellow")
        syntax = Syntax(meta_value, 'python')
        console.print(syntax)


@endpoints.command("delete")
@click.argument('endpoint_name')
def delete_bundle(endpoint_name):
    """Delete a model bundle"""
    client = init_client()

    console = Console()
    endpoint = Endpoint(name=endpoint_name)
    dummy_endpoint = AsyncModelEndpoint(endpoint=endpoint, client=client)
    res = client.delete_model_endpoint(dummy_endpoint)
    console.print(res)
