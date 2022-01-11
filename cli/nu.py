import click

from cli.datasets import datasets
from cli.helpers.web_helper import launch_web_or_show_help
from cli.install_completion import install_completion
from cli.jobs import jobs
from cli.models import models
from cli.reference import reference
from cli.slices import slices
from cli.tests import tests


@click.group("cli", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def nu(ctx, web):
    """Nucleus CLI (nu)

        \b
    ███╗   ██╗██╗   ██╗ ██████╗██╗     ███████╗██╗   ██╗███████╗
    ████╗  ██║██║   ██║██╔════╝██║     ██╔════╝██║   ██║██╔════╝
    ██╔██╗ ██║██║   ██║██║     ██║     █████╗  ██║   ██║███████╗
    ██║╚██╗██║██║   ██║██║     ██║     ██╔══╝  ██║   ██║╚════██║
    ██║ ╚████║╚██████╔╝╚██████╗███████╗███████╗╚██████╔╝███████║
    ╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝

        `nu` is a command line interface to interact with Scale Nucleus (https://dashboard.scale.com/nucleus)
    """
    launch_web_or_show_help(sub_url="", ctx=ctx, launch_browser=web)


nu.add_command(datasets)  # type: ignore
nu.add_command(install_completion)  # type: ignore
nu.add_command(jobs)  # type: ignore
nu.add_command(models)  # type: ignore
nu.add_command(reference)  # type: ignore
nu.add_command(slices)  # type: ignore
nu.add_command(tests)  # type: ignore

if __name__ == "__main__":
    """To debug, run this script followed by request command tree e.g. `cli/nu.py datasets list`"""
    nu()
