import click

from nucleus.deploy.cli.bundles import bundles
from nucleus.deploy.cli.endpoints import endpoints


@click.group("cli", invoke_without_command=True)
@click.pass_context
def entry_point(ctx):
    """Launch CLI

        \b
    ██╗      █████╗ ██╗   ██╗███╗   ██╗ ██████╗██╗  ██╗
    ██║     ██╔══██╗██║   ██║████╗  ██║██╔════╝██║  ██║
    ██║     ███████║██║   ██║██╔██╗ ██║██║     ███████║
    ██║     ██╔══██║██║   ██║██║╚██╗██║██║     ██╔══██║
    ███████╗██║  ██║╚██████╔╝██║ ╚████║╚██████╗██║  ██║
    ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝

        `scale-launch` is a command line interface to interact with Scale Launch
    """
    pass


entry_point.add_command(bundles)  # type: ignore
entry_point.add_command(endpoints)  # type: ignore

if __name__ == "__main__":
    """To debug, run this script followed by request command tree e.g. `cli/nu.py datasets list`"""
    entry_point()
