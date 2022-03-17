import click

from nucleus.deploy.cli.bundles import bundles
from nucleus.deploy.cli.endpoints import endpoints


@click.group("cli")
def entry_point():
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


entry_point.add_command(bundles)  # type: ignore
entry_point.add_command(endpoints)  # type: ignore

if __name__ == "__main__":
    entry_point()
