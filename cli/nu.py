import os.path
import shutil

import click
from rich.console import Console
from rich.table import Column, Table
from shellingham import detect_shell

import nucleus


def compose_client():
    # TODO: Use env var!
    client = nucleus.NucleusClient("test_0b302578b4164aed9f2454a107cb1915")
    return client


@click.group("cli")
def nu():
    pass


@nu.command("install-completion")
def install_completion():
    """TODO: https://click.palletsprojects.com/en/8.0.x/shell-completion/, click_completion is outdated"""
    shell, _ = detect_shell()
    if shell == "zsh":
        rc_path = "~/.zshrc"
        append_to_file = 'eval "$(_NU_COMPLETE=zsh_source nu)"'
    elif shell == "bash":
        rc_path = "~/.bashrc"
        append_to_file = 'eval "$(_NU_COMPLETE=bash_source nu)"'
    elif shell == "fish":
        rc_path = "~/.config/fish/completions/foo-bar.fish"
        append_to_file = "eval (env _NU_COMPLETE=fish_source nu)"
    else:
        raise RuntimeError(f"Unsupported shell {shell} for completions")

    rc_path_expanded = os.path.expanduser(rc_path)
    rc_bak = f"{rc_path_expanded}.bak"
    shutil.copy(rc_path_expanded, rc_bak)
    click.echo(f"Backed up {rc_path} to {rc_bak}")
    with open(rc_path_expanded, mode="a") as rc_file:
        rc_file.write("\n")
        rc_file.write("# Shell completion for nu\n")
        rc_file.write(append_to_file)
    click.echo(f"Completion script added to {rc_path}")
    click.echo(f"Don't forget to `source {rc_path}")


@nu.command("datasets")
def datasets():
    console = Console()
    with console.status("Finding your Datasets!", spinner="dots4"):
        client = compose_client()
        datasets = client.datasets
        table = Table(
            "Name",
            "id",
            Column("url", overflow="fold"),
            title=":fire: :fire: Datasets",
            title_justify="left",
        )
        for ds in datasets:
            table.add_row(
                ds.name, ds.id, f"https://dashboard.scale.com/nucleus/{ds.id}"
            )
    console.print(table)


@nu.group("modelci")
def modelci():
    pass


@modelci.group("unit-tests")
def unit_tests():
    pass


@unit_tests.command("list")
def list_unit_tests():
    # TODO: Read from env
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
            table.add_row(
                ut.name,
                ut.id,
                ut.slice_id,
                f"https://dashboard.scale.com/nucleus/{ut.id}",
            )
    console.print(table)


if __name__ == "__main__":
    nu()
