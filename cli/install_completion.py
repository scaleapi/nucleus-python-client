import os
import shutil
import subprocess

import click
from shellingham import detect_shell


def generate_completions(shell, completion_path):
    command = f"_NU_COMPLETE={shell}_source nu > {completion_path}"
    subprocess.run(command, shell=True)
    click.echo(f"Generated completions for {shell}: {completion_path}")


@click.command("install-completion")
def install_completion():
    """Install shell completion script to your rc file"""
    shell, _ = detect_shell()
    os.makedirs("~/.config", exist_ok=True)
    append_to_file = None
    if shell == "zsh":
        rc_path = "~/.zshrc"
        completion_path = "~/.config/nu-cli-completions.zsh"
        append_to_file = f". {completion_path}"
    elif shell == "bash":
        rc_path = "~/.bashrc"
        completion_path = "~/.config/nu-cli-completions.bash"
        append_to_file = f". {completion_path}"
    elif shell == "fish":
        rc_path = "~/.config/fish/completions/foo-bar.fish"
        completion_path = "~/.config/fish/completions/nu.fish"
    else:
        raise RuntimeError(f"Unsupported shell {shell} for completions")

    generate_completions(shell, completion_path)

    if append_to_file is not None:
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
