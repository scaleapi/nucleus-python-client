import os
import shutil

import click
from shellingham import detect_shell


@click.command("install-completion")
def install_completion():
    """Install shell completion script to your rc file"""
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
