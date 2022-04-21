import os.path

import click


def is_release_head(line):
    if line.startswith("## ["):
        return True
    else:
        return False


def check_if_start_of_line_factory(version):
    if version is None:
        return lambda line: is_release_head(line)
    else:
        return lambda line: is_release_head(line) and version in line


@click.command("find-version")
@click.option(
    "--version", help="Target version to output, if none, the topmost"
)
@click.option(
    "--changelog-path",
    default="./CHANGELOG.md",
    help="Path to changelog, defaults to ./CHANGELOG.md (works if you run it from root of repo)",
)
def find_version_changelog(version, changelog_path):
    target_changes = []
    in_version = False
    is_start_of_version = check_if_start_of_line_factory(version)
    with open(os.path.expanduser(changelog_path), "r") as ch_file:
        content = ch_file.read()
        for line in content.splitlines():
            if is_release_head(line) and in_version:
                break
            elif is_start_of_version(line):
                in_version = True
            if in_version:
                target_changes.append(line)

    changes = "\n".join(target_changes)
    click.echo(changes)


if __name__ == "__main__":
    find_version_changelog()
