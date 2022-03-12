import click

from cli.helpers.nucleus_url import nucleus_url


def launch_web_or_show_help(
    sub_url: str, ctx: click.Context, launch_browser: bool
):
    """Launches the sub_url (composed with nuclues_url(sub_url)) in the browser if requested"""
    if not ctx.invoked_subcommand:
        if launch_browser:
            url = nucleus_url(sub_url)
            click.launch(url)
        else:
            click.echo(ctx.get_help())
    else:
        if launch_browser:
            click.echo(click.style("--web does not work with sub-commands"))
            ctx.abort()


def launch_web_or_invoke(
    sub_url: str,
    ctx: click.Context,
    launch_browser: bool,
    command: click.BaseCommand,
):
    """Launches the sub_url (composed with nuclues_url(sub_url)) in the browser if requested, otherwise invokes
    the passed command
    """
    if not ctx.invoked_subcommand:
        if launch_browser:
            url = nucleus_url(sub_url)
            click.launch(url)
        else:
            ctx.invoke(command)
    else:
        if launch_browser:
            click.echo(click.style("--web does not work with sub-commands"))
            ctx.abort()
