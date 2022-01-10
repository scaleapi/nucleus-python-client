import click


@click.command("docs")
def docs():
    """ View the Nucleus documentation in the browser"""
    click.launch("https://nucleus.scale.com/docs")
