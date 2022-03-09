import click


# NOTE: Named reference instead of docs to not clash with dataset autocomplete
@click.command("reference")
def reference():
    """View the Nucleus reference documentation in the browser"""
    click.launch("https://nucleus.scale.com/docs")
