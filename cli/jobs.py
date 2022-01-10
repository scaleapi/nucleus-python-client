import click
from rich.live import Live
from rich.spinner import Spinner

from cli.client import init_client
from cli.helpers.web_helper import launch_web_or_invoke


@click.group("jobs", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def jobs(ctx, web):
    """Jobs are a wrapper around various long-running tasks withing Nucleus

    https://dashboard.scale.com/nucleus/jobs
    """
    launch_web_or_invoke("jobs", ctx, web, list_jobs)


@jobs.command("list")
def list_jobs():
    """List all of your Jobs"""
    client = init_client()
    with Live(Spinner("dots4", text="Finding your Jobs!")) as live:
        all_jobs = client.jobs
        live.update(all_jobs)
