import click
from rich.live import Live
from rich.spinner import Spinner

from cli.client import init_client
from cli.helpers.web_helper import launch_web_or_show_help


@click.group("jobs", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def jobs(ctx, web):
    """Jobs are a wrapper around various long-running tasks withing Nucleus

    https://dashboard.scale.com/nucleus/jobs
    """
    launch_web_or_show_help("jobs", ctx, web)


@jobs.command("list")
def list_jobs():
    """List all of your Jobs"""
    client = init_client()
    with Live(Spinner("dots4", text="Finding your Jobs!")) as live:
        all_jobs = client.jobs
        all_jobs_2 = client.list_jobs(show_completed=True)
        live.update(all_jobs)
