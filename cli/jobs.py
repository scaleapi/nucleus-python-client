import click
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Column, Table

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
    table = Table(
        Column("id", overflow="fold", min_width=24),
        "status",
        "type",
        "created at",
        title=":satellite: Jobs",
        title_justify="left",
    )
    with Live(Spinner("dots4", text="Finding your Jobs!")) as live:
        all_jobs = client.jobs
        for job in all_jobs:
            table.add_row(
                job.job_id,
                job.job_last_known_status,
                job.job_type,
                job.job_creation_time,
            )
            live.update(table)
