import click
from rich.console import Console
from rich.live import Live
from rich.table import Column, Table
from rich.tree import Tree

from cli.client import init_client
from cli.helpers.nucleus_url import nucleus_url
from cli.helpers.web_helper import launch_web_or_invoke
from nucleus import NucleusAPIError
from nucleus.modelci import (
    AvailableEvalFunctions,
    ThresholdComparison,
    UnitTestMetric,
)


@click.group("tests", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def tests(ctx, web):
    """Scenario Tests allow you to test your Models

    https://dashboard.scale.com/nucleus/unit-tests
    """
    launch_web_or_invoke("unit-tests", ctx, web, list_tests)


@tests.command("list")
def list_tests():
    """List all your Scenario Tests"""
    console = Console()
    with console.status("Finding your Scenario Tests", spinner="dots4"):
        client = init_client()
        unit_tests = client.modelci.list_unit_tests()
        table = Table(
            Column("id", overflow="fold", min_width=24),
            "Name",
            "slice_id",
            Column("url", overflow="fold"),
            title=":chart_with_upwards_trend: Scenario Tests",
            title_justify="left",
        )
        for ut in unit_tests:
            table.add_row(ut.id, ut.name, ut.slice_id, nucleus_url(ut.id))
    console.print(table)


def format_criterion(
    criterion: UnitTestMetric, eval_functions: AvailableEvalFunctions
):
    op_map = {
        ThresholdComparison.GREATER_THAN: ">",
        ThresholdComparison.GREATER_THAN_EQUAL_TO: ">=",
        ThresholdComparison.LESS_THAN: "<",
        ThresholdComparison.LESS_THAN_EQUAL_TO: "<=",
    }
    eval_function_name = eval_functions.from_id(
        criterion.eval_function_id
    ).name
    return f"{eval_function_name} {op_map[criterion.threshold_comparison]} {criterion.threshold}"
    pass


@tests.command("describe")
@click.argument("unit-test-id", default=None, required=False)
@click.option(
    "--all", "-a", is_flag=True, help="View details about all Scenario Tests"
)
def describe_test(unit_test_id, all):
    """View detailed information about a test or all tests"""
    console = Console()
    # unit_test = client.modelci.get_unit_test(unit_test_id)
    assert unit_test_id or all, "Must pass a unit_test_id or --all"
    client = init_client()
    unit_tests = client.modelci.list_unit_tests()
    if all:
        tree = Tree(":chart_with_upwards_trend: All Scenario Tests")
        with Live(
            "Fetching description of all Scenario Tests",
            vertical_overflow="visible",
        ) as live:
            for idx, ut in enumerate(unit_tests):
                test_branch = tree.add(f"{idx}: Unit Test")
                build_unit_test_info_tree(client, ut, test_branch)
                live.update(tree)
    else:
        with console.status("Fetching Scenario Test information"):
            unit_test = [ut for ut in unit_tests if ut.id == unit_test_id][0]
            tree = Tree(":chart_with_upwards_trend: Unit Test")
            build_unit_test_info_tree(client, unit_test, tree)
            console.print(tree)


def build_unit_test_info_tree(client, unit_test, tree):
    try:
        slc = client.get_slice(unit_test.slice_id)
        info_branch = tree.add(":mag: Details")
        info_branch.add(f"id: '{unit_test.id}'")
        info_branch.add(f"name: '{unit_test.name}'")
        unit_test_url = nucleus_url(unit_test.id)
        info_branch.add(f"url: {unit_test_url}")
        slice_url = nucleus_url(f"{slc.dataset_id}/{slc.slice_id}")
        slice_branch = tree.add(":cake: Slice")
        slice_branch.add(f"id: '{slc.id}'")
        slice_info = slc.info()
        slice_branch.add(f"name: '{slice_info['name']}'")
        slice_branch.add(f"len: {len(slc.items)}")
        slice_branch.add(f"url: {slice_url}")
        criteria = unit_test.get_criteria()
        criteria_branch = tree.add(":crossed_flags: Criteria")
        for criterion in criteria:
            pretty_criterion = format_criterion(
                criterion, client.modelci.eval_functions
            )
            criteria_branch.add(pretty_criterion)
    except NucleusAPIError as e:
        error_branch = tree.add(":x: Error")
        error_branch.add(f"detail: {str(e)}")
