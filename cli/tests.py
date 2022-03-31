import click
from rich.console import Console
from rich.live import Live
from rich.table import Column, Table
from rich.tree import Tree

from cli.client import init_client
from cli.helpers.nucleus_url import nucleus_url
from cli.helpers.web_helper import launch_web_or_invoke
from nucleus import NucleusAPIError
from nucleus.validate import (
    AvailableEvalFunctions,
    ScenarioTestMetric,
    ThresholdComparison,
)


@click.group("tests", invoke_without_command=True)
@click.option("--web", is_flag=True, help="Launch browser")
@click.pass_context
def tests(ctx, web):
    """Scenario Tests allow you to test your Models

    https://dashboard.scale.com/nucleus/scenario-tests
    """
    launch_web_or_invoke("scenario-tests", ctx, web, list_tests)


@tests.command("list")
def list_tests():
    """List all your Scenario Tests"""
    console = Console()
    with console.status("Finding your Scenario Tests", spinner="dots4"):
        client = init_client()
        scenario_tests = client.validate.scenario_tests
        table = Table(
            Column("id", overflow="fold", min_width=24),
            "Name",
            "slice_id",
            Column("url", overflow="fold"),
            title=":chart_with_upwards_trend: Scenario Tests",
            title_justify="left",
        )
        for ut in scenario_tests:
            table.add_row(ut.id, ut.name, ut.slice_id, nucleus_url(ut.id))
    console.print(table)


def format_criterion(
    criterion: ScenarioTestMetric, eval_functions: AvailableEvalFunctions
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
@click.argument("scenario-test-id", default=None, required=False)
@click.option(
    "--all", "-a", is_flag=True, help="View details about all Scenario Tests"
)
def describe_test(scenario_test_id, all):
    """View detailed information about a test or all tests"""
    console = Console()
    # scenario_test = client.validate.get_scenario_test(scenario_test_id)
    assert scenario_test_id or all, "Must pass a scenario_test_id or --all"
    client = init_client()
    scenario_tests = client.validate.scenario_tests
    if all:
        tree = Tree(":chart_with_upwards_trend: All Scenario Tests")
        with Live(
            "Fetching description of all Scenario Tests",
            vertical_overflow="visible",
        ) as live:
            for idx, ut in enumerate(scenario_tests):
                test_branch = tree.add(f"{idx}: Scenario Test")
                build_scenario_test_info_tree(client, ut, test_branch)
                live.update(tree)
    else:
        with console.status("Fetching Scenario Test information"):
            scenario_test = [
                ut for ut in scenario_tests if ut.id == scenario_test_id
            ][0]
            tree = Tree(":chart_with_upwards_trend: Scenario Test")
            build_scenario_test_info_tree(client, scenario_test, tree)
            console.print(tree)


def build_scenario_test_info_tree(client, scenario_test, tree):
    try:
        slc = client.get_slice(scenario_test.slice_id)
        info_branch = tree.add(":mag: Details")
        info_branch.add(f"id: '{scenario_test.id}'")
        info_branch.add(f"name: '{scenario_test.name}'")
        scenario_test_url = nucleus_url(scenario_test.id)
        info_branch.add(f"url: {scenario_test_url}")
        slice_url = nucleus_url(f"{slc.dataset_id}/{slc.slice_id}")
        slice_branch = tree.add(":cake: Slice")
        slice_branch.add(f"id: '{slc.id}'")
        slice_info = slc.info()
        slice_branch.add(f"name: '{slice_info['name']}'")
        slice_branch.add(f"len: {len(slc.items)}")
        slice_branch.add(f"url: {slice_url}")
        criteria = scenario_test.get_eval_functions()
        criteria_branch = tree.add(":crossed_flags: Criteria")
        for criterion in criteria:
            pretty_criterion = format_criterion(
                criterion, client.validate.eval_functions
            )
            criteria_branch.add(pretty_criterion)
    except NucleusAPIError as e:
        error_branch = tree.add(":x: Error")
        error_branch.add(f"detail: {str(e)}")
