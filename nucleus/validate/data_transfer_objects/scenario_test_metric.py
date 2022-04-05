from nucleus.pydantic_base import ImmutableModel


class AddScenarioTestFunction(ImmutableModel):
    """Data transfer object to add a scenario test."""

    scenario_test_name: str
    eval_function_id: str
