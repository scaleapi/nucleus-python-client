from pydantic import BaseModel

from nucleus.modelci import ThresholdComparison


class AddUnitTestMetric(BaseModel):
    """ Data transfer object to add a unit test."""
    unit_test_name: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison