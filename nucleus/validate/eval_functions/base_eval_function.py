import abc
from typing import Any, Dict

from ..constants import ThresholdComparison
from ..data_transfer_objects.eval_function import (
    EvalFunctionEntry,
    EvaluationCriterion,
)


class EvalFunctionConfig(abc.ABC):
    """Abstract base class for concrete implementations of EvalFunctionsConfigs

    Operating on this class with comparison operators produces an EvaluationCriterion
    """

    def __init__(self, eval_func_entry: EvalFunctionEntry):
        self.eval_func_entry = eval_func_entry
        self.id = eval_func_entry.id
        self.name = eval_func_entry.name
        self.eval_func_arguments: Dict[str, Any] = {}

    def __repr__(self):
        return f"<EvalFunction: name={self.name}, id={self.id}>"

    @classmethod
    @abc.abstractmethod
    def expected_name(cls) -> str:
        """Name to look for in the EvalFunctionDefinitions"""

    def __call__(self, **kwargs) -> "EvalFunctionConfig":
        """Adding call to prepare for being able to pass parameters to function

        Notes:
            Technically now you could do something like eval_function > 0.5 but we want it
            to look like eval_function() > 0.5 to support eval_function(parameters) > 0.5
            in the future
        """
        self.eval_func_arguments.update(**kwargs)
        return self

    def __gt__(self, other) -> EvaluationCriterion:
        return self._op_to_test_metric(ThresholdComparison.GREATER_THAN, other)

    def __ge__(self, other) -> EvaluationCriterion:
        return self._op_to_test_metric(
            ThresholdComparison.GREATER_THAN_EQUAL_TO, other
        )

    def __lt__(self, other) -> EvaluationCriterion:
        return self._op_to_test_metric(ThresholdComparison.LESS_THAN, other)

    def __le__(self, other) -> EvaluationCriterion:
        return self._op_to_test_metric(
            ThresholdComparison.LESS_THAN_EQUAL_TO, other
        )

    def _op_to_test_metric(self, comparison: ThresholdComparison, value):
        return EvaluationCriterion(
            eval_function_id=self.eval_func_entry.id,
            threshold_comparison=comparison,
            threshold=value,
            eval_func_arguments=self.eval_func_arguments,
        )
