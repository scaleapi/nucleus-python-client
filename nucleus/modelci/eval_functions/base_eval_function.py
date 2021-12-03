import abc

from ..data_transfer_objects.eval_function import (
    EvaluationCriteria,
)
from ..data_transfer_objects.eval_function import EvalFunctionEntry

from ..constants import ThresholdComparison


class BaseEvalFunction(abc.ABC):
    """Abstract base class for concrete implementations of EvalFunctions

    Operating on this class with comparison operators produces an EvaluationCriteria
    """

    def __init__(self, eval_func_entry: EvalFunctionEntry):
        self.eval_func_entry = eval_func_entry

    @classmethod
    @property
    @abc.abstractmethod
    def name(cls) -> str:
        """Name to look for in the EvalFunctionDefinitions"""

    def __call__(self) -> "BaseEvalFunction":
        """Adding call to prepare for being able to pass parameters to function

        Notes:
            Technically now you could do something like eval_function > 0.5 but we want it
            to look like eval_function() > 0.5 to support eval_function(parameters) > 0.5
            in the future
        """
        return self

    def __gt__(self, other) -> EvaluationCriteria:
        return self._op_to_test_metric(ThresholdComparison.GREATER_THAN, other)

    def __ge__(self, other) -> EvaluationCriteria:
        return self._op_to_test_metric(
            ThresholdComparison.GREATER_THAN_EQUAL_TO, other
        )

    def __lt__(self, other) -> EvaluationCriteria:
        return self._op_to_test_metric(ThresholdComparison.LESS_THAN, other)

    def __le__(self, other) -> EvaluationCriteria:
        return self._op_to_test_metric(
            ThresholdComparison.LESS_THAN_EQUAL_TO, other
        )

    def _op_to_test_metric(self, comparison: ThresholdComparison, value):
        return EvaluationCriteria(
            eval_function_id=self.eval_func_entry.id,
            threshold_comparison=comparison,
            threshold=value,
        )
