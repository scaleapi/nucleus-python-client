import abc
from typing import List, Dict, Union, Type

from .data_transfer_objects.eval_function_condition import EvalFunctionCondition
from .constants import ThresholdComparison
from .eval_function import EvalFunctionDefinition

MEAN_AVG_PRECISION_NAME = "mean_average_precision_boxes"
IOU_NAME = "IOU"


class BaseEvalFunction(abc.ABC):
    def __init__(self, eval_func_response: EvalFunctionDefinition):
        self.eval_func_definition = eval_func_response

    @classmethod
    @abc.abstractmethod
    def from_def(cls, response: EvalFunctionDefinition):
        pass

    @classmethod
    @property
    @abc.abstractmethod
    def name(self):
        """Name to look for in the EvalFunctionDefinitions"""
        pass

    def __call__(self) -> "BaseEvalFunction":
        """ Adding call to prepare for being able to pass parameters to function

        Notes:
            Technically now you could do something like eval_function > 0.5 but we want it
            to look like eval_function() > 0.5 to support eval_function(parameters) > 0.5
            in the future
        """
        return self

    def __gt__(self, other) -> EvalFunctionCondition:
        return self._op_to_test_metric(ThresholdComparison.GREATER_THAN, other)

    def __ge__(self, other) -> EvalFunctionCondition:
        return self._op_to_test_metric(
            ThresholdComparison.GREATER_THAN_EQUAL_TO, other
        )

    def __lt__(self, other) -> EvalFunctionCondition:
        return self._op_to_test_metric(ThresholdComparison.LESS_THAN, other)

    def __le__(self, other) -> EvalFunctionCondition:
        return self._op_to_test_metric(
            ThresholdComparison.LESS_THAN_EQUAL_TO, other
        )

    def __eq__(self, other) -> EvalFunctionCondition:
        # TODO(gunnar): We should support comparison for custom evaluation functions
        raise NotImplementedError(
            "Equals has not been implemented yet, thresholding only supported."
        )

    def _op_to_test_metric(self, comparison: ThresholdComparison, value):
        return EvalFunctionCondition(
            eval_function_id=self.eval_func_definition.id,
            threshold_comparison=comparison,
            threshold=value,
        )


class IoU(BaseEvalFunction):
    @classmethod
    @property
    def name(self):
        return "IOU"

    @classmethod
    def from_def(cls, response: EvalFunctionDefinition):
        return cls(response)

    def __eq__(self, other) -> EvalFunctionCondition:
        return super().__eq__(other)


class MeanAveragePrecisionForBoxes(BaseEvalFunction):
    @classmethod
    def from_def(cls, response: EvalFunctionDefinition):
        return cls(response)

    def __eq__(self, other) -> EvalFunctionCondition:
        return super().__eq__(other)


EvalFunction = Union[Type[IoU], Type[MeanAveragePrecisionForBoxes]]


class AvailableEvaluationFunctions:
    """Collection class that acts as a common entrypoint to access evaluation function.
    """

    def __init__(self, available_functions: List[EvalFunctionDefinition]):
        """Class that holds all evaluation functions"""
        # Names taken from: https://github.com/scaleapi/models/pull/2100/files
        self.name_to_def = {f.name: f for f in available_functions}
        self.name_to_function = {}
        # TODO(gunnar): Extract common function that assigns to self.name_to_function and checks for existence
        self.iou = self._assign_eval_function_if_defined(self.name_to_def, IoU)
        self.mean_average_precision_for_boxes = (
            self._assign_eval_function_if_defined(
                self.name_to_def, MeanAveragePrecisionForBoxes
            )
        )

    def __repr__(self):
        return (
            f"<AvailableEvaluationFunctions: standard:{list(self.name_to_function.keys())}, "
            f"custom: {set(self.name_to_def.keys())-set(self.name_to_function.keys())}>"
        )

    def all(self) -> Dict[str, BaseEvalFunction]:
        return self.name_to_function

    def _assign_eval_function_if_defined(
        self, name_to_definition: dict, eval_function_constructor: EvalFunction
    ):
        # TODO(gunnar): Too convoluted .. simplify
        if eval_function_constructor.name in name_to_definition:
            definition = name_to_definition[eval_function_constructor.name]
            eval_function = eval_function_constructor(definition)
            self.name_to_function[eval_function_constructor.name] = eval_function
            return eval_function
        else:
            return None
