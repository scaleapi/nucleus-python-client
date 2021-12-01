from typing import List, Dict, Union, Type

from nucleus.modelci.eval_functions.base_eval_function import BaseEvalFunction
from nucleus.modelci.eval_function import EvalFunctionEntry

MEAN_AVG_PRECISION_NAME = "mean_average_precision_boxes"
IOU_NAME = "IOU"


class IoU(BaseEvalFunction):
    """Intersection over union for all bounding boxes
    """
    @classmethod
    @property
    def name(self) -> str:
        return "IOU"


class MeanAveragePrecisionForBoxes(BaseEvalFunction):
    """Mean average precision for boxes fitting the provided parameters"""
    @classmethod
    @property
    def name(self) -> str:
        return "mean_average_precision_for_boxes"


EvalFunction = Union[Type[IoU], Type[MeanAveragePrecisionForBoxes]]


class AvailableEvalFunctions:
    """Collection class that acts as a common entrypoint to access evaluation functions. Standard evaluation functions
    provided by Scale are attributes of this class.

    Examples:
        e = client.modelci.eval_functions
        unit_test_metrics = [e.iou() > 5, e.map() > 0.95, e.custom["customer_function"]() == True]
    """

    def __init__(self, available_functions: List[EvalFunctionEntry]):
        # Names taken from: https://github.com/scaleapi/models/pull/2100/files
        self._name_to_entry: Dict[str, EvalFunctionEntry] = {f.name: f for f in available_functions}
        self._name_to_function: Dict[str, EvalFunction] = {}
        # TODO(gunnar): Extract common function that assigns to self.name_to_function and checks for existence
        self.iou = self._assign_eval_function_if_defined(self._name_to_entry, IoU)
        self.mean_average_precision_for_boxes = (
            self._assign_eval_function_if_defined(
                self._name_to_entry, MeanAveragePrecisionForBoxes
            )
        )

    def __repr__(self):
        """Standard functions are ones Scale provides and custom ones customer defined"""
        return (
            f"<AvailableEvaluationFunctions: standard:{list(self._name_to_function.keys())}, "
            f"custom: {set(self._name_to_entry.keys()) - set(self._name_to_function.keys())}>"
        )

    def all(self) -> Dict[str, EvalFunction]:
        return self._name_to_function

    def _assign_eval_function_if_defined(
        self, name_to_definition: dict, eval_function_constructor: EvalFunction
    ):
        """Helper function for book-keeping and assignment of standard Scale provided functions that are accessible
        via attribute access
        """
        # TODO(gunnar): Too convoluted .. simplify
        if eval_function_constructor.name in name_to_definition:
            definition = name_to_definition[eval_function_constructor.name]
            eval_function = eval_function_constructor(definition)
            self._name_to_function[eval_function_constructor.name] = eval_function
            return eval_function
        else:
            return None
